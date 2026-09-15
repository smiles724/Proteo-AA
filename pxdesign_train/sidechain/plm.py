"""Frozen ESM-2 sequence conditioning, in APM's form.

APM (`apm/models/folding_model.py`, `apm/models/side_chain_model.py`) conditions
its packer on a protein language model exactly the way ESMFold does:

    plm_s = (plm_s_combine.softmax(0) @ all_layer_representations)   # [B, L, H]
    plm_s = plm_s_mlp(plm_s)                                          # -> c_node
    node_embed += plm_s

`plm_s_combine` is a LEARNED softmax over the layer axis, which is why all
layers have to be available at once -- and therefore why the PLM has to run
inside the training job rather than being cached. Caching would need
(n_layers+1) x L x 1280 per chain; for the 47,622-chain monomer set that is
tens of terabytes. Measured cost of running it instead, ESM-2 650M in bf16 on
one H200 at L=384: 12-17 ms per forward and 2.7 GB, against ~850 ms per
optimizer step. That is the whole reason this is affordable.

The default model is APM's: `faESM2-650M` in `apm/configs/base.yaml`, i.e.
ESM-2 650M (33 layers, width 1280). FAESM is a flash-attention drop-in for
speed; this uses stock fair-esm, which produces the same representations.

WHAT IS AND IS NOT A PARAMETER OF THIS MODEL. `plm_s_combine` and `plm_s_mlp`
are trained (they are APM's too). The ESM itself is frozen, and it is held by a
PLAIN OBJECT rather than as a submodule, deliberately: `stage4.apply_phase`
marks everything under `sidechain_module.` trainable, so a registered ESM would
silently become 651M trainable parameters with Adam state to match. A test pins
that the state dict stays clean.
"""
import argparse
import logging
from typing import Optional

import torch
import torch.nn as nn

from .instantiate import STD_AA_3
from .ipa import Linear

logger = logging.getLogger(__name__)

# ESM-2 token id for each type in STD_AA_3 order, then the fallback for
# non-canonical/unknown. Identical to the first 21 entries of APM's
# `folding_model.tk_mapping`; verified against fair-esm's alphabet, which also
# confirms STD_AA_3 is in openfold's restype order for the 20 canonical types.
ESM2_TOKENS = torch.tensor(
    [5, 10, 17, 13, 23, 16, 9, 6, 21, 12, 4, 15, 20, 18, 14, 8, 11, 22, 19, 7],
    dtype=torch.long,
)
ESM2_MASK_TOKEN = 32   # <mask>, what APM maps unknown residues to
ESM2_CLS_TOKEN = 0
ESM2_EOS_TOKEN = 2
ESM2_PAD_TOKEN = 1

DEFAULT_ESM2_NAME = "esm2_t33_650M_UR50D"
DEFAULT_ESM2_LAYERS = 33
DEFAULT_ESM2_DIM = 1280


class FrozenESM2:
    """Frozen ESM-2 returning ALL layer representations.

    Not an `nn.Module` on purpose -- see the module docstring. Loads lazily so
    CPU tests and config validation never touch the 2.5 GB checkpoint.
    """

    def __init__(self, checkpoint: str, dtype: torch.dtype = torch.bfloat16):
        self.checkpoint = str(checkpoint)
        self.dtype = dtype
        self._model = None          # plain attribute, never registered
        self._device = None
        self.num_layers = DEFAULT_ESM2_LAYERS
        self.embed_dim = DEFAULT_ESM2_DIM

    def _ensure(self, device):
        if self._model is not None and self._device == device:
            return
        import esm  # fair-esm; imported here so the dependency is runtime-only

        # fair-esm's checkpoints pickle an argparse.Namespace, which torch>=2.6
        # refuses under the default weights_only=True. These are the official
        # Meta weights fetched from dl.fbaipublicfiles.com; allow that one class
        # rather than turning weights_only off wholesale.
        torch.serialization.add_safe_globals([argparse.Namespace])
        model, _alphabet = esm.pretrained.load_model_and_alphabet_local(self.checkpoint)
        model = model.eval().to(device=device, dtype=self.dtype)
        model.requires_grad_(False)
        self.num_layers = int(model.num_layers)
        self.embed_dim = int(model.embed_dim)
        self._model = model
        self._device = device
        logger.info(
            "frozen ESM-2 loaded: %s, %d layers, width %d, dtype %s",
            self.checkpoint, self.num_layers, self.embed_dim, self.dtype,
        )

    def tokens(self, type_idx: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """[B, L] residue types -> [B, L+2] ESM tokens with BOS/EOS."""
        table = ESM2_TOKENS.to(type_idx.device)
        canonical = (type_idx >= 0) & (type_idx < len(STD_AA_3))
        body = torch.where(canonical, table[type_idx.clamp(0, len(STD_AA_3) - 1)],
                           torch.full_like(type_idx, ESM2_MASK_TOKEN))
        if mask is not None:
            # A position the packer does not own is still a real residue of the
            # chain the PLM is reading, so it stays in the sequence. Only
            # genuinely absent rows become padding.
            body = torch.where(mask.bool(), body, torch.full_like(body, ESM2_PAD_TOKEN))
        B = body.shape[0]
        bos = body.new_full((B, 1), ESM2_CLS_TOKEN)
        eos = body.new_full((B, 1), ESM2_EOS_TOKEN)
        return torch.cat([bos, body, eos], dim=1)

    @torch.no_grad()
    def representations(self, type_idx: torch.Tensor,
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """[B, L] types -> [B, L, num_layers + 1, embed_dim], BOS/EOS stripped."""
        self._ensure(type_idx.device)
        toks = self.tokens(type_idx, mask)
        out = self._model(toks, repr_layers=list(range(self.num_layers + 1)),
                          return_contacts=False)
        reps = torch.stack(
            [out["representations"][i] for i in range(self.num_layers + 1)], dim=2)
        return reps[:, 1:-1]


class PLMConditioner(nn.Module):
    """APM's `plm_s_combine` + `plm_s_mlp`, verbatim in form."""

    def __init__(self, num_layers: int, embed_dim: int, c_node: int):
        super().__init__()
        # Zero init -> a uniform softmax over layers at step 0, which is what
        # APM starts from (`torch.zeros(PLM_info[0] + 1)`).
        self.plm_s_combine = nn.Parameter(torch.zeros(num_layers + 1))
        self.plm_s_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            Linear(embed_dim, c_node, init="relu"),
            nn.ReLU(),
            Linear(c_node, c_node, init="final"),
        )

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        """[B, L, n_layer+1, H] -> [B, L, c_node]."""
        weights = self.plm_s_combine.softmax(0).to(representations.dtype)
        combined = (weights[None, None, :, None] * representations).sum(dim=2)
        return self.plm_s_mlp(combined.float())


__all__ = ["FrozenESM2", "PLMConditioner", "ESM2_TOKENS", "DEFAULT_ESM2_NAME"]
