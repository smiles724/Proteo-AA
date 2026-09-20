"""The real PXDesign backbone driver, via ``pxdesign_train``.

Provides what :class:`pxf.couple.controller.BackboneDenoiser` needs: one
denoising evaluation of the official PXDesign diffusion module, returning both
the denoised coordinates and the token features ``a_token`` the coupling reads,
and accepting a residual to inject before the atom decoder.

Why not PXDesign's own runner: it takes a target plus a binder to design and
emits CIF files. It cannot denoise a given monomer, and its sampler is
incompatible with the Protenix revision this repo pins. ``pxdesign_train``
supplies the missing half -- a featurizer that accepts an arbitrary structure and
scrubs the design region to backbone -- while the *model* remains official
PXDesign loaded from the published donor checkpoint.

The donor loads with zero missing and zero unexpected keys, so the backbone here
is the released network, not a reimplementation.
"""

import inspect
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

# Protenix's gradient checkpointing reaches for torch.utils.checkpoint, which
# torch does not import eagerly; without this the first denoise raises
# AttributeError: module 'torch.utils' has no attribute 'checkpoint'.
import torch.utils.checkpoint  # noqa: F401

from pxf import atom37
from pxf.backbone import proteoaa
from pxf.couple.controller import Topology
from pxf.couple.pxdesign_iface import (
    BackboneTap,
    Conditioning,
    conditioning_widths,
    token_feature_dim,
)

logger = logging.getLogger("pxf.backbone.driver")

DONOR_MODEL_NAME = "pxdesign_v0.1.0"
# The featurizer defaults that make a monomer's whole chain the design region.
MONOMER_DATASET = dict(
    compute_sidechain=True,
    backbone_only_binder=True,
    inference_safe_binder=True,
    ref_pos_augment=False,
    hotspot_force_zero_prob=0.0,
    aa_mask_mode="all",
    aa_mask_prob=1.0,
    max_crop_retries=1,
    max_binder_fraction=1.0,
)


def load_backbone_model(donor_checkpoint, *, device=None, proteoaa_root=None):
    """Official ``ProtenixDesign`` with the published donor weights.

    Returns ``(model, configs, record)``. The load is strict in effect: any
    missing or unexpected tensor is refused, because a partially initialized
    backbone still emits plausible coordinates.
    """
    bundle = proteoaa.load(proteoaa_root)
    from protenix.config.config import parse_configs
    from pxdesign.model.pxdesign import ProtenixDesign

    configs = parse_configs(bundle.configs.training_configs, arg_str="")
    model = ProtenixDesign(configs)

    path = Path(donor_checkpoint).resolve()
    if not path.is_file():
        raise ValueError(f"PXDesign donor checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    tensors = state.get("model", state.get("state_dict", state))
    tensors = {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in tensors.items()
    }
    missing, unexpected = model.load_state_dict(tensors, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"Donor {path.name} does not match the backbone exactly: "
            f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}. A "
            "partially loaded backbone still produces plausible coordinates, so "
            "this is refused rather than warned about."
        )
    model.eval()
    model.requires_grad_(False)
    if device is not None:
        model.to(device)
    from pxf import provenance

    c_s, c_z = conditioning_widths(model)
    record = dict(
        backend="pxdesign",
        model_name=DONOR_MODEL_NAME,
        weights=provenance.weight_record(path),
        c_token=token_feature_dim(model),
        c_s=c_s,
        c_z=c_z,
        sigma_data=float(model.diffusion_module.sigma_data),
        driver="pxdesign_train",
        proteoaa=bundle.record(),
    )
    return model, configs, record


def disable_activation_checkpointing(module):
    """Clear every ``blocks_per_ckpt`` in the tree; returns how many were cleared."""
    cleared = 0
    for child in module.modules():
        if getattr(child, "blocks_per_ckpt", None):
            child.blocks_per_ckpt = None
            cleared += 1
    return cleared


def _denoiser_accepts(model):
    """Which optional cache arguments this Protenix revision requires."""
    signature = inspect.signature(type(model.diffusion_module).forward)
    return [name for name in ("pair_z", "p_lm", "c_l") if name in signature.parameters]


class PXDesignBackboneDriver:
    """One-step PXDesign denoising with a token-feature tap and feedback port.

    Satisfies the :class:`~pxf.couple.controller.BackboneDenoiser` protocol once
    bound to a target's conditioning via :meth:`bind`.
    """

    def __init__(
        self, model, *, chunk_size=None, inplace_safe=False, activation_checkpointing=False
    ):
        self.model = model
        self.activation_checkpointing = bool(activation_checkpointing)
        self.blocks_cleared = 0
        if not activation_checkpointing:
            # Activation checkpointing recomputes the forward during backward, and
            # the feedback injection hook makes the recomputation diverge from the
            # original pass:
            #   CheckpointError: A different number of tensors was saved during
            #   the original forward and recomputation.
            # The adapters are small and the backbone is frozen, so the memory
            # saving is not needed; correctness is.
            self.blocks_cleared = disable_activation_checkpointing(model)
        self.chunk_size = chunk_size
        self.inplace_safe = bool(inplace_safe)
        self.c_token = token_feature_dim(model)
        # The early injection site's widths, read off the loaded conditioning
        # module. Neither equals c_token (384 and 128 against 768 on this
        # donor), which is why a conditioner is sized from these rather than
        # from the adapter's backbone dimension.
        self.c_s, self.c_z = conditioning_widths(model)
        self.sigma_data = float(model.diffusion_module.sigma_data)
        # Protenix 2.0 made pair_z/p_lm/c_l required parameters even though all
        # three are optional caches computed on demand; older revisions do not
        # have them at all. Pass exactly what this build expects.
        self._cache_args = {name: None for name in _denoiser_accepts(model)}

    def identity(self):
        """What a run should record about how the backbone was driven.

        ``blocks_cleared`` in particular: "checkpointing is off" is a default,
        and a default is not evidence. A run whose feedback silently trained
        against a recomputed forward would be hard to diagnose after the fact,
        so the count of ``blocks_per_ckpt`` flags actually cleared is recorded
        rather than assumed.
        """
        return dict(
            c_token=self.c_token,
            c_s=self.c_s,
            c_z=self.c_z,
            sigma_data=self.sigma_data,
            activation_checkpointing=self.activation_checkpointing,
            blocks_cleared=self.blocks_cleared,
            chunk_size=self.chunk_size,
            inplace_safe=self.inplace_safe,
            cache_args=sorted(self._cache_args),
        )

    @property
    def device(self):
        return next(self.model.parameters()).device

    def prepare_features(self, feature_dict):
        """Add the derived features the diffusion module expects.

        ``pxdesign_train``'s featurizer emits the reference-atom features but not
        ``relp`` or the atom-pair block (``d_lm``, ``v_lm``, ``pad_info``), which
        Protenix computes on the fly. Both steps are Protenix's own, in the order
        Proteo-AA uses; without them the encoder raises ``KeyError: 'd_lm'``.
        """
        from protenix.model.protenix import update_input_feature_dict

        prepared = self.model.diffusion_module.diffusion_conditioning.relpe.generate_relp(
            feature_dict
        )
        return update_input_feature_dict(prepared)

    def conditioning(self, feature_dict, *, prepare=True):
        """Compute the per-target conditioning once; reused across evaluations.

        The dict is moved onto the model's device first, so the derived features
        ``prepare_features`` builds land there too rather than being recomputed
        on the CPU and then mismatching.
        """
        feature_dict = move_to_device(feature_dict, self.device)
        if prepare:
            feature_dict = self.prepare_features(feature_dict)
        return Conditioning.build(self.model, feature_dict, chunk_size=self.chunk_size)

    def denoise_direct(self, conditioning, x_noisy, sigma):
        """One diffusion-module call, no hooks, coordinates only.

        :meth:`bind` exists to capture ``a_token`` and to inject feedback, and it
        does both with temporary forward hooks. A training path that wants
        gradients through the backbone needs neither, and the hooks are actively
        in the way there: they are what makes activation-checkpointing
        recomputation diverge (``CheckpointError: A different number of tensors
        was saved``), so a hook-free forward is the prerequisite for ever turning
        checkpointing back on.

        Gradients flow to whichever backbone parameters require them; nothing
        here detaches.
        """
        sigma = torch.as_tensor(
            sigma, device=x_noisy.device, dtype=x_noisy.dtype
        ).reshape(-1)
        return self.model.diffusion_module(
            x_noisy=x_noisy,
            t_hat_noise_level=sigma,
            input_feature_dict=conditioning.input_feature_dict,
            s_inputs=conditioning.s_inputs,
            s_trunk=conditioning.s_trunk,
            z_trunk=conditioning.z_trunk,
            chunk_size=self.chunk_size,
            inplace_safe=self.inplace_safe,
            **self._cache_args,
        )

    def bind(self, conditioning, *, tap=None):
        """A ``BackboneDenoiser`` closed over one target's conditioning."""
        owned_tap = tap or BackboneTap(self.model.diffusion_module)

        def denoise(x_noisy, sigma, *, feedback=None):
            installed = bool(owned_tap._handles)
            if not installed:
                owned_tap.install()
            try:
                owned_tap.feedback = feedback
                sigma = torch.as_tensor(
                    sigma, device=x_noisy.device, dtype=x_noisy.dtype
                ).reshape(-1)
                x_denoised = self.model.diffusion_module(
                    x_noisy=x_noisy,
                    t_hat_noise_level=sigma,
                    input_feature_dict=conditioning.input_feature_dict,
                    s_inputs=conditioning.s_inputs,
                    s_trunk=conditioning.s_trunk,
                    z_trunk=conditioning.z_trunk,
                    chunk_size=self.chunk_size,
                    inplace_safe=self.inplace_safe,
                    **self._cache_args,
                )
                return x_denoised, owned_tap.a_token
            finally:
                if not installed:
                    owned_tap.remove()

        denoise.tap = owned_tap
        return denoise


# ---- featurization ---------------------------------------------------------


@dataclass
class FeaturizedStructure:
    """One featurized structure, ready for the coupled cycle."""

    sample_id: str
    feature_dict: dict
    label_dict: dict
    topology: Topology
    aatype: torch.Tensor  # [L] native identities (teacher-forced)
    design_mask: torch.Tensor  # [L] bool
    backbone_target: torch.Tensor  # [N_atom, 3] native coordinates
    num_tokens: int = 0

    def to(self, device):
        """Move every tensor onto ``device``, leaving the string columns alone.

        ``pxdesign_train``'s featurizer emits CPU tensors and five numpy string
        columns (``structure_atom_name`` and friends). The model is wherever it
        was loaded, so without this the first ``F.linear`` inside the condition
        embedder fails on mixed devices -- and it fails there rather than at the
        obvious place, because everything up to the first weight multiply is
        pure indexing that tolerates a CPU index.
        """
        from dataclasses import replace

        device = torch.device(device)
        return replace(
            self,
            feature_dict=move_to_device(self.feature_dict, device),
            label_dict=move_to_device(self.label_dict, device),
            topology=self.topology.to(device),
            aatype=self.aatype.to(device),
            design_mask=self.design_mask.to(device),
            backbone_target=self.backbone_target.to(device),
        )


def move_to_device(mapping, device):
    """Tensors in a feature dict onto ``device``; anything else passed through."""
    return {
        key: (value.to(device) if torch.is_tensor(value) else value)
        for key, value in mapping.items()
    }


def featurize_structures(
    cif_paths,
    *,
    crop_size=256,
    source_name="pxf",
    proteoaa_root=None,
    binder_chain_ids=None,
    parser_dataset="WeightedPDB",
    **overrides,
):
    """Featurize structures through ``pxdesign_train``, one item per path.

    Each path gets its own dataset so a failure is isolated to that structure
    rather than aborting the whole set.

    ``parser_dataset`` selects Protenix's parser. ``"WeightedPDB"`` (the
    default) expects a full mmCIF and reads ``pdbx_struct_assembly`` to build
    the bioassembly. ``"Distillation"`` treats the file as *already assembled*
    and is what a stripped or predicted structure needs -- the
    proteina-complexa mirror's CIFs carry no assembly category, so the default
    parser raises ``KeyError: 'pdbx_struct_assembly'``, which the provider
    re-tags and the dataset then reports as "parsed without
    atom_array/token_array". That message names a symptom two layers from the
    cause, which is why this is a parameter rather than a guess.
    """
    bundle = proteoaa.load(proteoaa_root)
    settings = dict(MONOMER_DATASET, **overrides)
    items = []
    for index, path in enumerate(cif_paths):
        provider = bundle.CifFileProvider(
            cif_paths=[str(path)],
            binder_chain_ids=[binder_chain_ids[index]] if binder_chain_ids else None,
            dataset=parser_dataset,
        )
        dataset = bundle.DesignSourceDataset(
            provider, source_name=source_name, crop_size=int(crop_size), **settings
        )
        items.append((provider.sample_id(0), dataset))
    return items


def to_featurized(sample_id, item):
    """Turn a ``DesignSourceDataset`` item into a :class:`FeaturizedStructure`."""
    feature_dict = item["input_feature_dict"]
    label_dict = item.get("label_dict", {})
    num_tokens = int(feature_dict["token_index"].reshape(-1).shape[0])
    aatype = feature_dict["aa_clean"].reshape(-1).long()[:num_tokens]
    design = feature_dict["design_token_mask"].reshape(-1).bool()[:num_tokens]
    # The design region's residue names are scrubbed to GLY by the featurizer, so
    # identities come from aa_clean -- the native sequence -- not from
    # structure_res_name, which would teacher-force glycine everywhere.
    from fampnn.data import residue_constants as rc

    # aa_clean uses -100 for tokens that are not amino acids at all -- ligands,
    # ions, waters. `int(a) < 20` is true for -100 and AA_ORDER[-100] raises
    # IndexError, so the bound has to be two-sided. CASP14 targets are
    # protein-only, which is why this only appears on PDB entries with hetero
    # groups (101m: 154 residues plus 49 HEM tokens).
    non_protein = int(((aatype < 0) | (aatype >= 20)).sum())
    per_token = [
        rc.restype_1to3[atom37.AA_ORDER[int(a)]] if 0 <= int(a) < 20 else "UNK"
        for a in aatype
    ]
    tokens = feature_dict["atom_to_token_idx"].reshape(-1).long()
    res_names = [per_token[int(t)] for t in tokens]
    topology = Topology(
        atom_names=list(feature_dict["structure_atom_name"]),
        atom_to_token_idx=tokens,
        num_tokens=num_tokens,
        res_names=res_names,
        residue_index=feature_dict.get("residue_index"),
        chain_index=feature_dict.get("asym_id"),
    )
    if non_protein:
        # Not raised here: the structure is still usable for backbone work. But
        # the side-chain module only accepts the canonical twenty, and
        # _native_atom37's sequence-equality check would compare a
        # protein-only parse against this token count. Say so where it is
        # visible rather than failing later with a length mismatch.
        logger.warning(
            "%s: %d of %d tokens are not amino acids (ligands/ions/waters). The "
            "coupling path needs a protein-only entry; its side-chain targets "
            "will not align.",
            sample_id,
            non_protein,
            num_tokens,
        )
    return FeaturizedStructure(
        sample_id=sample_id,
        feature_dict=feature_dict,
        label_dict=label_dict,
        topology=topology,
        aatype=aatype,
        design_mask=design,
        backbone_target=label_dict.get("coordinate"),
        num_tokens=num_tokens,
    )
