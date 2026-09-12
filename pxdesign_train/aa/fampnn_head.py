"""Strictly loaded official sequence network; no FaMPNN packing calls."""
from pathlib import Path
import hashlib
import subprocess
import torch
from torch import nn
from .atom_mapping import AA_ORDER, MAPPING_VERSION, assert_upstream_mapping

UPSTREAM_REVISION = "aaf788b1502ad95d5c5a84455cfc53f2544f3b45"


class FaMPNNHead(nn.Module):
    def __init__(self, checkpoint_path=None, source_revision=UPSTREAM_REVISION, identity=None):
        super().__init__()
        from fampnn.model import fampnn as upstream_module
        from fampnn.data import residue_constants as rc
        from fampnn.model.sd_model import SeqDenoiser
        from omegaconf import OmegaConf
        assert_upstream_mapping(rc)
        root = Path(upstream_module.__file__).resolve().parents[2]
        actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if actual != source_revision or actual != UPSTREAM_REVISION:
            raise ValueError(f"FaMPNN source revision mismatch: {actual}")
        if subprocess.check_output(["git", "-C", str(root), "diff", "--name-only", "HEAD", "--", "fampnn"], text=True).strip():
            raise ValueError("FaMPNN tracked source has local modifications")
        if identity is None:
            path = Path(checkpoint_path).resolve()
            with path.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            full = SeqDenoiser(checkpoint["model_cfg"])
            full.load_state_dict(checkpoint["state_dict"], strict=True)
            identity = dict(backend="fampnn", upstream_revision=actual,
                checkpoint_sha256=checksum, model_config=OmegaConf.to_container(checkpoint["model_cfg"], resolve=True),
                mapping_version=MAPPING_VERSION)
        else:
            from pxdesign_train.checkpoints import plain_config
            identity = plain_config(identity)
            if identity["upstream_revision"] != actual or identity["mapping_version"] != MAPPING_VERSION:
                raise ValueError("Saved FAMPNN source/mapping mismatch")
            full = SeqDenoiser(OmegaConf.create(identity["model_config"]))
        self.sequence_network = full.denoiser.seq_design_module
        if self.sequence_network.autoregressive:
            raise ValueError("Stage IV block decoding requires the released non-autoregressive configuration")
        self.register_buffer("canonical_indices", torch.tensor([rc.restype_order[a] for a in AA_ORDER]))
        self.identity = dict(identity)

    def forward(self, *, denoised_coords, aatype_noised, seq_mask,
                atom_mask_noised, residue_index, chain_encoding):
        lead = denoised_coords.shape[:-3]
        length = denoised_coords.shape[-3]
        if len(lead) != 2:
            raise ValueError("FaMPNN boundary requires [batch,sample,residue,37,3]")
        inputs = dict(denoised_coords=denoised_coords.float().reshape(-1, length, 37, 3),
            aatype_noised=aatype_noised.reshape(-1, length), seq_mask=seq_mask.float().reshape(-1, length),
            atom_mask_noised=atom_mask_noised.float().reshape(-1, length, 37),
            residue_index=residue_index.reshape(-1, length), chain_encoding=chain_encoding.reshape(-1, length))
        # Geometry and GVP normalization use fp32; preserve autograd in training.
        with torch.autocast(device_type=denoised_coords.device.type, enabled=False):
            logits, features = self.sequence_network(**inputs)
        logits = logits.reshape(*lead, length, 21)
        return logits.index_select(-1, self.canonical_indices), features
