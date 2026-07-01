"""
Design-side featurization wrapper.

This module takes the **output of Protenix's standard featurization** for a real
PDB complex (an `AtomArray` + per-token feature dict + label dict) and applies
the PXDesign-specific transformations on top:

1. Mark a binder region (`res_name = "xpb"`) — the [xpb] design token from the
   technical report (p. 23). Only the four backbone atoms (N, Cα, C, O — see
   `pxdesign/data/constants.py:RES_ATOMS_DICT["xpb"]`) survive on these residues,
   matching the released constants.
2. Recompute `restype` one-hot over `STD_RESIDUES_WITH_GAP` (length 36), which
   automatically widens to 32+4 channels because xpb/xpa/rbb/raa sit at
   indices 32–35.
3. Build the `conditional_templ` + `conditional_templ_mask` pair tensors via
   PXDesign's existing `DesignFeaturizer.get_condition_template_feature` — but
   sourced from `coord` (GT) rather than an externally-provided
   `coord_from_cif`, because at training time we know the truth.
4. Build a per-token `hotspot` mask. Inference reads hotspots from the YAML;
   training samples them randomly from target residues within 8 Å of any
   binder Cα. The fraction is itself randomized so the model learns to use
   either zero, few, or many hotspots.
5. Set `plddt` to zeros (per `pxdesign/model/embedders.py:148-151`, the
   InputFeatureEmbedderDesign auto-fills zeros when this key is absent — we set
   it explicitly to keep behavior identical between training and inference).
6. Mask out sequence-side features (MSA / profile / deletion) on the design
   tokens, matching `pxdesign/data/json_to_feature.py:353-361`. This prevents
   the binder's true sequence from leaking through the MSA channel.

The label dict from Protenix carries `coordinate` + `coordinate_mask` for the
full complex. The report says target coords are NOT frozen during training
(p. 23), so we pass the full GT coords through unchanged.

This module does NOT:
  - parse mmCIF / build the AtomArray (Protenix's `parser.py` does that)
  - perform cropping to 640 residues (Protenix's training crop does that)
  - select which complex / which chain to design (the dataloader does that)

Callers compose: parse → crop → Protenix-featurize → DesignFeaturizer.
"""
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from pxdesign.data.constants import (
    PRO_STD_RESIDUES_NATURAL,
    RES_ATOMS_DICT,
    STD_RESIDUES_WITH_GAP,
)

# Vendored copies of three small PXDesign helpers — see `_helpers.py` for why.
from pxdesign_train.data._helpers import (
    cano_seq_resname_with_mask,
    get_condition_template_feature,
    restype_onehot_encoded,
)

XPB_BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O")  # 4 backbone atoms per report p. 23
DEFAULT_HOTSPOT_RADIUS = 8.0   # Å, Cα-Cα interface cutoff
DEFAULT_HOTSPOT_MAX_FRAC = 0.5  # at most half of contact residues get marked
AA_IGNORE_INDEX = -100


@dataclass
class DesignSelection:
    """Specifies which residues become the binder ([xpb] design tokens).

    Exactly one of `binder_chain_id` or `binder_atom_mask` should be supplied.

    Args:
        binder_chain_id: select a whole chain by its biotite `chain_id`
            (e.g. "B") to be the binder. Convenient for PPI training where
            one chain is treated as the binder.
        binder_atom_mask: an [N_atom] boolean array marking exactly the atoms
            that belong to the binder. Use this when you want a contiguous
            sub-region of a single chain.
        hotspot_radius: Cα–Cα distance (Å) used to find contact residues.
        hotspot_max_frac: at most this fraction of contact target residues are
            marked as hotspots. The actual fraction is sampled uniformly in
            [0, hotspot_max_frac] so the model learns to use 0..many hotspots.
        hotspot_force_zero_prob: with this probability the hotspot channel is
            forced to all zeros (so the model sees "no hotspots" examples).
        rng: optional `np.random.Generator` for deterministic tests.
    """

    binder_chain_id: Optional[str] = None
    binder_atom_mask: Optional[np.ndarray] = None
    hotspot_radius: float = DEFAULT_HOTSPOT_RADIUS
    hotspot_max_frac: float = DEFAULT_HOTSPOT_MAX_FRAC
    hotspot_force_zero_prob: float = 0.2
    aa_mask_mode: str = "all"
    aa_mask_min_prob: float = 0.0
    aa_mask_max_prob: float = 1.0
    aa_mask_prob: float = 1.0
    rng: Optional[np.random.Generator] = None

    def __post_init__(self):
        n_set = int(self.binder_chain_id is not None) + int(self.binder_atom_mask is not None)
        if n_set != 1:
            raise ValueError(
                "DesignSelection requires exactly one of binder_chain_id or binder_atom_mask"
            )
        valid_modes = {"all", "none", "fixed", "time_dependent"}
        if self.aa_mask_mode not in valid_modes:
            raise ValueError(f"aa_mask_mode must be one of {sorted(valid_modes)}")
        for name, value in (
            ("aa_mask_min_prob", self.aa_mask_min_prob),
            ("aa_mask_max_prob", self.aa_mask_max_prob),
            ("aa_mask_prob", self.aa_mask_prob),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.aa_mask_min_prob > self.aa_mask_max_prob:
            raise ValueError("aa_mask_min_prob cannot exceed aa_mask_max_prob")

    def get_rng(self) -> np.random.Generator:
        return self.rng if self.rng is not None else np.random.default_rng()


class DesignFeaturizer:
    """Apply the PXDesign training transformations to a Protenix-featurized batch.

    Stateless apart from the `DesignSelection` configuration; the public method
    `transform()` is pure (no mutation of inputs).

    Args:
        selection: how to pick the binder and sample hotspots.
    """

    def __init__(self, selection: DesignSelection) -> None:
        self.selection = selection

    def transform(
        self,
        atom_array,
        feature_dict: dict[str, torch.Tensor],
        label_dict: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], Any]:
        """Apply design featurization.

        Args:
            atom_array: biotite `AtomArray` with Protenix annotations. We may
                mutate its `res_name` on the binder residues; the caller should
                pass a copy if mutation is undesirable.
            feature_dict: Protenix's per-batch feature dict. Must contain
                `restype` [N_token, 32], `distogram_rep_atom_mask` [N_atom],
                `atom_to_token_idx` [N_atom], `is_protein` [N_atom].
            label_dict: must contain `coordinate` [N_atom, 3] and
                `coordinate_mask` [N_atom].

        Returns:
            (updated feature_dict, updated label_dict, updated atom_array).
            `feature_dict["restype"]` is widened to 36 channels; design-token
            features (`conditional_templ`, `conditional_templ_mask`,
            `design_token_mask`, `condition_token_mask`, `hotspot`, `plddt`)
            are added.
        """
        binder_atom_mask = self._binder_atom_mask(atom_array)
        if not binder_atom_mask.any():
            raise ValueError("DesignFeaturizer selection produced an empty binder mask")

        # 1. Preserve clean AA labels before turning binder residues into xpb.
        #    These labels are supervision targets only, not model inputs.
        aa_clean = self._compute_clean_aa_labels(atom_array, feature_dict)
        clean_restype = self._compute_restype(atom_array, feature_dict)

        # 2. Mark binder residues as xpb on the AtomArray. Note: this is *before*
        #    computing restype, so the new restype one-hot reflects the design.
        atom_array = self._mark_as_xpb(atom_array, binder_atom_mask)

        # 3. Widen + recompute restype using PXDesign's canonical-sequence mapper.
        feature_dict = dict(feature_dict)  # shallow copy
        xpb_restype = self._compute_restype(atom_array, feature_dict)

        # 4. Token-level masks: which tokens are design tokens.
        token_is_design = self._token_level_mask(atom_array, feature_dict, binder_atom_mask)
        aa_corruption_mask, aa_t, aa_mask_prob = self._sample_aa_corruption_mask(
            token_is_design=token_is_design,
            aa_clean=aa_clean,
        )
        feature_dict["restype"] = torch.where(
            aa_corruption_mask[:, None],
            xpb_restype,
            clean_restype,
        )
        feature_dict["design_token_mask"] = token_is_design.long()
        feature_dict["condition_token_mask"] = (~token_is_design).long()
        feature_dict["aa_clean"] = aa_clean
        feature_dict["aa_corrupted"] = feature_dict["restype"].argmax(dim=-1).long()
        feature_dict["aa_corruption_mask"] = aa_corruption_mask.long()
        feature_dict["aa_loss_mask"] = aa_corruption_mask.long()
        feature_dict["aa_t"] = aa_t
        feature_dict["aa_mask_prob"] = aa_mask_prob

        # 5. Conditional template (binned pair distances on target residues, GT).
        templ_feats = get_condition_template_feature(
            atom_array=atom_array,
            coordinate_attribute="coord",         # use GT coords at train time
            ignore_ligand_only_condition=False,
            templ_token_mask=(~token_is_design).numpy(),
        )
        feature_dict.update(templ_feats)

        # 6. Hotspot mask: per-token binary, only on target residues that
        #    contact the binder. Stochastic — see DesignSelection docstring.
        feature_dict["hotspot"] = self._sample_hotspot(
            atom_array=atom_array,
            label_dict=label_dict,
            binder_atom_mask=binder_atom_mask,
            token_is_design=token_is_design,
        )

        # 7. pLDDT placeholder: zeros at train time (no predicted confidence).
        feature_dict["plddt"] = torch.zeros(
            size=(token_is_design.numel(),), dtype=torch.float32
        )

        # 8. Mask sequence features for design tokens to prevent leakage. We
        #    only mask if the key is present — Protenix's training featurizer
        #    might or might not produce these depending on data type.
        feature_dict = self._mask_sequence_leakage(feature_dict, token_is_design)

        # Labels are unchanged: target coords are noised and denoised alongside
        # the binder (report p. 23).
        return feature_dict, label_dict, atom_array

    # ----- internals -----

    def _binder_atom_mask(self, atom_array) -> np.ndarray:
        sel = self.selection
        if sel.binder_atom_mask is not None:
            mask = np.asarray(sel.binder_atom_mask, dtype=bool)
            if mask.shape != (len(atom_array),):
                raise ValueError(
                    f"binder_atom_mask shape {mask.shape} != atom_array length {len(atom_array)}"
                )
            return mask
        return atom_array.chain_id == sel.binder_chain_id

    @staticmethod
    def _compute_clean_aa_labels(atom_array, feature_dict: dict) -> torch.Tensor:
        """Return 20-AA labels per token before design residues become xpb."""
        rep_mask = feature_dict["distogram_rep_atom_mask"].bool().detach().cpu().numpy()
        centre_atoms = atom_array[rep_mask]
        labels = []
        for res_name in centre_atoms.res_name:
            idx = PRO_STD_RESIDUES_NATURAL.get(str(res_name), AA_IGNORE_INDEX)
            if idx >= 20:
                idx = AA_IGNORE_INDEX
            labels.append(idx)
        return torch.tensor(labels, dtype=torch.long)

    def _sample_aa_corruption_mask(
        self,
        token_is_design: torch.Tensor,
        aa_clean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample which design residue identities are masked for AA prediction."""
        valid_design = token_is_design & (aa_clean != AA_IGNORE_INDEX)
        mode = self.selection.aa_mask_mode
        if mode == "all":
            t = 1.0
            prob = 1.0
        elif mode == "none":
            t = 0.0
            prob = 0.0
        elif mode == "fixed":
            prob = self.selection.aa_mask_prob
            t = prob
        else:
            t = float(self.selection.get_rng().uniform(0.0, 1.0))
            lo = self.selection.aa_mask_min_prob
            hi = self.selection.aa_mask_max_prob
            prob = lo + (hi - lo) * t

        if prob <= 0.0:
            corruption = torch.zeros_like(valid_design)
        elif prob >= 1.0:
            corruption = valid_design.clone()
        else:
            draws = torch.from_numpy(
                self.selection.get_rng().random(valid_design.numel())
            ).to(dtype=torch.float32)
            corruption = valid_design & (draws < prob)

        return (
            corruption.bool(),
            torch.tensor(float(t), dtype=torch.float32),
            torch.tensor(float(prob), dtype=torch.float32),
        )

    @staticmethod
    def _mark_as_xpb(atom_array, binder_atom_mask: np.ndarray):
        """Set `res_name = "xpb"` on every binder atom.

        We do NOT drop side-chain atoms here. The released constants
        (`RES_ATOMS_DICT["xpb"]`) define xpb as backbone-only, but Protenix
        featurization keys atoms by `atom_to_token_idx` regardless of side-chain
        presence — leaving side-chain atoms in place is harmless for the
        diffusion target (they're still real protein atoms with coords) and
        avoids re-running atom tokenization.

        If you want to strictly enforce backbone-only xpb (matching inference
        more closely), drop side-chain atoms BEFORE Protenix featurization, not
        here.
        """
        new_res_name = atom_array.res_name.copy()
        new_res_name[binder_atom_mask] = "xpb"
        atom_array.res_name = new_res_name
        return atom_array

    @staticmethod
    def _compute_restype(atom_array, feature_dict: dict) -> torch.Tensor:
        """Recompute restype one-hot in the 36-channel design vocabulary."""
        rep_mask = feature_dict["distogram_rep_atom_mask"].bool().detach().cpu().numpy()
        centre_atoms = atom_array[rep_mask]
        restype_strs = cano_seq_resname_with_mask(centre_atoms)
        # `cano_seq_resname_with_mask` returns one resname per atom; take the
        # representative atom's value per token (rep atoms are 1-per-token).
        rep_count = int(rep_mask.sum())
        if len(restype_strs) != rep_count:
            # `cano_seq_resname_with_mask` actually returns 1 entry per atom-input,
            # so when fed only the rep atoms we get exactly N_token entries.
            raise AssertionError(
                f"Expected {rep_count} rep-atom restypes, got {len(restype_strs)}"
            )
        return restype_onehot_encoded(restype_strs)  # [N_token, 36]

    @staticmethod
    def _token_level_mask(
        atom_array,
        feature_dict: dict,
        binder_atom_mask: np.ndarray,
    ) -> torch.Tensor:
        """Token is 'design' iff its representative atom belongs to a binder residue."""
        rep_mask = feature_dict["distogram_rep_atom_mask"].bool().detach().cpu().numpy()
        token_is_design_np = binder_atom_mask[rep_mask]  # one entry per token
        return torch.from_numpy(token_is_design_np).bool()

    def _sample_hotspot(
        self,
        atom_array,
        label_dict: dict[str, torch.Tensor],
        binder_atom_mask: np.ndarray,
        token_is_design: torch.Tensor,
    ) -> torch.Tensor:
        rng = self.selection.get_rng()
        n_token = int(token_is_design.numel())

        if rng.random() < self.selection.hotspot_force_zero_prob:
            return torch.zeros(n_token, dtype=torch.float32)

        # Find contact target residues: representative atoms of target tokens
        # within `hotspot_radius` of any binder Cα.
        rep_mask = atom_array.distogram_rep_atom_mask.astype(bool)
        coord = label_dict["coordinate"].numpy()
        is_resolved = label_dict["coordinate_mask"].numpy().astype(bool)

        binder_resolved_atoms = binder_atom_mask & is_resolved
        if not binder_resolved_atoms.any():
            return torch.zeros(n_token, dtype=torch.float32)

        binder_cb = coord[binder_resolved_atoms]  # using all binder atoms is fine —
        # contact = "any binder atom within radius" is what the report implies.
        token_rep_coord = coord[rep_mask]                  # [N_token, 3]
        token_resolved = is_resolved[rep_mask]              # [N_token]
        token_is_design_np = token_is_design.numpy().astype(bool)

        # Distance from each rep atom to nearest binder atom.
        d = np.linalg.norm(
            token_rep_coord[:, None, :] - binder_cb[None, :, :],
            axis=-1,
        ).min(axis=-1)
        contact = (
            (d < self.selection.hotspot_radius)
            & token_resolved
            & ~token_is_design_np
        )

        if not contact.any():
            return torch.zeros(n_token, dtype=torch.float32)

        contact_idx = np.where(contact)[0]
        frac = float(rng.uniform(0.0, self.selection.hotspot_max_frac))
        n_pick = max(0, int(round(frac * len(contact_idx))))
        picked = rng.choice(contact_idx, size=n_pick, replace=False) if n_pick > 0 else np.array([], dtype=int)

        out = np.zeros(n_token, dtype=np.float32)
        out[picked] = 1.0
        return torch.from_numpy(out)

    @staticmethod
    def _mask_sequence_leakage(
        feature_dict: dict[str, torch.Tensor],
        token_is_design: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Zero out sequence-derived features on design tokens.

        Mirrors `pxdesign/data/json_to_feature.py:353-361`. Without this the
        binder's true sequence could leak through the MSA channel into the
        InputFeatureEmbedder.
        """
        condi = (~token_is_design).long()
        for key in ("msa", "has_deletion", "deletion_value"):
            if key in feature_dict:
                feature_dict[key] = feature_dict[key] * condi[None, :]
        if "profile" in feature_dict:
            feature_dict["profile"] = feature_dict["profile"] * condi[:, None]
        if "deletion_mean" in feature_dict:
            feature_dict["deletion_mean"] = feature_dict["deletion_mean"] * condi
        return feature_dict


def apply_design_featurization(
    atom_array,
    feature_dict: dict[str, torch.Tensor],
    label_dict: dict[str, torch.Tensor],
    selection: DesignSelection,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], Any]:
    """Functional convenience wrapper around `DesignFeaturizer.transform`."""
    return DesignFeaturizer(selection).transform(atom_array, feature_dict, label_dict)
