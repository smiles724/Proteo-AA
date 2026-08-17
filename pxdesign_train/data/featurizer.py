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
import logging
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
    compute_sidechain: bool = False
    # Optional labels captured from the native (pre-scrub) binder.  The strict
    # training pipeline removes binder side-chain atoms and canonicalises every
    # binder residue to the same GLY backbone *before* Protenix featurization;
    # without this override the AA supervision would therefore also become GLY.
    # This tensor is supervision only: masked design tokens still receive xpb in
    # feature_dict["restype"].
    aa_clean_override: Optional[torch.Tensor] = None
    # M1: emit `backbone_loss_mask` that excludes binder (design-token)
    # side-chain atoms from the backbone (L_bb) coordinate target, so B_theta is
    # backbone-only and S_phi is the sole side-chain generator. This also matches
    # de-novo inference, where the binder has no side-chain atoms.
    backbone_only_binder: bool = False
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

    # Set once the frame guard has reported; keeps a per-example diagnostic from
    # printing on every item of every epoch (this runs inside the dataloader).
    _warned_no_frame = False

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
        aa_clean = (
            self._compute_clean_aa_labels(atom_array, feature_dict)
            if self.selection.aa_clean_override is None
            else self.selection.aa_clean_override.detach().cpu().long().clone()
        )
        if aa_clean.shape != (int(feature_dict["restype"].shape[-2]),):
            raise ValueError(
                "aa_clean_override must have one label per token; "
                f"got {tuple(aa_clean.shape)} for {feature_dict['restype'].shape[-2]} tokens"
            )
        if self.selection.aa_clean_override is not None:
            override = torch.as_tensor(
                self.selection.aa_clean_override, dtype=torch.long
            ).detach().clone()
            if override.shape != aa_clean.shape:
                raise ValueError(
                    "aa_clean_override shape "
                    f"{tuple(override.shape)} != token labels {tuple(aa_clean.shape)}"
                )
            aa_clean = override
        clean_restype = self._compute_restype(atom_array, feature_dict)
        # Restore the native AA one-hot only for tokens deliberately left
        # unmasked by a partial-corruption schedule.  Fully masked Stage I uses
        # xpb below, so the label never crosses into the model input.
        valid_aa = (aa_clean >= 0) & (aa_clean < 20)
        if valid_aa.any():
            clean_restype = clean_restype.clone()
            clean_restype[valid_aa] = 0
            clean_restype[valid_aa, aa_clean[valid_aa]] = 1

        # 1b. Side-chain targets (Stage II-A): computed from the ORIGINAL atom
        #     array (real res_name + GT side-chain coords), BEFORE xpb marking.
        #     GT side chains are the label; the model input is Gaussian noise.
        if self.selection.compute_sidechain:
            feature_dict = dict(feature_dict)
            feature_dict.update(
                self._compute_sidechain_targets(atom_array, feature_dict, binder_atom_mask)
            )

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
        # A design token shows [xpb] if it is corrupted OR carries no usable
        # label. The second half matters: `_sample_aa_corruption_mask` only draws
        # from tokens with a valid 20-AA label, so a design residue whose native
        # type is non-standard (UNK / unmapped CCD -> aa_clean == -100) is never
        # corrupted and would otherwise fall through to `clean_restype` — showing
        # the model a concrete residue (GLY under strict re-featurization, the
        # native 3-letter code without it) at a position that is ALWAYS xpb at
        # inference. It is excluded from the AA loss either way; this only fixes
        # what the model is shown.
        show_xpb = aa_corruption_mask | (
            token_is_design & (aa_clean == AA_IGNORE_INDEX)
        )
        feature_dict["restype"] = torch.where(
            show_xpb[:, None],
            xpb_restype,
            clean_restype,
        )
        feature_dict["design_token_mask"] = token_is_design.long()
        feature_dict["condition_token_mask"] = (~token_is_design).long()
        feature_dict.update(self._eval_atom_masks(atom_array, label_dict))
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

        # 6b. Backbone-only binder (M1 + P1): B_theta must be backbone-only for
        #     the design region — both in the loss AND in the diffusion INPUT.
        #     (i) backbone_loss_mask: exclude binder side-chain atoms from L_bb.
        #     (ii) scrub the binder side-chain COORDINATES to the residue Cα so the
        #     denoiser never sees noisy GT side-chain geometry (which would leak
        #     into a_token / h_res). GT side-chain targets for S_phi were already
        #     captured above (step 1b) from the real coords, so this is safe.
        if self.selection.backbone_only_binder:
            feature_dict["backbone_loss_mask"] = self._backbone_loss_mask(
                atom_array, binder_atom_mask
            )
            feature_dict["design_sidechain_atom_mask"] = ~self._backbone_loss_mask(
                atom_array, binder_atom_mask
            )
            label_dict = self._scrub_design_sidechain_coords(
                atom_array, label_dict, binder_atom_mask
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
    def _eval_atom_masks(atom_array, label_dict: dict) -> dict[str, torch.Tensor]:
        """Atom masks for evaluation-only backbone structural metrics."""
        n_atom = int(label_dict["coordinate"].shape[-2])
        atom_name = np.asarray(atom_array.atom_name)[:n_atom]
        mol_type = np.asarray(getattr(atom_array, "mol_type", np.array([""] * len(atom_array))))[:n_atom]
        is_protein = mol_type == "protein"
        # Synthetic/unit-test AtomArrays may not carry a useful mol_type annotation.
        if not is_protein.any():
            is_protein = np.ones_like(atom_name, dtype=bool)
        ca = is_protein & (atom_name == "CA")
        bb = is_protein & np.isin(atom_name, np.asarray(XPB_BACKBONE_ATOM_NAMES))
        return {
            "eval_ca_atom_mask": torch.from_numpy(ca.astype(bool)),
            "eval_backbone_atom_mask": torch.from_numpy(bb.astype(bool)),
        }

    def _compute_sidechain_targets(
        self, atom_array, feature_dict: dict, binder_atom_mask: np.ndarray
    ) -> dict:
        """GT side-chain targets in residue-local frames, for design tokens.

        Returns per-token tensors [N_token, MAX_SC, *]: `sc_gt_local` (GT side
        chain in the residue's local frame), `sc_atom_mask` (valid/resolved
        atoms), `sc_atom_name_ids` (atom-name embedding ids). Non-binder tokens
        and GLY are all-zero/all-False. CCD-free: uses the AtomArray's own coords
        + atom names (called BEFORE xpb marking, so res_name is real).
        """
        from collections import defaultdict

        from pxdesign_train.sidechain.frames import build_frame, to_local
        from pxdesign_train.sidechain.instantiate import (
            ATOM_NAME_TO_ID,
            MAX_SC,
            sidechain_atoms,
        )

        rep_mask = feature_dict["distogram_rep_atom_mask"].bool().detach().cpu().numpy()
        rep_idx = np.nonzero(rep_mask)[0]
        n_token = len(rep_idx)
        coord = np.asarray(atom_array.coord, dtype=np.float32)
        atom_name = np.asarray(atom_array.atom_name)
        res_name = np.asarray(atom_array.res_name)
        chain_id = np.asarray(atom_array.chain_id)
        res_id = np.asarray(atom_array.res_id)
        binder = np.asarray(binder_atom_mask, dtype=bool)

        # An UNRESOLVED atom still occupies a row (the residue keeps its full
        # complement of atom names) but its coordinate is a placeholder (0, 0, 0).
        # Supervising such a row asks S_phi to put the atom at the coordinate
        # ORIGIN, which is wherever the depositor happened to centre the assembly —
        # 400 A away for 7us9, 37 A for 7lmv. Measured on the 491-protein
        # validation set before this guard: 9.84% of all supervised side-chain
        # atoms (59497/604774) across 461/491 structures were such origin targets,
        # contributing an unlearnable error that dominated the loss (7us9 read
        # 21860 A^2 == 133 atoms x ~416 A) and flat-lined training.
        #
        # `is_resolved` is the authoritative flag and survives cropping; on 7us9 it
        # marks all 1034 zero-coordinate atoms correctly. The exact-zero coordinate
        # test is a belt-and-braces fallback for arrays that lack the annotation (a
        # real atom sitting exactly on the origin is not physically meaningful, so
        # dropping it costs nothing).
        if "is_resolved" in atom_array.get_annotation_categories():
            resolved = np.asarray(atom_array.is_resolved, dtype=bool)
        else:
            resolved = np.ones(len(atom_array), dtype=bool)
        resolved &= np.abs(coord).max(axis=1) > 1e-3

        res_atoms: dict = defaultdict(dict)
        for idx in range(len(atom_array)):
            res_atoms[(chain_id[idx], res_id[idx])][str(atom_name[idx])] = idx

        sc_gt_local = np.zeros((n_token, MAX_SC, 3), dtype=np.float32)
        sc_mask = np.zeros((n_token, MAX_SC), dtype=bool)
        sc_ids = np.zeros((n_token, MAX_SC), dtype=np.int64)
        sc_frame_R = np.tile(np.eye(3, dtype=np.float32), (n_token, 1, 1))
        sc_frame_t = np.zeros((n_token, 3), dtype=np.float32)
        sc_bb_coords = np.zeros((n_token, 4, 3), dtype=np.float32)  # N,CA,C,O
        # Atom indices (into the N_atom axis) of this token's N, CA, C, O — the SAME
        # axis every Protenix per-atom tensor is indexed by (coordinates, and the
        # AtomAttentionEncoder's per-atom `q` features). Two consumers:
        #   (a) frames: gather the PREDICTED backbone (x_denoised) at columns 0:3
        #       (N, CA, C) to build F_hat for Stage II-B (S_phi conditions on
        #       x_hat_0^bb);
        #   (b) atom-level side-chain -> backbone feedback: gather/scatter the 4
        #       backbone atoms' per-atom features q_bb.
        # COLUMN ORDER IS (N, CA, C, O) and is load-bearing: the first three are the
        # frame atoms, so any 3-atom consumer must slice [..., :3]. -1 marks
        # non-binder / missing (skipped downstream). Column 3 (O) can be -1 while
        # 0:3 are valid — never test validity with `.all()` over all four columns.
        sc_bb_atom_idx = np.full((n_token, 4), -1, dtype=np.int64)

        n_no_frame = 0
        for ti, ai in enumerate(rep_idx):
            if not binder[ai]:
                continue
            atoms = res_atoms[(chain_id[ai], res_id[ai])]
            # The frame atoms must be PRESENT and RESOLVED. Name presence alone was
            # the same oversight that made unresolved side-chain atoms supervised
            # targets above, but here it does not show up as a masked atom: the side
            # chain may be perfectly well resolved while the frame it hangs off is
            # built from a (0,0,0) placeholder. `t` is CA and the basis comes from
            # N/CA/C, so one placeholder among the three puts the whole residue's
            # local geometry in a frame belonging to nothing.
            #
            # Stage II hides this: the coordinate loss rebuilds the target with the
            # SAME frame it was stored under, and to_global(to_local(x, R, t), R, t)
            # is exactly x for any orthonormal R, so the bad frame cancels. Stage III
            # does not cancel — there the target is rebuilt through the PREDICTED
            # frame — and `gt_local` then acts as a lever arm: with a placeholder CA
            # it grows from ~3 A to |x_true| (hundreds of A), multiplying the
            # backbone's own angular error by two orders of magnitude.
            #
            # Skipping is the treatment a missing atom NAME already got, so this
            # widens an existing guard rather than introducing a new behaviour.
            if not all(a in atoms and resolved[atoms[a]] for a in ("N", "CA", "C")):
                n_no_frame += 1
                continue
            n = torch.from_numpy(coord[atoms["N"]])[None]
            ca = torch.from_numpy(coord[atoms["CA"]])[None]
            c = torch.from_numpy(coord[atoms["C"]])[None]
            R, t = build_frame(n, ca, c)
            sc_frame_R[ti] = R[0].numpy()
            sc_frame_t[ti] = t[0].numpy()
            # Indices and coords come from the SAME by-name lookup in the SAME
            # order, so a gather at sc_bb_atom_idx reproduces sc_bb_coords exactly.
            #
            # An UNRESOLVED backbone atom is marked ABSENT (index -1) rather than
            # indexed-with-a-placeholder. Every consumer of these four slots decides
            # validity from `sc_bb_atom_idx >= 0`, which is true whenever the atom
            # NAME exists, so an unresolved-but-named atom used to pass as real while
            # carrying (0,0,0). Marking it absent routes it to the bounded fallback
            # that was already intended: v4 zeroes it and bb_local becomes 0, i.e.
            # the frame origin (== CA). It keeps q_bs's gather and the physical
            # loss's bb_valid consistent too, since both key off the same index.
            #
            # After the frame guard above this can only ever be O -- a residue
            # missing any of N/CA/C is dropped whole.
            for bi, bn in enumerate(XPB_BACKBONE_ATOM_NAMES):  # (N, CA, C, O)
                if bn in atoms and resolved[atoms[bn]]:
                    sc_bb_atom_idx[ti, bi] = atoms[bn]
                    sc_bb_coords[ti, bi] = coord[atoms[bn]]
            for j, nm in enumerate(sidechain_atoms(str(res_name[ai]))):
                if j >= MAX_SC:
                    break
                # `sc_ids` stays type-derived: it names the SLOT, which exists for
                # this residue type whether or not the structure resolved it. Only
                # the supervision mask is gated on having a real coordinate.
                sc_ids[ti, j] = ATOM_NAME_TO_ID[nm]
                if nm in atoms and resolved[atoms[nm]]:
                    g = torch.from_numpy(coord[atoms[nm]])[None, None]  # [1,1,3]
                    sc_gt_local[ti, j] = to_local(g, R, t)[0, 0].numpy()
                    sc_mask[ti, j] = True

        # Say how much the frame guard actually costs, ONCE per process. Whether
        # the Stage II checkpoint is worth retraining on the corrected data turns
        # on this number, and nobody has measured it — the 9.84% figure in the
        # unresolved-atom guard above counts side-chain atoms, not frame atoms.
        if n_no_frame and not DesignFeaturizer._warned_no_frame:
            DesignFeaturizer._warned_no_frame = True
            n_binder_tok = int(sum(1 for ai in rep_idx if binder[ai]))
            logging.getLogger(__name__).info(
                "side-chain targets: %d/%d binder tokens dropped for having an "
                "unresolved N/CA/C (no usable local frame). Logged once per process.",
                n_no_frame, n_binder_tok,
            )

        return {
            "sc_gt_local": torch.from_numpy(sc_gt_local),
            "sc_atom_mask": torch.from_numpy(sc_mask),
            "sc_atom_name_ids": torch.from_numpy(sc_ids),
            "sc_frame_R": torch.from_numpy(sc_frame_R),
            "sc_frame_t": torch.from_numpy(sc_frame_t),
            "sc_bb_coords": torch.from_numpy(sc_bb_coords),
            "sc_bb_atom_idx": torch.from_numpy(sc_bb_atom_idx),
            # Representative (CA, for protein) atom of EVERY token — binder AND
            # receptor/motif/ligand. `sc_bb_atom_idx` is binder-only (-1 elsewhere),
            # so it cannot give S_phi a position for the context tokens it must
            # attend to. Same N_atom axis, so it is remapped on crop like the others.
            "sc_token_center_idx": torch.from_numpy(np.asarray(rep_idx, dtype=np.int64)),
        }

    @staticmethod
    def _backbone_loss_mask(atom_array, binder_atom_mask: np.ndarray) -> torch.Tensor:
        """Per-atom [N_atom] bool: True everywhere except binder side-chain atoms.

        Backbone atoms (N/CA/C/O) of binder residues stay True (they ARE the
        backbone target); non-binder atoms all stay True; binder heavy side-chain
        atoms become False so `L_bb` never supervises them — S_phi is the sole
        side-chain generator (M1). Called on the (real-atom-name) AtomArray;
        res_name/xpb marking does not affect atom_name, so ordering is safe.
        """
        atom_name = np.asarray(atom_array.atom_name)
        binder = np.asarray(binder_atom_mask, dtype=bool)
        is_backbone = np.isin(atom_name, np.asarray(XPB_BACKBONE_ATOM_NAMES))
        keep = ~(binder & ~is_backbone)  # drop only binder side-chain atoms
        return torch.from_numpy(keep.astype(bool))

    @staticmethod
    def _scrub_design_sidechain_coords(atom_array, label_dict, binder_atom_mask):
        """Replace binder (design-region) side-chain atom coordinates with their
        residue Cα (P1). This removes GT side-chain geometry from the coordinate
        diffusion INPUT so B_theta / a_token / h_res cannot see it — the atoms
        remain as tokens (no re-tokenisation) but carry no side-chain signal.
        Backbone atoms and all non-binder atoms are untouched. Idempotent-safe:
        returns a new label_dict with a cloned `coordinate`.
        """
        if "coordinate" not in label_dict:
            return label_dict
        coord = label_dict["coordinate"]
        new = coord.clone()
        atom_name = np.asarray(atom_array.atom_name)
        chain = np.asarray(atom_array.chain_id)
        res = np.asarray(atom_array.res_id)
        binder = np.asarray(binder_atom_mask, dtype=bool)
        is_backbone = np.isin(atom_name, np.asarray(XPB_BACKBONE_ATOM_NAMES))
        ca_of = {}
        for i in range(len(atom_array)):
            if atom_name[i] == "CA":
                ca_of[(chain[i], res[i])] = i
        n = new.shape[-2]
        for i in range(len(atom_array)):
            if i < n and binder[i] and not is_backbone[i]:
                ci = ca_of.get((chain[i], res[i]))
                if ci is not None and ci < n:
                    new[..., i, :] = coord[..., ci, :]
        return {**label_dict, "coordinate": new}

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
