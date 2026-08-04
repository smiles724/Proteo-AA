"""The paired A/B/C/D and coordinate-path 2x2 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pxdesign_train.sidechain.chi_constants import (
    CHI_ATOM_IDX,
    CHI_MASK,
    IDEAL_BB_LOCAL,
)
from pxdesign_train.sidechain.feedback import HResFeedback
from pxdesign_train.sidechain.frames import dihedral, to_local
from pxdesign_train.sidechain.instantiate import N_BB

from .bundle import SidechainInputBundle
from .geometry import apply_rigid, random_rotation, rotate_about_axis, rotate_frames
from .metrics import (
    aggregate_numeric,
    equivariant_rmsd,
    rmsd,
    summarize_invariant,
)


@dataclass(frozen=True)
class CoordinateArm:
    local_input: bool
    frame_aware_head: bool

    @property
    def name(self) -> str:
        return (
            f"local_input={str(self.local_input).lower()},"
            f"frame_aware_head={str(self.frame_aware_head).lower()}"
        )


ARMS = (
    CoordinateArm(False, False),
    CoordinateArm(True, False),
    CoordinateArm(False, True),
    CoordinateArm(True, True),
)


class _IntermediateRecorder:
    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.values: dict[str, torch.Tensor] = {}
        self.handles: list[Any] = []

    def _save(self, name):
        def hook(_module, _args, output):
            value = output[0] if isinstance(output, tuple) else output
            self.values[name] = value.detach()

        return hook

    def __enter__(self) -> "_IntermediateRecorder":
        fixed = {
            "atom_embedding": self.module.atom_embed,
            "h_res_projected": self.module.w_res,
            "aa_projected": self.module.w_aa,
            "time_projected": self.module.w_t,
            "xyz_projected": self.module.w_xyz,
            "head_pre_reconstruction": self.module.out,
        }
        for name, submodule in fixed.items():
            self.handles.append(submodule.register_forward_hook(self._save(name)))
        for i, block in enumerate(self.module.blocks):
            self.handles.append(block.register_forward_hook(self._save(f"intra_block_{i:02d}")))
        for i, block in enumerate(self.module.cross_res_blocks):
            self.handles.append(block.register_forward_hook(self._save(f"cross_block_{i:02d}")))
        return self

    def __exit__(self, *_exc) -> None:
        for handle in self.handles:
            handle.remove()


def _full_atom_mask(bundle: SidechainInputBundle) -> torch.Tensor:
    if bundle.bb_local is None:
        return bundle.atom_mask.bool()
    if bundle.res_mask is None:
        bb_mask = torch.ones(
            *bundle.atom_mask.shape[:2],
            N_BB,
            dtype=torch.bool,
            device=bundle.atom_mask.device,
        )
    else:
        bb_mask = bundle.res_mask.bool()[..., None].expand(*bundle.res_mask.shape, N_BB)
    return torch.cat([bb_mask, bundle.atom_mask.bool()], dim=-1)


def _initial_u(values: dict[str, torch.Tensor]) -> torch.Tensor:
    atom = values["atom_embedding"]
    return (
        atom
        + values["h_res_projected"][:, :, None, :]
        + values["aa_projected"][:, :, None, :]
        + values["xyz_projected"]
        + values["time_projected"][:, None, None, :]
    )


class RotationAblation:
    """Evaluate one SideChainModule/HResFeedback pair without redrawing noise."""

    def __init__(
        self,
        module: torch.nn.Module,
        feedback: HResFeedback,
        bundle: SidechainInputBundle,
        *,
        invariant_tolerance: float = 1e-4,
        coordinate_tolerance: float = 1e-3,
    ) -> None:
        self.module = module.eval()
        self.feedback = feedback.eval()
        self.bundle = bundle.validate()
        self.invariant_tolerance = float(invariant_tolerance)
        self.coordinate_tolerance = float(coordinate_tolerance)

    @torch.no_grad()
    def _run(
        self,
        arm: CoordinateArm,
        *,
        h_res: torch.Tensor,
        aa_logits: torch.Tensor,
        noisy_local: torch.Tensor,
        noisy_global: torch.Tensor,
        ca_coords: torch.Tensor,
        frame_r: torch.Tensor,
        frame_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        coords = noisy_local if arm.local_input else noisy_global
        kwargs: dict[str, Any] = {}
        if self.bundle.bb_local is not None:
            kwargs.update(bb_local=self.bundle.bb_local, res_mask=self.bundle.res_mask)
        with _IntermediateRecorder(self.module) as recorder:
            result = self.module(
                h_res,
                aa_logits,
                self.bundle.atom_name_ids,
                self.bundle.atom_mask,
                coords,
                self.bundle.time,
                ca_coords=ca_coords,
                frame_R=frame_r if arm.frame_aware_head else None,
                frame_t=frame_t if arm.frame_aware_head else None,
                ctx_mask=self.bundle.ctx_mask,
                **kwargs,
            )
        prediction, atom_feats = result[:2]
        values = recorder.values
        values["initial_atom_feature"] = _initial_u(values)
        values["atom_feats"] = atom_feats.detach()

        m = self.bundle.atom_mask[..., None].to(atom_feats.dtype)
        pooled = (atom_feats * m).sum(dim=2) / (m.sum(dim=2) + self.feedback.eps)
        g = self.feedback.pool_proj(pooled)
        delta = self.feedback.update(torch.cat([h_res, g], dim=-1))
        values["a_sc"] = pooled.detach()
        values["g"] = g.detach()
        values["delta_h_res"] = delta.detach()
        values["h_res_prime"] = (h_res + delta).detach()
        values["prediction_global"] = prediction.detach()
        values["h_res_input"] = h_res.detach()
        values["aa_logits_input"] = aa_logits.detach()
        return values

    def _compare(
        self,
        reference: dict[str, torch.Tensor],
        candidate: dict[str, torch.Tensor],
        *,
        q: torch.Tensor,
        t: torch.Tensor,
        arm: CoordinateArm,
    ) -> dict[str, Any]:
        residue_mask = (
            self.bundle.res_mask
            if self.bundle.res_mask is not None
            else self.bundle.atom_mask.bool().any(dim=-1)
        )
        full_atom_mask = _full_atom_mask(self.bundle)
        invariant: dict[str, Any] = {}
        for name, ref in reference.items():
            if name in ("prediction_global", "head_pre_reconstruction"):
                continue
            cand = candidate.get(name)
            if cand is None or cand.shape != ref.shape:
                continue
            mask = None
            if ref.dim() >= 3 and ref.shape[:2] == self.bundle.atom_mask.shape[:2]:
                if ref.dim() >= 4 and ref.shape[2] == full_atom_mask.shape[2]:
                    mask = full_atom_mask
                elif ref.dim() >= 4 and ref.shape[2] == self.bundle.atom_mask.shape[2]:
                    mask = self.bundle.atom_mask
            invariant[name] = summarize_invariant(
                ref, cand, mask=mask, residue_mask=residue_mask
            )

        head_ref = reference["head_pre_reconstruction"]
        head_cand = candidate["head_pre_reconstruction"]
        if arm.frame_aware_head:
            head_metrics = {
                "semantics": "local invariant coordinates",
                **summarize_invariant(
                    head_ref,
                    head_cand,
                    mask=self.bundle.atom_mask,
                    residue_mask=residue_mask,
                ),
                "rmsd": rmsd(head_ref, head_cand, self.bundle.atom_mask),
            }
        else:
            zero = torch.zeros_like(t)
            head_metrics = {
                "semantics": "global offset vector",
                **equivariant_rmsd(
                    head_ref, head_cand, q, zero, self.bundle.atom_mask
                ),
                "raw_rmsd": rmsd(head_ref, head_cand, self.bundle.atom_mask),
            }
        coords = equivariant_rmsd(
            reference["prediction_global"],
            candidate["prediction_global"],
            q,
            t,
            self.bundle.atom_mask,
        )
        return {
            "invariant": invariant,
            "coordinate_head": head_metrics,
            "prediction_global": coords,
        }

    def _geometry(
        self, q: torch.Tensor, t: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return {
            # The exact same local epsilon/template state is reused.
            "noisy_local": self.bundle.noisy_local,
            "noisy_global": apply_rigid(self.bundle.noisy_global, q, t),
            "ca_coords": apply_rigid(self.bundle.ca_coords, q, t),
            "frame_r": rotate_frames(self.bundle.frame_R, q),
            "frame_t": apply_rigid(self.bundle.frame_t, q, t),
        }

    @torch.no_grad()
    def one_transform(
        self,
        q: torch.Tensor,
        t: torch.Tensor,
        *,
        arms: tuple[CoordinateArm, ...] = ARMS,
        use_recomputed_pair: bool = False,
    ) -> dict[str, Any]:
        original_geometry = {
            "noisy_local": self.bundle.noisy_local,
            "noisy_global": self.bundle.noisy_global,
            "ca_coords": self.bundle.ca_coords,
            "frame_r": self.bundle.frame_R,
            "frame_t": self.bundle.frame_t,
        }
        transformed_geometry = self._geometry(q, t)
        result: dict[str, Any] = {}
        for arm in arms:
            a = self._run(
                arm,
                h_res=self.bundle.h_res,
                aa_logits=self.bundle.aa_logits,
                **original_geometry,
            )
            c = self._run(
                arm,
                h_res=self.bundle.h_res,
                aa_logits=self.bundle.aa_logits,
                **transformed_geometry,
            )
            arm_result: dict[str, Any] = {
                "C_coordinate_only": self._compare(a, c, q=q, t=t, arm=arm),
            }
            if use_recomputed_pair and self.bundle.h_res_q is not None:
                q_logits = (
                    self.bundle.aa_logits_q
                    if self.bundle.aa_logits_q is not None
                    else self.bundle.aa_logits
                )
                b = self._run(
                    arm,
                    h_res=self.bundle.h_res_q,
                    aa_logits=q_logits,
                    **transformed_geometry,
                )
                d = self._run(
                    arm,
                    h_res=self.bundle.h_res_q,
                    aa_logits=q_logits,
                    **original_geometry,
                )
                arm_result["B_end_to_end"] = self._compare(a, b, q=q, t=t, arm=arm)
                arm_result["D_h_res_only"] = self._compare(a, d, q=q, t=t, arm=arm)
                arm_result["diagnosis"] = self._diagnose(arm_result)
            else:
                arm_result["B_end_to_end"] = {"available": False}
                arm_result["D_h_res_only"] = {"available": False}
                arm_result["diagnosis"] = "coordinate-path C intervention"
            result[arm.name] = arm_result
        return result

    def _diagnose(self, result: dict[str, Any]) -> str:
        def error(run: str) -> float:
            return float(
                result[run]["invariant"]["h_res_prime"]["relative_frobenius"]
            )

        b, c, d = error("B_end_to_end"), error("C_coordinate_only"), error("D_h_res_only")
        large = lambda x: x > self.invariant_tolerance
        if large(c) and not large(d):
            return "coordinate pathway is responsible"
        if large(d) and not large(c):
            return "h_res/AA pathway is responsible"
        if large(c) and large(d):
            return "both coordinate and h_res/AA pathways contribute"
        if large(b):
            return "individual interventions are small; nonlinear interaction is large"
        return "no material paired sensitivity at h_res_prime"

    def _quality(self, prediction_global: torch.Tensor) -> dict[str, Any]:
        """Optional packing metrics when the bundle carries targets/context."""
        out: dict[str, Any] = {}
        pred_local = to_local(
            prediction_global.float(),
            self.bundle.frame_R.float(),
            self.bundle.frame_t.float(),
        )
        if self.bundle.gt_local is not None:
            out["local_sc_rmsd"] = rmsd(
                self.bundle.gt_local, pred_local, self.bundle.atom_mask
            )
            per_residue_sq = (
                (pred_local - self.bundle.gt_local.float()).square().sum(dim=-1)
            )
            denom = self.bundle.atom_mask.sum(dim=-1).clamp_min(1)
            per_residue = torch.sqrt(
                (per_residue_sq * self.bundle.atom_mask).sum(dim=-1) / denom
            )
            if self.bundle.residue_category is not None:
                categories = {0: "buried", 1: "interface", 2: "surface"}
                out["local_sc_rmsd_by_category"] = {
                    name: float(per_residue[self.bundle.residue_category == code].mean())
                    for code, name in categories.items()
                    if (self.bundle.residue_category == code).any()
                }
            if self.bundle.aa_correct is not None:
                out["local_sc_rmsd_by_aa_prediction"] = {
                    "correct": (
                        float(per_residue[self.bundle.aa_correct.bool()].mean())
                        if self.bundle.aa_correct.bool().any()
                        else float("nan")
                    ),
                    "incorrect": (
                        float(per_residue[~self.bundle.aa_correct.bool()].mean())
                        if (~self.bundle.aa_correct.bool()).any()
                        else float("nan")
                    ),
                }
            out.update(self._chi_accuracy(pred_local))
        if self.bundle.gt_global is not None:
            out["global_sc_rmsd"] = rmsd(
                self.bundle.gt_global, prediction_global, self.bundle.atom_mask
            )
        if self.bundle.context_coords is not None:
            ctx = self.bundle.context_coords.float()
            ctx_mask = (
                self.bundle.context_mask.bool()
                if self.bundle.context_mask is not None
                else torch.ones(ctx.shape[:-1], dtype=torch.bool, device=ctx.device)
            )
            pred = prediction_global.float().reshape(prediction_global.shape[0], -1, 3)
            pred_mask = self.bundle.atom_mask.reshape(self.bundle.atom_mask.shape[0], -1)
            distance = torch.cdist(pred, ctx)
            valid_pair = pred_mask[..., None] & ctx_mask[:, None, :]
            clashing_atom = ((distance < 2.0) & valid_pair).any(dim=-1)
            out["clash_rate"] = float(
                (clashing_atom & pred_mask).sum() / pred_mask.sum().clamp_min(1)
            )
        return out

    def _chi_accuracy(self, pred_local: torch.Tensor) -> dict[str, Any]:
        if self.bundle.gt_local is None or self.bundle.residue_type_idx is None:
            return {
                "chi1_accuracy_20deg": float("nan"),
                "chi2_accuracy_20deg": float("nan"),
            }
        types = self.bundle.residue_type_idx.long()
        if self.bundle.bb_local is not None:
            bb = self.bundle.bb_local[..., :3, :].float()
        else:
            bb = IDEAL_BB_LOCAL.to(pred_local.device)[types]
        pred_all = torch.cat([bb, pred_local], dim=-2)
        gt_all = torch.cat([bb, self.bundle.gt_local.float()], dim=-2)
        indices = CHI_ATOM_IDX.to(types.device)[types]
        chi_mask = CHI_MASK.to(types.device)[types]
        batch_shape = indices.shape[:-2]
        gather_idx = indices.reshape(*batch_shape, -1)
        pred_points = pred_all.gather(
            -2, gather_idx[..., None].expand(*gather_idx.shape, 3)
        ).reshape(*batch_shape, 4, 4, 3)
        gt_points = gt_all.gather(
            -2, gather_idx[..., None].expand(*gather_idx.shape, 3)
        ).reshape(*batch_shape, 4, 4, 3)
        pred_chi = dihedral(
            pred_points[..., 0, :],
            pred_points[..., 1, :],
            pred_points[..., 2, :],
            pred_points[..., 3, :],
        )
        gt_chi = dihedral(
            gt_points[..., 0, :],
            gt_points[..., 1, :],
            gt_points[..., 2, :],
            gt_points[..., 3, :],
        )
        delta = torch.atan2(torch.sin(pred_chi - gt_chi), torch.cos(pred_chi - gt_chi)).abs()
        correct = delta <= torch.deg2rad(torch.tensor(20.0, device=delta.device))
        metrics = {}
        for chi_i in range(2):
            valid = chi_mask[..., chi_i]
            metrics[f"chi{chi_i + 1}_accuracy_20deg"] = (
                float(correct[..., chi_i][valid].float().mean())
                if valid.any()
                else float("nan")
            )
        return metrics

    @torch.no_grad()
    def chi_positive_control(self, arm: CoordinateArm) -> dict[str, Any]:
        """Rotate a downstream side-chain subtree by +60 degrees.

        The first valid S_phi atom is treated as CB. The axis is CA->CB and all
        later valid atoms in that residue are rotated. This is a true local
        conformation change, unlike a global SE(3) transform.
        """
        mask = self.bundle.atom_mask.bool()
        chosen = None
        for b in range(mask.shape[0]):
            for i in range(mask.shape[1]):
                valid = torch.nonzero(mask[b, i], as_tuple=False).flatten()
                if valid.numel() >= 3:
                    chosen = (b, i, valid)
                    break
            if chosen is not None:
                break
        if chosen is None:
            return {"available": False, "reason": "no residue has >=3 side-chain atoms"}
        b, i, valid = chosen
        selection = torch.zeros_like(mask)
        selection[b, i, valid[1:]] = True
        changed_global = rotate_about_axis(
            self.bundle.noisy_global,
            self.bundle.ca_coords,
            self.bundle.noisy_global[..., 0, :],
            selection,
            60.0,
        )
        changed_local = to_local(
            changed_global.float(), self.bundle.frame_R.float(), self.bundle.frame_t.float()
        ).to(changed_global)
        geometry = {
            "ca_coords": self.bundle.ca_coords,
            "frame_r": self.bundle.frame_R,
            "frame_t": self.bundle.frame_t,
        }
        base = self._run(
            arm,
            h_res=self.bundle.h_res,
            aa_logits=self.bundle.aa_logits,
            noisy_local=self.bundle.noisy_local,
            noisy_global=self.bundle.noisy_global,
            **geometry,
        )
        changed = self._run(
            arm,
            h_res=self.bundle.h_res,
            aa_logits=self.bundle.aa_logits,
            noisy_local=changed_local,
            noisy_global=changed_global,
            **geometry,
        )
        return {
            "available": True,
            "batch_index": b,
            "residue_index": i,
            "input_change_rmsd": rmsd(
                self.bundle.noisy_global, changed_global, selection
            ),
            "xyz_feature_change": summarize_invariant(
                base["xyz_projected"], changed["xyz_projected"], mask=_full_atom_mask(self.bundle)
            ),
            "atom_feature_change": summarize_invariant(
                base["atom_feats"], changed["atom_feats"], mask=self.bundle.atom_mask
            ),
            "prediction_change_rmsd": rmsd(
                base["prediction_global"], changed["prediction_global"], self.bundle.atom_mask
            ),
        }

    @torch.no_grad()
    def run(
        self,
        *,
        n_rotations: int = 20,
        seed: int = 0,
        translation_scale: float = 10.0,
    ) -> dict[str, Any]:
        device = self.bundle.h_res.device
        dtype = self.bundle.noisy_global.dtype
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        rotation_rows = []
        for _ in range(int(n_rotations)):
            q = random_rotation(generator=generator, device=device, dtype=dtype)
            zero = torch.zeros(3, device=device, dtype=dtype)
            rotation_rows.append(self.one_transform(q, zero))

        identity = torch.eye(3, device=device, dtype=dtype)
        translation = torch.randn(
            3, generator=generator, device=device, dtype=dtype
        ) * float(translation_scale)
        translation_only = self.one_transform(identity, translation)
        paired_abcd = None
        if self.bundle.h_res_q is not None:
            paired_abcd = self.one_transform(
                self.bundle.paired_rotation,
                self.bundle.paired_translation,
                use_recomputed_pair=True,
            )

        arm_summary = aggregate_numeric(rotation_rows)
        verdicts: dict[str, Any] = {}
        for arm in ARMS:
            summary = arm_summary[arm.name]["C_coordinate_only"]
            hidden_max = summary["invariant"]["atom_feats"]["relative_frobenius"]["max"]
            coord_max = summary["prediction_global"]["eq_rmsd"]["max"]
            verdicts[arm.name] = {
                "rotation_sensitive": bool(
                    hidden_max > self.invariant_tolerance
                    or coord_max > self.coordinate_tolerance
                ),
                "max_atom_feature_relative_error": hidden_max,
                "max_coordinate_equivariance_rmsd": coord_max,
            }

        return {
            "schema_version": 1,
            "n_rotations": int(n_rotations),
            "seed": int(seed),
            "tolerances": {
                "invariant": self.invariant_tolerance,
                "coordinate_angstrom": self.coordinate_tolerance,
            },
            "has_recomputed_h_res_pair": self.bundle.h_res_q is not None,
            "paired_abcd": paired_abcd,
            "rotation_summary": arm_summary,
            "translation_only": translation_only,
            "chi_positive_control": {
                arm.name: self.chi_positive_control(arm) for arm in ARMS
            },
            "reference_quality": {
                arm.name: self._quality(
                    self._run(
                        arm,
                        h_res=self.bundle.h_res,
                        aa_logits=self.bundle.aa_logits,
                        noisy_local=self.bundle.noisy_local,
                        noisy_global=self.bundle.noisy_global,
                        ca_coords=self.bundle.ca_coords,
                        frame_r=self.bundle.frame_R,
                        frame_t=self.bundle.frame_t,
                    )["prediction_global"]
                )
                for arm in ARMS
            },
            "verdicts": verdicts,
            "metadata": self.bundle.metadata or {},
        }
