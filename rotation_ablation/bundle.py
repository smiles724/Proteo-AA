"""Serializable, canonical inputs for one paired S_phi experiment."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch

from pxdesign_train.sidechain.frames import to_global, to_local


@dataclass
class SidechainInputBundle:
    """One fixed protein/noise/sigma row at the S_phi boundary.

    Coordinates are stored in both representations so the coordinate 2x2 arms
    change only the representation handed to ``w_xyz``. ``h_res_q`` and
    ``aa_logits_q`` are optional recomputed-backbone values for the physically
    paired B/D interventions. When absent, the runner still performs the exact
    coordinate-path C test and labels B/D as unavailable.
    """

    h_res: torch.Tensor
    aa_logits: torch.Tensor
    atom_name_ids: torch.Tensor
    atom_mask: torch.Tensor
    noisy_local: torch.Tensor
    noisy_global: torch.Tensor
    time: torch.Tensor
    ca_coords: torch.Tensor
    frame_R: torch.Tensor
    frame_t: torch.Tensor
    h_res_q: torch.Tensor | None = None
    aa_logits_q: torch.Tensor | None = None
    paired_rotation: torch.Tensor | None = None
    paired_translation: torch.Tensor | None = None
    bb_local: torch.Tensor | None = None
    res_mask: torch.Tensor | None = None
    ctx_mask: torch.Tensor | None = None
    gt_local: torch.Tensor | None = None
    gt_global: torch.Tensor | None = None
    residue_type_idx: torch.Tensor | None = None
    context_coords: torch.Tensor | None = None
    context_mask: torch.Tensor | None = None
    residue_category: torch.Tensor | None = None
    aa_correct: torch.Tensor | None = None
    metadata: dict[str, Any] | None = None

    def validate(self) -> "SidechainInputBundle":
        tensors = {
            "h_res": self.h_res,
            "aa_logits": self.aa_logits,
            "atom_name_ids": self.atom_name_ids,
            "atom_mask": self.atom_mask,
            "noisy_local": self.noisy_local,
            "noisy_global": self.noisy_global,
            "ca_coords": self.ca_coords,
            "frame_R": self.frame_R,
            "frame_t": self.frame_t,
        }
        batch_residue = self.h_res.shape[:2]
        for name in ("aa_logits", "atom_name_ids", "atom_mask", "noisy_local", "noisy_global"):
            if tensors[name].shape[:2] != batch_residue:
                raise ValueError(
                    f"{name} begins {tuple(tensors[name].shape[:2])}, expected {tuple(batch_residue)}"
                )
        if self.noisy_local.shape != self.noisy_global.shape:
            raise ValueError("noisy_local and noisy_global must have the same shape")
        if self.noisy_local.shape[-1] != 3:
            raise ValueError("coordinate tensors must end in 3")
        if self.atom_mask.shape != self.atom_name_ids.shape:
            raise ValueError("atom_mask and atom_name_ids must have the same shape")
        if self.noisy_local.shape[:-1] != self.atom_mask.shape:
            raise ValueError("coordinate atom axis must match atom_mask")
        if self.frame_R.shape[:2] != batch_residue or self.frame_R.shape[-2:] != (3, 3):
            raise ValueError("frame_R must be [B,L,3,3]")
        if self.frame_t.shape != self.ca_coords.shape or self.frame_t.shape[-1] != 3:
            raise ValueError("frame_t and ca_coords must both be [B,L,3]")
        if self.h_res_q is not None and self.h_res_q.shape != self.h_res.shape:
            raise ValueError("h_res_q must match h_res")
        if self.aa_logits_q is not None and self.aa_logits_q.shape != self.aa_logits.shape:
            raise ValueError("aa_logits_q must match aa_logits")
        if self.h_res_q is not None:
            if self.paired_rotation is None or self.paired_translation is None:
                raise ValueError(
                    "h_res_q requires paired_rotation [3,3] and paired_translation [3]"
                )
            if self.paired_rotation.shape != (3, 3) or self.paired_translation.shape != (3,):
                raise ValueError("paired_rotation/paired_translation must be [3,3]/[3]")
        return self

    def to(self, device: torch.device | str) -> "SidechainInputBundle":
        values = {}
        for f in fields(self):
            value = getattr(self, f.name)
            values[f.name] = value.to(device) if isinstance(value, torch.Tensor) else value
        return SidechainInputBundle(**values)

    @classmethod
    def from_module_call(
        cls,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        local_coord_input: bool,
        h_res_q: torch.Tensor | None = None,
        aa_logits_q: torch.Tensor | None = None,
        paired_rotation: torch.Tensor | None = None,
        paired_translation: torch.Tensor | None = None,
        frame_R: torch.Tensor | None = None,
        frame_t: torch.Tensor | None = None,
        ca_coords: torch.Tensor | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "SidechainInputBundle":
        names = [
            "h_res",
            "restype_logits",
            "atom_name_ids",
            "atom_mask",
            "noisy_local",
            "t",
            "ca_coords",
            "frame_R",
            "frame_t",
            "bb_local",
            "res_mask",
            "ctx_mask",
        ]
        call = dict(zip(names, args))
        call.update(kwargs)
        required = names[:6]
        missing = [name for name in required if call.get(name) is None]
        if missing:
            raise ValueError(f"captured S_phi call is missing {missing}")
        frame_r = frame_R if frame_R is not None else call.get("frame_R")
        frame_t_value = frame_t if frame_t is not None else call.get("frame_t")
        ca = ca_coords if ca_coords is not None else call.get("ca_coords")
        if frame_r is None or frame_t_value is None:
            raise ValueError(
                "rotation ablation requires frame_R/frame_t; with the default CA-anchored "
                "head pass output['sc_frame_R']/output['sc_frame_t'] as overrides"
            )
        # The enclosing model restores its collapsed batch dimension on output
        # in reduced/warmup mode, while the captured S_phi call itself is [1,L,...].
        if call["h_res"].dim() == 3 and call["h_res"].shape[0] == 1:
            if frame_r.dim() == 3:
                frame_r = frame_r.unsqueeze(0)
            if frame_t_value.dim() == 2:
                frame_t_value = frame_t_value.unsqueeze(0)
            if ca is not None and ca.dim() == 2:
                ca = ca.unsqueeze(0)
        coords = call["noisy_local"]
        if local_coord_input:
            noisy_local = coords
            noisy_global = to_global(
                coords.float(), frame_r.float(), frame_t_value.float()
            ).to(coords)
        else:
            noisy_global = coords
            noisy_local = to_local(
                coords.float(), frame_r.float(), frame_t_value.float()
            ).to(coords)
        if ca is None:
            ca = frame_t_value
        return cls(
            h_res=call["h_res"].detach(),
            aa_logits=call["restype_logits"].detach(),
            atom_name_ids=call["atom_name_ids"].detach(),
            atom_mask=call["atom_mask"].detach(),
            noisy_local=noisy_local.detach(),
            noisy_global=noisy_global.detach(),
            time=torch.as_tensor(call["t"]).detach(),
            ca_coords=ca.detach(),
            frame_R=frame_r.detach(),
            frame_t=frame_t_value.detach(),
            h_res_q=h_res_q.detach() if h_res_q is not None else None,
            aa_logits_q=aa_logits_q.detach() if aa_logits_q is not None else None,
            paired_rotation=(
                paired_rotation.detach() if paired_rotation is not None else None
            ),
            paired_translation=(
                paired_translation.detach() if paired_translation is not None else None
            ),
            bb_local=call.get("bb_local").detach() if call.get("bb_local") is not None else None,
            res_mask=call.get("res_mask").detach() if call.get("res_mask") is not None else None,
            ctx_mask=call.get("ctx_mask").detach() if call.get("ctx_mask") is not None else None,
            metadata=metadata,
        ).validate()


class SidechainCallCapture:
    """Context manager capturing the first call into a SideChainModule."""

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.call: tuple[tuple[Any, ...], dict[str, Any]] | None = None
        self._handle = None

    def _hook(self, _module, args, kwargs):
        if self.call is None:
            self.call = (args, kwargs)

    def __enter__(self) -> "SidechainCallCapture":
        self._handle = self.module.register_forward_pre_hook(self._hook, with_kwargs=True)
        return self

    def __exit__(self, *_exc) -> None:
        if self._handle is not None:
            self._handle.remove()


def save_bundle(bundle: SidechainInputBundle, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f.name: getattr(bundle, f.name) for f in fields(bundle)}
    torch.save(payload, path)


def load_bundle(path: str | Path, *, map_location: str | torch.device = "cpu") -> SidechainInputBundle:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("bundle file must contain a dictionary")
    valid = {f.name for f in fields(SidechainInputBundle)}
    unknown = set(payload) - valid
    if unknown:
        raise ValueError(f"unknown bundle fields: {sorted(unknown)}")
    return SidechainInputBundle(**payload).validate()
