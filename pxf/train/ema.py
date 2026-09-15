"""Weight averaging for FaMPNN training.

Appendix B.2 reports that the PDB models used the *post-hoc* EMA procedure of
Karras et al. (2024), choosing the EMA length after training had finished --- 1%
for the 0.3 A model and 25% for the 0.0 A one.

Two modes are provided, and the distinction is deliberate:

``relative_length``
    A running average whose window is a fixed fraction ``f`` of training so far,
    i.e. decay ``1 - 1/(f * step)``. This realizes "EMA length of 25%" in the
    direct sense and needs no post-hoc machinery, but it is *not* bit-identical to
    Karras's power-function profile.

``decay``
    A constant-decay EMA, the usual baseline.

For a faithful reproduction of the paper's procedure you need the length chosen
*after* training, which means keeping snapshots and reconstructing the profile
afterwards. :meth:`EMA.snapshot_state` exists for that; the reconstruction step
itself (Karras's sigma_rel to gamma mapping) is **not** implemented here, and
picking a length up front via ``relative_length`` is the approximation this code
makes. See :mod:`pxf.train.trainer` for how snapshots are written.
"""
import copy
import torch


class EMA:
    """Shadow copy of a model's floating-point parameters and buffers."""

    def __init__(self, model, *, decay=None, relative_length=None, device=None):
        if (decay is None) == (relative_length is None):
            raise ValueError("Give exactly one of decay or relative_length")
        if relative_length is not None and not 0 < float(relative_length) <= 1:
            raise ValueError(f"relative_length must be in (0, 1], got {relative_length}")
        self.decay = None if decay is None else float(decay)
        self.relative_length = None if relative_length is None else float(relative_length)
        self.step = 0
        self.shadow = {name: value.detach().clone().to(device or value.device)
                       for name, value in self._floats(model)}

    @staticmethod
    def _floats(model):
        for name, value in list(model.named_parameters()) + list(model.named_buffers()):
            if value.dtype.is_floating_point:
                yield name, value

    def current_decay(self):
        """Decay for the step about to be taken."""
        if self.decay is not None:
            return self.decay
        # Window of relative_length * step: at step n, average the last f*n updates.
        window = max(1.0, self.relative_length * max(self.step, 1))
        return max(0.0, 1.0 - 1.0 / window)

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        decay = self.current_decay()
        for name, value in self._floats(model):
            shadow = self.shadow[name]
            if value.shape != shadow.shape:
                raise ValueError(f"EMA shape mismatch for {name}")
            shadow.mul_(decay).add_(value.detach().to(shadow.device), alpha=1.0 - decay)
        return decay

    @torch.no_grad()
    def copy_to(self, model):
        """Write the averaged weights into ``model`` in place."""
        for name, value in self._floats(model):
            value.copy_(self.shadow[name].to(value.device))

    def swapped_into(self, model):
        """Context manager evaluating ``model`` under the averaged weights."""
        return _Swap(self, model)

    def state_dict(self):
        return dict(step=self.step, decay=self.decay,
                    relative_length=self.relative_length,
                    shadow={k: v.cpu() for k, v in self.shadow.items()})

    def load_state_dict(self, state):
        self.step = int(state["step"])
        self.decay = state["decay"]
        self.relative_length = state["relative_length"]
        for name, value in state["shadow"].items():
            if name in self.shadow:
                self.shadow[name].copy_(value.to(self.shadow[name].device))

    def snapshot_state(self):
        """A plain weight dict, for post-hoc EMA reconstruction after training."""
        return {name: value.detach().cpu().clone() for name, value in self.shadow.items()}


class _Swap:
    def __init__(self, ema, model):
        self.ema, self.model = ema, model
        self.backup = None

    def __enter__(self):
        self.backup = {name: value.detach().clone()
                       for name, value in EMA._floats(self.model)}
        self.ema.copy_to(self.model)
        return self.model

    def __exit__(self, *exc):
        with torch.no_grad():
            for name, value in EMA._floats(self.model):
                value.copy_(self.backup[name])
        self.backup = None
        return False
