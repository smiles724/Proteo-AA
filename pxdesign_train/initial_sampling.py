"""Backend-independent initial backbone sampling and isolated random streams."""
from contextlib import contextmanager
import torch
import numpy as np
import random


class RandomStream:
    """Own Torch, NumPy/SciPy and Python RNG state without consuming the caller's stream."""
    def __init__(self, seed):
        np_before, py_before = np.random.get_state(), random.getstate()
        try:
            np.random.seed(seed)
            random.seed(seed)
            self.numpy, self.python = np.random.get_state(), random.getstate()
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                self.cpu = torch.get_rng_state()
                self.cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        finally:
            np.random.set_state(np_before)
            random.setstate(py_before)

    @contextmanager
    def use(self):
        np_before, py_before = np.random.get_state(), random.getstate()
        with torch.random.fork_rng():
            torch.set_rng_state(self.cpu)
            np.random.set_state(self.numpy)
            random.setstate(self.python)
            if self.cuda:
                torch.cuda.set_rng_state_all(self.cuda)
            try:
                yield
            finally:
                self.cpu = torch.get_rng_state()
                self.cuda = torch.cuda.get_rng_state_all() if self.cuda else []
                self.numpy, self.python = np.random.get_state(), random.getstate()
                np.random.set_state(np_before)
                random.setstate(py_before)


def native_backbone(model, feat, s_inputs, s_trunk, z_trunk, schedule, observer=None):
    """Observe official sampling calls; do not alter coordinates or consume RNG."""
    captured = {}
    def denoise(**kwargs):
        for key in ("pair_z", "p_lm", "c_l"):
            kwargs.setdefault(key, None)
        captured.setdefault("first_input_xyz", kwargs["x_noisy"].clone())
        xyz = model.diffusion_module(**kwargs)
        captured.update(input_xyz=kwargs["x_noisy"].clone(), sigma=kwargs["t_hat_noise_level"].clone(),
            denoised_xyz=xyz.clone(), a_token=getattr(model, "_a_token_cache", None),
            q=getattr(model, "_q_skip_cache", None), frame="last_native_denoiser_input")
        if observer is not None:
            observer(kwargs, xyz)
        return xyz
    xyz = model.sample_diffusion(denoise_net=denoise, input_feature_dict=feat,
        s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk, noise_schedule=schedule, N_sample=1)
    if xyz.ndim != 3 or xyz.shape[0] != 1:
        raise ValueError(f"Native binding requires [1,atom,xyz], received {tuple(xyz.shape)}")
    if not captured:
        raise ValueError("Native sampler made no denoising calls")
    return xyz, captured


def sample_initial(model, feat, s_inputs, s_trunk, z_trunk, schedule, *, sampler, target_policy):
    if target_policy not in ("joint", "fixed_context"):
        raise ValueError(f"Unknown initial target policy {target_policy}")
    if sampler == "pxdesign_native":
        if target_policy != "joint":
            raise ValueError("Official native sampling requires joint denoising; choose minimal_euler for fixed-context adaptation")
        return native_backbone(model, feat, s_inputs, s_trunk, z_trunk, schedule)
    if sampler != "minimal_euler":
        raise ValueError(f"Unknown backbone sampler {sampler}")
    xyz = schedule[0] * torch.randn(1, feat["atom_to_token_idx"].numel(), 3, device=s_inputs.device, dtype=s_inputs.dtype)
    design = feat["design_token_mask"].bool()[feat["atom_to_token_idx"].long()]
    fixed = feat["fixed_atom_xyz"].to(xyz)[None] if target_policy == "fixed_context" else None
    def clamp(value):
        return torch.where(design[None,:,None], value, fixed) if fixed is not None else value
    xyz = clamp(xyz)
    for current, following in zip(schedule[:-1], schedule[1:]):
        sigma = current.reshape(1)
        denoised = model.diffusion_module(x_noisy=xyz, t_hat_noise_level=sigma,
            input_feature_dict=feat, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk,
            pair_z=None, p_lm=None, c_l=None)
        captured = dict(input_xyz=xyz, sigma=sigma, denoised_xyz=denoised,
            a_token=getattr(model, "_a_token_cache", None), q=getattr(model, "_q_skip_cache", None), frame="last_euler_input")
        xyz = clamp(xyz + (following-current) * (xyz-denoised) / current)
    return clamp(denoised), captured
