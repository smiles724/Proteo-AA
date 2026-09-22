"""Record a sampler state and resume from it exactly.

A one-event intervention has to compare arms that differ *only* in the
injected residual. That needs the solver resumable at a chosen denoiser
invocation, which in turn needs more than the coordinates:

**The sigma the denoiser receives is not the scheduled sigma.** Protenix churns
before denoising -- ``gamma = gamma0 if c_tau > gamma_min else 0``, then
``t_hat = c_tau_last * (gamma + 1)`` and noise is added to reach it. PXDesign's
published config sets ``gamma0 = 1.0, gamma_min = 0.01``, so across the whole
useful range ``t_hat = 2 * c_tau_last``: a replay that fed the scheduled level
back in would denoise at half the noise the original saw. So the *actual*
``x_noisy`` and ``t_hat`` are recorded, after augmentation and after churn,
immediately before the call.

**Every step begins with a random rigid motion.** ``centre_random_augmentation``
re-centres and randomly rotates the whole coordinate set each step, drawing from
the global RNG. Two consequences: a resumed trajectory must restore the RNG or
its later steps diverge, and "the target chain stays fixed" cannot mean fixed in
this frame -- it is only fixed up to a per-step rigid motion, which is why
:class:`FixedTarget` re-imposes the reference by superposition rather than by
assignment.

**One loop serves the original trajectory and every replay.** Writing a separate
replay path would make "the replay reproduces the trajectory" a coincidence
between two implementations; here it is structural. What that trades away is
fidelity to upstream, so :func:`run_trajectory` is a transcription of
``generator.sample_diffusion``'s chunk loop and
``tests/test_couple_replay.py::test_the_transcription_matches_upstream`` pins it
against the real thing.

**BB and FaMPNN need separate streams.** Both draw from the global RNG, and
upstream takes no ``generator``. So a stream is a saved global state swapped in
around the calls that belong to it: running FaMPNN between two solver steps then
cannot shift the backbone trajectory, which it otherwise would even with the
feedback disabled.

**A stream is torch *and* numpy.** ``centre_random_augmentation`` draws its
translation from torch but its rotation from
``scipy.spatial.transform.Rotation.random``, which uses **numpy's** global RNG
(``protenix/model/utils.py::uniform_random_rotation``). Seeding torch alone
therefore reproduces nothing: measured, two augmentations under the same
``torch.manual_seed`` differ. A consequence beyond replay is that PXDesign
sampling was never reproducible from a torch seed by itself, whatever the
launcher set.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field, replace

import numpy as np
import torch


class RngStream:
    """A named stream over the global RNG, because upstream takes no generator.

    ``centre_random_augmentation`` and the churn draw both use the global RNG.
    Handing FaMPNN its own stream therefore means swapping the global state in
    and out around each subsystem's work, rather than passing a generator down.
    """

    def __init__(self, name, seed, *, device=None):
        self.name = str(name)
        self.seed = int(seed)
        self.device = device
        saved_cpu = torch.random.get_rng_state()
        saved_numpy = np.random.get_state()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._cpu = torch.random.get_rng_state()
        self._numpy = np.random.get_state()
        self._cuda = (
            torch.cuda.get_rng_state(device)
            if device is not None and torch.cuda.is_available()
            else None
        )
        torch.random.set_rng_state(saved_cpu)
        np.random.set_state(saved_numpy)

    @contextmanager
    def active(self):
        """Run a block with this stream's state installed globally."""
        outer_cpu = torch.random.get_rng_state()
        outer_numpy = np.random.get_state()
        outer_cuda = (
            torch.cuda.get_rng_state(self.device)
            if self.device is not None and torch.cuda.is_available()
            else None
        )
        torch.random.set_rng_state(self._cpu)
        np.random.set_state(self._numpy)
        if self._cuda is not None:
            torch.cuda.set_rng_state(self._cuda, self.device)
        try:
            yield self
        finally:
            self._cpu = torch.random.get_rng_state()
            self._numpy = np.random.get_state()
            if self.device is not None and torch.cuda.is_available():
                self._cuda = torch.cuda.get_rng_state(self.device)
            torch.random.set_rng_state(outer_cpu)
            np.random.set_state(outer_numpy)
            if outer_cuda is not None:
                torch.cuda.set_rng_state(outer_cuda, self.device)

    def capture(self, *, live=True):
        """Snapshot this stream. ``live`` reads the installed global state.

        Must be live when called from inside :meth:`active`: ``self._cpu`` is
        only written back on exit, so a mid-trajectory snapshot of it would be
        the state from the *start* of the trajectory. That reads as a replay
        that diverges for no visible reason.
        """
        cpu = torch.random.get_rng_state() if live else self._cpu
        numpy_state = np.random.get_state() if live else self._numpy
        cuda = self._cuda
        if live and self.device is not None and torch.cuda.is_available():
            cuda = torch.cuda.get_rng_state(self.device)
        return dict(
            name=self.name,
            seed=self.seed,
            cpu=cpu.clone(),
            numpy=tuple(v.copy() if hasattr(v, "copy") else v for v in numpy_state),
            cuda=None if cuda is None else cuda.clone(),
        )

    @contextmanager
    def protected(self):
        """Run a block without letting it advance this stream.

        For the feedback callback, which runs FaMPNN. FaMPNN draws from the
        global RNG, so without this it consumes the backbone's randomness and
        the remaining trajectory diverges -- an arm difference produced by the
        side computation rather than by the residual.
        """
        saved_cpu = torch.random.get_rng_state()
        saved_numpy = np.random.get_state()
        saved_cuda = (
            torch.cuda.get_rng_state(self.device)
            if self.device is not None and torch.cuda.is_available()
            else None
        )
        try:
            yield
        finally:
            torch.random.set_rng_state(saved_cpu)
            np.random.set_state(saved_numpy)
            if saved_cuda is not None:
                torch.cuda.set_rng_state(saved_cuda, self.device)

    def restore(self, captured, *, live=True):
        self._cpu = captured["cpu"].clone()
        self._numpy = tuple(
            v.copy() if hasattr(v, "copy") else v for v in captured["numpy"]
        )
        self._cuda = None if captured["cuda"] is None else captured["cuda"].clone()
        if live:
            torch.random.set_rng_state(self._cpu)
            np.random.set_state(self._numpy)
            if self._cuda is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(self._cuda, self.device)
        return self


@dataclass
class SamplerState:
    """One denoiser invocation, recorded well enough to resume from it.

    ``sigma`` is the churn-adjusted ``t_hat`` the denoiser is actually called
    with. ``c_tau_last`` and ``c_tau`` are the scheduled levels the step spans
    and are kept because the Euler update needs ``c_tau``, not because the
    denoiser sees either of them.
    """

    x_noisy: torch.Tensor
    sigma: torch.Tensor
    step: int
    substage: int
    c_tau_last: float
    c_tau: float
    rng: dict
    n_steps: int
    identity: dict = field(default_factory=dict)
    target_reference: torch.Tensor | None = None
    target_atom_mask: torch.Tensor | None = None

    @property
    def key(self):
        """The event key: a solver position, not a noise threshold.

        A sigma threshold fires on whichever step happens to fall below it and
        silently moves when the schedule or the churn factor changes.
        """
        return (int(self.step), int(self.substage))

    def to(self, device):
        return replace(
            self,
            x_noisy=self.x_noisy.to(device),
            sigma=self.sigma.to(device),
            target_reference=(
                None if self.target_reference is None else self.target_reference.to(device)
            ),
            target_atom_mask=(
                None if self.target_atom_mask is None else self.target_atom_mask.to(device)
            ),
        )

    def detach_cpu(self):
        return replace(
            self,
            x_noisy=self.x_noisy.detach().cpu(),
            sigma=self.sigma.detach().cpu(),
            target_reference=(
                None
                if self.target_reference is None
                else self.target_reference.detach().cpu()
            ),
            target_atom_mask=(
                None
                if self.target_atom_mask is None
                else self.target_atom_mask.detach().cpu()
            ),
        )


class FixedTarget:
    """Hold the target chain's atoms on their reference coordinates.

    **Zero residual on target tokens is not the same thing.** The adapter can
    be masked to the design region and the target's *predicted* coordinates
    still move: attention carries generated-chain changes into target tokens
    inside the decoder, and the solver's Euler update then writes them. So the
    policy is enforced on the coordinates themselves, in every arm including
    the baseline, rather than inferred from the residual being zero.

    Enforced by superposition, not assignment, because every solver step
    re-centres and randomly rotates the whole system: the reference is only
    meaningful up to a rigid motion, so the target is rotated onto its
    reference and the same transform applied to the generated chain, which
    keeps the two in one frame.

    **Applied at three points per step, and the first one matters most.**
    Pinning only after the Euler update -- the obvious place -- leaves the churn
    to noise the target before the next denoise: at ``t_hat = 2*sigma`` the churn
    adds ``sigma*sqrt(3)`` per coordinate, about 1.5 A at sigma = 0.86. The
    denoiser then conditions the generated chain against a smeared target for
    the whole trajectory, which shows up as nonsense interface geometry rather
    than as an error. So the target is pinned after the augmentation, again
    after the churn (immediately before the denoiser reads it), and on the
    denoised output -- the last of which is where ``pxdesign_train/stage4.py``
    applies its own ``torch.where(fixed_atom_mask, fixed_atom_xyz, xyz)``.
    """

    def __init__(self, reference, atom_mask):
        self.reference = reference
        self.atom_mask = atom_mask.bool()
        if not bool(self.atom_mask.any()):
            raise ValueError("FixedTarget needs at least one target atom")

    def apply(self, coords):
        """Return ``coords`` rigidly moved so the target sits on its reference."""
        flat = coords.reshape(-1, coords.shape[-1])
        keep = self.atom_mask.reshape(-1)
        moving = flat[keep].double()
        fixed = self.reference.reshape(-1, 3)[keep].double()
        if moving.shape[0] < 3:
            return coords
        mu_m, mu_f = moving.mean(0, keepdim=True), fixed.mean(0, keepdim=True)
        u, _s, vh = torch.linalg.svd((moving - mu_m).T @ (fixed - mu_f))
        sign = torch.sign(torch.det(u @ vh))
        correction = torch.eye(3, dtype=torch.float64, device=coords.device)
        correction[2, 2] = sign
        rotation = u @ correction @ vh
        moved = ((flat.double() - mu_m) @ rotation + mu_f).to(coords.dtype)
        out = moved.reshape(coords.shape).clone()
        # The target is then pinned exactly, absorbing the residual rotation
        # error; the generated chain keeps the transform only.
        out.reshape(-1, 3)[keep] = self.reference.reshape(-1, 3)[keep].to(out.dtype)
        return out

    def displacement(self, coords):
        """Worst-atom displacement of the target from its reference, Angstroms."""
        flat = coords.reshape(-1, coords.shape[-1])[self.atom_mask.reshape(-1)]
        ref = self.reference.reshape(-1, 3)[self.atom_mask.reshape(-1)]
        return float((flat.float() - ref.float()).norm(dim=-1).max())


def _event_key_set(event):
    """The set of solver invocations that are events.

    ``event`` is one ``(step, substage)`` key, or several. A single key stays
    a single key: ``(350, 0)`` is two ints, not two keys, and reading it as a
    sequence of keys would silently arm steps 350 and 0. So a flat pair of
    ints is treated as ONE key, and anything whose first element is itself a
    pair is treated as a collection.
    """
    if event is None:
        return frozenset()
    items = list(event)
    if not items:
        return frozenset()
    if all(isinstance(part, int) for part in items):
        if len(items) != 2:
            raise ValueError(
                f"a single event key must be (step, substage); got {event!r}"
            )
        return frozenset({(int(items[0]), int(items[1]))})
    keys = set()
    for item in items:
        pair = tuple(int(part) for part in item)
        if len(pair) != 2:
            raise ValueError(f"event key must be (step, substage); got {item!r}")
        keys.add(pair)
    return frozenset(keys)


def run_trajectory(
    *,
    denoise,
    schedule,
    n_atom,
    device,
    dtype=torch.float32,
    batch_shape=(),
    n_sample=1,
    gamma0=1.0,
    gamma_min=0.01,
    noise_scale_lambda=1.003,
    # PXDesign's setting, not Protenix's. `configs_base` gives the design model
    # `eta_schedule = {type: piecewise_65, min: 1.0, max: 2.5}` and the
    # `pxdesign` CLI overrides it to a constant 2.5; 1.5 is Protenix's generic
    # default and was never PXDesign's. The Euler step is scaled by this at
    # every level, so the difference is not cosmetic.
    step_scale_eta=2.5,
    stream,
    record_steps=(),
    resume=None,
    event=None,
    feedback=None,
    after_event=None,
    fixed_target=None,
    identity=None,
):
    """Protenix's sampler loop, recordable and resumable.

    A transcription of ``generator.sample_diffusion``'s ``_chunk_sample_diffusion``
    (Protenix ``model/generator.py``), pinned against it by test. Returns
    ``(x0, records, stats)``.

    ``resume`` starts at a recorded state instead of from pure noise: its RNG is
    installed, its ``x_noisy``/``sigma`` are denoised directly, and the loop then
    continues from the following step. ``event`` is a ``(step, substage)`` key, or a collection of them; at each
    matching invocation ``feedback(state)`` supplies the residual for that
    invocation. With several keys the callback is invoked once per event, in
    trajectory order, and ``stats["injections"]`` counts them. Optional ``after_event(state, x_denoised)``
    observes that invocation's result before the solver advances. It runs
    under RNG protection, including when the event has no feedback. Its
    return value does not change the solver estimate.
    """
    schedule = schedule.to(device=device, dtype=torch.float32)
    record_steps = {int(s) for s in record_steps}
    records, stats = [], dict(calls=0, injections=0, augmentations=0)
    event_keys = _event_key_set(event)
    stats["events"] = len(event_keys)

    def one_call(x_noisy, sigma, step, substage, c_tau_last, c_tau):
        """Record, optionally inject, denoise. The only denoiser entry point."""
        state = SamplerState(
            x_noisy=x_noisy,
            sigma=sigma,
            step=step,
            substage=substage,
            c_tau_last=float(c_tau_last),
            c_tau=float(c_tau),
            rng=stream.capture(),
            n_steps=int(schedule.numel() - 1),
            identity=dict(identity or {}),
            target_reference=(None if fixed_target is None else fixed_target.reference),
            target_atom_mask=(None if fixed_target is None else fixed_target.atom_mask),
        )
        if step in record_steps:
            records.append(state.detach_cpu())
        residual = None
        on_event = state.key in event_keys
        if on_event and feedback is not None:
            # Protected: the callback runs FaMPNN, which draws from the global
            # RNG. Letting it advance this stream would shift every later step.
            # With several events this matters more, not less: an unprotected
            # decode at event k would shift every event after it as well.
            with stream.protected():
                residual = feedback(state)
            if residual is not None:
                stats["injections"] += 1
        stats["calls"] += 1
        x_denoised = denoise(x_noisy, sigma, feedback=residual)
        if on_event and after_event is not None:
            with stream.protected():
                after_event(state, x_denoised)
        return x_denoised

    with stream.active():
        if resume is None:
            x_l = schedule[0] * torch.randn(
                size=(*batch_shape, n_sample, n_atom, 3), device=device, dtype=dtype
            )
            start = 0
        else:
            stream.restore(resume.rng, live=True)
            x_noisy = resume.x_noisy.to(device=device, dtype=dtype)
            sigma = resume.sigma.to(device=device, dtype=torch.float32)
            if fixed_target is not None:
                x_noisy = fixed_target.apply(x_noisy)
            x_denoised = one_call(
                x_noisy,
                sigma,
                resume.step,
                resume.substage,
                resume.c_tau_last,
                resume.c_tau,
            )
            if fixed_target is not None:
                x_denoised = fixed_target.apply(x_denoised)
            drift = (x_noisy - x_denoised) / sigma.reshape(-1)[0]
            x_l = (
                x_noisy
                + step_scale_eta * (resume.c_tau - float(sigma.reshape(-1)[0])) * drift
            )
            if fixed_target is not None:
                x_l = fixed_target.apply(x_l)
            start = resume.step + 1

        from protenix.model.utils import centre_random_augmentation

        n_levels = int(schedule.numel())
        for step_i in range(start, n_levels - 1):
            # Tensor arithmetic, matching upstream: a float64 round-trip through
            # .tolist() shifts t_hat in the last bits and the trajectories part.
            c_tau_last, c_tau = schedule[step_i], schedule[step_i + 1]
            x_l = (
                centre_random_augmentation(x_input_coords=x_l, N_sample=1)
                .squeeze(dim=-3)
                .to(dtype)
            )
            stats["augmentations"] += 1
            if fixed_target is not None:
                # The augmentation just moved everything; put the target back.
                x_l = fixed_target.apply(x_l)
            gamma = float(gamma0) if float(c_tau) > gamma_min else 0.0
            t_hat = c_tau_last * (gamma + 1.0)
            delta_noise = torch.sqrt(t_hat**2 - c_tau_last**2)
            x_noisy = x_l + noise_scale_lambda * delta_noise * torch.randn(
                size=x_l.shape, device=device, dtype=dtype
            )
            if fixed_target is not None:
                # Before the denoiser reads it. The churn just put sigma*sqrt(3)
                # of noise on every atom including the target's.
                x_noisy = fixed_target.apply(x_noisy)
            sigma = t_hat.reshape(1).to(device=device, dtype=torch.float32)
            x_denoised = one_call(x_noisy, sigma, step_i, 0, c_tau_last, c_tau)
            if fixed_target is not None:
                # stage4.py's placement: force the fixed atoms in the prediction.
                x_denoised = fixed_target.apply(x_denoised)
            drift = (x_noisy - x_denoised) / t_hat
            x_l = x_noisy + step_scale_eta * (c_tau - t_hat) * drift
            if fixed_target is not None:
                x_l = fixed_target.apply(x_l)

    if fixed_target is not None:
        stats["target_max_displacement"] = fixed_target.displacement(x_l)
    return x_l, records, stats
