"""Phase 0: expose PXDesign's token features for reading and for feedback.

There are **two** coupling points, and they are deliberately different places
in the same forward pass.

**Late, at the decoder input.** The original SC -> BB adapter's site:

    a_token = self.layernorm_a(a_token)        <-- read here
    ...                                        <-- inject the residual here
    r_update = self.atom_attention_decoder(atom_to_token_idx, a_token, ...)

So ``a_token`` is captured as the output of ``layernorm_a`` and the residual is
added to the ``a`` argument of ``atom_attention_decoder``, which is positional
index 1 of its forward call. Everything between the conditioning and the decoder
has already run by then, so a correction here can only re-mix the final token
features.

**Early, at the conditioning output.** The new site:

    s_single, z_pair = self.diffusion_conditioning(...)   <-- inject here
    ...
    a_token = a_token + linear_no_bias_s(layernorm_s(s_single))
    a_token = self.diffusion_transformer(a=a_token, s=s_single, z=z_pair, ...)

which is upstream of the atom encoder, all the transformer blocks and the
decoder, so the correction is *conditioning* rather than a last-layer nudge.
:class:`ConditioningFeedback` carries it. Note what is not touched: ``s_trunk``,
``z_trunk``, ``ref_pos``, the noisy coordinates and the final ``a_token``. The
pretrained :class:`DiffusionConditioning` runs unmodified and its output is added
to, never replaced.

One :class:`BackboneTap` serves both. The payload's *type* selects the site -- a
tensor goes to the decoder, a :class:`ConditioningFeedback` to the conditioning
output -- so the two cannot be applied at once, and an arm that meant to inject
early but handed over a bare tensor lands somewhere that fails a width check
rather than silently at the old site.

This is done with hooks rather than by editing the submodule, deliberately:
:mod:`pxf.provenance` requires Protenix to be pristine and allows PXDesign only
the recorded embedders patch. Hooks keep that guarantee intact, and
:class:`BackboneTap` is a context manager so they are always removed --- a
leaked hook would silently perturb every later call, including evaluation.
"""

from dataclasses import dataclass

import torch

# Where the token features sit in AtomAttentionDecoder.forward:
#   forward(self, atom_to_token_idx, a, q_skip, c_skip, p_skip, ...)
# Protenix calls this BOTH ways depending on activation checkpointing --
# positionally through checkpoint_fn, and with a= as a keyword otherwise -- so the
# hook has to handle each. Assuming one form silently stops injecting under the
# other, which looks like an adapter that learned nothing.
DECODER_A_ARG = 1
DECODER_A_KWARG = "a"


@dataclass
class ConditioningFeedback:
    """Residuals for the pretrained conditioning's own outputs.

    ``delta_single``  ``[1, L, c_s]``, added to ``s_single``
    ``delta_pair``    ``[L, L, c_z]``, added to ``z_pair``, or ``None``

    Both widths come from the loaded conditioning module and **neither is
    ``c_token``**. On this donor ``c_s`` is 384, ``c_z`` is 128 and ``c_token``
    is 768, so a payload built against ``c_token`` is refused by the width check
    below rather than broadcast into place -- which is the point of checking.

    The singleton dimensions are the initial scope, not a convenience: one
    unpadded protein and one diffusion sample per forward. Several independently
    packed samples would each want their own pair residual, and broadcasting one
    across them would apply a correction computed from sample 0's side chains to
    sample 3's backbone.
    """

    delta_single: torch.Tensor | None = None
    delta_pair: torch.Tensor | None = None

    @property
    def requires_grad(self):
        """True if either residual carries a gradient back to the conditioner."""
        return any(
            t is not None and t.requires_grad
            for t in (self.delta_single, self.delta_pair)
        )

    def detach(self):
        from dataclasses import replace

        return replace(
            self,
            delta_single=(
                None if self.delta_single is None else self.delta_single.detach()
            ),
            delta_pair=None if self.delta_pair is None else self.delta_pair.detach(),
        )

    def norms(self):
        """Per-residue / per-pair residual norms, for the logs. Detached."""
        record = {}
        if self.delta_single is not None:
            record["delta_s_norm"] = float(
                self.delta_single.detach().norm(dim=-1).mean()
            )
        if self.delta_pair is not None:
            norm = self.delta_pair.detach().norm(dim=-1)
            record["delta_z_norm"] = float(norm.mean())
            record["delta_z_norm_max"] = float(norm.max()) if norm.numel() else 0.0
        return record


def add_conditioning_residual(target, delta, *, name, width_name, trailing):
    """``target + delta`` with the singleton contract enforced, not broadcast.

    ``trailing`` is how many of the target's axes the residual genuinely
    addresses: two for ``s_single`` (tokens, channels) and three for ``z_pair``
    (tokens, tokens, channels). Everything in front of them is a batch or sample
    axis, and every one of those must be 1 -- that is the contract, and checking
    it is what stops one protein's pair residual being silently broadcast across
    several independently packed samples.

    The distinction matters because the two payloads are both 3-D: ``[1, L, c_s]``
    carries a sample axis, ``[L, L, c_z]`` does not. Treating them alike would
    accept a two-sample ``s_single`` by matching its sample axis against the
    residual's own leading 1.
    """
    if delta is None:
        return target
    if delta.dim() != 3:
        raise ValueError(f"{name} must be a 3-D tensor, got {tuple(delta.shape)}")
    real = tuple(delta.shape[-trailing:])
    if any(int(size) != 1 for size in delta.shape[:-trailing]):
        raise ValueError(
            f"{name} is shaped {tuple(delta.shape)}; the axes in front of its "
            f"last {trailing} must be singleton"
        )
    if real[-1] != target.shape[-1]:
        raise ValueError(
            f"{name} has width {real[-1]} but the conditioning's {width_name} is "
            f"{target.shape[-1]}; c_token is not either of them"
        )
    if target.dim() < trailing or tuple(target.shape[-trailing:]) != real:
        raise ValueError(
            f"{name} addresses {real} but the conditioning output is "
            f"{tuple(target.shape)}"
        )
    leading = target.shape[:-trailing]
    if any(int(size) != 1 for size in leading):
        raise ValueError(
            f"the conditioning output has non-singleton leading dimensions "
            f"{tuple(leading)}. The conditioner is scoped to one protein and one "
            "diffusion sample per call; broadcasting one pair residual across "
            "several independently packed samples would apply one sample's side "
            "chains to another's backbone"
        )
    delta = delta.to(target.dtype).to(target.device)
    return target + delta.reshape((1,) * len(leading) + real)


class BackboneTap:
    """Capture ``a_token`` and optionally add a residual before the atom decoder.

    Usage::

        with BackboneTap(model.diffusion_module) as tap:
            x0 = denoise(...)              # tap.a_token now holds the features
            tap.feedback = delta_a         # applies to subsequent calls
            x1 = denoise(...)

    The tap records the features of the *most recent* denoiser evaluation, which
    is what a single coupling cycle needs. ``calls`` counts evaluations so a
    caller can assert it intercepted exactly the pass it meant to.
    """

    def __init__(self, diffusion_module, *, feedback=None):
        self.diffusion_module = diffusion_module
        self.layernorm = diffusion_module.layernorm_a
        self.decoder = diffusion_module.atom_attention_decoder
        # The early site. Absent on the stub modules the tap's own tests use, so
        # it is optional here and required only when a ConditioningFeedback
        # actually arrives.
        self.conditioning = getattr(diffusion_module, "diffusion_conditioning", None)
        self.feedback = feedback
        self.a_token = None
        self.calls = 0
        self.injections = 0
        self.conditioning_injections = 0
        self._handles = []

    # ---- hooks -----------------------------------------------------------

    def _capture(self, module, inputs, output):
        # Detaching here would cut BB -> SC gradients; keep the graph and let the
        # caller decide, since phase 1 trains an adapter fed by these features.
        self.a_token = output
        self.calls += 1
        return None

    def _condition(self, module, args, output):
        """Add the early residuals to ``(s_single, z_pair)``. Returns the new pair.

        A forward hook rather than a wrapper so the pretrained module runs
        exactly as it does uncoupled and the addition is provably *after* its
        output and *before* anything consumes it -- the atom encoder, the
        pre-transformer token addition and the transformer all read the values
        this hook returns.
        """
        feedback = self.feedback
        if not isinstance(feedback, ConditioningFeedback):
            return None
        if feedback.delta_single is None and feedback.delta_pair is None:
            return None
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError(
                "DiffusionConditioning returned "
                f"{type(output).__name__} rather than (s_single, z_pair); the "
                "early injection site has moved and would corrupt the call"
            )
        s_single, z_pair = output
        s_single = add_conditioning_residual(
            s_single,
            feedback.delta_single,
            name="delta_single",
            width_name="c_s",
            trailing=2,
        )
        z_pair = add_conditioning_residual(
            z_pair, feedback.delta_pair, name="delta_pair", width_name="c_z", trailing=3
        )
        self.conditioning_injections += 1
        return s_single, z_pair

    def _inject(self, module, args, kwargs):
        if self.feedback is None:
            return None
        if isinstance(self.feedback, ConditioningFeedback):
            # Handled at the early site. Returning None here is what keeps the
            # two mutually exclusive: an early arm never also perturbs a_token.
            return None
        delta = self.feedback
        if DECODER_A_KWARG in kwargs:
            target = kwargs[DECODER_A_KWARG]
            updated_kwargs = dict(kwargs)
            updated_kwargs[DECODER_A_KWARG] = self._add(target, delta)
            self.injections += 1
            return args, updated_kwargs
        if len(args) > DECODER_A_ARG:
            updated = list(args)
            updated[DECODER_A_ARG] = self._add(args[DECODER_A_ARG], delta)
            self.injections += 1
            return tuple(updated), kwargs
        raise ValueError(
            f"AtomAttentionDecoder was called with {len(args)} positional and "
            f"{sorted(kwargs)} keyword arguments; the token features are at "
            f"neither index {DECODER_A_ARG} nor {DECODER_A_KWARG!r}, so feedback "
            "injection would corrupt the call"
        )

    @staticmethod
    def _add(target, delta):
        if delta.shape[-1] != target.shape[-1]:
            raise ValueError(
                f"Feedback width {delta.shape[-1]} does not match token "
                f"features {target.shape[-1]}"
            )
        delta = delta.to(target.dtype).to(target.device)
        return target + (delta.expand_as(target) if delta.shape != target.shape else delta)

    # ---- lifecycle -------------------------------------------------------

    def install(self):
        if self._handles:
            raise RuntimeError("BackboneTap is already installed")
        self._handles = [
            self.layernorm.register_forward_hook(self._capture),
            self.decoder.register_forward_pre_hook(self._inject, with_kwargs=True),
        ]
        if self.conditioning is not None:
            self._handles.append(
                self.conditioning.register_forward_hook(self._condition)
            )
        return self

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.remove()
        return False

    def reset(self):
        self.a_token = None
        self.feedback = None
        self.calls = self.injections = self.conditioning_injections = 0
        return self


def conditioning_widths(model):
    """``(c_s, c_z)`` read off the loaded conditioning module.

    Read rather than configured, and never assumed equal to ``c_token``: the
    early residuals are added to ``s_single`` and ``z_pair``, which are 384 and
    128 wide on this donor while ``c_token`` is 768. A conditioner built against
    ``c_token`` would be a shape error at the hook -- but only if the widths came
    from the model, which is why they do.
    """
    module = getattr(model, "diffusion_module", model)
    conditioning = getattr(module, "diffusion_conditioning", None)
    if conditioning is None:
        raise ValueError(
            "this diffusion module has no diffusion_conditioning submodule, so "
            "there is no early injection site to read widths from"
        )
    c_s, c_z = getattr(conditioning, "c_s", None), getattr(conditioning, "c_z", None)
    if not c_s or not c_z:
        raise ValueError(
            f"DiffusionConditioning reports c_s={c_s!r} c_z={c_z!r}; the early "
            "conditioner cannot be sized"
        )
    return int(c_s), int(c_z)


def token_feature_dim(model):
    """Width of PXDesign's token features, i.e. the adapter's backbone dimension."""
    module = getattr(model, "diffusion_module", model)
    for attribute in ("c_token",):
        value = getattr(module, attribute, None)
        if value:
            return int(value)
    weight = getattr(module.layernorm_a, "weight", None)
    if weight is not None:
        return int(weight.shape[-1])
    raise ValueError("Could not determine the token feature width")


@dataclass
class Conditioning:
    """Conditioning computed once per target, reused across denoiser evaluations.

    PXDesign's sampler recomputes nothing per step; it threads these through
    every call, so a coupled controller that evaluates the denoiser itself must
    hold them too.
    """

    s_inputs: torch.Tensor
    s_trunk: torch.Tensor
    z_trunk: torch.Tensor
    input_feature_dict: dict

    @classmethod
    def build(cls, model, input_feature_dict, *, chunk_size=None):
        s_inputs, s_trunk, z_trunk = model.get_condition_embedding(
            input_feature_dict=input_feature_dict, chunk_size=chunk_size
        )
        return cls(
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            input_feature_dict=input_feature_dict,
        )


def denoise(
    model,
    conditioning,
    x_noisy,
    sigma,
    *,
    tap=None,
    feedback=None,
    chunk_size=None,
    inplace_safe=False,
):
    """One denoiser evaluation, returning ``(x_denoised, a_token)``.

    Called with the same keyword set PXDesign's own sampler uses, so the coupled
    controller is doing exactly what an ordinary sampling step does, plus the tap.
    """
    owned = tap is None
    tap = tap or BackboneTap(model.diffusion_module)
    if owned:
        tap.install()
    try:
        tap.feedback = feedback
        x_denoised = model.diffusion_module(
            x_noisy=x_noisy,
            t_hat_noise_level=sigma,
            input_feature_dict=conditioning.input_feature_dict,
            s_inputs=conditioning.s_inputs,
            s_trunk=conditioning.s_trunk,
            z_trunk=conditioning.z_trunk,
            chunk_size=chunk_size,
            inplace_safe=inplace_safe,
        )
        return x_denoised, tap.a_token
    finally:
        if owned:
            tap.remove()
