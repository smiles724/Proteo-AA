"""Phase 0: expose PXDesign's token features for reading and for feedback.

The coupling point inside the diffusion module is narrow and specific:

    a_token = self.layernorm_a(a_token)        <-- read here
    ...                                        <-- inject the residual here
    r_update = self.atom_attention_decoder(atom_to_token_idx, a_token, ...)

So ``a_token`` is captured as the output of ``layernorm_a`` and the SC -> BB
residual is added to the ``a`` argument of ``atom_attention_decoder``, which is
positional index 1 of its forward call.

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
        self.feedback = feedback
        self.a_token = None
        self.calls = 0
        self.injections = 0
        self._handles = []

    # ---- hooks -----------------------------------------------------------

    def _capture(self, module, inputs, output):
        # Detaching here would cut BB -> SC gradients; keep the graph and let the
        # caller decide, since phase 1 trains an adapter fed by these features.
        self.a_token = output
        self.calls += 1
        return None

    def _inject(self, module, args, kwargs):
        if self.feedback is None:
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
        self.calls = self.injections = 0
        return self


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
