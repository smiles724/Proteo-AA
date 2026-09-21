"""Apply the A_BS residual inside FaMPNN's ITERATIVE design loop.

`pxf.couple.shared_prelogit` defines where the residual goes; this puts it
there during `FaMPNNFullAtomDesigner.design`, which calls FaMPNN's shipped
``SeqDenoiser.sample`` and therefore evaluates the encoder ~101 times behind a
single call this process does not drive.

The seam is exact. ``SeqDesignModule.forward`` ends (``fampnn/model/fampnn.py``)::

    logits = None if self.no_aatype_pred else self.W_out(h_V)
    mpnn_feature_dict = {"h_V": h_V, ...}
    return logits, mpnn_feature_dict

and ``h_V`` in that dict is the only encoder output the side-chain module
reads. So a forward hook that replaces the returned pair with
``shared_prelogit.condition(...)`` conditions the sequence logits and the
side-chain branch from one residual, which is what `shared_prelogit` means.
FaMPNN itself is untouched, so `pxf.provenance` still pins it unpatched.

**Why a hook and not `pxf.couple.codesign`.** `codesign` is the single-pass
variant and says so: one encoder evaluation over a fully masked sequence. The
benchmark arms specify ``seq_steps: 100``, the iterative scheme, because that
is what the designability numbers are comparable to. Reusing the single-pass
path here would answer a different question at lower recovery.

**Accumulation.** The residual is constant within a design and the loop runs
~101 times, so an in-place add would compound it ~101-fold and look exactly
like a strong adapter. :func:`pxf.couple.shared_prelogit.condition` derives
``h'`` from the original ``h_V`` and writes into a shallow copy every call;
:class:`ResidualConditioning` additionally counts calls so a run can assert the
hook fired the number of times it expected instead of hoping it did.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch

from pxf.couple.binder_residual import ChainRoles, binder_masked_residual
from pxf.couple.shared_prelogit import check_exclusive, condition


def build_residual(
    adapters,
    *,
    binder_mask: torch.Tensor,
    a_token: torch.Tensor,
    sigma: float,
    source: str = "matched",
    gate=None,
    mean=None,
) -> torch.Tensor:
    """The ``[1, L, c]`` residual for one design, zero on target rows.

    ``sigma`` must be the payload's ``actual_sigma``. The scheduled value is
    half of it (churn 2.0), and conditioning the adapter on that would query
    it at half the noise the denoiser actually saw.
    """
    roles = ChainRoles(binder=binder_mask.reshape(-1).bool())
    delta = binder_masked_residual(
        adapters, source, roles=roles,
        a_token=a_token, sigma=sigma, gate=gate, mean=mean,
    )
    if delta is None:
        raise ValueError(
            f"residual source {source!r} produced no residual; the uncoupled "
            "arm is a separate arm, not this one with the adapter off"
        )
    return delta


class ResidualConditioning:
    """Context manager installing the shared-pre-logit hook, with a call count."""

    def __init__(self, model, delta: Optional[torch.Tensor]):
        self.model = model
        self.delta = delta
        self.calls = 0
        self._handle = None

    @property
    def seq_module(self):
        return self.model.denoiser.seq_design_module

    def __enter__(self):
        if self.delta is None:
            return self  # the uncoupled arm takes the untouched path
        check_exclusive("shared_prelogit", packing_hook_active=False)
        module = self.seq_module

        def hook(mod, args, kwargs, output):
            logits, features = output
            if not isinstance(features, dict) or "h_V" not in features:
                raise TypeError(
                    "the sequence module returned something without an h_V "
                    f"feature dict ({type(features).__name__}); FaMPNN's "
                    "contract changed and the residual has nowhere to land"
                )
            self.calls += 1
            new_logits, conditioned = condition(mod, features, self.delta)
            return new_logits, conditioned

        self._handle = module.register_forward_hook(hook, with_kwargs=True)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False

    def assert_fired(self, minimum: int = 1) -> None:
        """A coupled arm whose hook never fired is an uncoupled arm mislabelled."""
        if self.delta is None:
            return
        if self.calls < minimum:
            raise AssertionError(
                f"the shared-pre-logit hook fired {self.calls} time(s), expected "
                f"at least {minimum}. This arm would be reported as coupled "
                "while having run the donor unchanged."
            )


@contextmanager
def conditioned(model, delta: Optional[torch.Tensor], *, expect_calls: int = 1):
    """`with conditioned(model, delta): designer.design(...)`."""
    ctx = ResidualConditioning(model, delta)
    with ctx:
        yield ctx
    ctx.assert_fired(expect_calls)
