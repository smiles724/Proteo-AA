"""The coupled forward pass: one bidirectional cycle.

    (X_BB^0, a_BB) = B(X_BB_sigma, sigma)                    backbone proposal
    h_FA           = E(X_BB^0, S, masked SC)                 BB -> SC encode
    h~_FA          = h_FA + A_BS(a_BB, sigma)                BB -> SC residual
    X_SC           = G(h~_FA, S)                             pack
    h_packed       = E(X_BB^0, X_SC, S)                      re-encode predicted SC
    a~_BB          = a_BB + A_SB(h_packed, sigma)            SC -> BB residual
    X_BB^1         = D(a~_BB)                                backbone correction

The re-encode is the point of the design: ``h_packed`` sees the side chains the
model actually realized, not native ones, so the feedback carries information
about *this* packing. Making that true takes a second mask -- the input's
``missing_atom_mask`` marks every side-chain slot absent, which is right for the
first encode and wrong for the second, where it would keep the generated atoms
masked and leave ``h_packed == h_base``. :mod:`pxf.couple.visibility` computes
the post-packing availability instead, and the cycle records it so a run can be
audited on how many atoms the feedback actually saw.

One corrective event, both PXDesign calls at the *same* noisy state:

    (B0, a0) = D(x_sigma, sigma; 0)
    S0       = P(B0, s)                      packing, the selected BB->SC policy
    z        = R(B0, S0, s)                  feedback readout
    B1       = D(x_sigma, sigma; A_SB(z, sigma))

``bb0``/``sc0``/``bb1`` are kept apart on the output, and ``sc1`` -- a fresh
packing on the corrected backbone -- is produced by :meth:`repack_on` rather
than inferred, because reporting ``bb1`` combined with the unchanged ``sc0`` as
the final structure would credit the correction with side chains built for a
different backbone.

Gradient routing is explicit and staged, not implicit. Phase 2 trains ``A_SB``
with the packing detached -- backbone gradients do not flow back through the
side-chain sampler -- because credit assignment through a 50-step diffusion
rollout is both expensive and noisy. :class:`GradientPolicy` makes that a
declared choice rather than an accident of where ``detach()`` happened to land.

The backbone module is injected as a callable rather than imported, for two
reasons: the controller's logic is independent of how PXDesign is driven, and
PXDesign's official inference runner cannot generate monomers at all, so the
driver is expected to change. Any callable with the
:class:`BackboneDenoiser` protocol works, including a stub in tests.
"""

import logging
import time
from dataclasses import dataclass, field, replace
from typing import Protocol

import torch

from pxf.couple import fampnn_iface as iface
from pxf.couple import visibility as vis
from pxf.couple.converter import PXFaRepresentationConverter

logger = logging.getLogger("pxf.couple.controller")

PHASES = ("frozen", "bb_to_sc", "sc_to_bb", "joint")


class BackboneDenoiser(Protocol):
    """One denoising evaluation, optionally with a token-feature residual.

    Returns ``(x_denoised, a_token)`` where ``x_denoised`` is on PXDesign's flat
    atom axis ``[..., N_atom, 3]`` and ``a_token`` is ``[..., L, c_token]``.
    """

    def __call__(
        self,
        x_noisy: torch.Tensor,
        sigma: torch.Tensor,
        *,
        feedback: torch.Tensor | None = None,
    ) -> tuple: ...


@dataclass
class Topology:
    """The flat-atom <-> per-residue mapping, constant for one target."""

    atom_names: list
    atom_to_token_idx: torch.Tensor
    num_tokens: int
    res_names: list | None = None
    residue_index: torch.Tensor | None = None
    chain_index: torch.Tensor | None = None

    def to(self, device):
        """The same topology with its index tensors on ``device``.

        Indexing a CUDA tensor with a CPU index happens to work, but arithmetic
        between them does not, so the mapping has to travel with the batch.
        """
        from dataclasses import replace

        moved = {
            name: getattr(self, name).to(device)
            for name in ("atom_to_token_idx", "residue_index", "chain_index")
            if torch.is_tensor(getattr(self, name))
        }
        return replace(self, **moved)


@dataclass
class GradientPolicy:
    """Where the graph is cut. Declared per phase; see the module docstring."""

    detach_sidechains: bool = True  # X_SC -> h_packed
    detach_h_packed: bool = True  # h_packed -> A_SB
    detach_a_token_for_feedback: bool = False

    @classmethod
    def for_phase(cls, phase):
        if phase not in PHASES:
            raise ValueError(f"Unknown phase {phase!r}; choose from {PHASES}")
        if phase == "bb_to_sc":
            # Phase 1 trains A_BS through the packing loss; the feedback branch
            # is not run at all, so its cuts are irrelevant.
            return cls(detach_sidechains=True, detach_h_packed=True)
        if phase in ("sc_to_bb", "joint"):
            return cls(detach_sidechains=True, detach_h_packed=True)
        return cls()


@dataclass
class CycleOutput:
    """Every intermediate the staged losses, the ablations and the report need.

    The four structural stages are stored separately and never merged:

    ``bb0``  the initial clean backbone estimate
    ``sc0``  side chains packed on ``bb0``; this is what the feedback reads
    ``bb1``  the corrected clean backbone estimate
    ``sc1``  a fresh packing on ``bb1``, for the final side-chain evaluation

    ``sc1`` is only populated by :meth:`CoupledDenoiser.repack_on`. It is not
    filled in by the cycle because a corrective event does not need it and the
    packing is the expensive half; an evaluator that wants a final structure has
    to ask for it, which is also what stops ``bb1 + sc0`` being reported as one.
    """

    bb0_flat: torch.Tensor  # [..., N_atom, 3]
    bb0_dense: torch.Tensor  # [B, L, 37, 3]
    a_token: torch.Tensor
    h_base: torch.Tensor | None = None
    delta_h: torch.Tensor | None = None
    h_cond: torch.Tensor | None = None
    sidechains: torch.Tensor | None = None  # sc0, [B, L, 33, 3] global
    h_packed: torch.Tensor | None = None
    packed: object | None = None  # visibility.PackedStructure for the re-encode
    delta_a: torch.Tensor | None = None
    feedback_stats: dict = field(default_factory=dict)
    bb1_flat: torch.Tensor | None = None
    bb1_dense: torch.Tensor | None = None  # [B, L, 37, 3]
    sc1: torch.Tensor | None = None  # fresh packing on bb1, [B, L, 33, 3]
    aux: dict = field(default_factory=dict)

    @property
    def ran_feedback(self):
        return self.bb1_flat is not None

    # `sidechains` predates the stage naming and is load-bearing in the phase-1
    # paths, so it stays the field and `sc0` is the alias, not the reverse.
    @property
    def sc0(self):
        return self.sidechains

    @property
    def visibility(self):
        return None if self.packed is None else self.packed.visibility


# `delta_h=None` means bypass, which is a real choice, so "not specified" needs
# its own value rather than reusing None.
UNSET = object()


@dataclass
class Proposal:
    """Everything the arms share at one ``(structure, sigma, replicate)``.

    The backbone pass and the FaMPNN encoding do not depend on the residual, so
    computing them once and packing from them repeatedly is both cheaper and
    stronger: the arms then provably see identical conditioning rather than
    identical-by-determinism conditioning. Phase 1 happens to satisfy the latter
    (A_SB is zero, so the backbone is a pure function of x_noisy and sigma), but
    that is a property of the phase, not a guarantee of the comparison.

    Nothing here may be mutated by packing. ``iface.with_residual`` returns a
    shallow copy rather than writing into ``features``, and
    ``tests/test_couple_controller.py`` pins that a second pack from the same
    proposal sees byte-identical inputs.
    """

    topology: object
    x_noisy: torch.Tensor
    sigma: torch.Tensor
    aatype: torch.Tensor
    bb0_flat: torch.Tensor
    a_token: torch.Tensor
    inputs: object
    h_base: torch.Tensor
    features: dict


class CoupledDenoiser:
    """Runs one coupling cycle over an injected backbone denoiser and FaMPNN."""

    def __init__(
        self,
        backbone: BackboneDenoiser,
        fampnn,
        adapters,
        *,
        converter=None,
        phase="joint",
        pack_steps=None,
        gradient_policy=None,
        bs_gate=None,
    ):
        self.backbone = backbone
        self.fampnn = fampnn
        self.adapters = adapters
        self.converter = converter or PXFaRepresentationConverter()
        self.pack_steps = pack_steps
        # The selected Phase-1 BB->SC policy's gate. Held fixed across every
        # SC->BB comparison, so it belongs to the controller rather than being
        # re-chosen per call. `pxf.couple.bs_policy.Gate` shapes; None is
        # ungated, and the bypass is `delta_h=None` at the call site.
        self.bs_gate = bs_gate
        self.set_phase(phase)
        if gradient_policy is not None:
            self.policy = gradient_policy

    def set_phase(self, phase):
        self.phase = phase
        self.policy = GradientPolicy.for_phase(phase)
        record = (
            self.adapters.set_phase(phase) if hasattr(self.adapters, "set_phase") else {}
        )
        return dict(phase=phase, policy=vars(self.policy), adapters=record)

    # ---- the cycle -------------------------------------------------------

    def forward(
        self,
        topology,
        x_noisy,
        sigma,
        aatype,
        *,
        seq_mask=None,
        run_feedback=None,
        sidechain_context=None,
        a_token_override=None,
    ):
        """One cycle. ``run_feedback`` defaults to whether SC->BB is enabled.

        ``a_token_override`` feeds ``A_BS`` token features from somewhere other
        than this structure's own backbone pass. It is an experimental control,
        not a model capability: substituting another protein's ``a_token`` asks
        whether the adapter's gain comes from *sample-specific* information or
        from behaving as a generic regularizer on ``h_V``.

        It deliberately affects only the BB->SC residual. The backbone proposal,
        the encoder features and the SC->BB feedback all keep this structure's
        own ``a_token``, so the substitution changes one quantity and the arms
        stay comparable.
        """
        proposal = self.propose(topology, x_noisy, sigma, aatype, seq_mask=seq_mask)
        return self.pack_proposal(
            proposal, a_token_override=a_token_override, run_feedback=run_feedback
        )

    def propose(self, topology, x_noisy, sigma, aatype, *, seq_mask=None):
        """The residual-independent half: backbone proposal, then encoding.

        Deterministic given ``(x_noisy, sigma)`` and free of packing randomness,
        so it can be computed once and reused by every arm.
        """
        converter = self.converter
        # --- backbone proposal ---
        bb0_flat, a_token = self.backbone(x_noisy, sigma)
        if a_token is None:
            raise ValueError(
                "The backbone denoiser returned no token features; "
                "coupling needs a_token (see pxf.couple.pxdesign_iface)"
            )

        inputs = converter.px_backbone_to_fampnn(
            bb0_flat,
            topology.atom_names,
            topology.atom_to_token_idx,
            topology.num_tokens,
            res_names=topology.res_names,
            residue_index=topology.residue_index,
            chain_index=topology.chain_index,
            aatype=aatype,
        )
        if seq_mask is not None:
            inputs = replace(
                inputs,
                seq_mask=converter.px_residue_mask_to_fampnn(
                    seq_mask, topology.num_tokens, batch=inputs.batch
                ),
            )

        # --- BB -> SC encoding ---
        _, h_base, features = iface.encode(
            self.fampnn,
            inputs.coords_af2,
            inputs.aatype,
            seq_mask=inputs.seq_mask,
            missing_atom_mask=inputs.missing_atom_mask,
            residue_index=inputs.residue_index,
            chain_index=inputs.chain_index,
        )
        return Proposal(
            topology=topology,
            x_noisy=x_noisy,
            sigma=sigma,
            aatype=aatype,
            bb0_flat=bb0_flat,
            a_token=a_token,
            inputs=inputs,
            h_base=h_base,
            features=features,
        )

    def pack_proposal(
        self, proposal, *, a_token_override=None, run_feedback=None, delta_h=UNSET
    ):
        """The residual-dependent half: apply A_BS, then pack.

        One arm per call, from shared conditioning. Packing randomness is the
        caller's to control -- reseed immediately before each arm, or the two
        arms differ by the sampler as well as by the residual.
        """
        inputs = proposal.inputs
        a_token, sigma = proposal.a_token, proposal.sigma
        h_base, features = proposal.h_base, proposal.features
        x_noisy = proposal.x_noisy

        out = CycleOutput(
            bb0_flat=proposal.bb0_flat,
            bb0_dense=inputs.coords_af2,
            a_token=a_token,
            aux=dict(inputs=inputs),
        )
        out.h_base = h_base
        # L_SC re-runs the side-chain denoiser from these features, so the cycle
        # records the dict itself rather than just h_V.
        out.aux["features"] = features
        if delta_h is not UNSET:
            # The caller supplied the residual outright -- a gated one, a shared
            # mean, or an explicit bypass. `pxf.couple.bs_policy` builds these,
            # and routing them here keeps every arm on one packing path.
            out.delta_h = delta_h
        else:
            # Only the residual reads the override; out.a_token stays this
            # structure's own, so the recorded provenance is not falsified.
            source = a_token if a_token_override is None else a_token_override
            out.delta_h = self._delta_h(source, sigma, h_base)
        out.h_cond = h_base if out.delta_h is None else h_base + out.delta_h

        # --- pack ---
        sidechains, pack_aux = iface.pack_from_features(
            self.fampnn,
            iface.with_residual(features, out.delta_h),
            inputs.aatype,
            seq_mask=inputs.seq_mask,
            residue_index=inputs.residue_index,
            chain_index=inputs.chain_index,
            num_steps=self.pack_steps,
        )
        out.sidechains = sidechains
        out.aux["pack"] = pack_aux

        if run_feedback is None:
            run_feedback = bool(getattr(self.adapters, "enable_sc_to_bb", False))
        if not run_feedback:
            return out

        # --- re-encode the predicted packing ---
        packed = self.encode_predicted_packing(
            inputs,
            sidechains.detach() if self.policy.detach_sidechains else sidechains,
            h_base=h_base,
            psce=pack_aux.get("psce") if isinstance(pack_aux, dict) else None,
        )
        out.packed = packed.detach() if self.policy.detach_h_packed else packed
        out.h_packed = out.packed.h_packed

        # --- SC -> BB ---
        out.delta_a, out.feedback_stats = self._delta_a(out.packed, sigma, a_token)
        if out.delta_a is not None:
            out.bb1_flat, _ = self.backbone(x_noisy, sigma, feedback=out.delta_a)
            out.bb1_dense = self.densify(out.bb1_flat, proposal.topology, proposal.aatype)
        return out

    # ---- one corrective event, with the gradient boundary stated ---------

    def corrective_event(
        self,
        topology,
        x_noisy,
        sigma,
        aatype,
        *,
        seq_mask=None,
        bs_delta_h=UNSET,
        a_token_override=None,
        upstream=None,
    ):
        """One SC -> BB correction at a *fixed* noisy state. Returns the cycle.

        The gradient boundary is written out rather than left to whichever
        ``detach()`` happens to be reached first:

            with no_grad:                  frozen, cacheable
                bb0, a0 = D(x_sigma, sigma; 0)
                sc0     = P(bb0, s; selected BB->SC policy)
                packed  = E(bb0, sc0, s)
            z       = R(packed)            trainable
            delta_a = A_SB(z, sigma)       trainable
            bb1, _  = D(x_sigma, sigma; delta_a)     NOT under no_grad

        The corrective call must stay outside ``no_grad``: PXDesign's parameters
        are frozen, but its atom decoder has to differentiate with respect to
        ``delta_a`` or the loss reaches nothing. That is the failure mode this
        method exists to make impossible, and it is asserted below rather than
        trusted.

        Both denoiser calls receive the same ``x_noisy``, the same ``sigma`` and
        the same conditioning. ``bs_delta_h`` is the selected Phase-1 BB->SC
        policy, held fixed across every SC->BB comparison; pass ``None`` for the
        bypass.

        ``upstream`` supplies a previously computed (and detached) frozen half,
        so a pilot can pay for the packing once per example rather than once per
        step. See :mod:`pxf.couple.pilot`.
        """
        if upstream is None:
            upstream = self.frozen_half(
                topology,
                x_noisy,
                sigma,
                aatype,
                seq_mask=seq_mask,
                bs_delta_h=bs_delta_h,
                a_token_override=a_token_override,
            )

        out = CycleOutput(
            bb0_flat=upstream.bb0_flat,
            bb0_dense=upstream.packed.coords37,
            a_token=upstream.a_token,
            h_base=upstream.packed.h_base,
            delta_h=upstream.delta_h,
            sidechains=upstream.sidechains,
            packed=upstream.packed,
            h_packed=upstream.packed.h_packed,
            aux=dict(inputs=upstream.inputs, upstream=upstream.identity()),
        )
        # bb0_dense carries sc0 in its side-chain slots, which is what the
        # readout reads; the *backbone* stages are bb0_flat and bb1_flat.
        out.delta_a, out.feedback_stats = self._delta_a(
            upstream.packed, sigma, upstream.a_token
        )
        if out.delta_a is None:
            return out
        if torch.is_grad_enabled() and not out.delta_a.requires_grad:
            trainable = [
                name
                for name, p in getattr(self.adapters, "named_parameters", lambda: [])()
                if p.requires_grad
            ]
            if trainable:
                raise RuntimeError(
                    "delta_a does not require grad although "
                    f"{len(trainable)} adapter parameter(s) do (e.g. "
                    f"{trainable[:3]}). The trainable half has been captured by a "
                    "no_grad context, so L_BB would reach nothing while still "
                    "producing a loss curve."
                )
        out.bb1_flat, _ = self.backbone(x_noisy, sigma, feedback=out.delta_a)
        out.bb1_dense = self.densify(out.bb1_flat, topology, aatype)
        return out

    @torch.no_grad()
    def frozen_half(
        self,
        topology,
        x_noisy,
        sigma,
        aatype,
        *,
        seq_mask=None,
        bs_delta_h=UNSET,
        a_token_override=None,
    ):
        """The frozen, cacheable half of a corrective event, fully detached.

        Decorated rather than wrapped at the call site so there is one place
        where "this is the part that does not train" is stated, and so a caller
        cannot forget it.
        """
        from pxf.couple.pilot import UpstreamState

        # Timed per stage, because the arms need different subsets of it and a
        # cost comparison that charges every arm for the packing is wrong in the
        # direction that flatters feedback: a BB-only alternative needs `bb0`
        # and nothing else, while a feedback arm needs the 50-step rollout and
        # the re-encode too.
        clock = time.perf_counter()
        proposal = self.propose(topology, x_noisy, sigma, aatype, seq_mask=seq_mask)
        denoise_seconds = time.perf_counter() - clock

        clock = time.perf_counter()
        packing = self.pack_proposal(
            proposal,
            a_token_override=a_token_override,
            run_feedback=False,
            delta_h=bs_delta_h,
        )
        pack_seconds = time.perf_counter() - clock

        clock = time.perf_counter()
        packed = self.encode_predicted_packing(
            proposal.inputs,
            packing.sidechains,
            h_base=proposal.h_base,
            psce=(packing.aux.get("pack") or {}).get("psce"),
        )
        reencode_seconds = time.perf_counter() - clock

        return UpstreamState(
            timings=dict(
                denoise=denoise_seconds,
                pack=pack_seconds,
                reencode=reencode_seconds,
            ),
            packed=packed.detach(),
            bb0_flat=proposal.bb0_flat.detach(),
            a_token=proposal.a_token.detach(),
            delta_h=None if packing.delta_h is None else packing.delta_h.detach(),
            sidechains=packing.sidechains.detach(),
            inputs=proposal.inputs,
            sigma=torch.as_tensor(sigma).detach().clone(),
            pack_steps=self.pack_steps,
            bs_policy="bypass" if packing.delta_h is None else "residual",
        )

    # ---- the four stages -------------------------------------------------

    def encode_predicted_packing(self, inputs, sidechains, *, h_base=None, psce=None):
        """``E(bb0, sc0, s)`` with the post-packing availability mask.

        The one place the second encode happens. ``inputs.missing_atom_mask`` is
        deliberately *not* forwarded: it is the input's observation mask, it
        marks all 33 side-chain slots absent for a backbone-only proposal, and
        ``build_atom_mask``'s ``1 - missing_atom_mask`` factor would therefore
        keep every generated atom masked -- the failure this method exists to
        make impossible. Availability is recomputed from the fixed sequence, the
        supplied backbone and the packer's own output.
        """
        visibility = vis.predicted_availability(
            inputs.aatype,
            inputs.seq_mask,
            inputs.atom_mask,
            inputs.coords_af2,
            sidechains=sidechains,
        )
        padded = int((inputs.seq_mask <= 0).sum())
        if padded:
            # pxf.couple.probes measures a 2.6e-4 to 1.2e-3 relative leak from
            # padded rows into real residues' node features, upstream in
            # FaMPNN's encoder, plus a larger shift on whichever residue is no
            # longer the terminus. One structure per forward at its own length
            # avoids both, which is what the converter does; say so if that ever
            # changes.
            logger.warning(
                "%d padded residue(s) in the re-encode. FaMPNN's encoder leaks "
                "~1e-3 of their coordinates into real residues' features (see "
                "pxf.couple.probes), so the feedback features are no longer a "
                "function of this structure alone.",
                padded,
            )
        _, h_packed, features = iface.encode(
            self.fampnn,
            inputs.coords_af2,
            inputs.aatype,
            sidechains=sidechains,
            atom_availability=visibility.available,
            seq_mask=inputs.seq_mask,
            residue_index=inputs.residue_index,
            chain_index=inputs.chain_index,
        )
        coords37 = inputs.coords_af2.clone()
        coords37[..., list(self.converter.sidechain_slots), :] = sidechains.to(
            coords37.dtype
        )
        packed = vis.PackedStructure(
            h_packed=h_packed,
            coords37=coords37,
            aatype=inputs.aatype,
            seq_mask=inputs.seq_mask,
            visibility=visibility,
            psce=psce,
            h_base=h_base,
        )
        packed.features = features
        return packed

    def densify(self, flat, topology, aatype):
        """A flat PXDesign coordinate tensor as FaMPNN's ``[B, L, 37, 3]`` block."""
        return self.converter.px_backbone_to_fampnn(
            flat,
            topology.atom_names,
            topology.atom_to_token_idx,
            topology.num_tokens,
            res_names=topology.res_names,
            residue_index=topology.residue_index,
            chain_index=topology.chain_index,
            aatype=aatype,
        ).coords_af2

    def repack_on(
        self,
        bb_dense,
        aatype,
        *,
        seq_mask=None,
        residue_index=None,
        chain_index=None,
        supplied_atom_mask=None,
        num_steps=None,
        delta_h=UNSET,
    ):
        """``sc1``: a fresh packing on a corrected backbone. Returns ``(sc, aux)``.

        Separate from the cycle on purpose. The corrected backbone's side chains
        have to be *rebuilt*, not carried over: ``sc0`` was packed onto ``bb0``
        and reporting it attached to ``bb1`` would mix a structure that was never
        produced, flattering or penalizing the correction at random.
        """
        backbone = list(self.converter.backbone_slots)
        if supplied_atom_mask is None:
            supplied_atom_mask = torch.zeros(
                *bb_dense.shape[:2], 37, device=bb_dense.device
            )
            supplied_atom_mask[..., backbone] = 1.0
        start = vis.predicted_availability(
            aatype,
            torch.ones(bb_dense.shape[:2], device=bb_dense.device)
            if seq_mask is None
            else seq_mask,
            supplied_atom_mask,
            bb_dense,
        )
        _, _h, features = iface.encode(
            self.fampnn,
            bb_dense,
            aatype,
            atom_availability=start.available,
            seq_mask=seq_mask,
            residue_index=residue_index,
            chain_index=chain_index,
        )
        return iface.pack_from_features(
            self.fampnn,
            features if delta_h is UNSET else iface.with_residual(features, delta_h),
            aatype,
            seq_mask=seq_mask,
            residue_index=residue_index,
            chain_index=chain_index,
            num_steps=self.pack_steps if num_steps is None else num_steps,
        )

    __call__ = forward

    # ---- adapter plumbing -------------------------------------------------

    def _delta_h(self, a_token, sigma, h_base):
        if not getattr(self.adapters, "enable_bb_to_sc", True):
            return None
        delta = self.adapters.delta_h(self._per_residue(a_token, h_base.shape[1]), sigma)
        if delta is None:
            return None
        if self.bs_gate is not None:
            scale = self.bs_gate(
                float(sigma.reshape(-1)[0]) if torch.is_tensor(sigma) else float(sigma)
            )
            if scale == 0.0:
                # Exactly the bypass, not a scaled-to-zero approximation of it:
                # `with_residual(features, None)` takes the same path the
                # uncoupled packing does.
                return None
            delta = delta * scale
        return self._match(delta, h_base)

    def _delta_a(self, packed, sigma, a_token):
        """``A_SB(z, sigma)``, matched to the token axis. Returns ``(delta, stats)``.

        The adapter is handed the whole :class:`~pxf.couple.visibility.PackedStructure`
        rather than ``h_packed`` alone, because a readout over predicted side
        chains needs the coordinates, the sequence and the availability masks as
        well. Adapters that only want ``h_V`` declare so with
        ``reads_packing = False`` and still work unchanged.
        """
        if not getattr(self.adapters, "enable_sc_to_bb", True):
            return None, {}
        length = packed.h_packed.shape[1]
        reference = self._per_residue(a_token, length)
        result = self.adapters.delta_a(packed, sigma, reference=reference)
        delta, stats = result if isinstance(result, tuple) else (result, {})
        if delta is None:
            return None, dict(stats)
        return self._match(delta, reference), dict(stats)

    @staticmethod
    def _per_residue(tensor, length):
        """Collapse PXDesign's ``[..., N_sample, L, c]`` to ``[B, L, c]``."""
        if tensor.dim() == 3:
            return tensor
        return tensor.reshape(-1, length, tensor.shape[-1])

    @staticmethod
    def _match(delta, reference):
        if delta.shape[0] == 1 and reference.shape[0] > 1:
            delta = delta.expand(reference.shape[0], *delta.shape[1:])
        if delta.shape[:2] != reference.shape[:2]:
            raise ValueError(
                f"adapter output {tuple(delta.shape)} does not match "
                f"target latent {tuple(reference.shape)}"
            )
        return delta

    def identity(self):
        return dict(
            phase=self.phase,
            policy=vars(self.policy),
            pack_steps=self.pack_steps,
            bs_gate=(
                self.bs_gate.identity() if hasattr(self.bs_gate, "identity") else None
            ),
            adapters=self.adapters.identity() if hasattr(self.adapters, "identity") else {},
            converter=self.converter.identity(),
        )
