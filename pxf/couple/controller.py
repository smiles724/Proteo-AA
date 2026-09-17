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
about *this* packing.

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

from dataclasses import dataclass, field, replace
from typing import Protocol

import torch

from pxf.couple import fampnn_iface as iface
from pxf.couple.converter import PXFaRepresentationConverter

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
    """Every intermediate the staged losses and the ablations need."""

    bb0_flat: torch.Tensor  # [..., N_atom, 3]
    bb0_dense: torch.Tensor  # [B, L, 37, 3]
    a_token: torch.Tensor
    h_base: torch.Tensor | None = None
    delta_h: torch.Tensor | None = None
    h_cond: torch.Tensor | None = None
    sidechains: torch.Tensor | None = None  # [B, L, 33, 3] global
    h_packed: torch.Tensor | None = None
    delta_a: torch.Tensor | None = None
    bb1_flat: torch.Tensor | None = None
    aux: dict = field(default_factory=dict)

    @property
    def ran_feedback(self):
        return self.bb1_flat is not None


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
    ):
        self.backbone = backbone
        self.fampnn = fampnn
        self.adapters = adapters
        self.converter = converter or PXFaRepresentationConverter()
        self.pack_steps = pack_steps
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

    def pack_proposal(self, proposal, *, a_token_override=None, run_feedback=None):
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
        visible = sidechains.detach() if self.policy.detach_sidechains else sidechains
        _, h_packed, _ = iface.encode(
            self.fampnn,
            inputs.coords_af2,
            inputs.aatype,
            sidechains=visible,
            seq_mask=inputs.seq_mask,
            missing_atom_mask=inputs.missing_atom_mask,
            residue_index=inputs.residue_index,
            chain_index=inputs.chain_index,
        )
        out.h_packed = h_packed

        # --- SC -> BB ---
        source = h_packed.detach() if self.policy.detach_h_packed else h_packed
        out.delta_a = self._delta_a(source, sigma, a_token)
        if out.delta_a is not None:
            out.bb1_flat, _ = self.backbone(x_noisy, sigma, feedback=out.delta_a)
        return out

    __call__ = forward

    # ---- adapter plumbing -------------------------------------------------

    def _delta_h(self, a_token, sigma, h_base):
        if not getattr(self.adapters, "enable_bb_to_sc", True):
            return None
        delta = self.adapters.delta_h(self._per_residue(a_token, h_base.shape[1]), sigma)
        if delta is None:
            return None
        return self._match(delta, h_base)

    def _delta_a(self, h_packed, sigma, a_token):
        if not getattr(self.adapters, "enable_sc_to_bb", True):
            return None
        delta = self.adapters.delta_a(h_packed, sigma)
        if delta is None:
            return None
        return self._match(delta, self._per_residue(a_token, h_packed.shape[1]))

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
            adapters=self.adapters.identity() if hasattr(self.adapters, "identity") else {},
            converter=self.converter.identity(),
        )
