"""The early SC -> BB conditioners: residuals for ``s_single`` and ``z_pair``.

The late adapter (:class:`pxf.couple.readout.FeedbackPath`) injects after the
diffusion transformer, where the only thing left to change is how the final
token features are re-mixed by three decoder blocks. It converged to
approximately the sigma-conditioned bias its own ``generic`` control fits
directly -- the representation it reads responds 29% to a 60 degree rotamer flip
and its output responds 1.1% (``docs/sb_pilot_results.md``). That is a result
about *this injection site*, not about the information, so these two
architectures move the site and hold everything else fixed.

**E1** (:class:`EarlySingleConditioner`) reuses the existing
:class:`~pxf.couple.readout.FeedbackReadout` unchanged -- same feature groups,
same normalization, same sequence embedding, same ``full``/``bb_only``/``generic``
controls -- and changes only where its output lands. It is the controlled
version of "the site was the problem".

**E2** (:class:`AtomConditioner`) additionally changes the representation:
predicted atoms are encoded directly into a residue term ``U_i`` and a directed
pair term ``V_ij``, and the pair term is what the late site had no way to
express at all -- ``z_pair`` reaches the atom encoder and every transformer
block, so a side chain can say something about a *pair* of residues rather than
only about its own.

Neither runs together with the late adapter. The payload type is what enforces
that (see :mod:`pxf.couple.pxdesign_iface`): a
:class:`~pxf.couple.pxdesign_iface.ConditioningFeedback` is injected at the
conditioning output and the decoder hook declines it.

**What is frozen here and why it is written down.** Widths, neighbour limits and
the RBF are proposed defaults, not findings. They are constants in this module
and recorded in :meth:`identity`, so an arm cannot be compared against another
that quietly used different ones. Two of them were specified twice with
different values and the disagreement is resolved once, here, rather than per
call site: the RBF is the standard Gaussian ``exp(-(d-c)^2 / 2*0.8^2)``, and
local coordinates enter the atom encoder divided by 10 rather than in raw
Angstroms. Both alternatives differ only by a scale the first linear layer can
absorb, which is exactly why leaving the choice implicit would be a silent
inconsistency between arms rather than a visible one.

**Zero output, live gate.** Every final projection is zero-initialized and the
sigma gate is a fixed function equal to one across the training window, for the
same reason as the late adapter: the product of two zero-initialized factors has
no gradient in either. So step 0 reproduces PXDesign exactly *and* the encoders
can start moving -- but only after the output projection has, which is why the
gradient test checks the projection first and the encoders second.
"""

import torch
from torch import nn

from pxf import atom37
from pxf.couple import frames
from pxf.couple.adapters import NoiseEmbedding
from pxf.couple.pxdesign_iface import ConditioningFeedback
from pxf.couple.readout import (
    N_CHI_FEATURES,
    N_RELIABILITY,
    PSCE_SCALE,
    FeedbackReadout,
    chi_features,
    reliability_features,
)

# Bumped when a change would make two checkpoints incomparable rather than
# merely differently trained. Refused on load; see `check_compatible`.
CONDITIONER_VERSION = "pxf-early-conditioner-v1"

# --- frozen architecture constants ------------------------------------------

D_NOISE = 64  # sigma embedding, shared by both architectures and both heads
D_HIDDEN = 256  # head hidden width

D_ATOM_SLOT = 16  # learned atom37-slot embedding
D_ATOM = 64  # per-atom embedding
D_SEQUENCE = 32  # learned residue-type embedding (E2's own, not the readout's)
D_RESIDUE = 128  # U_i
D_EDGE = 64  # per-atom-pair embedding
D_PAIR = 64  # V_ij

# Neighbourhoods. Computational defaults for the pilot, recorded so the coverage
# and truncation counts a run reports can be read against them.
MAX_NEIGHBOURS = 16
NEIGHBOUR_RADIUS = 20.0  # Angstroms, between CA atoms
MAX_ATOM_PAIRS = 32
ATOM_PAIR_RADIUS = 12.0  # Angstroms

# Distance RBF: 16 centres uniform on [0, 12] A, Gaussian with sigma 0.8 A.
N_RBF = 16
RBF_SIGMA = 0.8

# Pair-type channels, in order. Directed, and BB-BB is deliberately absent: in
# the SC variants every retained atom pair has at least one side-chain atom, so
# BB-BB cannot occur, and in the BB-only variant every pair is BB-BB and these
# three channels are structural zeros. A fourth channel would be constant in
# every arm.
PAIR_TYPES = ("sc_sc", "sc_bb", "bb_sc")

# Per-atom encoder input: slot embedding, local coordinates, confidence, and
# whether there was a confidence to report.
N_ATOM_FEATURES = D_ATOM_SLOT + 3 + 1 + 1
# Per-atom-pair encoder input.
N_EDGE_FEATURES = 2 * D_ATOM_SLOT + N_RBF + 3 + 2 + 2 + len(PAIR_TYPES)
# U_i input: masked mean and max over the pooled atoms, plus the residue terms.
N_RESIDUE_FEATURES = 2 * D_ATOM + D_SEQUENCE + N_CHI_FEATURES + N_RELIABILITY

# Local coordinates are divided by this before entering the atom encoder.
COORDINATE_SCALE = 10.0

# The arms of the two experiments, and what each one is. `arch` selects the
# architecture, `variant` what it is allowed to read, `pair` whether the pair
# branch runs at all. Named here rather than assembled from flags at each call
# site so a launcher names one row and cannot produce a combination nobody chose.
ARMS = {
    # E0: the existing late adapter, for reference. Not part of E1 or E2 and
    # never combined with them.
    "late_full": dict(arch="late", variant="full", pair=False),
    "late_bb_only": dict(arch="late", variant="bb_only", pair=False),
    "late_generic": dict(arch="late", variant="generic", pair=False),
    # E1: the existing readout, injected early.
    "early_s_full": dict(arch="early_s", variant="full", pair=False),
    "early_s_bb_only": dict(arch="early_s", variant="bb_only", pair=False),
    "early_s_generic": dict(arch="early_s", variant="generic", pair=False),
    # E2: predicted atoms -> residue and pair conditioning.
    "atom_sz_full": dict(arch="atom", variant="full", pair=True),
    "atom_sz_bb_only": dict(arch="atom", variant="bb_only", pair=True),
    "atom_s_full": dict(arch="atom", variant="full", pair=False),
}
EARLY_ARMS = tuple(name for name, spec in ARMS.items() if spec["arch"] != "late")


# --- shared pieces ----------------------------------------------------------


def _zero_output(linear):
    """A projection that starts as an exact no-op but still has a gradient."""
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    return linear


def _head(d_in, d_out, *, d_hidden=D_HIDDEN):
    return nn.Sequential(
        nn.Linear(int(d_in), int(d_hidden)),
        nn.SiLU(),
        _zero_output(nn.Linear(int(d_hidden), int(d_out))),
    )


def _mlp(d_in, d_hidden, d_out):
    return nn.Sequential(
        nn.Linear(int(d_in), int(d_hidden)), nn.SiLU(), nn.Linear(int(d_hidden), int(d_out))
    )


def _broadcast_noise(embedded, length):
    """``[1, d]`` sigma embedding as ``[1, L, d]``."""
    if embedded.shape[0] != 1:
        raise ValueError(
            f"the conditioner takes one sigma per call, got {embedded.shape[0]}. "
            "One protein and one diffusion sample per forward is the initial scope"
        )
    return embedded[:, None, :].expand(1, int(length), embedded.shape[-1])


def masked_mean_max(values, mask):
    """``(mean, max)`` over ``dim=-2`` under ``mask``; both zero on an empty set.

    The max is the part that has to be written out. ``values.max()`` over a
    tensor whose absent rows are zeros returns 0 for an all-negative set and the
    absent row's own value for anything else -- so a slot the residue does not
    have would decide the feature. Absent rows are sent to ``-inf`` before the
    reduction and the result is zeroed where nothing was present.
    """
    keep = mask[..., None].to(values.dtype)
    count = keep.sum(dim=-2).clamp_min(1.0)
    mean = (values * keep).sum(dim=-2) / count
    filled = torch.where(mask[..., None], values, torch.full_like(values, float("-inf")))
    largest = filled.max(dim=-2).values
    any_present = mask.any(dim=-1, keepdim=True)
    largest = torch.where(any_present, largest, torch.zeros_like(largest))
    mean = torch.where(any_present, mean, torch.zeros_like(mean))
    return mean, largest


def distance_rbf(distance):
    """``[..., 16]`` Gaussian RBF, centres uniform on ``[0, 12]`` A, sigma 0.8 A."""
    centres = torch.linspace(
        0.0, ATOM_PAIR_RADIUS, N_RBF, device=distance.device, dtype=distance.dtype
    )
    return torch.exp(-((distance[..., None] - centres) ** 2) / (2.0 * RBF_SIGMA**2))


def atom_confidence(packed):
    """``([1, L, 37], [1, L, 37])`` scaled psCE and its presence indicator.

    FaMPNN reports psCE per *side-chain* slot, so it is scattered onto the
    atom37 axis and backbone slots keep zero in both channels -- a backbone atom
    has no predicted side-chain error, and representing that as "confidence 0"
    without the second channel would make it indistinguishable from a side-chain
    atom the packer was certain about.

    The scaling is the readout's, ``/4`` then clipped, so the two arms of E1 and
    E2 read the same quantity on the same scale.
    """
    coords = packed.coords37
    shape = coords.shape[:-1]  # [B, L, 37]
    value = coords.new_zeros(shape)
    present = coords.new_zeros(shape)
    if packed.psce is None:
        return value, present
    sidechain = list(atom37.SIDECHAIN_SLOTS)
    psce = packed.psce.to(device=coords.device, dtype=coords.dtype)
    if psce.shape[-1] != len(sidechain):
        raise ValueError(
            f"psce must cover {len(sidechain)} side-chain slots, got {psce.shape[-1]}"
        )
    finite = torch.isfinite(psce)
    # A missing or nonfinite confidence is zero *with a zero indicator*, never a
    # plausible-looking number.
    psce = torch.where(finite, psce, torch.zeros_like(psce))
    value[..., sidechain] = (psce / PSCE_SCALE).clamp(min=0.0, max=4.0)
    present[..., sidechain] = finite.to(coords.dtype)
    available = packed.available.to(coords.dtype)
    return value * available, present * available


def residue_frames(packed):
    """``(R, t, valid)`` from the predicted N, CA, C of every residue.

    ``valid`` is ``seq_mask & frame_valid`` -- backbone and presence only. The
    controls must not inherit a validity decision that depends on the side
    chains, or a "BB-only" arm would already be reading which residues packed
    successfully.
    """
    coords = packed.coords37.detach()
    n, ca, c = (coords[..., slot, :] for slot in frames_slots())
    rotation, translation = frames.build_frame(
        torch.nan_to_num(n), torch.nan_to_num(ca), torch.nan_to_num(c)
    )
    geometric = frames.frame_is_valid(n, ca, c)
    valid = packed.valid_residues.bool() & geometric
    eye = torch.eye(3, device=coords.device, dtype=coords.dtype)
    rotation = torch.where(valid[..., None, None], torch.nan_to_num(rotation), eye)
    return rotation, torch.nan_to_num(translation), valid


def frames_slots():
    """The atom37 slots of N, CA, C, in that order."""
    return (atom37.ATOM37.index("N"), atom37.ATOM37.index("CA"), atom37.ATOM37.index("C"))


def pooled_slots(variant):
    """Which atom37 slots a variant's residue encoder pools over."""
    return (
        list(atom37.BACKBONE_SLOTS)
        if variant == "bb_only"
        else list(atom37.SIDECHAIN_SLOTS)
    )


# --- E1: the existing readout, injected early -------------------------------


class EarlySingleConditioner(nn.Module):
    """``Delta s_i = m_i g(sigma) W_out SiLU(W_in [r_i ; e(sigma)])``.

    The readout is :class:`~pxf.couple.readout.FeedbackReadout`, imported and
    used as it stands -- same feature-group order, same LayerNorm, same sequence
    embedding, same three variants. Nothing about *what* is read changes, so the
    only difference from the late adapter is that the residual is added to
    ``s_single`` instead of to ``a_token``. That is the whole hypothesis, and
    keeping the representation fixed is what lets the comparison test it.

    ``delta_pair`` is ``None``: E1 makes no pair correction.
    """

    reads_packing = True
    produces_conditioning = True
    arch = "early_s"

    def __init__(
        self,
        c_h_V,
        c_s,
        *,
        variant="full",
        gate=None,
        d_hidden=D_HIDDEN,
        d_noise=D_NOISE,
        sequence_width=None,
    ):
        super().__init__()
        readout_kwargs = {} if sequence_width is None else dict(sequence_width=sequence_width)
        self.readout = FeedbackReadout(c_h_V, variant=variant, **readout_kwargs)
        self.c_s = int(c_s)
        self.noise = NoiseEmbedding(int(d_noise))
        self.single_head = _head(
            self.readout.width + int(d_noise), self.c_s, d_hidden=int(d_hidden)
        )
        self.gate = gate

    @property
    def variant(self):
        return self.readout.variant

    @property
    def pair(self):
        return False

    def forward(self, packed, sigma, *, reference=None):
        readout, stats = self.readout(packed)
        if readout.shape[0] != 1:
            raise ValueError(
                f"the early conditioner takes one protein per call, got "
                f"{readout.shape[0]}"
            )
        embedded = self.noise(sigma).to(readout.dtype).to(readout.device)
        noise = _broadcast_noise(embedded, readout.shape[1])
        delta_single = self.single_head(torch.cat([readout, noise], dim=-1))
        delta_single = delta_single * packed.valid_residues[..., None] * gate_scale(
            self.gate, sigma, delta_single
        )
        feedback = ConditioningFeedback(delta_single=delta_single, delta_pair=None)
        stats.update(feedback.norms())
        stats["residues_conditioned"] = int(packed.valid_residues.sum())
        return feedback, stats

    @torch.no_grad()
    def is_identity(self):
        final = self.single_head[-1]
        return bool(torch.all(final.weight == 0)) and bool(torch.all(final.bias == 0))

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def identity(self):
        return dict(
            kind="EarlySingleConditioner",
            version=CONDITIONER_VERSION,
            arch=self.arch,
            variant=self.variant,
            pair=False,
            c_s=self.c_s,
            c_z=None,
            d_hidden=int(self.single_head[0].out_features),
            d_noise=int(self.noise.dim),
            readout=self.readout.identity(),
            gate=self.gate.identity() if self.gate is not None else None,
            zero_initialized=self.is_identity(),
            parameters=sum(p.numel() for p in self.parameters()),
        )


def gate_scale(gate, sigma, like):
    """``g(sigma)`` shaped to multiply a ``[1, ..., d]`` residual."""
    if gate is None:
        return torch.ones((), dtype=like.dtype, device=like.device)
    scale = gate(sigma).to(like.dtype).to(like.device).reshape(-1)
    if scale.numel() != 1:
        raise ValueError(
            f"one sigma per call: the gate returned {scale.numel()} values"
        )
    return scale.reshape((1,) * like.dim())


# --- E2: predicted atoms -> residue and pair conditioning -------------------


class AtomConditioner(nn.Module):
    """Predicted atoms -> ``U_i`` and ``V_ij`` -> ``Delta s`` and ``Delta z``.

    Deliberately *not* fed ``h_packed`` or the E1 readout's node group. FaMPNN's
    node summary is the encoder's own view of the structure and mixing it in
    would make "what did the side-chain geometry contribute" unanswerable: E1
    already tests that representation at this site. Here the only structural
    input is ``coords37`` plus the availability mask, so a result is attributable
    to the geometry the packer produced.

    The pair branch is the capability the late site could not have. ``z_pair``
    is read by the atom encoder and by every transformer block as an attention
    bias, so ``V_ij`` can state that *these two residues* interact -- which is
    what a side chain bridging them is evidence of. ``V_ij`` is directed and is
    not symmetrized: its direction feature is expressed in residue ``i``'s frame,
    so ``V_ji`` is a different quantity rather than a redundant copy.
    """

    reads_packing = True
    produces_conditioning = True
    arch = "atom"

    def __init__(
        self,
        c_s,
        c_z,
        *,
        variant="full",
        pair=True,
        gate=None,
        d_hidden=D_HIDDEN,
        d_noise=D_NOISE,
        max_neighbours=MAX_NEIGHBOURS,
        neighbour_radius=NEIGHBOUR_RADIUS,
        max_atom_pairs=MAX_ATOM_PAIRS,
        atom_pair_radius=ATOM_PAIR_RADIUS,
    ):
        super().__init__()
        if variant not in ("full", "bb_only"):
            raise ValueError(
                f"unknown atom-conditioner variant {variant!r}; choose from "
                "('full', 'bb_only'). There is no sigma-only arm here: that "
                "control belongs to E1, where it shares a code path with the "
                "readout it is a control for"
            )
        self.variant = variant
        self.pair = bool(pair)
        self.c_s, self.c_z = int(c_s), int(c_z)
        self.max_neighbours = int(max_neighbours)
        self.neighbour_radius = float(neighbour_radius)
        self.max_atom_pairs = int(max_atom_pairs)
        self.atom_pair_radius = float(atom_pair_radius)

        self.atom_slot = nn.Embedding(atom37.NUM_ATOM37, D_ATOM_SLOT)
        self.sequence = nn.Embedding(atom37.UNKNOWN_AA_INDEX + 1, D_SEQUENCE)
        self.atom_mlp = _mlp(N_ATOM_FEATURES, D_ATOM, D_ATOM)
        self.residue_mlp = _mlp(N_RESIDUE_FEATURES, 2 * D_RESIDUE, D_RESIDUE)
        self.noise = NoiseEmbedding(int(d_noise))
        self.single_head = _head(D_RESIDUE + int(d_noise), self.c_s, d_hidden=int(d_hidden))
        if self.pair:
            self.edge_mlp = _mlp(N_EDGE_FEATURES, D_EDGE, D_EDGE)
            self.pair_mlp = _mlp(2 * D_EDGE + 2 * D_RESIDUE, D_RESIDUE, D_PAIR)
            self.pair_head = _head(D_PAIR + int(d_noise), self.c_z, d_hidden=int(d_hidden))
        self.gate = gate
        self.register_buffer(
            "is_sidechain",
            torch.zeros(atom37.NUM_ATOM37, dtype=torch.bool).index_fill_(
                0, torch.tensor(list(atom37.SIDECHAIN_SLOTS)), True
            ),
            persistent=False,
        )

    # ---- inputs ----------------------------------------------------------

    def _pooled_mask(self, packed):
        """``[L, 37]`` bool: which atoms this variant's residue encoder pools.

        Slots outside the pooled set are False, so they cannot reach the mean,
        the max, or -- through the pair branch -- any edge. A residue type that
        does not have a slot is already excluded by ``available``.
        """
        available = packed.available[0] > 0
        keep = torch.zeros_like(available)
        keep[:, pooled_slots(self.variant)] = True
        return available & keep

    def _atom_features(self, packed, rotation, translation, valid):
        """``[L, 37, 21]`` per-atom inputs, and the mask of atoms that exist."""
        coords = packed.coords37.detach()[0]  # [L, 37, 3]
        finite = torch.isfinite(coords).all(dim=-1)
        local = frames.to_local(torch.nan_to_num(coords), rotation[0], translation[0])
        local = torch.nan_to_num(local) / COORDINATE_SCALE
        confidence, present = atom_confidence(packed)
        if self.variant == "bb_only":
            # Structural zeros, not "happens to be zero": a BB-only control that
            # could read the packer's confidence would be reading the packing.
            confidence = torch.zeros_like(confidence)
            present = torch.zeros_like(present)
        slots = torch.arange(atom37.NUM_ATOM37, device=coords.device)
        embedded = self.atom_slot(slots)[None].expand(coords.shape[0], -1, -1)
        features = torch.cat(
            [
                embedded,
                local,
                confidence[0][..., None],
                present[0][..., None],
            ],
            dim=-1,
        )
        mask = self._pooled_mask(packed) & finite & valid[0][:, None]
        return features * mask[..., None].to(features.dtype), mask

    # ---- the residue term -------------------------------------------------

    def residue_representation(self, packed, rotation, translation, valid):
        """``([1, L, 128], stats)`` -- ``U_i``, zero on invalid residues."""
        features, mask = self._atom_features(packed, rotation, translation, valid)
        embedded = self.atom_mlp(features)
        mean, largest = masked_mean_max(embedded, mask)
        sequence = self.sequence(
            packed.aatype[0].clamp(0, atom37.UNKNOWN_AA_INDEX).long()
        )
        chi, chi_valid = chi_features(packed)
        reliability = reliability_features(packed, chi_valid)
        if self.variant == "bb_only":
            chi = torch.zeros_like(chi)
            reliability = torch.zeros_like(reliability)
        parts = torch.cat([mean, largest, sequence, chi[0], reliability[0]], dim=-1)
        residue = self.residue_mlp(parts) * valid[0][:, None].to(parts.dtype)
        stats = dict(
            pooled_atoms=int(mask.sum()),
            pooled_residues=int(mask.any(dim=-1).sum()),
            valid_residues=int(valid.sum()),
        )
        return residue[None], mask, stats

    # ---- the pair term ----------------------------------------------------

    @torch.no_grad()
    def neighbour_graph(self, packed, valid):
        """Directed ``(source, target)`` residue pairs from predicted CA atoms.

        Detached preprocessing: which residues are neighbours is a selection,
        not a differentiable quantity, and treating it as one would put a
        gradient on a ``topk`` boundary. Ties are broken by residue index via a
        stable sort, so two runs on the same structure select the same edges.
        """
        ca = packed.coords37.detach()[0][:, atom37.ATOM37.index("CA"), :]
        length = ca.shape[0]
        distance = torch.cdist(torch.nan_to_num(ca), torch.nan_to_num(ca))
        eligible = valid[0][:, None] & valid[0][None, :]
        eligible = eligible & ~torch.eye(length, dtype=torch.bool, device=ca.device)
        eligible = eligible & (distance <= self.neighbour_radius)
        distance = torch.where(eligible, distance, torch.full_like(distance, float("inf")))
        keep = min(self.max_neighbours, length)
        order = torch.sort(distance, dim=-1, stable=True).indices[:, :keep]
        selected = torch.gather(eligible, 1, order)
        source = (
            torch.arange(length, device=ca.device)[:, None].expand_as(order)[selected]
        )
        target = order[selected]
        return source, target, dict(
            edges=int(selected.sum()),
            residues_with_edges=int(selected.any(dim=-1).sum()),
            edges_truncated=int(
                (eligible.sum(dim=-1) > keep).sum()
            ),
        )

    @torch.no_grad()
    def atom_pairs(self, coords, available, source, target):
        """``([P, 32] flat atom-pair index, [P, 32] mask, stats)``.

        Selection only, on detached geometry, in chunks so the candidate block
        (``P x 37 x 37``) never has to exist all at once for a long chain.
        """
        pairs = source.shape[0]
        slots = atom37.NUM_ATOM37
        keep = min(self.max_atom_pairs, slots * slots)
        chosen = coords.new_zeros((pairs, keep), dtype=torch.long)
        mask = torch.zeros((pairs, keep), dtype=torch.bool, device=coords.device)
        sidechain = self.is_sidechain
        truncated = 0
        chunk = max(1, 4_194_304 // (slots * slots))
        for start in range(0, pairs, chunk):
            stop = min(start + chunk, pairs)
            i, j = source[start:stop], target[start:stop]
            distance = torch.cdist(coords[i], coords[j])  # [n, 37, 37]
            eligible = available[i][:, :, None] & available[j][:, None, :]
            if self.variant == "bb_only":
                # Both atoms backbone. `available` is already restricted to the
                # pooled set, so this is the whole of it.
                pass
            else:
                eligible = eligible & (
                    sidechain[None, :, None] | sidechain[None, None, :]
                )
            eligible = eligible & (distance <= self.atom_pair_radius)
            flat = distance.reshape(stop - start, -1)
            flat = torch.where(
                eligible.reshape(stop - start, -1), flat, torch.full_like(flat, float("inf"))
            )
            order = torch.sort(flat, dim=-1, stable=True).indices[:, :keep]
            chosen[start:stop] = order
            mask[start:stop] = torch.gather(
                eligible.reshape(stop - start, -1), 1, order
            )
            truncated += int((eligible.reshape(stop - start, -1).sum(-1) > keep).sum())
        return chosen, mask, dict(atom_pairs=int(mask.sum()), atom_pairs_truncated=truncated)

    def pair_representation(self, packed, residue, rotation, valid):
        """``(index, V_ij, stats)`` for the retained directed residue pairs."""
        coords = packed.coords37.detach()[0]
        available = self._pooled_mask(packed) if self.variant == "bb_only" else (
            (packed.available[0] > 0) & torch.isfinite(coords).all(dim=-1)
        )
        available = available & valid[0][:, None]
        source, target, graph_stats = self.neighbour_graph(packed, valid)
        if source.numel() == 0:
            return None, None, dict(graph_stats, atom_pairs=0, atom_pairs_truncated=0)
        chosen, mask, pair_stats = self.atom_pairs(
            torch.nan_to_num(coords), available, source, target
        )
        slot_a, slot_b = chosen // atom37.NUM_ATOM37, chosen % atom37.NUM_ATOM37

        gathered = torch.nan_to_num(coords)
        x_ia = gathered[source[:, None].expand_as(slot_a), slot_a]
        x_jb = gathered[target[:, None].expand_as(slot_b), slot_b]
        offset = x_jb - x_ia
        # Squared-then-clamped rather than `.norm()`: the norm's gradient at
        # zero is NaN, and two atoms can coincide in a predicted structure.
        distance = offset.pow(2).sum(-1).clamp_min(1e-12).sqrt()
        direction = frames.rotate_to_local(
            offset / distance.clamp_min(1e-6)[..., None], rotation[0][source]
        )

        confidence, present = atom_confidence(packed)
        if self.variant == "bb_only":
            confidence = torch.zeros_like(confidence)
            present = torch.zeros_like(present)
        conf_a = confidence[0][source[:, None].expand_as(slot_a), slot_a]
        conf_b = confidence[0][target[:, None].expand_as(slot_b), slot_b]
        present_a = present[0][source[:, None].expand_as(slot_a), slot_a]
        present_b = present[0][target[:, None].expand_as(slot_b), slot_b]

        sc_a, sc_b = self.is_sidechain[slot_a], self.is_sidechain[slot_b]
        if self.variant == "bb_only":
            kinds = torch.zeros(*slot_a.shape, len(PAIR_TYPES), device=coords.device)
        else:
            kinds = torch.stack(
                [(sc_a & sc_b), (sc_a & ~sc_b), (~sc_a & sc_b)], dim=-1
            ).to(coords.dtype)

        features = torch.cat(
            [
                self.atom_slot(slot_a),
                self.atom_slot(slot_b),
                distance_rbf(distance),
                direction,
                conf_a[..., None],
                conf_b[..., None],
                present_a[..., None],
                present_b[..., None],
                kinds,
            ],
            dim=-1,
        )
        features = features * mask[..., None].to(features.dtype)
        edges = self.edge_mlp(features)
        mean, largest = masked_mean_max(edges, mask)
        pooled = torch.cat([mean, largest, residue[0][source], residue[0][target]], dim=-1)
        values = self.pair_mlp(pooled)
        keep = mask.any(dim=-1)
        values = values * keep[:, None].to(values.dtype)
        return (source, target, keep), values, dict(graph_stats, **pair_stats)

    # ---- the heads --------------------------------------------------------

    def forward(self, packed, sigma, *, reference=None):
        if packed.coords37.shape[0] != 1:
            raise ValueError(
                "the atom conditioner takes one protein per call, got "
                f"{packed.coords37.shape[0]}"
            )
        rotation, translation, valid = residue_frames(packed)
        residue, _pooled, stats = self.residue_representation(
            packed, rotation, translation, valid
        )
        length = residue.shape[1]
        embedded = self.noise(sigma).to(residue.dtype).to(residue.device)
        noise = _broadcast_noise(embedded, length)
        delta_single = self.single_head(torch.cat([residue, noise], dim=-1))
        delta_single = delta_single * valid[..., None].to(delta_single.dtype)
        delta_single = delta_single * gate_scale(self.gate, sigma, delta_single)

        delta_pair = None
        if self.pair:
            index, values, pair_stats = self.pair_representation(
                packed, residue, rotation, valid
            )
            stats.update(pair_stats)
            delta_pair = torch.zeros(
                length * length, self.c_z, dtype=delta_single.dtype, device=residue.device
            )
            if index is not None:
                source, target, keep = index
                paired = torch.cat(
                    [values, noise[0][source]], dim=-1
                )
                written = self.pair_head(paired) * keep[:, None].to(values.dtype)
                written = written * gate_scale(self.gate, sigma, written)
                # index_add rather than an in-place scatter: the destination is
                # created inside the graph and the indices are unique, so this
                # is a differentiable placement, not an accumulation.
                delta_pair = delta_pair.index_add(
                    0, source * length + target, written
                )
            delta_pair = delta_pair.reshape(length, length, self.c_z)

        feedback = ConditioningFeedback(delta_single=delta_single, delta_pair=delta_pair)
        stats.update(feedback.norms())
        return feedback, stats

    @torch.no_grad()
    def is_identity(self):
        heads = [self.single_head] + ([self.pair_head] if self.pair else [])
        return all(
            bool(torch.all(head[-1].weight == 0)) and bool(torch.all(head[-1].bias == 0))
            for head in heads
        )

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def identity(self):
        return dict(
            kind="AtomConditioner",
            version=CONDITIONER_VERSION,
            arch=self.arch,
            variant=self.variant,
            pair=self.pair,
            c_s=self.c_s,
            c_z=self.c_z if self.pair else None,
            d_hidden=int(self.single_head[0].out_features),
            d_noise=int(self.noise.dim),
            widths=dict(
                atom_slot=D_ATOM_SLOT,
                atom=D_ATOM,
                sequence=D_SEQUENCE,
                residue=D_RESIDUE,
                edge=D_EDGE,
                pair=D_PAIR,
                atom_features=N_ATOM_FEATURES,
                residue_features=N_RESIDUE_FEATURES,
                edge_features=N_EDGE_FEATURES,
            ),
            neighbourhood=dict(
                max_neighbours=self.max_neighbours,
                neighbour_radius=self.neighbour_radius,
                max_atom_pairs=self.max_atom_pairs,
                atom_pair_radius=self.atom_pair_radius,
                rbf_centres=N_RBF,
                rbf_sigma=RBF_SIGMA,
                coordinate_scale=COORDINATE_SCALE,
                pair_types=list(PAIR_TYPES),
            ),
            gate=self.gate.identity() if self.gate is not None else None,
            zero_initialized=self.is_identity(),
            parameters=sum(p.numel() for p in self.parameters()),
        )


# --- selection and checkpoint compatibility ---------------------------------


def build_conditioner(arm, *, c_h_V, c_token, c_s, c_z, gate=None, d_hidden=D_HIDDEN,
                      d_noise=D_NOISE):
    """The module one named arm asks for. ``c_s``/``c_z`` come from the model."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {sorted(ARMS)}")
    spec = ARMS[arm]
    if spec["arch"] == "late":
        from pxf.couple.readout import FeedbackPath

        return FeedbackPath(
            c_h_V, c_token, variant=spec["variant"], gate=gate, d_hidden=d_hidden,
            d_noise=d_noise,
        )
    if spec["arch"] == "early_s":
        return EarlySingleConditioner(
            c_h_V, c_s, variant=spec["variant"], gate=gate, d_hidden=d_hidden,
            d_noise=d_noise,
        )
    return AtomConditioner(
        c_s, c_z, variant=spec["variant"], pair=spec["pair"], gate=gate,
        d_hidden=d_hidden, d_noise=d_noise,
    )


# Fields that make two conditioners *incomparable* rather than merely differently
# trained. A mismatch here means the weights describe a different function of a
# different input, so loading them would silently mislabel an arm.
COMPATIBILITY_KEYS = ("kind", "version", "arch", "variant", "pair", "c_s", "c_z")


def check_compatible(recorded, built, *, path=None):
    """Refuse a checkpoint whose architecture metadata differs from the module.

    ``load_state_dict(strict=True)`` is not enough: E1's ``full`` and ``bb_only``
    arms have identical parameter shapes by construction -- that is the point of
    the controls -- so the wrong one loads cleanly and the run reports the wrong
    variant. The metadata is what distinguishes them.
    """
    recorded = recorded or {}
    mismatch = [
        (key, recorded.get(key), built.get(key))
        for key in COMPATIBILITY_KEYS
        if recorded.get(key) != built.get(key)
    ]
    if mismatch:
        where = f"{path}: " if path else ""
        raise ValueError(
            where
            + "this checkpoint records a different conditioner than the one "
            "built here, and the two have the same parameter shapes so the load "
            "would succeed and mislabel the arm: "
            + "; ".join(f"{k}: checkpoint={a!r} built={b!r}" for k, a, b in mismatch)
        )
    return True
