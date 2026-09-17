"""The SC -> BB feedback: what the backbone is told about the packing it produced.

    z_i     = [ LN(h_i^packed) ; Geom_i ; Emb(s_i) ; u_i ]
    delta_a = m_i * g_SB(sigma_B) * W2 SiLU( W1 [ z_i ; e(sigma_B) ] )

Four feature groups, each answering a different question, and each individually
switchable so the controls are the same code path as the candidate:

``node``         FaMPNN's *final node readout* for the re-encoded ``bb0 + sc0``.
                 Deliberately the encoder's own invariant summary rather than
                 raw geometric vectors: the encoder is the thing that knows how
                 to look at a full-atom structure, and feeding global GVP
                 vectors into a plain MLP would both lose that and break the
                 rotation invariance the rest of the features have.
``geometry``     sine/cosine of the chi torsions the packer chose, with their
                 validity mask. The local conformation, in the coordinate the
                 packer actually decided.
``environment``  compact contact, clash and burial summaries *involving the
                 predicted side chains*. What h_V summarizes per residue it
                 summarizes through the encoder's own receptive field; these are
                 the direct packing-quality signals a backbone correction would
                 want, stated in a few scalars rather than left implicit.
``reliability``  the packer's own psCE confidence, frame validity and
                 geometry-validity flags. Which residues' feedback to believe.

**Only inference-available quantities.** No native side chains, no native chi
agreement, no residual backbone error. Those are supervision; a readout that
could reach them would report a gain that does not exist at inference. The
inputs come from ``bb0``, ``sc0``, the fixed sequence and the packer's psCE, all
of which exist during sampling.

**The output projection is zero-initialized and the gate is not.** At step 0 the
correction is exactly zero, so the coupled system reproduces PXDesign
bit-for-bit -- but zero-initializing the multiplicative gate as well would make
the product's gradient vanish in both factors and the adapter would never leave
the origin. So ``W2`` and its bias start at zero and the gate is a *fixed*
function of sigma, equal to one throughout the training window. Learned
confidence gating is deliberately deferred: adding it before the basic
correction works would make a null result ambiguous between "no signal" and "the
gate closed".
"""

import math
from dataclasses import dataclass

import torch
from torch import nn

from pxf import atom37
from pxf.couple import torsions
from pxf.couple.adapters import NoiseEmbedding

# The feature groups, in the order they are concatenated into z.
GROUPS = ("node", "geometry", "environment", "sequence", "reliability")
# Which groups a variant may read. Groups outside the set are still *present* at
# full width and fed exact zeros, so every variant has the identical parameter
# count -- the controls differ in information, not in capacity.
VARIANTS = {
    # The candidate: everything.
    "full": set(GROUPS),
    # Trained BB/sequence-only control. `node` is re-pointed at the
    # side-chain-masked encoding (h_base) and every side-chain-derived group is
    # zeroed, so it tests "does a learned backbone correction help at all",
    # which is the comparison that makes an SC-specific claim possible.
    "bb_only": {"node", "sequence"},
    # Trained generic control: z is identically zero, so the correction can only
    # be a function of sigma and the adapter's own biases.
    "generic": set(),
}

# Feature-group widths.
N_CHI_FEATURES = 2 * torsions.MAX_CHI + torsions.MAX_CHI  # sin, cos, validity
N_ENVIRONMENT = 8
N_RELIABILITY = 6
DEFAULT_SEQUENCE_WIDTH = 32

# Contact and clash radii, in Angstroms. 4.5 is the usual heavy-atom contact
# shell; 2.8 is below any non-bonded heavy-atom contact and so counts real
# overlaps rather than close packing.
CONTACT_RADIUS = 4.5
CLASH_RADIUS = 2.8
NEIGHBOUR_RADIUS = 8.0
BURIAL_RADIUS = 10.0
# psCE is an error in Angstroms; 4 A is the top of the head's own bin range.
PSCE_SCALE = 4.0


# --- geometry and environment ----------------------------------------------


def chi_features(packed, *, tables=None):
    """``[B, L, 12]`` sine/cosine of the packed chis plus their validity."""
    chi, valid = torsions.chi_angles(
        packed.coords37,
        packed.aatype,
        available=packed.available,
        tables=tables,
    )
    return torch.cat([torsions.chi_sin_cos(chi, valid), valid.float()], dim=-1), valid


def environment_features(packed):
    """``[B, L, 8]`` contact, clash, burial and extension summaries.

    Computed over the *available* atoms only, flattened to a list rather than a
    dense ``[L, 37, L, 37]`` block -- that block is 200M entries at L = 384 and
    the atom list is about 3k, so the flattened form is the difference between
    36 MB and 800 MB.

    Same-residue pairs are excluded throughout: they are fixed by the residue's
    own chemistry, so counting them would swamp the contact numbers with a
    constant that depends only on residue type, which ``Emb(s_i)`` already says.
    """
    coords, available = packed.coords37, packed.available
    batch, length = coords.shape[0], coords.shape[1]
    device = coords.device
    sidechain = list(atom37.SIDECHAIN_SLOTS)
    ca = coords[..., atom37.ATOM37.index("CA"), :]
    out = coords.new_zeros(batch, length, N_ENVIRONMENT)

    for b in range(batch):
        keep = available[b] > 0  # [L, 37]
        if not bool(keep.any()):
            continue
        residue_of = torch.arange(length, device=device)[:, None].expand(
            length, atom37.NUM_ATOM37
        )[keep]
        is_sidechain = torch.zeros(atom37.NUM_ATOM37, dtype=torch.bool, device=device)
        is_sidechain[sidechain] = True
        atom_is_sc = is_sidechain[None, :].expand(length, atom37.NUM_ATOM37)[keep]
        points = coords[b][keep]  # [N, 3]

        subject = torch.nonzero(atom_is_sc, as_tuple=True)[0]
        if subject.numel() == 0:
            continue
        distance = torch.cdist(points[subject], points)  # [N_sc, N]
        different = residue_of[subject][:, None] != residue_of[None, :]
        distance = torch.where(different, distance, torch.full_like(distance, 1e4))
        owner = residue_of[subject]

        def per_residue(values, index, *, reduce="sum"):
            acc = coords.new_zeros(length)
            if reduce == "sum":
                return acc.index_add_(0, index, values.to(acc.dtype))
            acc = acc.fill_(1e4)
            return acc.scatter_reduce_(0, index, values.to(acc.dtype), reduce="amin")

        contacts = per_residue((distance < CONTACT_RADIUS).sum(-1).float(), owner)
        clashes = per_residue((distance < CLASH_RADIUS).sum(-1).float(), owner)
        closest = per_residue(distance.min(-1).values, owner, reduce="amin")
        closest = torch.where(closest > 1e3, torch.full_like(closest, 20.0), closest)

        # Neighbour count and burial from CA, over residues rather than atoms,
        # so they do not double-count long side chains.
        ca_distance = torch.cdist(ca[b], ca[b])
        ca_distance.fill_diagonal_(1e4)
        real = (packed.seq_mask[b] > 0).float()
        neighbours = ((ca_distance < NEIGHBOUR_RADIUS).float() * real[None, :]).sum(-1)
        burial = ((ca_distance < BURIAL_RADIUS).float() * real[None, :]).sum(-1)

        extension = torch.zeros(length, device=device, dtype=coords.dtype)
        sc_keep = keep[:, sidechain]
        if bool(sc_keep.any()):
            offset = (coords[b][:, sidechain, :] - ca[b][:, None, :]).norm(dim=-1)
            extension = (offset * sc_keep.float()).max(dim=-1).values
        n_atoms = sc_keep.float().sum(-1)

        out[b, :, 0] = (contacts / 20.0).clamp(max=4.0)
        out[b, :, 1] = torch.exp(-closest / 4.0)
        out[b, :, 2] = (clashes / 5.0).clamp(max=4.0)
        out[b, :, 3] = (CLASH_RADIUS - closest).clamp(min=0.0) / CLASH_RADIUS
        out[b, :, 4] = (extension / 8.0).clamp(max=2.0)
        out[b, :, 5] = n_atoms / 10.0
        out[b, :, 6] = (neighbours / 20.0).clamp(max=4.0)
        out[b, :, 7] = (burial / 100.0).clamp(max=4.0)
    return out * (packed.seq_mask > 0).float()[..., None]


def reliability_features(packed, chi_valid):
    """``[B, L, 6]`` how much to believe this residue's feedback.

    Every entry is available at inference: psCE is the packer's own confidence
    head, and the rest are validity flags derived from the availability mask.
    """
    sidechain = list(atom37.SIDECHAIN_SLOTS)
    device = packed.coords37.device
    batch, length = packed.coords37.shape[0], packed.coords37.shape[1]
    sc_available = packed.available[..., sidechain]
    count = sc_available.sum(-1)

    if packed.psce is None:
        psce_mean = torch.zeros(batch, length, device=device)
        psce_max = torch.zeros(batch, length, device=device)
        has_psce = torch.zeros(batch, length, device=device)
    else:
        psce = packed.psce.to(device).float()
        if psce.shape[-1] != len(sidechain):
            raise ValueError(
                f"psce must cover {len(sidechain)} side-chain slots, got {psce.shape[-1]}"
            )
        weighted = psce * sc_available
        psce_mean = weighted.sum(-1) / count.clamp_min(1.0)
        psce_max = (
            torch.where(sc_available > 0, psce, torch.zeros_like(psce)).max(-1).values
        )
        has_psce = torch.ones(batch, length, device=device)

    expected = packed.visibility.exists[..., sidechain].sum(-1)
    return (
        torch.stack(
            [
                (psce_mean / PSCE_SCALE).clamp(max=4.0),
                (psce_max / PSCE_SCALE).clamp(max=4.0),
                has_psce,
                packed.frame_valid.float(),
                chi_valid.float().sum(-1) / torsions.MAX_CHI,
                count / expected.clamp_min(1.0),
            ],
            dim=-1,
        )
        * (packed.seq_mask > 0).float()[..., None]
    )


# --- the gate ---------------------------------------------------------------


@dataclass(frozen=True)
class SigmaWindow:
    """``g_SB(sigma)``: exactly 1 on ``[sigma_min, sigma_max]``, tapering outside.

    A fixed function, with no parameters, on purpose. The gate multiplies a
    zero-initialized projection, so making it learnable *and* starting it near
    zero would put the product at a stationary point in both factors. Equal to
    one throughout the training window means the adapter's gradient at step 0 is
    whatever the projection's is, which is the property that lets it start
    learning at all.

    The taper exists for deployment rather than training: outside the window the
    correction was never fitted, so it decays smoothly instead of being applied
    at full strength or switching off discontinuously at a sigma the sampler
    visits. Smoothstep in log sigma, matching :class:`pxf.couple.bs_policy.Gate`.
    """

    sigma_min: float
    sigma_max: float
    taper: float = 2.0  # multiplicative margin over which the gate falls to 0

    def __post_init__(self):
        if not 0 < self.sigma_min < self.sigma_max:
            raise ValueError(
                f"need 0 < sigma_min < sigma_max, got [{self.sigma_min}, {self.sigma_max}]"
            )
        if self.taper <= 1.0:
            raise ValueError(f"taper must exceed 1, got {self.taper}")

    def __call__(self, sigma):
        sigma = torch.as_tensor(sigma, dtype=torch.float32)
        log = torch.log(sigma.clamp_min(1e-12))
        lo_off = math.log(self.sigma_min / self.taper)
        lo_on = math.log(self.sigma_min)
        hi_on = math.log(self.sigma_max)
        hi_off = math.log(self.sigma_max * self.taper)

        def smoothstep(u):
            u = u.clamp(0.0, 1.0)
            return u * u * (3.0 - 2.0 * u)

        rising = smoothstep((log - lo_off) / (lo_on - lo_off))
        falling = smoothstep((hi_off - log) / (hi_off - hi_on))
        return rising * falling

    def identity(self):
        return dict(
            kind="sigma_window",
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            taper=self.taper,
        )


# --- the readout ------------------------------------------------------------


class FeedbackReadout(nn.Module):
    """``z_i`` from a packed structure. One width, whatever the variant reads."""

    def __init__(self, c_h_V, *, variant="full", sequence_width=DEFAULT_SEQUENCE_WIDTH):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
        self.variant = variant
        self.reads = VARIANTS[variant]
        self.c_h_V = int(c_h_V)
        self.norm = nn.LayerNorm(self.c_h_V)
        self.sequence = nn.Embedding(atom37.UNKNOWN_AA_INDEX + 1, int(sequence_width))
        self.widths = dict(
            node=self.c_h_V,
            geometry=N_CHI_FEATURES,
            environment=N_ENVIRONMENT,
            sequence=int(sequence_width),
            reliability=N_RELIABILITY,
        )

    @property
    def width(self):
        return sum(self.widths[name] for name in GROUPS)

    def forward(self, packed):
        """``(z, stats)`` with ``z`` masked to zero on invalid residues."""
        node_source = packed.h_packed
        if self.variant == "bb_only":
            if packed.h_base is None:
                raise ValueError(
                    "the bb_only control needs h_base, the side-chain-masked "
                    "encoding; the cycle recorded none, so this arm would be "
                    "reading the packed encoding and would not be a control"
                )
            node_source = packed.h_base
        node = self.norm(node_source)

        geometry, chi_valid = chi_features(packed)
        parts = dict(
            node=node,
            geometry=geometry,
            environment=environment_features(packed),
            sequence=self.sequence(packed.aatype.clamp(0, atom37.UNKNOWN_AA_INDEX).long()),
            reliability=reliability_features(packed, chi_valid),
        )
        # Groups the variant does not read are zeroed rather than dropped, so the
        # controls have exactly the same parameter count as the candidate and any
        # difference between them is information rather than capacity.
        zeroed = [name for name in GROUPS if name not in self.reads]
        for name in zeroed:
            parts[name] = torch.zeros_like(parts[name])

        valid = packed.valid_residues[..., None]
        z = torch.cat([parts[name] for name in GROUPS], dim=-1) * valid
        stats = dict(
            z_norm=float(z.detach().norm(dim=-1).mean()),
            z_groups_zeroed=zeroed,
            chis_valid=int(chi_valid.sum()),
            valid_residues=int(packed.valid_residues.sum()),
        )
        return z, stats

    def identity(self):
        return dict(
            variant=self.variant,
            reads=sorted(self.reads),
            zeroed=sorted(set(GROUPS) - self.reads),
            widths=dict(self.widths),
            width=self.width,
            parameters=sum(p.numel() for p in self.parameters()),
        )


class FeedbackPath(nn.Module):
    """``A_SB``: readout plus the gated, zero-initialized residual generator.

    Declares ``reads_packing`` so :meth:`pxf.couple.adapters.CouplingAdapters.delta_a`
    hands it the whole packed structure instead of ``h_V`` alone. Drops into the
    same ``sc_to_bb`` attribute as :class:`~pxf.couple.adapters.ResidualAdapter`,
    so the trainer, the phase freezing and the checkpoint layout are unchanged.
    """

    reads_packing = True

    def __init__(
        self,
        c_h_V,
        c_token,
        *,
        d_hidden=256,
        d_noise=64,
        variant="full",
        gate=None,
        sequence_width=DEFAULT_SEQUENCE_WIDTH,
    ):
        super().__init__()
        self.readout = FeedbackReadout(
            c_h_V, variant=variant, sequence_width=sequence_width
        )
        self.c_token = int(c_token)
        self.noise = NoiseEmbedding(d_noise)
        self.project_in = nn.Linear(self.readout.width + d_noise, int(d_hidden))
        self.activation = nn.SiLU()
        self.project_out = nn.Linear(int(d_hidden), self.c_token)
        # Zero output, non-zero gate: the correction is exactly nothing at step 0
        # and still has a gradient. See the module docstring.
        nn.init.zeros_(self.project_out.weight)
        nn.init.zeros_(self.project_out.bias)
        self.gate = gate

    @property
    def variant(self):
        return self.readout.variant

    def forward(self, packed, sigma, *, reference=None):
        z, stats = self.readout(packed)
        embedded = self.noise(sigma).to(z.dtype).to(z.device)
        if embedded.shape[0] == 1 and z.shape[0] > 1:
            embedded = embedded.expand(z.shape[0], -1)
        if embedded.shape[0] != z.shape[0]:
            raise ValueError(
                f"got {embedded.shape[0]} sigma values for {z.shape[0]} examples"
            )
        embedded = embedded[:, None, :].expand(z.shape[0], z.shape[1], embedded.shape[-1])
        hidden = self.activation(self.project_in(torch.cat([z, embedded], dim=-1)))
        delta = self.project_out(hidden)

        scale = 1.0
        if self.gate is not None:
            scale = self.gate(sigma).to(delta.dtype).to(delta.device).reshape(-1)
            if scale.numel() == 1:
                scale = scale.expand(delta.shape[0])
            scale = scale[:, None, None]
        delta = delta * packed.valid_residues[..., None] * scale

        with torch.no_grad():
            norm = delta.norm(dim=-1)
            stats.update(
                delta_a_norm=float(norm.mean()),
                delta_a_norm_max=float(norm.max()) if norm.numel() else 0.0,
                gate=float(torch.as_tensor(scale).float().mean()),
                sigma=float(torch.as_tensor(sigma).float().mean()),
            )
            if reference is not None:
                # The residual's size relative to what it is added to. The
                # number that says whether the correction is a nudge or a
                # replacement, and the first thing to look at if the pilot
                # destabilizes.
                reference_norm = reference.detach().float().norm(dim=-1)
                stats["a_token_norm"] = float(reference_norm.mean())
                stats["relative_residual"] = float(
                    (norm.float() / reference_norm.clamp_min(1e-8)).mean()
                )
        return delta, stats

    @torch.no_grad()
    def is_identity(self):
        """True while the correction is still an exact no-op."""
        return bool(torch.all(self.project_out.weight == 0)) and bool(
            torch.all(self.project_out.bias == 0)
        )

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def identity(self):
        return dict(
            kind="FeedbackPath",
            variant=self.variant,
            c_token=self.c_token,
            readout=self.readout.identity(),
            gate=self.gate.identity() if self.gate is not None else None,
            d_hidden=int(self.project_in.out_features),
            zero_initialized=self.is_identity(),
            parameters=sum(p.numel() for p in self.parameters()),
        )
