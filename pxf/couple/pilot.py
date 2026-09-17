"""The SC -> BB pilot: cache the frozen half, train only the correction.

One corrective event costs a PXDesign conditioning pass, a PXDesign denoise, a
50-step FaMPNN packing rollout and a re-encode -- and all four are *frozen*. The
only thing that trains is the readout and ``A_SB``. So the frozen half is
computed once per example and cached, and the 2k pilot pays for the packing
once rather than once per step.

**What goes in the cache is exactly the frozen half.** The noisy state (or the
seed that reconstructs it), sigma, the sequence and topology, the predicted side
chains, the re-encoded features, the checkpoint identities and the Phase-1
BB->SC policy. The readout's *output* deliberately does not: ``z`` depends on
trainable parameters, so caching it would freeze the thing being optimized and
the run would report a flat loss with no explanation.

**The cache is keyed by what it depends on.** Two runs with different donors,
different packing lengths or a different Phase-1 policy did not compute the same
upstream, and silently reusing one for the other would make a controlled
comparison uncontrolled. :meth:`UpstreamCache.compatible` refuses rather than
warns.

The corrective call itself is never cached and never runs under ``no_grad``:
PXDesign's atom decoder has to differentiate with respect to ``delta_a``.
:meth:`pxf.couple.controller.CoupledDenoiser.corrective_event` asserts that.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from pxf import atom37


@dataclass
class UpstreamState:
    """The detached output of one corrective event's frozen half.

    Every tensor here is cut from the graph. Nothing in it trains, and nothing
    in it is native: see :class:`pxf.couple.visibility.PackedStructure` for what
    the readout is allowed to read.
    """

    packed: object  # visibility.PackedStructure, detached
    bb0_flat: torch.Tensor
    a_token: torch.Tensor
    delta_h: torch.Tensor | None
    sidechains: torch.Tensor
    inputs: object  # converter.CoupledInputs
    sigma: torch.Tensor
    pack_steps: int | None = None
    bs_policy: str = "bypass"
    # Seconds for the three stages, so an evaluator can charge each arm for the
    # work it actually needs rather than for the whole frozen half.
    timings: dict = field(default_factory=dict)

    def identity(self):
        return dict(
            sigma=float(torch.as_tensor(self.sigma).float().mean()),
            pack_steps=self.pack_steps,
            bs_policy=self.bs_policy,
            timings=dict(self.timings),
            residues=int(self.packed.seq_mask.sum()),
            visibility=self.packed.visibility.record(),
        )

    def to(self, device):
        from dataclasses import replace

        def move(value):
            return value.to(device) if torch.is_tensor(value) else value

        packed = self.packed
        packed = replace(
            packed,
            h_packed=move(packed.h_packed),
            coords37=move(packed.coords37),
            aatype=move(packed.aatype),
            seq_mask=move(packed.seq_mask),
            psce=move(packed.psce),
            h_base=move(packed.h_base),
        )
        from pxf.couple.visibility import Visibility

        packed.visibility = Visibility(
            available=move(packed.visibility.available),
            missing_atom_mask=move(packed.visibility.missing_atom_mask),
            frame_valid=move(packed.visibility.frame_valid),
            sidechain_visible=move(packed.visibility.sidechain_visible),
            exists=move(packed.visibility.exists),
            stats=packed.visibility.stats,
        )
        return replace(
            self,
            packed=packed,
            bb0_flat=move(self.bb0_flat),
            a_token=move(self.a_token),
            delta_h=move(self.delta_h),
            sidechains=move(self.sidechains),
            sigma=move(self.sigma),
        )


def backbone_supervision_mask(atom_names, *, coordinate_mask=None, device=None):
    """``[N_atom]`` 1 where the flat axis carries a supervisable backbone coordinate.

    Two filters: the atom is one of N, CA, C, O, and the reference resolved it.

    **Under the monomer configuration this is currently mostly the second.**
    ``MONOMER_DATASET`` sets ``backbone_only_binder=True`` with the whole chain
    as the design region, and the featurizer then emits a backbone-only flat
    axis -- measured on T1031 (95 tokens, 380 atoms) and on an AFDB entry (34
    tokens, 136 atoms): four distinct atom names, zero non-backbone atoms, every
    atom resolved. So an unmasked ``L_BB`` was *not* silently scoring side
    chains, and this is not a bug fix.

    It is supplied anyway because the axis is only backbone-only by
    configuration, not by construction. ``_scrub_design_sidechain_coords``
    replaces each design-region side-chain coordinate with its residue's CA
    rather than removing the atom, so any configuration that keeps those atoms
    -- a target chain alongside the binder, ``backbone_only_binder=False``, a
    partial design region, an entry with hetero groups -- puts CA-collapsed
    coordinates on the axis that ``label_dict["coordinate"]`` presents as
    targets. Supervising those would train the correction to put side chains on
    their CA, and nothing in the loss would say so. An explicit mask makes the
    supervised set a stated choice at the cost of one boolean per atom.
    """
    names = [str(name) for name in atom_names]
    keep = torch.tensor(
        [name in atom37.BACKBONE_ATOMS for name in names],
        dtype=torch.float32,
        device=device,
    )
    if coordinate_mask is not None:
        resolved = torch.as_tensor(coordinate_mask).reshape(-1).float()
        if resolved.numel() != keep.numel():
            raise ValueError(
                f"coordinate_mask covers {resolved.numel()} atoms but the "
                f"topology names {keep.numel()}"
            )
        keep = keep * resolved.to(keep.device)
    return keep


# --- the cache --------------------------------------------------------------


def _digest(payload):
    return hashlib.blake2b(
        json.dumps(payload, sort_keys=True, default=str).encode(), digest_size=16
    ).hexdigest()


@dataclass
class UpstreamCache:
    """Frozen upstream states, keyed by ``(structure, sigma, seed)``.

    ``identity`` records what the cache's contents are a function of. It is
    compared on load and on merge, so a cache built with one donor or one
    packing length cannot be reused under another.
    """

    identity: dict
    states: dict = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    @staticmethod
    def key(name, sigma, seed):
        return f"{name}|{float(sigma):.12g}|{int(seed)}"

    @property
    def fingerprint(self):
        return _digest(self.identity)

    def compatible(self, other):
        """Which identity fields disagree. Empty means the two may be merged."""
        keys = set(self.identity) | set(other)
        return [
            (k, self.identity.get(k), other.get(k))
            for k in sorted(keys)
            if self.identity.get(k) != other.get(k)
        ]

    def get(self, name, sigma, seed):
        state = self.states.get(self.key(name, sigma, seed))
        if state is None:
            self.misses += 1
        else:
            self.hits += 1
        return state

    def put(self, name, sigma, seed, state):
        self.states[self.key(name, sigma, seed)] = state
        return state

    def stats(self):
        total = self.hits + self.misses
        return dict(
            entries=len(self.states),
            hits=self.hits,
            misses=self.misses,
            hit_rate=(self.hits / total if total else 0.0),
            fingerprint=self.fingerprint,
        )

    def save(self, path):
        """Write the cache, on the CPU, with its identity alongside."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            identity=self.identity,
            fingerprint=self.fingerprint,
            states={k: v.to("cpu") for k, v in self.states.items()},
        )
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path, *, identity=None):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cache = cls(identity=payload["identity"], states=payload["states"])
        if identity is not None:
            mismatch = cache.compatible(identity)
            if mismatch:
                raise ValueError(
                    "this upstream cache was built from different frozen "
                    "components than the ones loaded now, so reusing it would "
                    "make the comparison uncontrolled: "
                    + "; ".join(f"{k}: cached={a!r} now={b!r}" for k, a, b in mismatch)
                )
        return cache


def cache_identity(*, frozen, pack_steps, bs_policy, sigma_schedule, seed_base):
    """What a cache's contents are a function of, and nothing else.

    Job id, node and wall time are deliberately absent: they change how a state
    was scheduled, not what it is.
    """
    return dict(
        frozen=frozen,
        pack_steps=pack_steps,
        bs_policy=bs_policy,
        sigma_schedule=sigma_schedule,
        seed_base=int(seed_base),
        version="sb-pilot-upstream-v1",
    )
