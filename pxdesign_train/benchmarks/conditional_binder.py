"""ConditionalBinderDesignBenchmark — the A-CODE binder-design test set.

Reproduces the test set of Section 4.2 / Appendix C.2 of *A-CODE: Fully Atomic
Protein Co-Design with Unified Multimodal Diffusion* (arXiv:2605.03360) so that
it can be run against a Proteo-AA Stage III (co-evolution) checkpoint.

What the paper specifies, and where each piece comes from:

  * **Targets** — "a test set comprising 10 protein targets with diverse
    structural properties, as proposed in Zambaldi et al." A-CODE does not print
    the crops or hotspots; it defers to PXDesign, which defers to AlphaProteo.
    The manifest (`targets/conditional_binder_design_v1.json`) carries the
    per-target PDB / crop / hotspot values with a `source` field for each, plus a
    `status` of `verified` or `pending_source`. All ten are now `verified`
    against AlphaProteo Table S1 and the local mmCIF mirror. The
    `pending_source` path stays live for the case where a target's definition is
    not sourced: a benchmark that silently ran on invented hotspots would be
    worse than one that refuses, so `tasks()` skips such an entry unless
    `include_pending=True`, which fails loudly instead.
  * **Conditioning** — target structure + sequence as the condition, hotspots as
    optional extra input. Hotspots here are the *fixed, published* residues, not
    the randomly sampled ones training uses, so the featurizer's stochastic
    hotspot channel must be forced off and overwritten; see
    `target_prep.apply_hotspots`.
  * **Sampling** — "we sample 328-728 binders with lengths ranging from 80 to
    130". The per-target length lists are AlphaProteo's and unpublished here, so
    the manifest ships a uniform 6-length x 64-sample grid (384, inside the
    paper's envelope) and honours per-target overrides.
  * **Metric** — Designability, the percentage of samples passing the AF2-IG
    filter, with "the success counts summed across different binder lengths for
    the same target". That summing is why `designability()` pools over the whole
    length grid rather than averaging per-length rates: a length grid with
    unequal sample counts would give a different answer.

The two reported variants ("the model-generated co-designed sequence and the
PMPNN-redesigned single sequence") are modelled as `SequenceVariant`, because on
a Stage III co-evolution checkpoint the co-designed sequence is the interesting
one and the PMPNN arm needs an external ProteinMPNN pass over the same backbone.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

MANIFEST_DIR = Path(__file__).resolve().parent / "targets"
DEFAULT_MANIFEST = MANIFEST_DIR / "conditional_binder_design_v1.json"

# Paper Table 4 column order, which is also the order the manifest lists and the
# order the summary table prints, so a run can be diffed against Table 4 by eye.
TABLE4_ORDER = (
    "BHRF1", "H1", "IL17A", "IL7RA", "IR",
    "PDL1", "SC2RBD", "TNFa", "TrkA", "VEGFA",
)


class SequenceVariant(str, Enum):
    """The two arms A-CODE reports separately in Table 4."""

    CO_DESIGN = "co_design"      # the sequence the model itself emits
    PMPNN = "pmpnn"              # one ProteinMPNN redesign of the same backbone


class TargetStatus(str, Enum):
    VERIFIED = "verified"
    PENDING_SOURCE = "pending_source"


@dataclass(frozen=True)
class AF2IGFilter:
    """The four-way conjunction of Appendix C.2.

    Thresholds are quoted verbatim from the paper and match the `AF2-IG-easy`
    row of PXDesign technical report Table 2 (BindCraft's thresholds). Note the
    paper writes pLDDT "greater than 80%": `plddt_min` is on the 0-1 scale and
    `is_designable` accepts either scale, because AF2 wrappers disagree about it.
    """

    ipae_max: float = 10.85
    iptm_min: float = 0.5
    plddt_min: float = 0.8
    binder_bound_unbound_rmsd_max: float = 3.5
    # CHOSEN VALUE: the paper says only "a large offset"; 200 is the
    # AF2-multimer / AF2-IG convention. Carried here so the number that shaped
    # the numbers is recorded next to them rather than buried in a shell script.
    chain_break_offset: int = 200

    REQUIRED_METRICS = ("ipae", "iptm", "plddt", "binder_bound_unbound_rmsd")

    @classmethod
    def from_manifest(cls, payload: dict[str, Any]) -> "AF2IGFilter":
        criteria = payload["criteria"]
        return cls(
            ipae_max=float(criteria["ipae_max"]),
            iptm_min=float(criteria["iptm_min"]),
            plddt_min=float(criteria["plddt_min"]),
            binder_bound_unbound_rmsd_max=float(
                criteria["binder_bound_unbound_rmsd_max"]
            ),
            chain_break_offset=int(payload.get("chain_break_offset", 200)),
        )

    def normalise_plddt(self, plddt: float) -> float:
        """Accept pLDDT on either the 0-1 or the 0-100 scale.

        AF2-IG wrappers report both. Guessing wrong silently turns criterion (c)
        into a no-op (every 0-100 value clears a 0.8 bar), which would inflate
        designability, so the conversion is explicit and one-way: only values
        that cannot be a 0-1 fraction are rescaled.
        """
        value = float(plddt)
        return value / 100.0 if value > 1.0 else value

    def is_designable(self, metrics: dict[str, Any]) -> bool:
        missing = [k for k in self.REQUIRED_METRICS if metrics.get(k) is None]
        if missing:
            raise KeyError(
                "AF2-IG filter needs "
                f"{list(self.REQUIRED_METRICS)}; missing {missing}"
            )
        return (
            float(metrics["ipae"]) < self.ipae_max
            and float(metrics["iptm"]) > self.iptm_min
            and self.normalise_plddt(metrics["plddt"]) > self.plddt_min
            and float(metrics["binder_bound_unbound_rmsd"])
            < self.binder_bound_unbound_rmsd_max
        )

    def describe(self) -> str:
        return (
            f"ipAE < {self.ipae_max}, ipTM > {self.iptm_min}, "
            f"pLDDT > {self.plddt_min}, binder bound/unbound RMSD < "
            f"{self.binder_bound_unbound_rmsd_max} A"
        )


@dataclass(frozen=True)
class BinderTarget:
    """One row of the test set, before the length grid is expanded."""

    name: str
    full_name: str
    status: TargetStatus
    source: str
    pdb_id: Optional[str]
    # chain_id -> list of inclusive [start, end] author-numbering ranges.
    chains: Optional[dict[str, list[tuple[int, int]]]]
    # (chain_id, author residue number) pairs.
    hotspots: Optional[tuple[tuple[str, int], ...]]
    numbering: str = "author"
    lengths: Optional[tuple[int, ...]] = None
    samples_per_length: Optional[int] = None
    note: str = ""

    @property
    def is_runnable(self) -> bool:
        return (
            self.status is TargetStatus.VERIFIED
            and bool(self.pdb_id)
            and bool(self.chains)
        )

    def require_runnable(self) -> None:
        if self.is_runnable:
            return
        raise ValueError(
            f"target {self.name!r} is {self.status.value}: no PDB/crop/hotspot "
            f"definition is recorded. {self.note} "
            "Fill it in the manifest (see `target_definition_sources`) before "
            "running it; the benchmark refuses to invent one."
        )

    def crop_ranges(self) -> list[tuple[str, int, int]]:
        self.require_runnable()
        out: list[tuple[str, int, int]] = []
        for chain_id, ranges in (self.chains or {}).items():
            for start, end in ranges:
                out.append((chain_id, int(start), int(end)))
        return out


@dataclass(frozen=True)
class BinderDesignTask:
    """One (target, binder length) cell of the benchmark grid."""

    target: BinderTarget
    binder_length: int
    n_samples: int

    @property
    def task_id(self) -> str:
        return f"{self.target.name}_L{self.binder_length}"

    def sample_ids(self) -> Iterator[str]:
        for i in range(self.n_samples):
            yield f"{self.task_id}_s{i:04d}"


@dataclass
class ConditionalBinderDesignBenchmark:
    """The A-CODE conditional binder-design test set.

    Usage::

        bench = ConditionalBinderDesignBenchmark.load()
        for task in bench.tasks():
            ...                      # generate task.n_samples binders
        bench.designability(rows)    # apply the AF2-IG filter, per target
    """

    targets: tuple[BinderTarget, ...]
    af2ig: AF2IGFilter
    lengths: tuple[int, ...]
    samples_per_length: int
    diffusion_steps: int
    version: str
    manifest_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)

    # ----- construction -----

    @classmethod
    def load(cls, manifest_path: Path | str = DEFAULT_MANIFEST) -> "ConditionalBinderDesignBenchmark":
        path = Path(manifest_path).expanduser().resolve()
        payload = json.loads(path.read_text())
        sampling = payload["sampling"]
        targets = tuple(cls._parse_target(entry) for entry in payload["targets"])

        names = [t.name for t in targets]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate target names in {path}: {names}")
        # The paper's set is exactly ten targets; a manifest that has drifted
        # from that is a different benchmark and should say so out loud.
        if len(targets) != 10:
            raise ValueError(
                f"{path} defines {len(targets)} targets; the A-CODE test set has 10"
            )

        return cls(
            targets=targets,
            af2ig=AF2IGFilter.from_manifest(payload["filter"]),
            lengths=tuple(int(x) for x in sampling["lengths"]),
            samples_per_length=int(sampling["samples_per_length"]),
            diffusion_steps=int(sampling["diffusion_steps"]),
            version=str(payload["version"]),
            manifest_path=path,
            metadata={
                "paper": payload.get("paper", {}),
                "target_definition_sources": payload.get(
                    "target_definition_sources", []
                ),
                "filter": payload.get("filter", {}),
                "sampling": sampling,
            },
        )

    @staticmethod
    def _parse_target(entry: dict[str, Any]) -> BinderTarget:
        chains = entry.get("chains")
        parsed_chains = (
            {
                str(chain): [(int(lo), int(hi)) for lo, hi in ranges]
                for chain, ranges in chains.items()
            }
            if chains
            else None
        )
        hotspots = entry.get("hotspots")
        parsed_hotspots = (
            tuple((str(chain), int(resid)) for chain, resid in hotspots)
            if hotspots
            else None
        )
        if parsed_chains and parsed_hotspots:
            # A hotspot outside every crop range can never reach the model: it
            # would be dropped by the crop and the run would quietly condition on
            # fewer hotspots than the paper. Catch it at load time.
            for chain, resid in parsed_hotspots:
                ranges = parsed_chains.get(chain)
                if ranges is None or not any(lo <= resid <= hi for lo, hi in ranges):
                    raise ValueError(
                        f"{entry['name']}: hotspot {chain}{resid} is outside the "
                        f"crop {parsed_chains}"
                    )
        lengths = entry.get("lengths")
        return BinderTarget(
            name=str(entry["name"]),
            full_name=str(entry.get("full_name", entry["name"])),
            status=TargetStatus(entry.get("status", "verified")),
            source=str(entry.get("source", "")),
            pdb_id=(str(entry["pdb_id"]).lower() if entry.get("pdb_id") else None),
            chains=parsed_chains,
            hotspots=parsed_hotspots,
            numbering=str(entry.get("numbering", "author")),
            lengths=tuple(int(x) for x in lengths) if lengths else None,
            samples_per_length=(
                int(entry["samples_per_length"])
                if entry.get("samples_per_length")
                else None
            ),
            note=str(entry.get("note", "")),
        )

    # ----- the test set -----

    def target(self, name: str) -> BinderTarget:
        for candidate in self.targets:
            if candidate.name == name:
                return candidate
        raise KeyError(f"unknown target {name!r}; have {[t.name for t in self.targets]}")

    def runnable_targets(self) -> tuple[BinderTarget, ...]:
        return tuple(t for t in self.targets if t.is_runnable)

    def pending_targets(self) -> tuple[BinderTarget, ...]:
        return tuple(t for t in self.targets if not t.is_runnable)

    def tasks(
        self,
        only: Optional[Iterable[str]] = None,
        include_pending: bool = False,
        lengths: Optional[Iterable[int]] = None,
        samples_per_length: Optional[int] = None,
    ) -> list[BinderDesignTask]:
        """Expand the (target x length) grid into concrete tasks.

        Args:
            only: restrict to these target names (still in Table 4 order).
            include_pending: if True, `pending_source` targets raise instead of
                being skipped. Use it to assert a manifest is complete.
            lengths / samples_per_length: override the manifest grid for a quick
                run. Per-target manifest overrides win over the global default
                but lose to these explicit arguments, which are what the user
                just typed.
        """
        wanted = set(only) if only is not None else None
        if wanted is not None:
            unknown = wanted - {t.name for t in self.targets}
            if unknown:
                raise KeyError(f"unknown target(s): {sorted(unknown)}")

        out: list[BinderDesignTask] = []
        for target in self._ordered_targets():
            if wanted is not None and target.name not in wanted:
                continue
            if not target.is_runnable:
                if include_pending:
                    target.require_runnable()
                continue
            grid = (
                tuple(int(x) for x in lengths)
                if lengths is not None
                else (target.lengths or self.lengths)
            )
            per_length = (
                int(samples_per_length)
                if samples_per_length is not None
                else (target.samples_per_length or self.samples_per_length)
            )
            for length in grid:
                out.append(
                    BinderDesignTask(
                        target=target, binder_length=int(length), n_samples=per_length
                    )
                )
        return out

    def _ordered_targets(self) -> list[BinderTarget]:
        rank = {name: i for i, name in enumerate(TABLE4_ORDER)}
        return sorted(self.targets, key=lambda t: rank.get(t.name, len(rank)))

    # ----- the metric -----

    def designability(
        self, rows: Iterable[dict[str, Any]], variant: Optional[str] = None
    ) -> dict[str, dict[str, Any]]:
        """Per-target designability from AF2-IG metric rows.

        Each row needs `target`, `binder_length` and the four AF2-IG metrics;
        `variant` (a `SequenceVariant` value) is optional and filters the rows.

        Returns `{target_name: {designability, n_designable, n_samples,
        per_length}}`. Designability pools successes over the whole length grid
        ("Across different binder lengths for the same target, the success counts
        are summed"), so a target evaluated at unequal per-length sample counts
        is still scored the way the paper scores it.
        """
        agg: dict[str, dict[str, Any]] = {}
        for row in rows:
            if variant is not None and row.get("variant", variant) != variant:
                continue
            name = str(row["target"])
            length = int(row["binder_length"])
            bucket = agg.setdefault(
                name, {"n_samples": 0, "n_designable": 0, "per_length": {}}
            )
            per_length = bucket["per_length"].setdefault(
                length, {"n_samples": 0, "n_designable": 0}
            )
            ok = bool(self.af2ig.is_designable(row))
            bucket["n_samples"] += 1
            per_length["n_samples"] += 1
            if ok:
                bucket["n_designable"] += 1
                per_length["n_designable"] += 1

        for bucket in agg.values():
            bucket["designability"] = (
                100.0 * bucket["n_designable"] / bucket["n_samples"]
                if bucket["n_samples"]
                else float("nan")
            )
            for per_length in bucket["per_length"].values():
                per_length["designability"] = (
                    100.0 * per_length["n_designable"] / per_length["n_samples"]
                    if per_length["n_samples"]
                    else float("nan")
                )
        return agg

    def summary_table(self, scores: dict[str, dict[str, Any]]) -> str:
        """Table 4-shaped one-liner: target columns in the paper's order."""
        names = [n for n in TABLE4_ORDER if n in scores]
        header = " ".join(f"{n:>8s}" for n in names)
        values = " ".join(f"{scores[n]['designability']:8.2f}" for n in names)
        return f"{'target':>10s} {header}\n{'design.%':>10s} {values}"
