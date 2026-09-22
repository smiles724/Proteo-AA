"""The folding half of the AF2-IG filter: everything that is not AlphaFold.

`score_af2ig_designability.py` consumes a CSV of per-design metrics and applies
`AF2IGFilter`. Nothing produced that CSV, which is what this module and
`scripts/evaluation/fold_af2ig.py` exist for. The split is deliberate:

  * **here** — design discovery, binder-chain resolution, the bound/unbound
    superposition, and the append-only metrics sink. All pure
    numpy + biotite, so it is testable on a laptop and pinned by
    `tests/test_af2ig_scoring.py`;
  * **in the driver** — the JAX/AlphaFold call, which needs a GPU, 4 GB of
    parameters and a second environment.

Keeping the arithmetic out of the GPU script matters because the arithmetic is
where the filter can silently become a different filter. An ipAE read off the
wrong normalisation, or a bound/unbound RMSD computed against the designed
backbone instead of the unbound prediction, both produce plausible-looking
numbers that are not the paper's.

### On the two RMSDs

A-CODE Appendix C.2 criterion (d) is *"binder bound/unbound RMSD"*: predict the
binder inside the complex, predict the same sequence alone, superimpose, and
measure. That is the number the filter uses and it is what
`bound_unbound_rmsd` computes.

It is **not** the RMSD most AF2-IG wrappers print. dl_binder_design and
ColabDesign's binder protocol report the *designed* RMSD — predicted binder vs
the backbone that was handed in, after aligning on the target — which measures
whether AF2 reproduces the design rather than whether the binder is
conformationally stable off its partner. Both are recorded (the second as
`binder_designed_rmsd`, an extra column the scorer ignores), because they answer
different questions and a run that passes one can fail the other. The designed
RMSD comes straight out of the folding backend, which already computes it; only
the bound/unbound one is defined here, because no backend reports it.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Columns `score_af2ig_designability.py` requires, in its order, followed by the
# ones this harness adds for traceability. The scorer ignores extras; the first
# seven are a contract and `tests/test_af2ig_scoring.py` pins them against it.
REQUIRED_METRIC_COLUMNS = (
    "sample_id",
    "target",
    "binder_length",
    "ipae",
    "iptm",
    "plddt",
    "binder_bound_unbound_rmsd",
)
EXTRA_METRIC_COLUMNS = (
    "variant",
    "binder_designed_rmsd",
    "ptm",
    "plddt_complex",
    "binder_chain",
    "sequence",
    "design_pdb",
    "num_recycles",
    "model_name",
    "seconds",
)
METRIC_COLUMNS = REQUIRED_METRIC_COLUMNS + EXTRA_METRIC_COLUMNS


# --------------------------------------------------------------- design inputs


@dataclass(frozen=True)
class Design:
    """One row of a generation run's `designs.csv`, resolved to files on disk."""

    sample_id: str
    target: str
    binder_length: int
    sequence: str
    pdb_path: Path
    binder_chain: str
    target_chains: tuple[str, ...]


def read_designs_csv(path: Path | str) -> list[dict[str, str]]:
    path = Path(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path}: no design rows")
    missing = [c for c in ("sample_id", "target", "binder_length", "sequence")
               if c not in rows[0]]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}")
    return rows


def load_prep_sidecars(inputs_dir: Path | str) -> dict[str, dict[str, Any]]:
    """task_id -> the `.prep.json` written next to each prepared input.

    This is the authoritative source for which chain is the binder: preparation
    chose it with `choose_binder_chain_id`, which is not always 'B' (it is the
    first letter the target does not already use, so a target occupying A and B
    pushes the binder to Z). Guessing here would silently score the wrong chain
    on exactly the multi-chain targets — IL17A, TNFa, VEGFA, H1 — where a
    mistake is least visible.
    """
    inputs_dir = Path(inputs_dir)
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(inputs_dir.glob("*.prep.json")):
        payload = json.loads(path.read_text())
        out[str(payload.get("task_id", path.stem.removesuffix(".prep")))] = payload
    return out


def chain_residue_counts(pdb_path: Path | str) -> dict[str, int]:
    """chain id -> number of distinct residues, CA atoms only."""
    from biotite.structure.io.pdb import PDBFile

    array = PDBFile.read(str(pdb_path)).get_structure(model=1)
    array = array[array.atom_name == "CA"]
    counts: dict[str, int] = {}
    for chain_id in array.chain_id:
        counts[str(chain_id)] = counts.get(str(chain_id), 0) + 1
    return counts


def resolve_binder_chain(
    pdb_path: Path | str,
    binder_length: int,
    prep: Optional[dict[str, Any]] = None,
) -> tuple[str, tuple[str, ...]]:
    """Return (binder chain, target chains) for a design PDB.

    Prefers the `.prep.json` sidecar and verifies it against the file; falls
    back to "the one chain with exactly `binder_length` residues" and refuses
    when that is ambiguous. Refusing is the point — an ambiguous answer here
    silently redefines the task, and a target chain that happens to be
    binder-length is not a rare coincidence at 80-130 residues.
    """
    counts = chain_residue_counts(pdb_path)
    if not counts:
        raise ValueError(f"{pdb_path}: no CA atoms")

    if prep is not None and prep.get("binder_chain_id"):
        binder = str(prep["binder_chain_id"])
        if binder not in counts:
            raise ValueError(
                f"{pdb_path}: sidecar names binder chain {binder!r} but the file "
                f"has chains {sorted(counts)}"
            )
        if counts[binder] != binder_length:
            raise ValueError(
                f"{pdb_path}: binder chain {binder!r} has {counts[binder]} residues, "
                f"designs.csv says {binder_length}"
            )
        return binder, tuple(sorted(c for c in counts if c != binder))

    candidates = sorted(c for c, n in counts.items() if n == binder_length)
    if len(candidates) != 1:
        raise ValueError(
            f"{pdb_path}: cannot tell which chain is the binder — chains "
            f"{ {c: counts[c] for c in sorted(counts)} } contain "
            f"{len(candidates)} of length {binder_length}. Point --inputs-dir at "
            "the run's inputs/ so the .prep.json sidecars can be used."
        )
    binder = candidates[0]
    return binder, tuple(sorted(c for c in counts if c != binder))


# ------------------------------------------------------------------ geometry


def kabsch_rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
    """Optimally superimposed RMSD between two equal-length point sets.

    Plain Kabsch, no weighting and no reflection: `np.linalg.svd` can return a
    rotation with determinant -1, which is a mirror image and would report a
    left-handed helix as a perfect match for a right-handed one. The sign fix
    below is what rules that out.
    """
    mobile = np.asarray(mobile, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if mobile.shape != target.shape:
        raise ValueError(f"shape mismatch: {mobile.shape} vs {target.shape}")
    if mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError(f"expected (n, 3) coordinates, got {mobile.shape}")
    if mobile.shape[0] < 3:
        raise ValueError(f"need at least 3 points to superimpose, got {mobile.shape[0]}")

    p = mobile - mobile.mean(axis=0)
    q = target - target.mean(axis=0)
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, d]) @ vt
    diff = p @ rotation - q
    return float(np.sqrt((diff ** 2).sum() / diff.shape[0]))


def bound_unbound_rmsd(bound_ca: np.ndarray, unbound_ca: np.ndarray) -> float:
    """A-CODE criterion (d): binder in the complex vs the binder alone.

    Both arguments are the *predictions* — the binder as AF2 places it inside
    the complex, and the same sequence folded on its own. Superposition is
    unconstrained (the two predictions share no frame), so this is a pure
    conformational-change measure, which is what "bound/unbound" means.
    """
    return kabsch_rmsd(bound_ca, unbound_ca)


# -------------------------------------------------------------- metrics sink


@dataclass
class MetricRow:
    sample_id: str
    target: str
    binder_length: int
    ipae: Optional[float]
    iptm: Optional[float]
    plddt: Optional[float]
    binder_bound_unbound_rmsd: Optional[float]
    variant: str = "co_design"
    binder_designed_rmsd: Optional[float] = None
    ptm: Optional[float] = None
    plddt_complex: Optional[float] = None
    binder_chain: str = ""
    sequence: str = ""
    design_pdb: str = ""
    num_recycles: Optional[int] = None
    model_name: str = ""
    seconds: Optional[float] = None

    def to_csv_row(self) -> dict[str, Any]:
        row = asdict(self)
        return {k: ("" if row[k] is None else row[k]) for k in METRIC_COLUMNS}


class MetricsSink:
    """Append-only `af2ig_metrics.csv` with resume, keyed by (sample_id, variant).

    Append-only and resumable for the same reason `designs.csv` is: a paper-scale
    run is thousands of AF2 calls over many hours and will be interrupted. The
    key includes the variant because the two arms score the same `sample_id`
    twice, once per sequence, and collapsing them would drop the co-design arm
    the moment the PMPNN arm ran.
    """

    def __init__(self, path: Path | str, overwrite: bool = False) -> None:
        self.path = Path(path)
        self.done: set[tuple[str, str]] = set()
        exists = self.path.is_file() and not overwrite
        if exists:
            with self.path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                header = list(reader.fieldnames or [])
                if header != list(METRIC_COLUMNS):
                    raise ValueError(
                        f"{self.path}: existing header does not match this harness.\n"
                        f"  found:    {header}\n  expected: {list(METRIC_COLUMNS)}\n"
                        "Refusing to append rows under a different schema; move it "
                        "aside or pass --overwrite."
                    )
                for row in reader:
                    self.done.add(
                        (row["sample_id"], row.get("variant") or "co_design")
                    )
        self._handle = self.path.open("a" if exists else "w", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=list(METRIC_COLUMNS))
        if not exists:
            self._writer.writeheader()
            self._handle.flush()

    def has(self, sample_id: str, variant: str) -> bool:
        return (sample_id, variant) in self.done

    def write(self, row: MetricRow) -> None:
        self._writer.writerow(row.to_csv_row())
        self._handle.flush()
        self.done.add((row.sample_id, row.variant))

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "MetricsSink":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
