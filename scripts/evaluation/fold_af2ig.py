#!/usr/bin/env python3
"""Fold benchmark designs with AF2 initial-guess and emit the metrics CSV.

This is the missing middle of the ConditionalBinderDesignBenchmark:
`eval_conditional_binder_benchmark.py` writes `designs/*.pdb` + `designs.csv`,
`score_af2ig_designability.py` reads `af2ig_metrics.csv`, and nothing joined
them. This does.

    python scripts/evaluation/fold_af2ig.py \
        --run-dir /path/to/runs/cbdb/12345 \
        --data-dir /path/to/af2_params \
        --variants co_design pmpnn

It runs in the **AF2-IG environment**, not the training one — JAX + ColabDesign,
no torch, no Protenix. `scripts/utilities/bootstrap_af2ig.sh` builds it. The
only thing it shares with training is this repo's `pxdesign_train.benchmarks`
package, which is pure-python on the paths used here.

### What is computed, and from where

Per design, two AlphaFold passes:

1. **bound** — `mk_af_model(protocol="binder", initial_guess=True)` over the
   design PDB, sequence pinned to the design's own. This is AF2 initial guess:
   the binder's template is removed (`rm_binder=True`, ColabDesign's default)
   and the design's coordinates enter only as the recycling "previous"
   positions. It yields ipAE, ipTM, binder pLDDT, and the designed-vs-predicted
   RMSD.
2. **unbound** — the same sequence folded alone, `protocol="hallucination"`,
   no templates, no initial guess. Only this gives criterion (d): A-CODE asks
   for the *bound/unbound* RMSD, which is a property of the sequence off its
   partner and cannot be read off the complex prediction. It is cached by
   sequence, so it costs one fold per distinct binder, not one per design.

ColabDesign reports `i_pae` divided by 31 and `plddt` as a loss; both are
converted here (see `_metrics_from_log`), and getting either wrong yields
numbers that look reasonable and are not the paper's.

### Two chosen values worth knowing

**Chain-break offset.** The manifest records 200 and A-CODE says only "a large
offset". ColabDesign's binder protocol uses 50 between chains and this script
does not override it: AF2's monomer relative-position encoding clips at +/-32,
so every value above 32 is the same input to the model. The manifest's 200 and
ColabDesign's 50 are indistinguishable here, which is why this is a note rather
than a flag.

**PMPNN redesign.** `--variants pmpnn` redesigns the binder on the same backbone
with ColabDesign's bundled ProteinMPNN (v_48_020, the original weights) at
temperature 1e-4, the setting PXDesign uses, one sequence per structure, with the
target chain held fixed. No amino acid is excluded by default: A-CODE says only
"the PMPNN-redesigned single sequence", and BindCraft's habit of banning
cysteine is a different protocol. `--mpnn-rm-aa C` restores it.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# Imported as a top-level module, not as `pxdesign_train.benchmarks.af2ig`.
# `pxdesign_train/__init__.py` imports Protenix, which is a torch stack this
# environment deliberately does not have -- the split exists because PXDesign's
# AF2 path pins Protenix v0.5.0+pxd against this repo's v2.0.0, and the two
# cannot share an interpreter. `af2ig.py` itself needs only numpy and biotite,
# so putting its directory on the path gets it without the package __init__.
BENCHMARKS_DIR = REPO_ROOT / "pxdesign_train" / "benchmarks"
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

from af2ig import (  # noqa: E402
    Design,
    MetricRow,
    MetricsSink,
    bound_unbound_rmsd,
    load_prep_sidecars,
    read_designs_csv,
    resolve_binder_chain,
)

logger = logging.getLogger("af2ig")

# atom37 index of CA, the one atom both predictions always have.
CA = 1
# ColabDesign stores PAE divided by its 31 A cap; undo that to get Angstroms.
PAE_SCALE = 31.0


# ------------------------------------------------------------------ discovery


def _collect_designs(
    designs_csv: Path,
    designs_dir: Path,
    inputs_dir: Optional[Path],
    targets: Optional[list[str]],
    limit: Optional[int],
) -> list[Design]:
    rows = read_designs_csv(designs_csv)
    sidecars = load_prep_sidecars(inputs_dir) if inputs_dir and inputs_dir.is_dir() else {}
    if inputs_dir and not sidecars:
        logger.warning(
            "no .prep.json under %s; binder chains will be inferred from residue "
            "counts and an ambiguous design will abort the run",
            inputs_dir,
        )

    # (target, length) -> chain split. Every design in a cell was written from
    # one prepared input by one code path, so the split is a property of the
    # cell, not of the sample -- and resolving it per design means re-parsing a
    # 200-450 residue PDB a few thousand times just to count chains. The
    # per-design guard that actually protects correctness is in
    # `Af2Ig.prep_complex`, which checks AlphaFold's own binder length against
    # designs.csv for every single fold.
    chain_split: dict[str, tuple[str, tuple[str, ...]]] = {}

    designs: list[Design] = []
    for row in rows:
        if targets and row["target"] not in targets:
            continue
        pdb_path = Path(row.get("design_pdb") or "")
        if not pdb_path.is_file():
            # The generation run records absolute paths; a run directory that
            # has since moved is the common case, so fall back to the local
            # designs/ dir rather than failing.
            pdb_path = designs_dir / f"{row['sample_id']}.pdb"
        if not pdb_path.is_file():
            raise SystemExit(
                f"{row['sample_id']}: no PDB at {row.get('design_pdb')!r} or "
                f"{designs_dir / (row['sample_id'] + '.pdb')}"
            )
        binder_length = int(float(row["binder_length"]))
        task_id = f"{row['target']}_L{binder_length}"
        if task_id not in chain_split:
            chain_split[task_id] = resolve_binder_chain(
                pdb_path, binder_length, sidecars.get(task_id)
            )
        binder_chain, target_chains = chain_split[task_id]
        designs.append(
            Design(
                sample_id=row["sample_id"],
                target=row["target"],
                binder_length=binder_length,
                sequence=row["sequence"],
                pdb_path=pdb_path,
                binder_chain=binder_chain,
                target_chains=target_chains,
            )
        )
        if limit and len(designs) >= limit:
            break
    if not designs:
        raise SystemExit(f"{designs_csv}: no designs selected")
    # Group by (target, length): every change of total token count recompiles
    # the AlphaFold graph, which costs a minute or two each time.
    designs.sort(key=lambda d: (d.target, d.binder_length, d.sample_id))
    return designs


# --------------------------------------------------------------------- models


class Af2Ig:
    """Thin wrapper over the two ColabDesign models this filter needs."""

    def __init__(
        self,
        data_dir: Path,
        num_recycles: int,
        complex_model: str,
        monomer_model: str,
        mpnn_weights: str = "original",
    ) -> None:
        from colabdesign.af import mk_af_model

        self.num_recycles = num_recycles
        self.complex_model_name = complex_model
        self.monomer_model_name = monomer_model
        # initial_guess=True is the whole point: the design's coordinates seed
        # the recycling "previous" positions while its template stays removed.
        self.complex = mk_af_model(
            protocol="binder",
            initial_guess=True,
            use_multimer=False,
            model_names=[complex_model],
            data_dir=str(data_dir),
        )
        self.monomer = mk_af_model(
            protocol="hallucination",
            use_templates=False,
            use_multimer=False,
            model_names=[monomer_model],
            data_dir=str(data_dir),
        )
        self._mpnn = None
        self._mpnn_weights = mpnn_weights
        self._unbound_cache: dict[str, np.ndarray] = {}

    # -- bound ---------------------------------------------------------------

    def prep_complex(self, design: Design) -> None:
        self.complex.prep_inputs(
            str(design.pdb_path),
            target_chain=",".join(design.target_chains),
            binder_chain=design.binder_chain,
        )
        # ColabDesign's prep_pdb drops residues with no CA (`ignore_missing`).
        # If it ever dropped one from the binder, `positions[-binder_length:]`
        # below would slice into the target and the bound/unbound RMSD would be
        # computed on the wrong atoms -- while still returning a number.
        if self.complex._binder_len != design.binder_length:
            raise SystemExit(
                f"{design.sample_id}: AlphaFold prep kept "
                f"{self.complex._binder_len} binder residues, designs.csv says "
                f"{design.binder_length}. Chain {design.binder_chain!r} in "
                f"{design.pdb_path} has residues without a CA."
            )

    def predict_complex(self, sequence: str) -> dict[str, Any]:
        self.complex.predict(
            seq=sequence, num_recycles=self.num_recycles, verbose=False
        )
        return self.complex.aux

    # -- unbound -------------------------------------------------------------

    def unbound_ca(self, sequence: str) -> np.ndarray:
        cached = self._unbound_cache.get(sequence)
        if cached is not None:
            return cached
        self.monomer.prep_inputs(length=len(sequence))
        self.monomer.predict(
            seq=sequence, num_recycles=self.num_recycles, verbose=False
        )
        coords = np.asarray(self.monomer.aux["atom_positions"])[:, CA, :].copy()
        self._unbound_cache[sequence] = coords
        return coords

    # -- pmpnn ---------------------------------------------------------------

    def redesign(self, temperature: float, rm_aa: Optional[str], seed: int) -> str:
        """One ProteinMPNN sequence for the binder of the currently prepped complex.

        `get_af_inputs` must be called after `prep_complex`, because it copies
        the backbone, the chain indices and — for the binder protocol — the
        fixed-position mask that pins the whole target. That mask is why this
        redesigns the binder only.
        """
        from colabdesign.mpnn import mk_mpnn_model

        if self._mpnn is None:
            self._mpnn = mk_mpnn_model(weights=self._mpnn_weights)
        self._mpnn.set_seed(seed)
        self._mpnn.get_af_inputs(self.complex)
        if rm_aa:
            # ColabDesign only accepts rm_aa through `prep_inputs`, which this
            # path bypasses (the af model already holds the backbone and the
            # fixed-target mask). Replicate its bias by hand — and only over the
            # binder tail, because `get_af_inputs` has already written +1e7 into
            # the target rows to pin them, and subtracting from those would
            # start eroding the fixed target sequence.
            from colabdesign.af.alphafold.common import residue_constants

            bias = self._mpnn._inputs["bias"]
            for aa in rm_aa.replace(",", ""):
                bias[-self.complex._len:, residue_constants.restype_order[aa]] -= 1e6
        out = self._mpnn.sample(num=1, batch=1, temperature=temperature)
        # `seq` is the full complex with "/" chain separators; the binder is the
        # tail, since prep_inputs writes the target first.
        return out["seq"][0].replace("/", "")[-self.complex._len:]


def _metrics_from_log(log: dict[str, Any]) -> dict[str, float]:
    """Convert ColabDesign's log into the paper's units.

    Two conversions, both easy to get silently wrong:

    * `i_pae` is stored divided by AF2's 31 A PAE cap, so a raw value of 0.3
      reads as a spectacular 0.3 A against a 10.85 A threshold instead of the
      9.3 A it is — every design would pass criterion (a).
    * `plddt` in the log is already flipped back from the `1 - plddt` loss by
      ColabDesign itself (af/design.py), and is the mean over the **binder**
      only, which is what criterion (c) asks for. It is on the 0-1 scale;
      `AF2IGFilter.normalise_plddt` accepts either, but recording 0-1 keeps the
      CSV unambiguous.
    """
    return {
        "ipae": float(log["i_pae"]) * PAE_SCALE,
        "iptm": float(log["i_ptm"]),
        "plddt": float(log["plddt"]),
        "ptm": float(log.get("ptm", float("nan"))),
        "binder_designed_rmsd": float(log["rmsd"]) if "rmsd" in log else None,
    }


# ----------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", default=None,
                        help="generation run directory (designs.csv, designs/, inputs/)")
    parser.add_argument("--designs-csv", default=None)
    parser.add_argument("--designs-dir", default=None)
    parser.add_argument("--inputs-dir", default=None,
                        help="where the .prep.json sidecars live; names the binder chain")
    parser.add_argument("--metrics-csv", default=None,
                        help="default: <run-dir>/af2ig_metrics.csv")
    parser.add_argument("--data-dir", default=os.environ.get("AF2_PARAMS_DIR", ""),
                        help="directory holding params/params_<model>.npz")
    parser.add_argument("--variants", nargs="+", default=["co_design"],
                        choices=["co_design", "pmpnn"])
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="take the first N designs from designs.csv. Note this "
                             "applies before the resume check, so on a resumed run "
                             "it caps which designs are considered, not how many "
                             "are folded")
    parser.add_argument("--num-recycles", type=int, default=3,
                        help="AF2 recycles; 3 is the AF2-IG / dl_binder_design default")
    parser.add_argument("--complex-model", default="model_1_ptm",
                        help="templates are required for the binder protocol, so a *_ptm "
                             "model that has a template stack")
    parser.add_argument("--monomer-model", default="model_3_ptm",
                        help="no template stack needed for the unbound fold")
    parser.add_argument("--mpnn-temperature", type=float, default=0.0001)
    parser.add_argument("--mpnn-rm-aa", default=None,
                        help="amino acids ProteinMPNN may not sample (e.g. 'C'); "
                             "none by default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-failures", action="store_true",
                        help="log a design that fails to fold and carry on, instead "
                             "of aborting. No row is written for it, so a later "
                             "resume retries it -- which is the point: writing a "
                             "blank row would mark it done and the scorer would "
                             "count a crashed job as a non-designable sample")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve designs and binder chains, then exit without "
                             "touching JAX or the parameters")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None
    designs_csv = Path(args.designs_csv) if args.designs_csv else (
        run_dir / "designs.csv" if run_dir else None
    )
    if designs_csv is None:
        raise SystemExit("pass --run-dir or --designs-csv")
    designs_csv = designs_csv.expanduser().resolve()
    designs_dir = Path(args.designs_dir).expanduser().resolve() if args.designs_dir else (
        designs_csv.parent / "designs"
    )
    inputs_dir = Path(args.inputs_dir).expanduser().resolve() if args.inputs_dir else (
        designs_csv.parent / "inputs"
    )
    metrics_csv = Path(args.metrics_csv).expanduser().resolve() if args.metrics_csv else (
        designs_csv.parent / "af2ig_metrics.csv"
    )

    designs = _collect_designs(
        designs_csv, designs_dir, inputs_dir, args.targets, args.limit
    )
    logger.info(
        "%d designs over %d target(s); variants=%s",
        len(designs), len({d.target for d in designs}), args.variants,
    )
    if args.dry_run:
        for design in designs[:10]:
            logger.info(
                "%s target=%s L=%d binder_chain=%s target_chains=%s",
                design.sample_id, design.target, design.binder_length,
                design.binder_chain, ",".join(design.target_chains),
            )
        logger.info("dry run: resolved %d designs, wrote nothing", len(designs))
        return

    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else None
    if data_dir is None or not (data_dir / "params").is_dir():
        raise SystemExit(
            "--data-dir must point at the directory CONTAINING params/, i.e. one "
            f"with params/params_{args.complex_model}.npz in it (got {data_dir}). "
            "Run scripts/utilities/bootstrap_af2ig.sh to create one."
        )
    for name in (args.complex_model, args.monomer_model):
        npz = data_dir / "params" / f"params_{name}.npz"
        if not npz.is_file():
            raise SystemExit(f"missing AlphaFold parameters: {npz}")

    models = Af2Ig(
        data_dir=data_dir,
        num_recycles=args.num_recycles,
        complex_model=args.complex_model,
        monomer_model=args.monomer_model,
    )

    n_written = 0
    n_failed = 0
    with MetricsSink(metrics_csv, overwrite=args.overwrite) as sink:
        prepped: Optional[str] = None
        for index, design in enumerate(designs):
            pending = [v for v in args.variants if not sink.has(design.sample_id, v)]
            if not pending:
                continue
            if prepped != design.sample_id:
                models.prep_complex(design)
                prepped = design.sample_id

            for variant in pending:
                started = time.time()
                try:
                    row = _score_one(models, design, variant, args, index)
                except Exception:
                    if not args.skip_failures:
                        raise
                    n_failed += 1
                    logger.exception("%s/%s failed to fold; skipping",
                                     design.sample_id, variant)
                    continue
                sink.write(row)
                n_written += 1
                logger.info(
                    "%s/%s ipae=%.2f iptm=%.3f plddt=%.3f bu_rmsd=%.2f (%.1fs)",
                    design.sample_id, variant, row.ipae, row.iptm, row.plddt,
                    row.binder_bound_unbound_rmsd, time.time() - started,
                )

    logger.info("wrote %d new row(s) -> %s", n_written, metrics_csv)
    if n_failed:
        logger.warning(
            "%d (design, variant) pair(s) failed and were skipped; they have no "
            "row, so re-running this command retries them", n_failed
        )
    print(
        "\nnext:\n"
        f"  python scripts/evaluation/score_af2ig_designability.py "
        f"--metrics-csv {metrics_csv}"
    )


def _score_one(models, design, variant: str, args, index: int) -> MetricRow:
    """Fold one (design, variant) pair and turn the two AF2 passes into a row.

    Split out of the loop so a single failure can be caught and skipped without
    the `try` swallowing the sink write as well -- a half-written row is worse
    than no row, because the resume key would then treat it as done.
    """
    started = time.time()
    if variant == "pmpnn":
        sequence = models.redesign(
            temperature=args.mpnn_temperature,
            rm_aa=args.mpnn_rm_aa,
            seed=args.seed + index,
        )
    else:
        sequence = design.sequence
    if len(sequence) != design.binder_length:
        raise ValueError(
            f"{design.sample_id}/{variant}: sequence is {len(sequence)} aa, "
            f"binder is {design.binder_length}"
        )

    aux = models.predict_complex(sequence)
    metrics = _metrics_from_log(aux["log"])
    positions = np.asarray(aux["atom_positions"])
    bound_ca = positions[-design.binder_length:, CA, :]
    unbound = models.unbound_ca(sequence)
    bu_rmsd = bound_unbound_rmsd(bound_ca, unbound)
    plddt_all = np.asarray(aux["plddt"], dtype=float)

    return MetricRow(
        sample_id=design.sample_id,
        target=design.target,
        binder_length=design.binder_length,
        ipae=round(metrics["ipae"], 4),
        iptm=round(metrics["iptm"], 4),
        plddt=round(metrics["plddt"], 4),
        binder_bound_unbound_rmsd=round(bu_rmsd, 4),
        variant=variant,
        binder_designed_rmsd=(
            None if metrics["binder_designed_rmsd"] is None
            else round(metrics["binder_designed_rmsd"], 4)
        ),
        ptm=round(metrics["ptm"], 4),
        plddt_complex=round(float(plddt_all.mean()), 4),
        binder_chain=design.binder_chain,
        sequence=sequence,
        design_pdb=str(design.pdb_path),
        num_recycles=args.num_recycles,
        model_name=args.complex_model,
        seconds=round(time.time() - started, 2),
    )


if __name__ == "__main__":
    main()
