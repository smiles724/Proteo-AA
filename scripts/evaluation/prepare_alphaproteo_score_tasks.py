#!/usr/bin/env python3
"""Build one PXDesignBench task per model/target/sequence arm."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generation-root", required=True)
    p.add_argument("--score-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mpnn-sequences", type=int, default=1)
    # Generation emits every sequence arm from one trajectory, but scoring each
    # is a separate GPU job. Restricting the arms here keeps the emitted task
    # indices contiguous, which is what the queue manager's TASK_START/TASK_END
    # range can actually select; filtering downstream cannot.
    p.add_argument(
        "--arms",
        default="",
        help="comma-separated sequence arms to score; empty means every arm",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    generation_root = Path(args.generation_root).expanduser().resolve()
    score_root = Path(args.score_root).expanduser().resolve()
    keep_arms = {arm for arm in args.arms.split(",") if arm.strip()}
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    manifests = sorted(generation_root.glob("*/manifest.csv"))
    if not manifests:
        raise FileNotFoundError(f"no */manifest.csv under {generation_root}")
    seen_arms: set[str] = set()
    for manifest in manifests:
        with manifest.open() as handle:
            for row in csv.DictReader(handle):
                arm = row["sequence_arm"]
                seen_arms.add(arm)
                if keep_arms and arm not in keep_arms:
                    continue
                groups[(row["model_label"], row["target"], arm)].append(row)
    if not groups:
        raise ValueError(
            f"--arms {args.arms!r} matched nothing; manifests contain: "
            f"{sorted(seen_arms)}"
        )
    unknown = keep_arms - seen_arms
    if unknown:
        raise ValueError(
            f"--arms names arms absent from every manifest: {sorted(unknown)}; "
            f"manifests contain: {sorted(seen_arms)}"
        )

    rows = []
    for task_index, ((model, target, arm), items) in enumerate(sorted(groups.items())):
        paths = [Path(row["cif_path"]) for row in items]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{model}/{target}/{arm}: {len(missing)} CIF files missing; first={missing[0]}"
            )
        input_dirs = {str(path.parent.resolve()) for path in paths}
        if len(input_dirs) != 1:
            raise ValueError(f"one task spans multiple input dirs: {input_dirs}")
        n_backbones = len({row["sample_name"] for row in items})
        n_sequences = n_backbones * (args.mpnn_sequences if arm == "proteinmpnn" else 1)
        rows.append(
            {
                "task_index": task_index,
                "model_label": model,
                "target": target,
                "sequence_arm": arm,
                "input_dir": next(iter(input_dirs)),
                "output_dir": str((score_root / model / target / arm).resolve()),
                "use_gt_seq": "false" if arm == "proteinmpnn" else "true",
                "n_backbones": n_backbones,
                "expected_sequences": n_sequences,
            }
        )

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"tasks={len(rows)}")
    print(f"task_file={output}")


if __name__ == "__main__":
    main()
