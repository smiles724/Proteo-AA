import csv
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_length_schedule_is_paired_across_models_and_inclusive():
    mod = _module(
        "alpha_generate", "scripts/evaluation/generate_alphaproteo_designs.py"
    )
    first = mod.length_schedule(
        seed=42, target_index=3, count=200,
        length_min=80, length_max=130, fixed_length=None,
    )
    second = mod.length_schedule(
        seed=42, target_index=3, count=200,
        length_min=80, length_max=130, fixed_length=None,
    )
    assert first == second
    assert min(first) == 80
    assert max(first) == 130
    assert mod.length_schedule(
        seed=1, target_index=0, count=3,
        length_min=80, length_max=130, fixed_length=105,
    ) == [105, 105, 105]


def test_prepare_tasks_and_summary_count_missing_scores_as_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    generation = tmp_path / "generation"
    model = generation / "ours"
    cif_dir = model / "pdl1" / "target_sigma"
    cif_dir.mkdir(parents=True)
    manifest_rows = []
    for i in range(2):
        cif = cif_dir / f"pdl1_L105_seed{i}.cif"
        cif.write_text("data_test\n")
        manifest_rows.append(
            {
                "model_label": "ours",
                "target": "pdl1",
                "sequence_arm": "target_sigma",
                "sample_name": cif.stem,
                "cif_path": str(cif),
            }
        )
    with (model / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader(); writer.writerows(manifest_rows)

    task_file = tmp_path / "tasks.tsv"
    score_root = tmp_path / "scores"
    prep = _module(
        "alpha_prepare", "scripts/evaluation/prepare_alphaproteo_score_tasks.py"
    )
    monkeypatch.setattr(
        sys, "argv",
        ["prepare", "--generation-root", str(generation),
         "--score-root", str(score_root), "--output", str(task_file)],
    )
    prep.main()
    with task_file.open() as handle:
        task = next(csv.DictReader(handle, delimiter="\t"))
    assert task["use_gt_seq"] == "true"
    assert task["expected_sequences"] == "2"

    output_dir = Path(task["output_dir"])
    output_dir.mkdir(parents=True)
    with (output_dir / "sample_level_output.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["name", "af2_opt_success", "unscaled_i_pAE", "pLDDT",
                        "af2_binder_pred_design_rmsd"],
        )
        writer.writeheader()
        writer.writerow(
            {"name": "one", "af2_opt_success": 1, "unscaled_i_pAE": 5.0,
             "pLDDT": 0.95, "af2_binder_pred_design_rmsd": 1.0}
        )

    summary_dir = tmp_path / "summary"
    summary = _module(
        "alpha_summary", "scripts/evaluation/summarize_alphaproteo_designability.py"
    )
    monkeypatch.setattr(
        sys, "argv",
        ["summary", "--task-file", str(task_file), "--output-dir", str(summary_dir)],
    )
    summary.main()
    with (summary_dir / "designability_by_target.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert float(row["coverage"]) == pytest.approx(0.5)
    assert float(row["designability"]) == pytest.approx(0.5)
