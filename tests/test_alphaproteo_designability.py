import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _atom_array(chains: list[tuple[str, str, str, int]]):
    """Create one CA atom per (chain_id, label_id, author_id, residue)."""
    struc = pytest.importorskip("biotite.structure")
    atoms = struc.AtomArray(len(chains))
    atoms.chain_id = np.array([row[0] for row in chains])
    atoms.res_id = np.array([row[3] for row in chains])
    atoms.res_name = np.full(len(chains), "ALA")
    atoms.atom_name = np.full(len(chains), "CA")
    atoms.element = np.full(len(chains), "C")
    atoms.coord = np.zeros((len(chains), 3), dtype=np.float32)
    atoms.set_annotation("label_asym_id", np.array([row[1] for row in chains]))
    atoms.set_annotation("auth_asym_id", np.array([row[2] for row in chains]))
    atoms.set_annotation(
        "auth_seq_id", np.array([str(row[3]) for row in chains])
    )
    atoms.set_annotation("mol_type", np.full(len(chains), "protein"))
    return atoms


def test_target_chain_mapping_drops_biological_assembly_copies():
    mod = _module(
        "alpha_single_mapping", "scripts/evaluation/design_binder_from_target.py"
    )
    atoms = _atom_array(
        [
            ("A", "A", "A", 5),
            ("A", "A", "A", 6),
            ("B", "B", "B", 501),
            ("B", "B", "B", 502),
            ("A.1", "A", "A", 5),
            ("A.1", "A", "A", 6),
            ("B.1", "B", "B", 501),
            ("B.1", "B", "B", 502),
        ]
    )
    target, hotspot = mod._select_and_normalize_target_chains(
        atoms,
        {
            "A": ([(5, 6)], []),
            "B": ([(501, 502)], [502]),
        },
    )
    assert target.array_length() == 4
    assert list(dict.fromkeys(target.chain_id)) == ["A", "B"]
    assert set(target.auth_asym_id) == {"A", "B"}
    assert set(target.label_asym_id) == {"A", "B"}
    assert hotspot.tolist() == [False, False, False, True]


def test_target_chain_mapping_uses_label_id_then_yaml_order():
    mod = _module(
        "alpha_single_author_mapping",
        "scripts/evaluation/design_binder_from_target.py",
    )
    atoms = _atom_array(
        [
            ("C", "C", "V", 14),
            ("D", "D", "W", 14),
            ("C.1", "C", "V", 14),
            ("D.1", "D", "W", 14),
        ]
    )
    target, _ = mod._select_and_normalize_target_chains(
        atoms,
        {"V": ([(14, 14)], []), "W": ([(14, 14)], [])},
    )
    assert target.chain_id.tolist() == ["A", "B"]


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


def test_summary_parses_scalar_and_list_scores():
    summary = _module("alpha_summary_forms", "scripts/evaluation/summarize_alphaproteo_designability.py")
    assert summary._float("[0.91]") == pytest.approx(.91)
    assert summary._float("[1.0, 3.0]") == 2.
    assert summary._float("bad") is None
    assert summary._float("[]") is None
    assert summary._float("[None]") is None
    assert summary._float("nan") is None
    assert summary._truth("[True]")
    assert not summary._truth("[False]")
    assert not summary.valid_score({"af2_opt_success": 1})
