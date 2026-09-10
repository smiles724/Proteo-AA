"""Regression checks for Stage IV launcher overrides and smoke supervision."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("override", ["default", "environment", "cli"])
def test_hai_launcher_argument_precedence(tmp_path, dry_run, override):
    # Execute both real shell layers, replacing only Python so this never trains.
    capture = tmp_path / "calls.jsonl"
    interpreter = tmp_path / "capture_python"
    interpreter.write_text(
        "#!" + sys.executable + "\n"
        "import json, os, sys\n"
        "with open(os.environ['CAPTURE_FILE'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    interpreter.chmod(0o755)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("BASH_FUNC_")}
    env.update(PROTEOAA_REPO=str(ROOT), PROTEOAA_DATA_ROOT=str(tmp_path / "data"),
               PROTEOAA_CODE_ROOT=str(tmp_path / "Code With Spaces"),
               PYTHON_BIN=str(interpreter), CAPTURE_FILE=str(capture),
               OUTPUT_DIR=str(tmp_path / "output"),
               EVAL_INTERVAL="2000", ITERS_TO_ACCUMULATE="8", NUM_WORKERS="4")
    options = ["--dry-run"] if dry_run else []
    expected = (2000, 8, 4)
    if override != "default":
        env.update(EVAL_INTERVAL="3", ITERS_TO_ACCUMULATE="2", NUM_WORKERS="1")
        expected = (3, 2, 1)
    if override == "cli":
        options += ["--eval-interval", "2", "--iters-to-accumulate", "1", "--num-workers", "0"]
        expected = (2, 1, 0)
    subprocess.run(["bash", str(ROOT / "scripts/training/slurm_stage4_fampnn_binder_hai.sh"),
                    *options], env=env, check=True, capture_output=True, text=True)
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    args = next(call for call in calls if call[0] == "scripts/training/train_protenix_monomer.py")
    def last_value(option):
        return [args[i + 1] for i, value in enumerate(args) if value == option][-1]
    assert tuple(int(last_value(key)) for key in
                 ("--eval-interval", "--iters-to-accumulate", "--num-workers")) == expected
    assert ("--dry-run" in args) == dry_run
    if dry_run:
        assert last_value("--device") == "cpu"
    assert last_value("--protenix-code-dir") == str(tmp_path / "Code With Spaces/Protenix")


def _smoke_selector():
    spec = importlib.util.spec_from_file_location(
        "stage4_smoke", ROOT / "scripts/utilities/smoke_stage4_fampnn.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.select_supervised_batch


def _batch(observed, design):
    return {"input_feature_dict": {"sc_atom_mask": torch.tensor(observed),
                                    "design_token_mask": torch.tensor(design)}}


def test_smoke_skips_empty_and_receptor_only_supervision():
    empty = _batch([[0, 0], [0, 0]], [1, 0])
    receptor_only = _batch([[0, 0], [1, 1]], [1, 0])
    binder = _batch([[1, 0], [1, 1]], [1, 0])
    batch, counts = _smoke_selector()([empty, receptor_only, binder])
    assert batch is binder
    assert counts == {"observed_sc_atoms": 1, "observed_sc_residues": 1,
                      "candidates_checked": 3}


def test_smoke_rejects_no_observed_binder_targets():
    with pytest.raises(RuntimeError, match="none of the 1 candidate items"):
        _smoke_selector()([_batch([[0, 0], [1, 1]], [1, 0])])
