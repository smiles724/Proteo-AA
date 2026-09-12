import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _eval_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/evaluation/eval_pinder_binder_backbone_inputs.py"
    )
    spec = importlib.util.spec_from_file_location("pinder_binder_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_per_class_summary_exposes_prediction_collapse():
    module = _eval_module()
    confusion = np.zeros((20, 20), dtype=np.int64)
    confusion[0, 0] = 5
    confusion[1, 0] = 3

    summary, rows = module._summarize_aa_confusion(confusion, "inference_style")

    assert summary["binder_aa_num_predicted_classes"] == 1
    assert summary["binder_aa_balanced_acc"] == pytest.approx(1.0 / 20.0)
    assert rows[0]["precision"] == pytest.approx(5.0 / 8.0)
    assert rows[0]["recall"] == pytest.approx(1.0)
    assert rows[1]["recall"] == pytest.approx(0.0)
    assert rows[0]["prediction_fraction"] == pytest.approx(1.0)
