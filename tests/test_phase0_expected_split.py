"""Phase 0 must abort when the featurized split is not the screened one.

Selecting the wrong chain is not an error the featurizer reports -- it returns
a smaller design region and everything downstream succeeds. The only evidence
is the token count recorded when the entry was chosen, so the comparison
against it has to be a hard stop rather than a logged note, and it has to
survive someone editing the config.

Importing the check alone keeps this free of the model load the rest of the
script does.
"""
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from phase0_pack_hook_check import _assert_expected_split  # noqa: E402


class StubTopology:
    def __init__(self, num_tokens):
        self.num_tokens = num_tokens


class StubStructure:
    """Only what the check reads: a token count and a design mask."""

    def __init__(self, tokens, design_tokens):
        self.topology = StubTopology(tokens)
        mask = torch.zeros(tokens, dtype=torch.bool)
        mask[:design_tokens] = True
        self.design_mask = mask


def test_agreement_records_the_observed_split():
    entry = {"id": "7f7p", "featurizer_tokens": 192,
             "featurizer_design_tokens": 96}
    out = {}
    _assert_expected_split(entry, StubStructure(192, 96), out)
    assert out["featurizer"] == {"tokens": 192, "design_tokens": 96}


def test_a_shrunken_design_region_aborts():
    """The 7p0s failure: a valid, smaller, wrong design region."""
    entry = {"id": "7p0s", "featurizer_tokens": 348,
             "featurizer_design_tokens": 132}
    with pytest.raises(SystemExit) as excinfo:
        _assert_expected_split(entry, StubStructure(348, 26), {})
    message = str(excinfo.value)
    assert "design_tokens: config 132, featurizer 26" in message
    assert "screen_dev_complexes.py" in message


def test_a_changed_crop_aborts_on_the_total_too():
    entry = {"id": "7ppb", "featurizer_tokens": 466,
             "featurizer_design_tokens": 125}
    with pytest.raises(SystemExit, match="tokens: config 466, featurizer 512"):
        _assert_expected_split(entry, StubStructure(512, 125), {})


def test_both_disagreeing_are_both_reported():
    entry = {"id": "7f91", "featurizer_tokens": 280,
             "featurizer_design_tokens": 140}
    with pytest.raises(SystemExit) as excinfo:
        _assert_expected_split(entry, StubStructure(300, 150), {})
    message = str(excinfo.value)
    assert "tokens: config 280, featurizer 300" in message
    assert "design_tokens: config 140, featurizer 150" in message


def test_an_entry_without_expectations_still_records_them():
    """A newly screened entry has nothing to compare against yet; it passes,
    and the observed numbers go into the report so they can be recorded."""
    out = {}
    _assert_expected_split({"id": "7new"}, StubStructure(210, 100), out)
    assert out["featurizer"] == {"tokens": 210, "design_tokens": 100}
