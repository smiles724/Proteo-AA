"""Reading a cached backbone payload, and the two bugs that reading it hid.

Gated on the collection being present, because it cannot be regenerated (0.55 A
between identical invocations) and a synthetic fixture would test my idea of
the schema rather than the schema. Where a property can be checked without the
files -- the chain-letter arithmetic -- it is checked unconditionally.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

BACKBONES = Path("/scratch/m000137-pm06/Proteo-AA/pxf/runs/binder_bench/backbones")
needs_collection = pytest.mark.skipif(
    not (BACKBONES / "backbones.json").is_file(),
    reason="the shared backbone collection is not on this filesystem",
)

# H1, IL17A and VEGFA have two target chains; TNFa has three. Any of them
# exercises the multi-chain path; BHRF1 is the single-target-chain control.
MULTI = "H1"
SINGLE = "BHRF1"


def _first(target):
    manifest = json.loads((BACKBONES / "backbones.json").read_text())
    ids = sorted(r["design_id"] for r in manifest["designs"]
                 if r["target"] == target)
    from pxf.bench.backbone_inputs import load_payload

    return load_payload(BACKBONES / "designs" / f"{ids[0]}.pt")


# ---------------------------------------------------------------- chain ids


def test_chain_letter_is_positional_not_binary():
    """The bug: 'asym_id 0 -> A, anything else -> B'.

    On a target with two chains that merged the binder into the target's
    second chain. ProteinMPNN then returned only the target's residues
    (283 for a 383-row complex) -- which is how it was caught, but it would
    have been an arm silently designed against the wrong chain if the lengths
    had happened to line up.
    """
    from pxf.bench.backbone_inputs import chain_letter

    assert [chain_letter(i) for i in range(4)] == ["A", "B", "C", "D"]


def test_chain_letter_refuses_past_the_single_letter_space():
    from pxf.bench.backbone_inputs import chain_letter

    with pytest.raises(ValueError, match="single-letter"):
        chain_letter(26)


# ------------------------------------------------------------------ payload


@needs_collection
@pytest.mark.parametrize("target,n_target_chains", [(SINGLE, 1), (MULTI, 2)])
def test_binder_is_the_last_chain_not_always_B(target, n_target_chains):
    from pxf.bench.backbone_inputs import (binder_chain_of, chains_of,
                                           target_chains_of, to_design_inputs)

    inputs = to_design_inputs(_first(target), context="complex")
    assert len(target_chains_of(inputs)) == n_target_chains
    assert binder_chain_of(inputs) not in target_chains_of(inputs)
    assert len(chains_of(inputs)) == n_target_chains + 1


@needs_collection
def test_design_mask_sums_to_binder_length():
    from pxf.bench.backbone_inputs import to_design_inputs

    payload = _first(SINGLE)
    inputs = to_design_inputs(payload, context="complex")
    assert int(inputs.binder_mask.sum()) == payload["binder_length"]


@needs_collection
def test_disagreeing_markers_are_refused():
    """design_mask, res_name=='xpb' and conditional_label must coincide.

    Corrupting one must be an error rather than a silent preference for
    whichever the code happens to read first.
    """
    from pxf.bench.backbone_inputs import _checked_design_mask

    payload = _first(SINGLE)
    payload["topology"] = dict(payload["topology"])
    mask = np.asarray(payload["topology"]["design_mask"]).copy()
    mask[0] = ~mask[0]
    payload["topology"]["design_mask"] = torch.from_numpy(mask)
    with pytest.raises(ValueError, match="markers of the|design mask selects"):
        _checked_design_mask(payload)


@needs_collection
def test_sigma_is_the_actual_one_not_the_scheduled_one():
    """Churn is 2.0 on this collection; the scheduled value is half."""
    from pxf.bench.backbone_inputs import to_design_inputs

    payload = _first(SINGLE)
    inputs = to_design_inputs(payload, context="complex")
    assert inputs.sigma == pytest.approx(float(payload["actual_sigma"]))
    assert inputs.sigma == pytest.approx(0.87109375)


@needs_collection
def test_binder_rows_never_carry_a_side_chain():
    """PXDesign generates a backbone; there is no side chain to leak."""
    from pxf import atom37
    from pxf.bench.backbone_inputs import to_design_inputs

    for context in ("complex", "complex_sc"):
        inputs = to_design_inputs(_first(SINGLE), context=context)
        binder = inputs.binder_mask[0].bool()
        sc = inputs.atom_mask[0][binder][:, list(atom37.SIDECHAIN_SLOTS)]
        assert float(sc.sum()) == 0.0, context


@needs_collection
def test_binder_rows_have_no_identity_to_teacher_force():
    from pxf import atom37
    from pxf.bench.backbone_inputs import to_design_inputs

    inputs = to_design_inputs(_first(SINGLE), context="complex_sc")
    binder = inputs.binder_mask[0].bool()
    assert bool((inputs.aatype[0][binder] == atom37.UNKNOWN_AA_INDEX).all())
    # ... and none of them is held fixed, or design() would refuse.
    assert int(inputs.fixed_sequence_mask[0][binder].sum()) == 0


@needs_collection
def test_sidechain_context_is_a_subset_of_fixed():
    """`design()` enforces this; violating it is a crash, not a silent skew."""
    from pxf.bench.backbone_inputs import to_design_inputs

    for context in ("complex", "complex_sc"):
        inputs = to_design_inputs(_first(SINGLE), context=context)
        extra = (inputs.sidechain_context_mask - inputs.fixed_sequence_mask) > 0
        assert not bool(extra.any()), context


@needs_collection
def test_binder_only_drops_the_target_entirely():
    from pxf.bench.backbone_inputs import to_design_inputs

    payload = _first(SINGLE)
    inputs = to_design_inputs(payload, context="binder_only")
    assert inputs.length == payload["binder_length"]
    assert int(inputs.binder_mask.sum()) == inputs.length
    assert int(inputs.fixed_sequence_mask.sum()) == 0
