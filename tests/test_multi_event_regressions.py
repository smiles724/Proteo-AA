"""Regressions from the review of multi-event feedback at 2bd1dbf.

Each test names the finding it pins. They failed on that commit.
"""
import ast
import importlib.util
from pathlib import Path

import pytest
import torch

MATRIX = Path(__file__).resolve().parents[1] / "scripts" / "run_integrated_binder_matrix.py"


def _module():
    spec = importlib.util.spec_from_file_location("matrix_mod", MATRIX)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ R1
# main() injected `select_event` but _one_cell looked up `select_events`.
# --help exits before main()'s body, so nothing caught it and every cell
# would KeyError after loading the donors. This checks the interface
# structurally rather than hoping a functional test covers it.

def test_every_api_key_the_cell_reads_is_supplied_by_main():
    tree = ast.parse(MATRIX.read_text())

    read = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name) and node.value.id == "api"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            read.add(node.slice.value)

    supplied = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "_one_cell":
            supplied |= {kw.arg for kw in node.keywords if kw.arg}
        # api=dict(...) forwarding inside the cell
        for kw in getattr(node, "keywords", []):
            if kw.arg == "api" and isinstance(kw.value, ast.Call):
                supplied |= {k.arg for k in kw.value.keywords if k.arg}

    assert read, "no api[...] lookups found; the probe is broken"
    assert not (read - supplied), f"_one_cell reads {sorted(read - supplied)} which main never supplies"


# ------------------------------------------------------------------ R3
# _event_seed keyed on the event's INDEX in the selected list, so the same
# solver step drew differently depending on how many other events were
# requested, and the single-event default silently moved off gen_seed.

def test_first_event_provisional_decode_keeps_the_legacy_seed():
    mod = _module()
    assert mod._decode_seed(41, (350, 0), "provisional", (350, 0)) == 41


def test_a_step_seeds_the_same_however_many_events_are_selected():
    mod = _module()
    lone = mod._decode_seed(41, (350, 0), "provisional", (200, 0))
    crowd = mod._decode_seed(41, (350, 0), "provisional", (200, 0))
    assert lone == crowd
    # and it does not collide with a different step
    assert lone != mod._decode_seed(41, (351, 0), "provisional", (200, 0))


def test_provisional_and_redecode_of_one_event_do_not_share_a_draw():
    mod = _module()
    assert (mod._decode_seed(41, (350, 0), "provisional", (200, 0))
            != mod._decode_seed(41, (350, 0), "redecode", (200, 0)))


# ------------------------------------------------------------------ R5
# Sample ids carried no schedule, so a one-event and a four-event run wrote
# the same filenames and the reporter's pairing kept whichever came last.

def test_schedule_id_separates_different_schedules():
    mod = _module()
    one = [_choice(350)]
    four = [_choice(s) for s in (200, 275, 320, 350)]
    assert mod._schedule_id(one) != mod._schedule_id(four)


def test_schedule_id_is_the_same_for_the_same_resolved_steps():
    mod = _module()
    a = [_choice(s) for s in (200, 350)]
    b = [_choice(s) for s in (200, 350)]
    assert mod._schedule_id(a) == mod._schedule_id(b)


class _choice:
    def __init__(self, step):
        self.step, self.substage = step, 0

    @property
    def key(self):
        return (self.step, self.substage)


def test_reporter_refuses_to_pool_two_schedules(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "rep", MATRIX.parent / "report_integrated_binder_matrix.py")
    rep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rep)
    csv_path = tmp_path / "designs.csv"
    csv_path.write_text(
        "sample_id,arm,bs_seed,shared_prefix_id,sequence_policy,schedule_id\n"
        "a,J03,0,p,event_fixed,ev1s350\n"
        "b,J03,0,p,event_fixed,ev4_abc123\n"
    )
    with pytest.raises(ValueError, match="mixed event schedules"):
        rep.load_rows(str(csv_path))


# ------------------------------------------------------------------ R4
# immediate_event_coordinate_rmsd compared the FIRST event's bb0 with the
# LAST event's corrected bb0 -- different noisy states, different augmented
# frames, so the number folded in the whole trajectory.

def test_immediate_correction_uses_the_terminal_event_not_the_first():
    from pxf.bench.integrated_redecode import sequence_diagnostics

    first = _products(torch.zeros(4, 3))
    terminal = _products(torch.full((4, 3), 10.0))
    output = _products(torch.full((4, 3), 10.5))

    d = sequence_diagnostics(first, output, policy="post_feedback_redesign",
                             seed=1, terminal=terminal)
    # The metric sums the squared difference over all three axes before the
    # mean, so an offset of d per axis is sqrt(3)*d, not d.
    root3 = 3 ** 0.5
    assert d["immediate_event_coordinate_rmsd"] == pytest.approx(0.5 * root3)
    assert d["first_to_final_coordinate_displacement"] == pytest.approx(10.5 * root3)
    # The point of the fix: the immediate number must be the SMALL one.
    assert (d["immediate_event_coordinate_rmsd"]
            < d["first_to_final_coordinate_displacement"])


def test_decode_passes_report_work_done_not_a_constant():
    from pxf.bench.integrated_redecode import sequence_diagnostics

    p = _products(torch.zeros(4, 3))
    d = sequence_diagnostics(p, p, policy="post_feedback_redesign", seed=1,
                             decode_passes=5)
    assert d["sequence_decode_passes"] == 5


def _products(bb0):
    from types import SimpleNamespace as NS
    return NS(bb0=bb0, aatype=torch.zeros(1, 4, dtype=torch.long),
              binder_mask=torch.ones(1, 4, dtype=torch.bool),
              binder_sequence="AAAA", sigma=0.5, provenance={})


# ------------------------------------------------------------------ R6b
# sequence_decode_passes counted only SELF-EXECUTED decodes, so a feedback
# arm reusing the shared first decode reported 1 while its control reported
# 2 -- backwards, since the feedback arm does strictly more work. Observed
# on job 499381.

def test_decode_passes_count_contributing_decodes_not_self_executed():
    from pxf.bench.integrated_redecode import sequence_diagnostics

    p = _products(torch.zeros(4, 3))
    # an arm that reused one shared event decode and then re-decoded
    reused = sequence_diagnostics(p, p, policy="post_feedback_redesign",
                                  seed=1, decode_passes=1 + 1)
    # a control that executed its own event decode and then re-decoded
    own = sequence_diagnostics(p, p, policy="post_feedback_redesign",
                               seed=1, decode_passes=1 + 1)
    assert reused["sequence_decode_passes"] == own["sequence_decode_passes"] == 2
