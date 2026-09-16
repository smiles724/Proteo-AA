"""Score packed side chains with the canonical Proteo-AA metrics.

This is an adapter, not a metric implementation. The pipeline works in AF2
atom37; the canonical metrics work in Proteo-AA's 10-slot per-residue local
frame. Everything here is the translation between the two, so the reported
numbers come out of the same functions as the earlier Proteo-AA runs.

Aggregation matters as much as the metrics. ``packing_metrics`` returns
*numerators and counts*, not ratios, precisely so a dataset figure can be formed
by summing counts across targets and dividing once -- atom-weighted, and not
distorted by short targets. :func:`aggregate` does that; :func:`score` also
returns the per-target ratios for the breakdown.
"""

import torch

from pxf import atom37
from pxf.eval.canonical import load

# atom37 slots for the frame-defining backbone atoms, and for the backbone
# partners the environment lDDT term needs.
NCAC = (0, 1, 2)
BACKBONE = atom37.BACKBONE_SLOTS
LDDT_KEYS = ("lddt_sc_sc", "lddt_sc_env")


def slot_tables(canonical, device=None):
    """``[20, 10]`` atom37 index per side-chain slot, plus its validity mask."""
    max_sc = canonical.instantiate.MAX_SC
    table = torch.full((20, max_sc), 0, dtype=torch.long, device=device)
    valid = torch.zeros((20, max_sc), dtype=torch.bool, device=device)
    for index, name3 in enumerate(canonical.instantiate.STD_AA_3):
        for slot, atom in enumerate(canonical.instantiate.sidechain_atoms(name3)):
            table[index, slot] = atom37.ATOM37.index(atom)
            valid[index, slot] = True
    return table, valid


def to_slots(coords37, mask37, aatype, table, valid):
    """Gather ``[L, 37, 3]`` / ``[L, 37]`` into the 10-slot side-chain layout."""
    index = table[aatype]  # [L, 10]
    exists = valid[aatype]  # [L, 10]
    coords = coords37.gather(-2, index[..., None].expand(*index.shape, 3))
    present = mask37.gather(-1, index).bool() & exists
    # Absent slots must not carry stale coordinates into a distance.
    return torch.where(present[..., None], coords, torch.nan), present


def score(
    pred37,
    pred_mask37,
    native37,
    native_mask37,
    aatype,
    *,
    canonical=None,
    residue_mask=None,
):
    """Canonical side-chain metrics for one structure.

    All inputs are AF2 atom37: ``[L, 37, 3]`` coordinates with ``[L, 37]`` masks.
    ``native37`` supplies the reference side chains and the shared backbone.
    ``residue_mask`` restricts scoring to a subset of residues.

    Returns ``(counts, summary)`` -- raw numerators/counts for aggregation, and
    the per-target ratios.
    """
    canonical = canonical or load()
    device = pred37.device
    aatype = aatype.reshape(-1).long()
    length = aatype.shape[0]
    if length and int(aatype.max()) >= 20:
        raise ValueError(
            "Side-chain metrics require canonical residue types only; found "
            f"{int((aatype >= 20).sum())} position(s) outside the twenty"
        )
    table, valid = slot_tables(canonical, device)

    pred_sc, generated = to_slots(pred37, pred_mask37, aatype, table, valid)
    native_sc, observed = to_slots(native37, native_mask37, aatype, table, valid)

    # The backbone is shared: FaMPNN does not move it, so one frame serves both.
    ncac = native37[:, list(NCAC), :]
    rotation, translation = canonical.frames.build_frame(ncac[:, 0], ncac[:, 1], ncac[:, 2])
    to_local = canonical.frames.to_local
    bb_local = to_local(ncac, rotation, translation)
    pred_local = to_local(torch.nan_to_num(pred_sc), rotation, translation)
    native_local = to_local(torch.nan_to_num(native_sc), rotation, translation)

    scored = torch.ones(length, dtype=torch.bool, device=device)
    if residue_mask is not None:
        scored &= residue_mask.reshape(-1).bool().to(device)

    counts = canonical.packing_metrics(
        aatype,
        pred_local,
        bb_local,
        generated,
        scored,
        target=native_local,
        observed=observed,
        target_bb=bb_local,
    )

    # lDDT is a global-coordinate, distance-difference metric: only atoms present
    # in both structures can contribute a pair.
    both = generated & observed & scored[:, None]
    bb_mask = native_mask37[:, list(BACKBONE)].bool() & scored[:, None]
    lddt = canonical.sidechain_lddt(
        torch.nan_to_num(pred_sc),
        torch.nan_to_num(native_sc),
        both,
        bb_coords=native37[:, list(BACKBONE), :],
        bb_mask=bb_mask,
    )

    counts = {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in counts.items()
    }
    for key in LDDT_KEYS:
        pairs = float(lddt[f"n_pairs_{key[5:]}"])
        # A target with no scored pairs -- a single supervised residue has no
        # side-chain neighbour to form one -- gets lDDT 0/0 = nan. Its weighted
        # contribution is nan * 0, which is nan rather than 0, and summing that
        # across the dataset turns the aggregate into nan no matter how many
        # good targets there are. On the recentPDB eval split exactly 10 of
        # 1,582 targets do this, and they took `lddt_sc_sc` -- a headline
        # metric, so a regression check on it silently compares nan to nan.
        # Contributing zero of zero pairs is the arithmetically correct answer:
        # the target simply does not participate in a pair-weighted mean.
        counts[f"{key}_pairs"] = torch.tensor(pairs)
        counts[f"{key}_sum"] = torch.tensor(float(lddt[key]) * pairs if pairs else 0.0)
    summary = {
        key: float(value)
        for key, value in canonical.summarize_metrics(
            {k: v for k, v in counts.items() if torch.is_tensor(v)}
        ).items()
    }
    summary.update({key: float(lddt[key]) for key in LDDT_KEYS})
    summary["scored_residues"] = int(scored.sum())
    return counts, summary


def aggregate(per_target, *, canonical=None):
    """Dataset figures: sum counts across targets, then form the ratios once."""
    canonical = canonical or load()
    if not per_target:
        raise ValueError("No targets to aggregate")
    keys = set().union(*(set(counts) for counts in per_target))
    totals = {}
    for key in keys:
        values = [counts[key] for counts in per_target if key in counts]
        totals[key] = torch.stack([v.reshape(()).float() for v in values]).sum()
    summary = {
        key: float(value) for key, value in canonical.summarize_metrics(totals).items()
    }
    # lDDT aggregates by pair count, the weighting its own definition implies.
    for key in LDDT_KEYS:
        pairs = float(totals.get(f"{key}_pairs", torch.tensor(0.0)))
        summary[key] = float(totals[f"{key}_sum"]) / pairs if pairs else float("nan")
        summary[f"n_pairs_{key[5:]}"] = pairs
        # How many targets were too small to contribute a pair. Reported rather
        # than inferred, so a shrinking denominator is visible instead of being
        # read as a dataset-wide score.
        summary[f"n_targets_without_pairs_{key[5:]}"] = sum(
            1 for counts in per_target if float(counts.get(f"{key}_pairs", 0.0)) == 0.0
        )
    summary["n_targets"] = len(per_target)
    return summary
