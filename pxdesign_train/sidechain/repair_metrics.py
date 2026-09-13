"""Repair-only clash diagnostics using the loss chemistry topology."""
import torch


@torch.no_grad()
def corrected_clash_diagnostics(coords, chemistry, *, overlap_allowance=0.4):
    """Count unique nonbonded overlaps with complete 1-/2-bond exclusions.

    The packed monomer contains generated SC atoms and fixed N/CA/C/O anchors.
    A pair is eligible when at least one endpoint is generated. This diagnostic
    is never part of the repair objective.
    """
    from .packing_objective import _inputs

    xyz = _inputs(coords, chemistry.valid_mask, chemistry.subject_mask, chemistry.group_id)
    atoms = xyz.shape[1]
    count = xyz.new_zeros(())
    clashes = count.clone()
    for start in range(0, atoms, 256):
        stop = min(start + 256, atoms)
        distance = torch.cdist(
            xyz[:, start:stop], xyz, compute_mode="donot_use_mm_for_euclid_dist"
        )
        eligible = chemistry.valid_mask[:, start:stop, None] & chemistry.valid_mask[:, None, :]
        eligible &= chemistry.subject_mask[:, start:stop, None] | chemistry.subject_mask[:, None, :]
        eligible &= (
            torch.arange(start, stop, device=xyz.device)[:, None]
            < torch.arange(atoms, device=xyz.device)[None, :]
        )
        for b in range(xyz.shape[0]):
            first = chemistry.excluded_pairs[b, :, 0]
            second = chemistry.excluded_pairs[b, :, 1]
            use = (first >= start) & (first < stop) & (second >= 0)
            eligible[b, first[use] - start, second[use]] = False
        threshold = (
            chemistry.radii[:, start:stop, None]
            + chemistry.radii[:, None, :]
            - float(overlap_allowance)
        )
        count += eligible.sum()
        clashes += (eligible & (distance < threshold)).sum()
    return {
        "eligible_pairs": count,
        "overlap_gt_0p4A_pairs": clashes,
        "generated_atoms": chemistry.subject_mask.sum(),
    }
