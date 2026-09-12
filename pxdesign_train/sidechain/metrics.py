"""Packing diagnostics with explicit numerators/counts and named atom chemistry.

These are evaluation metrics, not training targets. Symmetry swaps preserve atom
elements. Covalent errors use CCD ideal bond lengths, with a stated 0.2 A cutoff.
"""
import torch
from .instantiate import STD_AA_3, sidechain_atoms, instantiate_from_type_indices
from .templates import IDEAL_SC_LOCAL
from .chi_constants import IDEAL_BB_LOCAL, CHI_ATOM_IDX, CHI_MASK
from .buildsc import chi_from_local

SWAPS = {"ARG": [("NH1", "NH2")], "ASP": [("OD1", "OD2")], "GLU": [("OE1", "OE2")],
    "LEU": [("CD1", "CD2")], "VAL": [("CG1", "CG2")],
    "PHE": [("CD1", "CD2"), ("CE1", "CE2")], "TYR": [("CD1", "CD2"), ("CE1", "CE2")]}
BONDS = dict(ALA="", ARG="CB-CG CG-CD CD-NE NE-CZ CZ-NH1 CZ-NH2", ASN="CB-CG CG-OD1 CG-ND2",
    ASP="CB-CG CG-OD1 CG-OD2", CYS="CB-SG", GLN="CB-CG CG-CD CD-OE1 CD-NE2",
    GLU="CB-CG CG-CD CD-OE1 CD-OE2", GLY="", HIS="CB-CG CG-ND1 ND1-CE1 CE1-NE2 NE2-CD2 CD2-CG",
    ILE="CB-CG1 CB-CG2 CG1-CD1", LEU="CB-CG CG-CD1 CG-CD2", LYS="CB-CG CG-CD CD-CE CE-NZ",
    MET="CB-CG CG-SD SD-CE", PHE="CB-CG CG-CD1 CG-CD2 CD1-CE1 CD2-CE2 CE1-CZ CE2-CZ",
    PRO="CB-CG CG-CD CD-N", SER="CB-OG", THR="CB-OG1 CB-CG2",
    TRP="CB-CG CG-CD1 CG-CD2 CD1-NE1 NE1-CE2 CE2-CD2 CD2-CE3 CE3-CZ3 CZ3-CH2 CH2-CZ2 CZ2-CE2",
    TYR="CB-CG CG-CD1 CG-CD2 CD1-CE1 CD2-CE2 CE1-CZ CE2-CZ CZ-OH", VAL="CB-CG1 CB-CG2")


@torch.no_grad()
def packing_metrics(types, pred_local, bb_local, generation, design, *, target=None, observed=None, target_bb=None, selection=None):
    types, pred, bb = types.reshape(-1), pred_local.float().reshape(-1,10,3), bb_local.float().reshape(-1,3,3)
    mask, owned = generation.reshape(-1,10).bool(), design.reshape(-1).bool().clone()
    owned &= (types >= 0) & (types < 20)
    if selection is not None:
        owned &= selection.reshape(-1).bool()
    _, chemical = instantiate_from_type_indices(types.clamp(0,19))
    zero = pred.new_zeros(())
    result = dict(chemical_atoms=(chemical & owned[:,None]).sum().float(),
        generated_atoms=(mask & owned[:,None] & torch.isfinite(pred).all(-1)).sum().float(),
        bond_abs_error_sum=zero.clone(), bond_count=zero.clone(), bad_bond_count=zero.clone())
    combined = torch.cat((bb, pred), -2)
    combined_mask = torch.cat((torch.isfinite(bb).all(-1), mask), -1)
    for index, name in enumerate(STD_AA_3):
        selected = owned & (types == index)
        names = ["N", "CA", "C"] + sidechain_atoms(name)
        bonds = (["CA-CB"] if "CB" in names else []) + BONDS[name].split()
        ideal = torch.cat((IDEAL_BB_LOCAL[index], IDEAL_SC_LOCAL[index]), -2).to(pred)
        for bond in bonds:
            a, b = [names.index(atom) for atom in bond.split("-")]
            valid = selected & combined_mask[:,a] & combined_mask[:,b]
            error = ((combined[:,a]-combined[:,b]).norm(dim=-1) - (ideal[a]-ideal[b]).norm()).abs()
            result["bond_abs_error_sum"] += torch.where(valid, error, 0.).sum()
            result["bond_count"] += valid.sum()
            result["bad_bond_count"] += (valid & (error > .2)).sum()
    if target is None:
        return result
    target = target.float().reshape_as(pred)
    loss_mask = observed.reshape_as(mask).bool() & mask & owned[:,None] & torch.isfinite(target).all(-1)
    safe_target = torch.where(loss_mask[...,None], target, 0.)
    aligned = pred.clone()
    error = torch.where(loss_mask, (pred-safe_target).square().sum(-1), 0.).sum(-1)
    for index, name in enumerate(STD_AA_3):
        if name not in SWAPS:
            continue
        permutation = list(range(10))
        names = sidechain_atoms(name)
        for first, second in SWAPS[name]:
            a,b = names.index(first),names.index(second)
            permutation[a],permutation[b] = permutation[b],permutation[a]
        alternative = pred[:,permutation]
        alternate_error = torch.where(loss_mask, (alternative-safe_target).square().sum(-1), 0.).sum(-1)
        better = (types == index) & (alternate_error < error)
        aligned = torch.where(better[:,None,None], alternative, aligned)
        error = torch.where(better, alternate_error, error)
    result.update(symmetry_squared_error=error.sum(), observed_atoms=loss_mask.sum().float())
    pred_chi = chi_from_local(types, aligned, bb)
    gt_chi = chi_from_local(types, safe_target, target_bb.float().reshape_as(bb) if target_bb is not None else bb)
    observed_combined = torch.cat((torch.isfinite(bb).all(-1) & owned[:,None], loss_mask), -1)
    indices = CHI_ATOM_IDX.to(types.device)[types.clamp(0,19)]
    valid_chi = observed_combined.gather(-1, indices.reshape(-1,16)).reshape(-1,4,4).all(-1)
    valid_chi &= CHI_MASK.to(types.device)[types.clamp(0,19)] & torch.isfinite(pred_chi) & torch.isfinite(gt_chi)
    angle_error = torch.atan2(torch.sin(pred_chi-gt_chi), torch.cos(pred_chi-gt_chi)).abs()
    recovered = valid_chi & (angle_error < 40*torch.pi/180)
    rotamer_valid = valid_chi.any(-1) & ((~CHI_MASK.to(types.device)[types.clamp(0,19)]) | valid_chi).all(-1)
    result.update(chi_count=valid_chi.sum().float(), chi_recovered=recovered.sum().float(),
        rotamer_count=rotamer_valid.sum().float(),
        rotamer_recovered=(rotamer_valid & (~valid_chi | recovered).all(-1)).sum().float())
    return result


def summarize_metrics(counts):
    """No-observation entries remain identifiable by their zero counts."""
    result = dict(counts)
    def ratio(numerator, denominator):
        return counts[numerator] / counts[denominator].clamp_min(1)
    result["completeness"] = ratio("generated_atoms", "chemical_atoms")
    result["bond_mae"] = ratio("bond_abs_error_sum", "bond_count")
    result["bad_bond_fraction"] = ratio("bad_bond_count", "bond_count")
    if "observed_atoms" in counts:
        result["symmetry_rmsd"] = ratio("symmetry_squared_error", "observed_atoms").sqrt()
        result["chi_recovery"] = ratio("chi_recovered", "chi_count")
        result["rotamer_recovery"] = ratio("rotamer_recovered", "rotamer_count")
    return result


@torch.no_grad()
def diagnose_packing(feat, pack, types, xyz, *, observed=None):
    from .frames import to_local
    with torch.autocast(device_type=xyz.device.type, enabled=False):
        R, t = pack["sc_frame_R"].float().reshape(-1,3,3), pack["sc_frame_t"].float().reshape(-1,3)
        pred = to_local(pack["sc_pred_global"].float().reshape(-1,10,3), R, t)
        idx = feat["aa_bb_atom_idx"].long()[...,:3]
        bb = to_local(xyz.float().reshape(-1,3)[idx.clamp_min(0)], R, t)
        bb = torch.where((idx >= 0)[...,None], bb, float("nan"))
        kwargs = {}
        if observed is not None:
            native_bb = to_local(feat["sc_bb_coords"].float()[...,:3,:], feat["sc_frame_R"].float(), feat["sc_frame_t"].float())
            kwargs = dict(target=feat["sc_gt_local"], observed=observed, target_bb=native_bb)
        design = feat["design_token_mask"].bool()
        selections = dict(all=torch.ones_like(design))
        if "sc_interface_mask" in feat:
            interface = feat["sc_interface_mask"].bool()
            selections.update(interface=interface, noninterface=~interface)
        if 80 <= int(design.sum()) <= 130:
            selections["binder80_130"] = torch.ones_like(design)
        metrics = {}
        for name, selection in selections.items():
            counts = packing_metrics(types, pred, bb, pack["sc_generation_mask"], design,
                selection=selection, **kwargs)
            metrics.update({f"{name}/{key}": value for key,value in summarize_metrics(counts).items()})
        return metrics
