"""Immutable native RMS calibration identity and shared native packed mapping."""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import torch
from .chemistry import GeometryConfig, PackedAtom, build_packing_chemistry, registry_sha256
from .instantiate import STD_AA_3, sidechain_atoms


def validate_calibration(value, registry=None):
    if value['chemistry_registry_sha256'] != (registry or registry_sha256()):
        raise ValueError('Calibration chemistry registry SHA256 mismatch')
    if value['coordinate_units'] != 'angstrom' or value['angle_units'] != 'radians':
        raise ValueError('Calibration units must be angstrom/radians')
    digest = value['calibration_manifest_sha256']
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Calibration requires manifest SHA256')
    if value.get('partition') != 'train' or value.get('scale_definition') != 'native_rms_deviation':
        raise ValueError('Calibration must report native RMS deviations on training data')
    for name in ('bond_sc','bond_attach','angle_sc','angle_attach'):
        row = value['classes'][name]
        if row['count'] <= 0 or not math.isfinite(row['signed_mean']) or not math.isfinite(row['rms']) or row['rms'] <= 0:
            raise ValueError(f'Invalid calibration class {name}')
    for key, default in (('tolerance_multiplier',3.), ('cosine_scale_floor',1e-4)):
        if not math.isfinite(float(value.get(key,default))) or float(value.get(key,default)) <= 0:
            raise ValueError(f'Invalid calibration {key}')
    return value


@lru_cache(maxsize=8)
def load_calibration(path, expected_sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Calibration artifact SHA256 mismatch')
    import yaml
    return validate_calibration(yaml.safe_load(raw))


def pack_native_geometry(feat, pred, generation, *, native=False):
    """Pack one monomer: detached N/CA/C/O anchors plus generated canonical SC.

    Observation masks do not restrict generated inventories. Calibration calls
    this same mapper with native=True and then masks constraints by observations.
    Padded/unobserved native values are sanitized only in that offline path.
    """
    types = feat['aa_clean'].long().reshape(-1)
    length = types.numel()
    pred = pred.float().reshape(-1, length, 10, 3)
    generation = generation.bool().reshape(-1, length, 10)
    if pred.shape[0] != 1 or generation.shape[0] != 1:
        raise ValueError('Native monomer repair requires exactly one SC prediction')
    bb = feat['sc_bb_coords'].detach().float().reshape(length, 4, 3)
    bb_valid = feat['sc_bb_observed_mask'].bool().reshape(length, 4) & torch.isfinite(bb).all(-1)
    owned = feat['design_token_mask'].bool().reshape(-1) & feat['sc_frame_valid'].bool().reshape(-1)
    identities, rows, xyz, obs = {}, [], [], []
    observed = feat['sc_atom_mask'].bool().reshape(length,10)
    for i in range(length):
        aa = int(types[i])
        if not owned[i] or not 0 <= aa < 20: continue
        uid, name = str(i), STD_AA_3[aa]
        identities[uid] = name
        names = sidechain_atoms(name)
        if not generation[0,i,:len(names)].all():
            raise ValueError('Generated inventory missing chemically present SC atoms')
        for j, atom in enumerate(('N','CA','C','O')):
            rows.append(PackedAtom(uid, atom, valid=bool(bb_valid[i,j])))
            xyz.append(torch.where(bb_valid[i,j], bb[i,j], 0.)); obs.append(bool(bb_valid[i,j]))
        for j, atom in enumerate(names):
            rows.append(PackedAtom(uid, atom, generated=True))
            position = pred[0,i,j]
            seen = bool(observed[i,j]) and bool(torch.isfinite(position).all())
            xyz.append(torch.where(torch.tensor(seen,device=position.device),position,0.) if native else position)
            obs.append(seen)
    if not rows:
        raise ValueError('No eligible native residues in repair item')
    chemistry = build_packing_chemistry([identities], [rows],
        config=GeometryConfig(0.,1.,0.,1.), device=pred.device)
    return torch.stack(xyz)[None], chemistry, torch.tensor([obs],device=pred.device,dtype=torch.bool)
