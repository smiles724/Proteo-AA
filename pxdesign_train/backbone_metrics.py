"""Backbone geometry with explicit atom mappings and continuous-chain masks."""
import numpy as np


def backbone_geometry(coords, atom_rows, design, chain, residue_index, observed=None):
    """Measure named N/CA/C/O rows; never join chains, gaps or missing residues.

    bad_bond_fraction is the historical CA-spacing diagnostic: |d - 3.8| > 0.3 A.
    It is not an all-covalent-bond violation rate. Missing/nonfinite pairs have
    separate counts and are excluded from the measured-pair denominator.
    """
    xyz = np.asarray(coords, dtype=np.float64)
    rows = np.asarray(atom_rows, dtype=np.int64)
    design = np.asarray(design, dtype=bool)
    chain, residue_index = np.asarray(chain), np.asarray(residue_index)
    if xyz.ndim != 2 or xyz.shape[-1] != 3 or rows.shape != (design.size, 4):
        raise ValueError('Expected [atom,3] coordinates and [residue,4] N/CA/C/O indices')
    present = (rows >= 0) & (rows < len(xyz))
    safe = np.clip(rows, 0, max(0, len(xyz)-1))
    if not len(xyz):
        raise ValueError('No coordinates')
    bb = xyz[safe]
    if observed is not None:
        present &= np.asarray(observed, dtype=bool)[safe]
    finite = np.isfinite(bb).all(-1)
    adjacent = design[:-1] & design[1:] & (chain[:-1] == chain[1:]) & (residue_index[1:] == residue_index[:-1]+1)
    result = dict(design_residues=int(design.sum()), ca_ideal_angstrom=3.8, ca_tolerance_angstrom=0.3)
    def distances(name, left, right, eligible, available, finite_pair):
        valid = eligible & available & finite_pair
        d = np.linalg.norm(right[valid]-left[valid], axis=-1)
        result[name+'_count'] = int(valid.sum())
        result[name+'_missing_count'] = int((eligible & ~available).sum())
        result[name+'_nonfinite_count'] = int((eligible & available & ~finite_pair).sum())
        for label, fn in [('mean',np.mean),('median',np.median),('min',np.min),('max',np.max)]:
            result[name+'_'+label+'_angstrom'] = float(fn(d)) if len(d) else None
        return d
    ca = distances('ca_ca', bb[:-1,1],bb[1:,1],adjacent,present[:-1,1]&present[1:,1],finite[:-1,1]&finite[1:,1])
    result['bad_bond_count'] = int((np.abs(ca-3.8) > .3).sum())
    result['bad_bond_fraction'] = result['bad_bond_count']/len(ca) if len(ca) else None
    distances('peptide_cn',bb[:-1,2],bb[1:,0],adjacent,present[:-1,2]&present[1:,0],finite[:-1,2]&finite[1:,0])
    for name,l,r in [('n_ca',0,1),('ca_c',1,2),('c_o',2,3)]:
        distances(name,bb[:,l],bb[:,r],design,present[:,l]&present[:,r],finite[:,l]&finite[:,r])
    return result
