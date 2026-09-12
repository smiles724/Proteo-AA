"""Consistent named-atom assembly and mmCIF export of the final state."""
import numpy as np
import torch
from .sidechain.instantiate import ATOM_NAME_TO_ID, STD_AA_3


def assemble_atoms(state, feat, batch=0, sample=0):
    state.validate_final(packing_enabled=(state.protocol or {}).get("packing_enabled", True))
    a2t = feat["atom_to_token_idx"].long()
    design = state.design_mask[batch,sample]
    assigned = state.assigned_aa[batch,sample]
    xyz = state.backbone_xyz[batch,sample]
    names = feat["structure_atom_name"]
    res_names = feat["structure_res_name"]
    chain_ids = feat["structure_chain_id"]
    res_ids = feat["structure_res_id"]
    coordinates, atom_names, residues, chains, numbers, elements, insertions, hetero = [], [], [], [], [], [], [], []
    owner_tokens = []
    # Fixed context is copied verbatim; binder input topology contributes BB only.
    for atom, token in enumerate(a2t.tolist()):
        if design[token] and names[atom] not in ("N","CA","C","O"):
            continue
        if not design[token] and not feat["fixed_atom_mask"][atom]:
            continue  # absent experimental context stays absent
        coordinates.append(xyz[atom])
        owner_tokens.append(token)
        atom_names.append(names[atom])
        residues.append(STD_AA_3[int(assigned[token])] if design[token] else res_names[atom])
        chains.append(chain_ids[atom]); numbers.append(int(res_ids[atom]))
        elements.append(feat["structure_element"][atom]); insertions.append(feat["structure_ins_code"][atom])
        hetero.append(bool(feat["structure_hetero"][atom]))
    by_id = {value:key for key,value in ATOM_NAME_TO_ID.items()}
    for token in design.nonzero().flatten().tolist():
        representative = int((a2t == token).nonzero()[0])
        mask = state.generation_mask[batch,sample,token]
        for slot in mask.nonzero().flatten().tolist():
            coordinates.append(state.sc_xyz[batch,sample,token,slot])
            owner_tokens.append(token)
            atom_names.append(by_id[int(state.sc_atom_name_ids[batch,sample,token,slot])])
            residues.append(STD_AA_3[int(assigned[token])]); chains.append(chain_ids[representative]); numbers.append(int(res_ids[representative]))
            elements.append(atom_names[-1][0]); insertions.append(feat["structure_ins_code"][representative]); hetero.append(False)
    # Keep each residue contiguous: downstream structure readers infer residue
    # boundaries from atom order as well as identifiers.
    order = sorted(range(len(owner_tokens)), key=owner_tokens.__getitem__)
    coordinates, atom_names, residues, chains, numbers, elements, insertions, hetero = (
        [values[i] for i in order] for values in
        (coordinates, atom_names, residues, chains, numbers, elements, insertions, hetero)
    )
    keys = list(zip(chains,numbers,insertions,atom_names))
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate residue/atom identifiers in assembled output")
    return dict(coordinate=torch.stack(coordinates),atom_name=atom_names,
                res_name=residues,chain_id=chains,res_id=numbers,element=elements,ins_code=insertions,hetero=hetero)


def write_mmcif(atoms, path):
    import biotite.structure as struc
    from biotite.structure.io.pdbx import CIFFile, set_structure
    array = struc.AtomArray(len(atoms["atom_name"]))
    array.coord = atoms["coordinate"].detach().cpu().numpy()
    for key in ("atom_name","res_name","chain_id","res_id","element","ins_code","hetero"):
        setattr(array,key,np.asarray(atoms[key]))
    file = CIFFile()
    set_structure(file,array)
    file.write(str(path))
