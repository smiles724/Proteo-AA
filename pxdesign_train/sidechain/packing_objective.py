"""FP32 covalent objectives sharing one chemistry registry with calibration.

Repair evaluates no clash term. Each term averages constraints per residue,
then eligible residues per item, then eligible items. Angle penalties operate
in cosine space, not exactly squared angular error.
"""
import math
import torch
from .chemistry import SC_INTERNAL, SC_ATTACHMENT

TERM_CLASSES = dict(bond_sc=('bond', SC_INTERNAL), bond_attach=('bond', SC_ATTACHMENT),
                    angle_sc=('angle', SC_INTERNAL), angle_attach=('angle', SC_ATTACHMENT))


def _check_coords(coords, chemistry):
    if coords.shape != (*chemistry.valid_mask.shape, 3):
        raise ValueError('Coordinates must match exact packed chemistry [B,N,3]')
    if not torch.isfinite(coords[chemistry.valid_mask]).all():
        raise ValueError('Nonfinite generated atom or valid anchor; cannot remove it from geometry denominator')
    return torch.where(chemistry.valid_mask[..., None], coords.float(), 0.)


def geometry_values(coords, chemistry, kind, *, active=None):
    idx = getattr(chemistry, kind+'_idx')
    batch = torch.arange(coords.shape[0], device=coords.device)[:, None, None]
    points = coords[batch, idx.clamp_min(0)]
    if kind == 'bond':
        return (points[..., 0, :] - points[..., 1, :]).norm(dim=-1)
    u, v = points[..., 0, :] - points[..., 1, :], points[..., 2, :] - points[..., 1, :]
    nu, nv = u.norm(dim=-1), v.norm(dim=-1)
    regular = ((u*v).sum(-1) / (nu*nv).clamp_min(1e-16)).clamp(-1., 1.)
    return torch.where((nu > 1e-8) & (nv > 1e-8), regular, torch.zeros_like(regular))


OBJECTIVE_VERSION = 'sc_geometry_v1'


def _scalar(value, name, *, positive=False):
    value = float(value)
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f'{name} must be finite and positive (tolerances may be zero)')
    return value


def _inputs(coords, valid_mask, subject_mask, group_id):
    if coords.ndim != 3 or coords.shape[-1] != 3 or coords.shape[1] == 0:
        raise ValueError('coords must be [batch, atom>0, 3]')
    for name, value in (('valid_mask',valid_mask),('subject_mask',subject_mask),('group_id',group_id)):
        if value.shape != coords.shape[:2] or value.device != coords.device:
            raise ValueError(f'{name} must match coordinate axes/device')
    if valid_mask.dtype != torch.bool or subject_mask.dtype != torch.bool:
        raise ValueError('atom masks must be boolean')
    if group_id.dtype != torch.long or (group_id[valid_mask] < 0).any():
        raise ValueError('valid atoms require nonnegative int64 group IDs')
    if (subject_mask & ~valid_mask).any():
        raise ValueError('every generated subject must be valid')
    if not torch.isfinite(coords[valid_mask]).all():
        raise ValueError('valid coordinates must be finite; do not silently mask failed predictions')
    xyz = torch.where(valid_mask[...,None],coords.float(),0.)
    return torch.where(subject_mask[...,None],xyz,xyz.detach())


def _indices(index, batch, atoms, arity, device):
    if index.dtype != torch.long or index.device != device:
        raise ValueError('indices must be int64 on coordinate device')
    if index.ndim == 2: index = index.unsqueeze(0).expand(batch,-1,-1)
    elif index.ndim == 3 and index.shape[0] == 1: index = index.expand(batch,-1,-1)
    if index.ndim != 3 or index.shape[0] != batch or index.shape[-1] != arity:
        raise ValueError('invalid constraint index shape')
    if ((index < -1) | (index >= atoms)).any():
        raise ValueError('index outside atom axis; use -1 for padding')
    return index


def _gather(values, index):
    return values[torch.arange(values.shape[0],device=values.device)[:,None,None],index.clamp_min(0)]


def _terms(xyz,index,valid,subject,groups,arity):
    index = _indices(index,xyz.shape[0],xyz.shape[1],arity,xyz.device)
    present = (index>=0).all(-1) & _gather(valid,index).all(-1)
    movable = _gather(subject,index)
    active = present & movable.any(-1)
    distinct = torch.ones_like(active)
    for i in range(arity):
        for j in range(i): distinct &= index[...,i] != index[...,j]
    if (active & ~distinct).any(): raise ValueError('constraint cannot repeat an atom')
    owners = _gather(groups,index).gather(-1,movable.long().argmax(-1)[...,None]).squeeze(-1)
    return _gather(xyz,index),active,owners


def _table(value,active,name,*,positive=False,nonnegative=False):
    value = torch.as_tensor(value,device=active.device,dtype=torch.float32)
    try: value = torch.broadcast_to(value,active.shape)
    except RuntimeError as exc: raise ValueError(f'{name} must broadcast to [batch,term]') from exc
    checked = value[active]
    if not torch.isfinite(checked).all() or (positive and (checked<=0).any()) or (nonnegative and (checked<0).any()):
        raise ValueError(f'invalid active {name}')
    return torch.where(active,value,1. if positive else 0.).detach()


def _residue_mean(values,active,owners,*,return_counts=False):
    items, residue_count = [], 0
    for b in range(values.shape[0]):
        selected = active[b]
        if not selected.any(): continue
        _,inverse = torch.unique(owners[b,selected],return_inverse=True)
        count = torch.bincount(inverse).to(values.dtype)
        sums = values.new_zeros(count.shape).scatter_add(0,inverse,values[b,selected])
        items.append((sums/count).mean()); residue_count += count.numel()
    loss = torch.stack(items).mean() if items else values.sum()*0.
    if not return_counts: return loss
    return loss, dict(constraints=active.sum(),residues=active.new_tensor(residue_count,dtype=torch.long),
                      items=active.new_tensor(len(items),dtype=torch.long))


def _term_mask(active, term_mask):
    if term_mask is None: return active
    if term_mask.dtype != torch.bool or term_mask.device != active.device:
        raise ValueError('term_mask must be boolean on coordinate device')
    return active & torch.broadcast_to(term_mask,active.shape)


def bond_violation_loss(coords,bond_idx,ideal_lengths,*,tolerance,scale,valid_mask,subject_mask,group_id,
                        term_mask=None,return_counts=False):
    """Supplied flat-bottom bond helper, extended by independent term masking/counts."""
    with torch.autocast(device_type=coords.device.type,enabled=False):
        xyz = _inputs(coords,valid_mask,subject_mask,group_id)
        points,active,owners = _terms(xyz,bond_idx,valid_mask,subject_mask,group_id,2)
        active = _term_mask(active,term_mask)
        ideal = _table(ideal_lengths,active,'ideal_lengths',positive=True)
        tol = _table(tolerance,active,'bond tolerance',nonnegative=True)
        unit = _table(scale,active,'bond scale',positive=True)
        distance = (points[...,0,:]-points[...,1,:]).norm(dim=-1)
        return _residue_mean((torch.relu((distance-ideal).abs()-tol)/unit).square(),active,owners,return_counts=return_counts)


def angle_violation_loss(coords,angle_idx,cos_min,cos_max,*,scale,valid_mask,subject_mask,group_id,
                         eps=1e-8,term_mask=None,return_counts=False):
    """Cosine-space helper; collapsed arms remain scored with a finite safeguard.

    A generated zero-length arm must not disappear from the denominator. Its
    cosine is evaluated through the clamped denominator, while bond constraints
    provide the direct restoring force for the collapsed atoms.
    """
    eps = _scalar(eps,'eps',positive=True)
    with torch.autocast(device_type=coords.device.type,enabled=False):
        xyz = _inputs(coords,valid_mask,subject_mask,group_id)
        points,active,owners = _terms(xyz,angle_idx,valid_mask,subject_mask,group_id,3)
        active = _term_mask(active,term_mask)
        low,high = _table(cos_min,active,'cos_min'),_table(cos_max,active,'cos_max')
        if (active & ((low < -1)|(high>1)|(low>high))).any(): raise ValueError('invalid angle bounds')
        unit = _table(scale,active,'angle scale',positive=True)
        u,v = points[...,0,:]-points[...,1,:],points[...,2,:]-points[...,1,:]
        nu,nv = u.norm(dim=-1),v.norm(dim=-1)
        regular = ((u*v).sum(-1)/(nu*nv).clamp_min(eps*eps)).clamp(-1.,1.)
        cosine = torch.where((nu > eps) & (nv > eps), regular, torch.zeros_like(regular))
        return _residue_mean(((torch.relu(low-cosine)+torch.relu(cosine-high))/unit).square(),active,owners,return_counts=return_counts)


def _common(chemistry):
    return dict(valid_mask=chemistry.valid_mask,subject_mask=chemistry.subject_mask,group_id=chemistry.group_id)


def bond_loss(coords,chemistry,*,term_mask=None,tolerance=None,scale=None):
    return bond_violation_loss(coords,chemistry.bond_idx,chemistry.ideal_lengths,
        tolerance=chemistry.bond_tolerance if tolerance is None else tolerance,
        scale=chemistry.bond_scale if scale is None else scale,
        term_mask=_term_mask(chemistry.bond_valid,term_mask),return_counts=True,**_common(chemistry))


def angle_loss(coords,chemistry,*,term_mask=None,cos_min=None,cos_max=None,scale=None):
    return angle_violation_loss(coords,chemistry.angle_idx,
        chemistry.cos_min if cos_min is None else cos_min,chemistry.cos_max if cos_max is None else cos_max,
        scale=chemistry.angle_scale if scale is None else scale,
        term_mask=_term_mask(chemistry.angle_valid,term_mask),return_counts=True,**_common(chemistry))


def native_geometry_repair_loss(coords, chemistry, *, calibration, terms=None):
    """Four independent penalties using native RMS scales (not standard deviations).

    Bounds theta* +/- k*s radians. Cosine scale=max(abs(sin(theta*))*s,1e-4).
    This safeguard is explicit and frozen in calibration identity.
    """
    coords = _check_coords(coords, chemistry)
    from .repair_calibration import validate_calibration
    validate_calibration(calibration, chemistry.metadata['registry_sha256'])
    terms = tuple(TERM_CLASSES) if terms is None else tuple(terms)
    if set(terms) - TERM_CLASSES.keys():
        raise ValueError('Unknown repair term')
    result = dict(counts={})
    k = float(calibration.get('tolerance_multiplier', 3.))
    for name in terms:
        kind, cls = TERM_CLASSES[name]
        mask = getattr(chemistry, kind+'_class') == cls
        s = float(calibration['classes'][name]['rms'])
        if kind == 'bond':
            loss, counts = bond_loss(coords, chemistry, term_mask=mask, tolerance=k*s, scale=s)
        else:
            theta = chemistry.ideal_angles_rad
            loss, counts = angle_loss(coords, chemistry, term_mask=mask,
                cos_min=(theta+k*s).clamp(max=math.pi).cos(), cos_max=(theta-k*s).clamp(min=0.).cos(),
                scale=(theta.sin().abs()*s).clamp_min(float(calibration.get('cosine_scale_floor', 1e-4))))
        result[name], result['counts'][name] = loss, counts
    return result


@torch.no_grad()
def geometry_diagnostics(coords, chemistry, calibration, *, observed=None):
    """Signed native/model geometry errors and complete fixed threshold ladders."""
    coords = _check_coords(coords, chemistry)
    result = {}
    for name, (kind, cls) in TERM_CLASSES.items():
        idx = getattr(chemistry, kind+'_idx')
        valid = getattr(chemistry, kind+'_valid') & (getattr(chemistry, kind+'_class') == cls)
        if observed is not None:
            valid &= observed[torch.arange(coords.shape[0], device=coords.device)[:,None,None], idx.clamp_min(0)].all(-1)
        value = geometry_values(coords, chemistry, kind, active=valid)
        error = value-chemistry.ideal_lengths if kind == 'bond' else value.acos()-chemistry.ideal_angles_rad
        selected = error[valid]
        prefix = name+'/'
        result.update({prefix+'count': valid.sum(), prefix+'signed_sum': selected.sum(),
            prefix+'squared_sum': selected.square().sum(), prefix+'abs_sum': selected.abs().sum()})
        thresholds = (.02,.05,.1,.2,.3,.5,1.) if kind == 'bond' else tuple(math.radians(x) for x in (1,2,5,10,20,30,45,60))
        for threshold in thresholds:
            result[prefix+f'outliers_absolute_{threshold:.6g}'] = (selected.abs() > threshold).sum()
        for multiplier in (1,2,3,4,5,10):
            result[prefix+f'outliers_{multiplier}rms'] = (selected.abs() > multiplier*calibration['classes'][name]['rms']).sum()
    return result


def covalent_pair_exclusions(bond_idx,num_atoms):
    """Full supplied 1-/2-bond topology exclusions, before coordinate masks."""
    if num_atoms<=0 or bond_idx.ndim not in (2,3): raise ValueError('invalid topology shape')
    batch=1 if bond_idx.ndim==2 else bond_idx.shape[0]
    bonds=_indices(bond_idx,batch,num_atoms,2,bond_idx.device)
    rows=[]
    for item in bonds.detach().cpu().tolist():
        graph=[set() for _ in range(num_atoms)]
        for a,b in item:
            if a<0 or b<0: continue
            if a==b: raise ValueError('self-bonds are invalid')
            graph[a].add(b);graph[b].add(a)
        pairs=set()
        for a,neighbors in enumerate(graph):
            reachable=set(neighbors)
            for b in neighbors: reachable.update(graph[b])
            pairs.update((a,b) for b in reachable if a<b)
        rows.append(sorted(pairs))
    result=torch.full((batch,max(map(len,rows),default=0),2),-1,dtype=torch.long,device=bond_idx.device)
    for b,row in enumerate(rows):
        if row: result[b,:len(row)]=torch.tensor(row,device=bond_idx.device)
    return result


def steric_clash_loss(coords,radii,*,excluded_pairs,overlap_allowance,scale,
                      valid_mask,subject_mask,group_id,chunk_size=256,checkpoint_chunks=True):
    """Supplied opt-in radius-aware helper; never called by repair training."""
    from torch.utils.checkpoint import checkpoint
    allowance=_scalar(overlap_allowance,'overlap_allowance')
    scale=_scalar(scale,'clash scale',positive=True)
    if not isinstance(chunk_size,int) or chunk_size<=0: raise ValueError('invalid chunk_size')
    with torch.autocast(device_type=coords.device.type,enabled=False):
        xyz=_inputs(coords,valid_mask,subject_mask,group_id)
        radii=_table(radii,valid_mask,'radii',positive=True)
        batch,atoms=xyz.shape[:2]
        pairs=_indices(excluded_pairs,batch,atoms,2,xyz.device)
        chunks=[]
        for start in range(0,atoms,chunk_size):
            stop=min(start+chunk_size,atoms)
            def chunk_scores(x,start=start,stop=stop):
                distance=torch.cdist(x[:,start:stop],x,compute_mode='donot_use_mm_for_euclid_dist')
                eligible=subject_mask[:,start:stop,None] & valid_mask[:,None,:]
                row_ids=torch.arange(start,stop,device=x.device)
                eligible=eligible & (row_ids[:,None]!=torch.arange(atoms,device=x.device)[None,:])
                for b in range(batch):
                    for left,right in ((pairs[b,:,0],pairs[b,:,1]),(pairs[b,:,1],pairs[b,:,0])):
                        use=(left>=start)&(left<stop)&(right>=0)
                        eligible[b,left[use]-start,right[use]]=False
                threshold=radii[:,start:stop,None]+radii[:,None,:]-allowance
                penalty=(torch.relu(threshold-distance)/scale).square()
                pair_weight=torch.where(subject_mask[:,None,:],.5,1.)
                return torch.where(eligible,penalty*pair_weight,0.).sum(-1)
            if checkpoint_chunks and torch.is_grad_enabled() and xyz.requires_grad:
                chunks.append(checkpoint(chunk_scores,xyz,use_reentrant=False,preserve_rng_state=False))
            else: chunks.append(chunk_scores(xyz))
        return _residue_mean(torch.cat(chunks,dim=-1),subject_mask,group_id)


def combine_physical_losses(terms,*,weights):
    names={'clash','bond','angle'}
    if set(terms)!=names or set(weights)!=names: raise ValueError('expected clash, bond, angle')
    weighted={}
    for name in sorted(names):
        if terms[name].ndim!=0 or not torch.isfinite(terms[name]): raise ValueError('expected finite scalar')
        weighted[name]=terms[name]*_scalar(weights[name],name+' weight')
    return dict(total=sum(weighted.values()),raw=dict(terms),weighted=weighted)


def packing_geometry_loss(coords,chemistry,*,weights,overlap_allowance,clash_scale,
                          chunk_size=256,checkpoint_chunks=True):
    """Original combined API retained independently from the repair entry point."""
    common=_common(chemistry)
    terms=dict(clash=steric_clash_loss(coords,chemistry.radii,excluded_pairs=chemistry.excluded_pairs,
        overlap_allowance=overlap_allowance,scale=clash_scale,chunk_size=chunk_size,checkpoint_chunks=checkpoint_chunks,**common),
        bond=bond_loss(coords,chemistry)[0],angle=angle_loss(coords,chemistry)[0])
    combined=combine_physical_losses(terms,weights=weights)
    return dict(**terms,total=combined['total'],weighted=combined['weighted'],chemistry_metadata=chemistry.metadata)
