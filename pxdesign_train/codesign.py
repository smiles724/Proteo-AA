"""Shared masked sequence/one-step packing/backbone revision state machine.

Continuous coordinate/feature routes retain autograd. Hard AA commitments and
atom inventory selection are discrete. Targets are deliberately absent here.
"""
from dataclasses import dataclass, replace
from typing import Callable
import torch
from .aa.atom_mapping import BB37, scatter_named_atoms
from .aa.masking import visible_input, assign_aa
from .sidechain.frames import gather_backbone, frames_from_backbone_index, to_local, to_global
from .sidechain.instantiate import instantiate_from_type_indices


@dataclass(frozen=True)
class CoDesignState:
    backbone_xyz: torch.Tensor           # [B,S,N_atom,3], fixed + generated BB
    backbone_features: dict
    bb_atom_idx: torch.Tensor            # [B,S,L,4], all protein residues
    assigned_aa: torch.Tensor            # [B,S,L], 20 = X
    sc_xyz: torch.Tensor                 # [B,S,L,10,3]
    sc_atom_name_ids: torch.Tensor
    generation_mask: torch.Tensor
    fixed_context_xyz: torch.Tensor      # [B,S,L,37,3]
    fixed_context_mask: torch.Tensor
    fixed_atom_xyz: torch.Tensor         # [B,S,N_atom,3], also ligand/metal
    fixed_atom_mask: torch.Tensor
    residue_mask: torch.Tensor
    design_mask: torch.Tensor
    query_mask: torch.Tensor
    seq_visible: torch.Tensor
    sc_visible: torch.Tensor
    residue_index: torch.Tensor
    chain_index: torch.Tensor
    state_version: int = 0
    feedback: dict = None

    def updated(self, **kwargs):
        return replace(self, state_version=self.state_version + 1, **kwargs)

    def aa_input(self, sc_feedback=True):
        # Clear queried SC names BEFORE the mapping; even corrupt stored query
        # inventories cannot affect the visible structural representation.
        permitted = self.sc_visible & self.seq_visible & ~self.query_mask & self.design_mask
        if not sc_feedback:
            permitted = torch.zeros_like(permitted)
        mask = self.generation_mask & permitted[..., None]
        ids = torch.where(mask, self.sc_atom_name_ids, 0)
        sc, sc_mask = scatter_named_atoms(self.sc_xyz, ids, mask)
        xyz = torch.where(self.fixed_context_mask[..., None], self.fixed_context_xyz, sc)
        atom_mask = self.fixed_context_mask | sc_mask
        bb, present = gather_backbone(self.backbone_xyz, self.bb_atom_idx)
        bb_index = torch.tensor(BB37, device=bb.device)
        xyz = xyz.index_copy(-2, bb_index, bb)
        atom_mask = atom_mask.index_copy(-1, bb_index, present)
        out = visible_input(xyz, atom_mask, self.assigned_aa, self.residue_mask,
            self.design_mask, self.query_mask, self.seq_visible, self.sc_visible)
        return dict(**out, residue_index=self.residue_index, chain_encoding=self.chain_index)

    def commit(self, aa, selected):
        changed = selected & (aa != self.assigned_aa)
        assigned = torch.where(selected, aa, self.assigned_aa)
        # Every queried chain was hidden. It becomes SC context only after pack.
        clear = changed | self.query_mask
        return self.updated(assigned_aa=assigned, seq_visible=self.seq_visible | selected,
            query_mask=self.query_mask & ~selected,
            sc_visible=self.sc_visible & ~clear,
            sc_xyz=torch.where(clear[..., None, None], 0.0, self.sc_xyz),
            sc_atom_name_ids=torch.where(clear[..., None], 0, self.sc_atom_name_ids),
            generation_mask=self.generation_mask & ~clear[..., None], feedback=None)

    def on_new_backbone(self, xyz, features):
        xyz = torch.where(self.fixed_atom_mask[..., None], self.fixed_atom_xyz, xyz)
        old_R, old_t, old_ok = frames_from_backbone_index(self.backbone_xyz, self.bb_atom_idx)
        new_R, new_t, new_ok = frames_from_backbone_index(xyz, self.bb_atom_idx)
        keep = old_ok & new_ok
        transported = to_global(to_local(self.sc_xyz, old_R, old_t), new_R, new_t)
        mask = self.generation_mask & keep[..., None]
        return self.updated(backbone_xyz=xyz, backbone_features=features,
            sc_xyz=torch.where(mask[..., None], transported, 0.0), generation_mask=mask,
            sc_visible=self.sc_visible & keep, feedback=None)

    def validate_final(self):
        if (self.design_mask & ((self.assigned_aa < 0) | (self.assigned_aa >= 20))).any():
            raise ValueError("Final design contains unknown identities")
        ids, chemistry = instantiate_from_type_indices(self.assigned_aa)
        _, _, frame_ok = frames_from_backbone_index(self.backbone_xyz, self.bb_atom_idx)
        expected = chemistry & (self.design_mask & frame_ok)[..., None]
        if not torch.equal(expected, self.generation_mask):
            raise ValueError("Final atom inventory does not match sequence and frames")
        if (self.design_mask & ~frame_ok).any():
            raise ValueError("Final design contains invalid backbone frames")
        if not torch.equal(self.sc_atom_name_ids, torch.where(expected, ids, 0)):
            raise ValueError("Final atom names do not match the assigned sequence")
        if not torch.isfinite(self.sc_xyz[self.generation_mask]).all():
            raise ValueError("Final generated atoms are not finite")
        if not torch.equal(self.backbone_xyz[self.fixed_atom_mask], self.fixed_atom_xyz[self.fixed_atom_mask]):
            raise ValueError("Fixed coordinates changed")


@dataclass(frozen=True)
class CycleConfig:
    rounds: int = 3
    decode_blocks: int = 4
    query_fraction: float = 0.5
    whole_mask_probability: float = 0.1
    temperature: float = 0.0
    sc_to_aa: bool = True
    sc_to_bb: bool = True

    def __post_init__(self):
        if self.rounds < 1 or self.decode_blocks < 1:
            raise ValueError("At least one training round and decoding block are required")
        if not 0 < self.query_fraction <= 1 or not 0 <= self.whole_mask_probability <= 1:
            raise ValueError("Invalid query schedule")


def random_query(design_mask, cfg, generator=None):
    draws = torch.rand(design_mask.shape, device=design_mask.device, generator=generator)
    query = (draws < cfg.query_fraction) & design_mask
    whole = torch.rand(design_mask.shape[:-1], device=design_mask.device, generator=generator) < cfg.whole_mask_probability
    query = torch.where(whole[..., None], design_mask, query)
    # A nonempty design always queries at least one residue, including small crops.
    first = draws.masked_fill(~design_mask, 2).argmin(-1)
    fallback = torch.nn.functional.one_hot(first, design_mask.shape[-1]).bool() & design_mask
    return torch.where(query.any(-1, keepdim=True), query, fallback)


def decode(state, head, cfg, generator=None):
    query = state.query_mask
    state = state.updated(seq_visible=state.seq_visible & ~query, sc_visible=state.sc_visible & ~query)
    draws = torch.rand(query.shape, device=query.device, generator=generator).masked_fill(~query, 2)
    order = draws.argsort(-1).argsort(-1)
    count = query.sum(-1, keepdim=True)
    records = []
    for block in range(cfg.decode_blocks):
        selected = query & (order >= count * block // cfg.decode_blocks) & (order < count * (block + 1) // cfg.decode_blocks)
        if not selected.any():
            continue
        logits, _ = head(**state.aa_input(sc_feedback=cfg.sc_to_aa))
        records.append((logits, selected))
        aa = assign_aa(logits, cfg.temperature, generator)
        state = state.commit(aa, selected)
    return state, records


def run_cycle(state, head, pack: Callable, refine: Callable, cfg: CycleConfig,
              generator=None, queries=None):
    """Initial full decode/pack, R complete AA/SC/BB updates, final repack.

    Supplied query masks and RNG state give training/evaluation forward parity.
    No native target or auxiliary branch is accepted by this function.
    """
    state = state.updated(query_mask=state.design_mask)
    state, initial = decode(state, head, cfg, generator)
    state = pack(state)
    revisions, trace = [], []
    covered = torch.zeros_like(state.design_mask)
    for round_index in range(cfg.rounds):
        query = queries[round_index] if queries is not None else random_query(state.design_mask, cfg, generator)
        if (query & ~state.design_mask).any():
            raise ValueError("Revision selected a permanently fixed residue")
        old_aa, old_xyz = state.assigned_aa, state.backbone_xyz
        state = state.updated(query_mask=query)
        state, records = decode(state, head, cfg, generator)
        revisions.extend(records)
        state = pack(state)
        state = refine(state, cfg.sc_to_bb)
        covered = covered | query
        old_bb, _ = gather_backbone(old_xyz, state.bb_atom_idx)
        new_bb, _ = gather_backbone(state.backbone_xyz, state.bb_atom_idx)
        ca_shift = (new_bb[..., 1, :] - old_bb[..., 1, :]).square().sum(-1)
        rms_shift = ((ca_shift * state.design_mask).sum(-1) / state.design_mask.sum(-1).clamp_min(1)).sqrt()
        trace.append(dict(round=round_index + 1, query_mask=query.detach(),
            aa_changes=((old_aa != state.assigned_aa) & query).detach(),
            query_count=query.sum(-1).tolist(), coverage_count=covered.sum(-1).tolist(),
            change_count=((old_aa != state.assigned_aa) & query).sum(-1).tolist(),
            binder_ca_displacement=rms_shift.detach().tolist(), state_version=state.state_version))
    state = pack(state)  # final sequence is fixed before this call
    state.validate_final()
    return state, dict(initial=initial, revisions=revisions, trace=trace, stop_reason="fixed_budget")
