"""Versioned canonical heavy-atom chemistry and packed-coordinate mapping.

Topology precedes coordinate masks; peptide/disulfide links are explicit. No
native labels or predicted distances determine reference geometry. The shipped
CCD snapshot is the single reference for repair training and calibration.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence
import torch
from .instantiate import STD_AA_3, sidechain_atoms

CHEMISTRY_VERSION = 'canonical_heavy_ccd_v1'
_FIXED_NAMES = frozenset(('N', 'CA', 'C', 'O', 'OXT'))
SC_INTERNAL, SC_ATTACHMENT, CROSS_RESIDUE = 0, 1, 2


@dataclass(frozen=True)
class GeometryConfig:
    """Explicit lengths in angstrom; angle tolerance in degrees, scale in cosine units."""
    bond_tolerance: float
    bond_scale: float
    angle_tolerance_deg: float
    angle_scale: float

    def __post_init__(self):
        for name in ('bond_tolerance', 'bond_scale', 'angle_tolerance_deg', 'angle_scale'):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0 or ('scale' in name and value == 0):
                raise ValueError(f'{name} must be finite and positive (tolerances may be zero)')
        if self.angle_tolerance_deg > 180:
            raise ValueError('angle tolerance cannot exceed 180 degrees')


@dataclass(frozen=True)
class ResidueChemistry:
    name: str
    atom_names: tuple[str, ...]
    elements: tuple[str, ...]
    radii: tuple[float, ...]
    ideal_xyz: tuple[tuple[float, float, float], ...]
    bonds: tuple[tuple[str, str], ...]
    bond_orders: tuple[str, ...]
    ideal_lengths: tuple[float, ...]
    angles: tuple[tuple[str, str, str], ...]
    ideal_angles_rad: tuple[float, ...]


def _angle(a, centre, b):
    u, v = [tuple(x-y for x, y in zip(p, centre)) for p in (a, b)]
    norm = math.sqrt(sum(x*x for x in u) * sum(x*x for x in v))
    if norm <= 1e-12:
        raise ValueError('degenerate ideal angle')
    return math.acos(max(-1., min(1., sum(x*y for x, y in zip(u, v)) / norm)))


@lru_cache(maxsize=1)
def _snapshot():
    raw = Path(__file__).with_name('canonical_chemistry.json').read_bytes()
    data = json.loads(raw)
    if data['schema_version'] != CHEMISTRY_VERSION:
        raise ValueError('unsupported canonical chemistry schema')
    return data, hashlib.sha256(raw).hexdigest()


def registry_sha256():
    return _snapshot()[1]


@lru_cache(maxsize=1)
def canonical_registry() -> Mapping[str, ResidueChemistry]:
    data, _ = _snapshot()
    records = {}
    radius_table = data['radius_source']['values']
    if set(data['residues']) != set(STD_AA_3):
        raise ValueError('snapshot must contain exactly the 20 canonical amino acids')
    for name in STD_AA_3:
        record = data['residues'][name]
        names = tuple(a['name'] for a in record['atoms'])
        if names != ('N', 'CA', 'C', 'O', *sidechain_atoms(name), 'OXT') or len(names) != len(set(names)):
            raise ValueError(f'{name}: atom names/order differ from instantiate.py')
        xyz = {a['name']: tuple(a['ideal_xyz']) for a in record['atoms']}
        if any(len(p) != 3 or not all(math.isfinite(x) for x in p) for p in xyz.values()):
            raise ValueError(f'{name}: invalid ideal coordinates')
        elements = tuple(a['element'] for a in record['atoms'])
        radii = tuple(float(radius_table[e]) for e in elements)
        if any(not math.isfinite(r) or r <= 0 for r in radii):
            raise ValueError('invalid radius table')
        bonds = tuple(tuple(b['atoms']) for b in record['bonds'])
        if len({tuple(sorted(pair)) for pair in bonds}) != len(bonds):
            raise ValueError(f'{name}: duplicate bonds')
        graph = {a: set() for a in names}
        for a, b in bonds:
            if a == b or a not in graph or b not in graph:
                raise ValueError(f'{name}: invalid bond {a,b}')
            graph[a].add(b); graph[b].add(a)
        if any(not neighbors for neighbors in graph.values()):
            raise ValueError(f'{name}: disconnected atom')
        angles = tuple((a, centre, b) for centre in names for a, b in combinations(sorted(graph[centre]), 2))
        lengths = tuple(math.dist(xyz[a], xyz[b]) for a, b in bonds)
        if any(length <= 0 for length in lengths):
            raise ValueError(f'{name}: zero ideal bond')
        records[name] = ResidueChemistry(name, names, elements, radii,
            tuple(xyz[a] for a in names), bonds, tuple(b['order'] for b in record['bonds']),
            lengths, angles, tuple(_angle(xyz[a], xyz[c], xyz[b]) for a, c, b in angles))
    return MappingProxyType(records)


@dataclass(frozen=True)
class PackedAtom:
    residue_uid: str
    atom_name: str
    valid: bool = True
    generated: bool = False
    element: str | None = None


@dataclass(frozen=True)
class CovalentLink:
    first: tuple[str, str]
    second: tuple[str, str]
    ideal_length: float | None = None


@dataclass(frozen=True)
class LinkAngle:
    atoms: tuple[tuple[str, str], tuple[str, str], tuple[str, str]]
    ideal_angle_rad: float


@dataclass(frozen=True)
class PackingChemistry:
    radii: torch.Tensor
    valid_mask: torch.Tensor
    subject_mask: torch.Tensor
    group_id: torch.Tensor
    bond_idx: torch.Tensor
    ideal_lengths: torch.Tensor
    bond_tolerance: torch.Tensor
    bond_scale: torch.Tensor
    angle_idx: torch.Tensor
    cos_min: torch.Tensor
    cos_max: torch.Tensor
    angle_scale: torch.Tensor
    excluded_pairs: torch.Tensor
    metadata: dict
    bond_class: torch.Tensor
    angle_class: torch.Tensor
    bond_valid: torch.Tensor
    angle_valid: torch.Tensor
    ideal_angles_rad: torch.Tensor

    def to(self, device):
        return PackingChemistry(**{f.name: (getattr(self, f.name).to(device)
            if torch.is_tensor(getattr(self, f.name)) else getattr(self, f.name)) for f in fields(self)})


def build_packing_chemistry(
    residue_types: Sequence[Mapping[str, str | int]], atom_records: Sequence[Sequence[PackedAtom]],
    *, config: GeometryConfig, covalent_links: Sequence[Sequence[CovalentLink]] | None = None,
    link_angles: Sequence[Sequence[LinkAngle]] | None = None, device='cpu',
) -> PackingChemistry:
    """Extend the supplied full-topology builder with ownership classes/validity.

    Generated SC inventories must be complete even with incomplete observations.
    Classes describe chemical SC/BB ownership, independently of generation masks.
    Cross-residue constraints have their own class and are excluded from repair.
    Invalid anchors remove only constraints that touch them, never graph edges.
    """
    batch = len(atom_records)
    if not batch or len(residue_types) != batch or any(not rows for rows in atom_records):
        raise ValueError('one nonempty atom-record list and residue mapping required per batch item')
    links = [[] for _ in range(batch)] if covalent_links is None else covalent_links
    cross_angles = [[] for _ in range(batch)] if link_angles is None else link_angles
    if len(links) != batch or len(cross_angles) != batch:
        raise ValueError('link metadata must match batch size')
    registry = canonical_registry()
    n = max(map(len, atom_records))
    valid = torch.zeros(batch, n, dtype=torch.bool)
    subject = torch.zeros_like(valid)
    radii = torch.zeros(batch, n)
    groups = torch.full((batch, n), -1, dtype=torch.long)
    bond_rows, bond_targets, angle_rows, angle_targets, exclusion_rows = [], [], [], [], []
    bond_classes, angle_classes = [], []
    bond_valid_rows, angle_valid_rows = [], []
    for b, (identities, rows) in enumerate(zip(residue_types, atom_records)):
        records = {}
        for uid, name in identities.items():
            if not isinstance(uid, str) or not uid:
                raise ValueError('residue UIDs must be nonempty strings')
            if isinstance(name, int) and not isinstance(name, bool) and 0 <= name < 20:
                name = STD_AA_3[name]
            if name not in registry:
                raise ValueError(f'unsupported residue identity: {name}')
            records[uid] = registry[name]
        group_lookup = {uid: i for i, uid in enumerate(sorted(records))}
        graph = {(uid, atom): set() for uid, rec in records.items() for atom in rec.atom_names}
        for uid, rec in records.items():
            for first, second in rec.bonds:
                graph[uid, first].add((uid, second)); graph[uid, second].add((uid, first))
        lookup = {}
        for i, atom in enumerate(rows):
            key = (atom.residue_uid, atom.atom_name)
            if key not in graph or key in lookup:
                raise ValueError(f'unknown or duplicated packed atom: {key}')
            if not isinstance(atom.valid, bool) or not isinstance(atom.generated, bool):
                raise ValueError('PackedAtom masks must be bool')
            if atom.generated and (not atom.valid or atom.atom_name in _FIXED_NAMES):
                raise ValueError('generated atoms must be valid side-chain atoms')
            rec = records[atom.residue_uid]
            slot = rec.atom_names.index(atom.atom_name)
            if atom.element is not None and atom.element != rec.elements[slot]:
                raise ValueError(f'element mismatch for {key}')
            lookup[key] = i
            valid[b, i], subject[b, i] = atom.valid, atom.generated
            radii[b, i], groups[b, i] = rec.radii[slot], group_lookup[atom.residue_uid]
        for uid in {atom.residue_uid for atom in rows if atom.generated}:
            expected = set(sidechain_atoms(records[uid].name))
            actual = {atom.atom_name for atom in rows if atom.residue_uid == uid and atom.generated}
            if actual != expected:
                raise ValueError(f'{uid}: incomplete generated SC inventory')
        seen_links = set()
        for link in links[b]:
            a, c = link.first, link.second
            if a not in graph or c not in graph or a == c or a[0] == c[0]:
                raise ValueError(f'invalid cross-residue covalent link: {link}')
            key = tuple(sorted((a, c)))
            if key in seen_links:
                raise ValueError('duplicate cross-residue covalent link')
            seen_links.add(key); graph[a].add(c); graph[c].add(a)
        bonds, lengths, angles, theta, bc, ac, bv, av = [], [], [], [], [], [], [], []
        def mapped(keys):
            if any(k not in lookup for k in keys):
                return None
            return tuple(lookup[k] for k in keys)
        def ownership(keys):
            if len({key[0] for key in keys}) > 1:
                return CROSS_RESIDUE
            return SC_ATTACHMENT if any(key[1] in _FIXED_NAMES for key in keys) else SC_INTERNAL
        def add_bond(keys, ideal):
            indices = mapped(keys)
            if indices is None or not any(rows[i].generated for i in indices):
                return
            if ideal is None or not math.isfinite(ideal) or ideal <= 0:
                raise ValueError('generated cross-link bonds require a positive explicit ideal_length')
            bonds.append(indices); lengths.append(ideal); bc.append(ownership(keys))
            bv.append(all(rows[i].valid for i in indices))
        def add_angle(keys, ideal):
            indices = mapped(keys)
            if indices is None or not any(rows[i].generated for i in indices):
                return
            if not math.isfinite(ideal) or not 0 < ideal < math.pi:
                raise ValueError('ideal angle must lie strictly between 0 and pi')
            angles.append(indices); theta.append(ideal); ac.append(ownership(keys))
            av.append(all(rows[i].valid for i in indices))
        for uid, rec in records.items():
            for pair, ideal in zip(rec.bonds, rec.ideal_lengths):
                add_bond(tuple((uid, a) for a in pair), ideal)
            for triple, ideal in zip(rec.angles, rec.ideal_angles_rad):
                add_angle(tuple((uid, a) for a in triple), ideal)
        for link in links[b]:
            add_bond((link.first, link.second), link.ideal_length)
        seen_angles = set()
        for angle in cross_angles[b]:
            a, centre, c = angle.atoms
            if a not in graph or centre not in graph or c not in graph or a == c or len({a[0], centre[0], c[0]}) == 1:
                raise ValueError('invalid cross-residue angle')
            if a not in graph[centre] or c not in graph[centre]:
                raise ValueError('link angle must follow explicit bonds')
            key = (min(a, c), centre, max(a, c))
            if key in seen_angles:
                raise ValueError('duplicate link angle')
            seen_angles.add(key); add_angle(angle.atoms, angle.ideal_angle_rad)
        exclusions = set()
        for atom, i in lookup.items():
            reachable = set(graph[atom])
            for middle in graph[atom]:
                reachable.update(graph[middle])
            for other in reachable:
                if other in lookup and i < lookup[other]:
                    exclusions.add((i, lookup[other]))
        bond_rows.append(bonds); bond_targets.append(lengths); bond_classes.append(bc)
        angle_rows.append(angles); angle_targets.append(theta); angle_classes.append(ac)
        bond_valid_rows.append(bv); angle_valid_rows.append(av)
        exclusion_rows.append(sorted(exclusions))
    def pad_indices(items, arity):
        out = torch.full((batch, max(map(len, items), default=0), arity), -1, dtype=torch.long)
        for b, row in enumerate(items):
            if row: out[b, :len(row)] = torch.tensor(row, dtype=torch.long)
        return out
    def pad_values(items, fill=0.):
        out = torch.full((batch, max(map(len, items), default=0)), float(fill))
        for b, row in enumerate(items):
            if row: out[b, :len(row)] = torch.tensor(row)
        return out
    ideals, theta = pad_values(bond_targets, 1.), pad_values(angle_targets)
    bi, ai = pad_indices(bond_rows, 2), pad_indices(angle_rows, 3)
    tol = math.radians(config.angle_tolerance_deg)
    topology_payload = dict(residues=list(residue_types),
        atoms=[[asdict(atom) for atom in rows] for rows in atom_records],
        links=[[asdict(link) for link in item] for item in links],
        angles=[[asdict(angle) for angle in item] for item in cross_angles])
    topology_digest = hashlib.sha256(json.dumps(topology_payload, sort_keys=True).encode()).hexdigest()
    tables = PackingChemistry(radii, valid, subject, groups, bi, ideals,
        torch.full_like(ideals, config.bond_tolerance), torch.full_like(ideals, config.bond_scale),
        ai, (theta+tol).clamp(max=math.pi).cos(), (theta-tol).clamp(min=0.).cos(),
        torch.full_like(theta, config.angle_scale), pad_indices(exclusion_rows, 2),
        dict(version=CHEMISTRY_VERSION, registry_sha256=registry_sha256(), topology_sha256=topology_digest,
             geometry_config=asdict(config), generated_atoms=subject.sum(-1).tolist(),
             bond_counts=list(map(len, bond_rows)), angle_counts=list(map(len, angle_rows)),
             cross_link_angles='explicit_only', radius_source=_snapshot()[0]['radius_source']),
        pad_values(bond_classes, -1).long(), pad_values(angle_classes, -1).long(),
        pad_values(bond_valid_rows).bool(), pad_values(angle_valid_rows).bool(), theta)
    return tables.to(device)
