"""The pilot's cache and supervision mask.

The cache is the thing that makes three comparable arms affordable, and it is
also the thing that could quietly make them incomparable: a state built with one
donor, one packing length or one BB->SC policy is not the state another arm
should read. So the refusal is tested, not the reuse.
"""

import copy

import pytest
import torch

from pxf import atom37
from pxf.couple import pilot

FROZEN = {
    "fampnn": {"sha256": "aaaa", "variant": "0.0"},
    "backbone_driver": "pxdesign",
    "upstream": {"fampnn": {"revision": "abc"}},
    "pxdesign": {
        "weights": {"sha256": "bbbb"},
        "proteoaa": {"revision": "def"},
        # Descriptive: recorded for provenance, does not change any state.
        "driver_settings": {
            "c_token": 768,
            "sigma_data": 16.0,
            "activation_checkpointing": False,
            "blocks_cleared": 4,
            "chunk_size": None,
            "inplace_safe": False,
        },
    },
}


def identity(**overrides):
    base = dict(
        frozen=FROZEN,
        pack_steps=50,
        bs_policy="bypass",
        sigma_schedule={"mode": "trajectory", "sigma_min": 0.1, "sigma_max": 2.0},
        seed_base=0,
        crop_size=512,
    )
    base.update(overrides)
    return pilot.cache_identity(**base)


# --- the supervision mask ---------------------------------------------------


def test_the_mask_keeps_only_backbone_names():
    names = ["N", "CA", "C", "O", "CB", "CG", "OXT"]
    mask = pilot.backbone_supervision_mask(names)
    assert mask.tolist() == [1, 1, 1, 1, 0, 0, 0]
    assert set(atom37.BACKBONE_ATOMS) == {"N", "CA", "C", "O"}


def test_unresolved_atoms_are_excluded():
    names = ["N", "CA", "C", "O"]
    mask = pilot.backbone_supervision_mask(
        names, coordinate_mask=torch.tensor([1.0, 1.0, 0.0, 1.0])
    )
    assert mask.tolist() == [1, 1, 0, 1]


def test_a_mismatched_coordinate_mask_is_refused():
    with pytest.raises(ValueError, match="covers 3 atoms"):
        pilot.backbone_supervision_mask(
            ["N", "CA", "C", "O"], coordinate_mask=torch.ones(3)
        )


# --- the cache --------------------------------------------------------------


def test_the_key_distinguishes_structure_sigma_and_seed():
    key = pilot.UpstreamCache.key
    assert key("a", 0.5, 1) != key("b", 0.5, 1)
    assert key("a", 0.5, 1) != key("a", 0.6, 1)
    assert key("a", 0.5, 1) != key("a", 0.5, 2)
    # Formatted, not repr'd: 0.5 and 0.50 are the same noise level.
    assert key("a", 0.5, 1) == key("a", 0.50, 1)


def test_hits_and_misses_are_counted():
    cache = pilot.UpstreamCache(identity=identity())
    assert cache.get("a", 0.5, 1) is None
    cache.put("a", 0.5, 1, "state")
    assert cache.get("a", 0.5, 1) == "state"
    stats = cache.stats()
    assert stats["hits"] == 1 and stats["misses"] == 1 and stats["entries"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("pack_steps", 10),
        ("bs_policy", "matched"),
        ("seed_base", 7),
        # A different FaMPNN donor, in the shape provenance.weight_record emits.
        ("frozen", {**FROZEN, "fampnn": {"sha256": "different", "variant": "0.3"}}),
    ],
)
def test_an_incompatible_cache_is_refused_not_warned(tmp_path, field, value):
    """Each of these changes what the frozen half *is*."""
    cache = pilot.UpstreamCache(identity=identity())
    path = cache.save(tmp_path / "cache.pt")
    with pytest.raises(ValueError, match="different frozen components"):
        pilot.UpstreamCache.load(path, identity=identity(**{field: value}))
    # ... and the matching one loads.
    assert pilot.UpstreamCache.load(path, identity=identity()).identity == cache.identity


def test_the_fingerprint_changes_with_the_identity():
    assert identity() == identity()
    a = pilot.UpstreamCache(identity=identity())
    b = pilot.UpstreamCache(identity=identity(pack_steps=10))
    assert a.fingerprint != b.fingerprint
    assert a.compatible(b.identity)


def test_the_version_is_part_of_the_identity():
    """So a change to what an UpstreamState holds invalidates old caches."""
    assert identity()["version"] == "sb-pilot-upstream-v2"


def test_descriptive_provenance_does_not_invalidate_a_cache():
    """The identity tracks what changes a state, not what a run records.

    Hashing the whole `frozen` blob made this sensitive to anything a run
    happened to log about itself: adding a descriptive driver_settings block
    refused every existing cache with a 40-line diff, though none of those
    fields alters a frozen state. Found by it happening.
    """
    import copy

    relabelled = copy.deepcopy(FROZEN)
    relabelled["pxdesign"]["driver_settings"]["blocks_cleared"] = 99
    relabelled["pxdesign"]["driver_settings"]["c_token"] = 1024
    relabelled["pxdesign"]["driver_settings"]["activation_checkpointing"] = True
    assert identity(frozen=relabelled) == identity()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda f: f["fampnn"].update(sha256="different"),
        lambda f: f["pxdesign"]["weights"].update(sha256="different"),
        lambda f: f["pxdesign"]["proteoaa"].update(revision="different"),
        lambda f: f.update(upstream={"fampnn": {"revision": "different"}}),
        lambda f: f.update(backbone_driver="stub"),
        # These two can move a float in the forward pass, so they must count.
        lambda f: f["pxdesign"]["driver_settings"].update(chunk_size=4),
        lambda f: f["pxdesign"]["driver_settings"].update(inplace_safe=True),
    ],
)
def test_anything_that_changes_a_state_does_invalidate_it(mutate):
    """The subset has to be wrong in neither direction."""
    changed = copy.deepcopy(FROZEN)
    mutate(changed)
    assert identity(frozen=changed) != identity()


def test_the_crop_size_is_part_of_the_identity():
    """It changes the featurization, so it changes the states.

    It was missing: only the cache *filename* carried it, so two runs at
    different crops pointed at the same path would have reused each other's
    states without complaint.
    """
    assert identity(crop_size=256) != identity(crop_size=512)


# --- .to(device) must move everything, not almost everything ----------------


def _tensor_devices(obj, prefix=""):
    """Every tensor reachable from a dataclass, as ``{path: device}``."""
    import dataclasses

    found = {}
    for field in dataclasses.fields(obj):
        value = getattr(obj, field.name)
        path = f"{prefix}{field.name}"
        if torch.is_tensor(value):
            found[path] = value.device.type
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            found.update(_tensor_devices(value, prefix=f"{path}."))
    return found


def test_upstream_state_to_moves_every_tensor_it_carries(monkeypatch):
    """A partial .to() is invisible until something uses the missed tensor.

    ``inputs`` was not moved. Nothing used those tensors with a model after the
    state came back from its CPU cache, so they sat on the wrong device
    indefinitely; the first arm that re-encoded from a cached state on a GPU
    failed inside FaMPNN's positional embedding, two hundred lines from the
    cause. This walks the whole object rather than naming fields, so the next
    field added is covered without anyone remembering to add it here.
    """
    state = _synthetic_upstream()
    before = _tensor_devices(state)
    assert "inputs.coords_af2" in before, "the walk is not reaching inputs"
    assert "packed.h_packed" in before, "the walk is not reaching packed"
    moved = state.to("cpu")  # no GPU needed: the point is that nothing is MISSED
    after = _tensor_devices(moved)
    assert set(after) == set(before), "a tensor disappeared from the state"
    assert all(d == "cpu" for d in after.values())
    # And the same walk on a real cached state would have caught the bug: every
    # path present before must be present after, including nested ones.
    assert len([k for k in after if k.startswith("inputs.")]) >= 5


def _synthetic_upstream():
    """The smallest UpstreamState with a populated `inputs` and `packed`."""
    from pxf.couple.converter import CoupledInputs
    from pxf.couple.pilot import UpstreamState
    from pxf.couple.visibility import PackedStructure, Visibility

    length = 4
    zeros37 = torch.zeros(1, length, 37)
    inputs = CoupledInputs(
        coords_af2=torch.zeros(1, length, 37, 3),
        atom_mask=zeros37.clone(),
        aatype=torch.zeros(1, length, dtype=torch.long),
        seq_mask=torch.ones(1, length),
        missing_atom_mask=zeros37.clone(),
        residue_index=torch.arange(length)[None],
        chain_index=torch.zeros(1, length, dtype=torch.long),
        design_mask=torch.ones(1, length, dtype=torch.bool),
        sequence_known=torch.ones(1, length, dtype=torch.bool),
        num_tokens=length,
    )
    visibility = Visibility(
        available=zeros37.clone(),
        missing_atom_mask=zeros37.clone(),
        frame_valid=torch.ones(1, length, dtype=torch.bool),
        sidechain_visible=torch.ones(1, length),
        exists=zeros37.clone(),
        stats={},
    )
    packed = PackedStructure(
        h_packed=torch.zeros(1, length, 8),
        coords37=torch.zeros(1, length, 37, 3),
        aatype=torch.zeros(1, length, dtype=torch.long),
        seq_mask=torch.ones(1, length),
        visibility=visibility,
        h_base=torch.zeros(1, length, 8),
    )
    return UpstreamState(
        packed=packed,
        bb0_flat=torch.zeros(1, length * 4, 3),
        a_token=torch.zeros(1, length, 16),
        delta_h=None,
        sidechains=torch.zeros(1, length, 33, 3),
        inputs=inputs,
        sigma=torch.tensor([0.5]),
    )


def test_coupled_inputs_to_moves_every_tensor():
    state = _synthetic_upstream()
    devices = _tensor_devices(state.inputs.to("cpu"))
    assert len(devices) == 9, f"expected every tensor field, got {sorted(devices)}"
    assert all(d == "cpu" for d in devices.values())
