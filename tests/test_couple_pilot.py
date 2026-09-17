"""The pilot's cache and supervision mask.

The cache is the thing that makes three comparable arms affordable, and it is
also the thing that could quietly make them incomparable: a state built with one
donor, one packing length or one BB->SC policy is not the state another arm
should read. So the refusal is tested, not the reuse.
"""

import pytest
import torch

from pxf import atom37
from pxf.couple import pilot


def identity(**overrides):
    base = dict(
        frozen={"fampnn": "0.0", "pxdesign": "v0.1.0"},
        pack_steps=50,
        bs_policy="bypass",
        sigma_schedule={"mode": "trajectory", "sigma_min": 0.1, "sigma_max": 2.0},
        seed_base=0,
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
        ("frozen", {"fampnn": "0.3"}),
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
    assert identity()["version"] == "sb-pilot-upstream-v1"
