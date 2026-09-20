"""The data contract for side-chain-supervised backbone fine-tuning.

One training example has to satisfy two modules that parsed the same file
differently. PXDesign sees it through ``pxdesign_train``'s featurizer: a flat
atom axis, a design region whose side chains are scrubbed onto CA, and
coordinates in whatever pose the featurizer produced. FaMPNN sees it through its
own parser: dense atom37, every native side chain, in the file's pose. The
backbone prediction lives in the first frame and the side-chain targets live in
the second, so a loss that compares them has to reconcile the two first.

What this module owns:

**The pose.** The featurizer centers the structure; FaMPNN's parse does not.
Scoring across that gap reports ~20 A on a perfectly good packing, which is how
the problem was found in the evaluator. :func:`recover_preprocessing_transform`
recovers the rigid map from matched *native backbone* atoms -- a function of the
two parses only, never of a model prediction -- and
:func:`align_native_to_features` applies it to the whole native atom37 block. A
residual above tolerance means preprocessing changed the geometry rather than
the pose, which is a rejected example, not something to fit away.

**The correspondence.** Residues are matched on stable keys (chain, residue
number), not on equal length or equal sequence: repeated chains can have
identical sequences in different places. Sequence equality is then asserted as a
second, independent check rather than used as the correspondence itself.

**The three masks**, which are genuinely different sets and are the thing most
likely to be silently conflated:

``encoder``    what the FaMPNN encoder may see. Derived from the *predicted*
    backbone's atom set, with every side-chain slot invisible. It says nothing
    about which atoms are supervised, and reusing it as a target mask would
    erase side-chain supervision entirely -- a predicted backbone has no
    observed side chains.
``local``      ``frame_targets(...)["loss_mask"]``: the diffusion target set.
    Includes **ghost** slots, whose local target is the origin, because that is
    what the original training code supervises. Excludes missing atoms and
    padding, and requires a native backbone frame.
``physical``   ``atom_mask * frames_exist``: real, present, quality-eligible
    side-chain atoms only. This is the mask for global placement, chemistry and
    any physical atom count. Using ``local`` there would score ghost slots
    parked at the frame origin against atoms that do not exist.

A quality-vetoed residue keeps ghost-zero supervision in ``local`` while
dropping out of ``physical``. That is intentional -- ``local`` is reproducing
the source objective, which vetoes through ``missing_atom_mask`` -- but it means
the two counts differ for reasons beyond glycine, so both are reported.
"""

from dataclasses import dataclass, field, replace

import torch

from pxf import atom37

# Preprocessing is expected to change pose, not geometry. Float32 coordinates
# through a rotation land well inside this; a real geometry change does not.
DEFAULT_TOLERANCE_ANGSTROM = 1e-2

# Enough matched backbone atoms for a meaningful superposition.
MIN_MATCHED_ATOMS = 12
# A superposition only determines the rotation about an axis the matched points
# actually span. Below this ratio between the smallest and largest principal
# extent, the point set is effectively planar or collinear: Kabsch still returns
# a rotation with a near-zero residual, and it is the wrong one for every atom
# off the degenerate axis.
MIN_EXTENT_RATIO = 1e-3


class PreprocessingMismatch(ValueError):
    """The two parses of one file cannot be reconciled by a rigid transform.

    Raised rather than absorbed: a residual above tolerance means the featurizer
    changed the structure's geometry, so the native side chains do not describe
    the backbone being trained on. The example is skippable; a fitted-away
    mismatch is not detectable later.
    """


def _rc():
    from fampnn.data import residue_constants as rc

    return rc


# ---- the rigid map between the two parses ----------------------------------


def rigid_transform(source, target):
    """Least-squares rotation and translation taking ``source`` onto ``target``.

    Both ``[N, 3]`` and in correspondence. Returns ``(rotation, translation,
    rmsd)`` with ``target ~ source @ rotation + translation``. Computed in
    float64: the residual is the accept/reject signal, so it must not be
    dominated by the arithmetic that produced it.
    """
    if source.shape != target.shape or source.dim() != 2 or source.shape[-1] != 3:
        raise ValueError(
            f"need matched [N, 3] point sets, got {tuple(source.shape)} and "
            f"{tuple(target.shape)}"
        )
    x = source.detach().to(torch.float64)
    y = target.detach().to(torch.float64)
    cx, cy = x.mean(0), y.mean(0)
    cov = (x - cx).T @ (y - cy)
    u, _s, vh = torch.linalg.svd(cov)
    sign = torch.sign(torch.det(u @ vh))
    correction = torch.eye(3, dtype=torch.float64, device=x.device)
    correction[2, 2] = sign
    rotation = u @ correction @ vh
    translation = cy - cx @ rotation
    residual = (x @ rotation + translation) - y
    rmsd = float(residual.pow(2).sum(-1).mean().sqrt())
    return rotation, translation, rmsd


def extent_ratio(points):
    """Smallest over largest principal extent of a point cloud, in ``[0, 1]``.

    Zero for a collinear or planar set. A superposition fitted on such a set has
    an unconstrained rotation about the missing direction, so the residual can
    be ~0 while every off-axis atom is placed wrong -- which is a silent failure,
    because the residual is the only thing that would otherwise be checked.
    """
    centered = points.detach().to(torch.float64)
    centered = centered - centered.mean(0)
    singular = torch.linalg.svdvals(centered)
    largest = float(singular[0])
    if largest <= 0:
        return 0.0
    return float(singular[-1]) / largest


def matched_backbone_atoms(structure, native):
    """Backbone atoms present in both parses, as ``(featurized, native)`` points.

    The flat axis carries ``(atom_name, token)`` pairs; the native parse carries
    ``[L, 37]``. An atom is matched when it is a backbone slot, the featurizer
    resolved it, and the native parse has it too.
    """
    names = [str(n) for n in structure.topology.atom_names]
    tokens = torch.as_tensor(structure.topology.atom_to_token_idx).reshape(-1).long()
    coordinates = structure.backbone_target
    if coordinates is None:
        raise PreprocessingMismatch(
            f"{structure.sample_id}: the featurizer produced no label coordinates, "
            "so there is nothing to align the native structure onto"
        )
    coordinates = coordinates.reshape(-1, 3)
    resolved = structure.label_dict.get("coordinate_mask")
    resolved = (
        torch.ones(coordinates.shape[0], device=coordinates.device)
        if resolved is None
        else torch.as_tensor(resolved).reshape(-1).float()
    )

    slots = {name: index for index, name in enumerate(atom37.ATOM37)}
    native_x = native["x"].reshape(-1, atom37.NUM_ATOM37, 3)
    native_present = 1.0 - native["missing_atom_mask"].reshape(
        -1, atom37.NUM_ATOM37
    )
    exists = _existence(native["aatype"].reshape(-1))
    native_present = native_present * exists

    left, right = [], []
    for atom_index, name in enumerate(names):
        if name not in atom37.BACKBONE_ATOMS or not float(resolved[atom_index]):
            continue
        token = int(tokens[atom_index])
        slot = slots[name]
        if token >= native_x.shape[0] or not float(native_present[token, slot]):
            continue
        left.append(native_x[token, slot])
        right.append(coordinates[atom_index])
    if len(left) < MIN_MATCHED_ATOMS:
        raise PreprocessingMismatch(
            f"{structure.sample_id}: only {len(left)} backbone atoms are present in "
            f"both parses (need {MIN_MATCHED_ATOMS}); the correspondence is too thin "
            "to recover the preprocessing transform"
        )
    return torch.stack(left), torch.stack(right)


def recover_preprocessing_transform(
    structure, native, *, tolerance=DEFAULT_TOLERANCE_ANGSTROM
):
    """The rigid map from the native parse's frame into the featurizer's.

    Depends only on the two parses of the same file. It is deliberately not a
    fit to the model's prediction: a transform that chased the prediction would
    absorb exactly the backbone error the experiment is trying to measure.
    """
    source, target = matched_backbone_atoms(structure, native)
    if not (torch.isfinite(source).all() and torch.isfinite(target).all()):
        raise PreprocessingMismatch(
            f"{structure.sample_id}: a matched backbone coordinate is not finite, "
            "so no transform between the parses is defined"
        )
    ratio = extent_ratio(source)
    if ratio < MIN_EXTENT_RATIO:
        raise PreprocessingMismatch(
            f"{structure.sample_id}: the matched backbone atoms span a degenerate "
            f"volume (extent ratio {ratio:.2e} < {MIN_EXTENT_RATIO:g}), so the "
            "rotation about the missing direction is unconstrained. The residual "
            "would look fine and the side chains would land in the wrong place"
        )
    rotation, translation, rmsd = rigid_transform(source, target)
    if not rmsd == rmsd or rmsd > float(tolerance):  # NaN-safe
        raise PreprocessingMismatch(
            f"{structure.sample_id}: the featurized and native backbones differ by "
            f"{rmsd:.4f} A after optimal superposition, above the {tolerance:g} A "
            "tolerance. Preprocessing changed the geometry, not just the pose, so "
            "the native side chains do not describe this backbone"
        )
    return rotation, translation, rmsd


def align_native_to_features(coords, rotation, translation, *, mask=None):
    """Apply the recovered transform to native ``[..., 37, 3]`` coordinates.

    ``mask`` zeroes absent slots afterwards, so an unfilled slot stays at the
    origin instead of being moved to the translation vector.
    """
    rotation = rotation.to(coords.dtype).to(coords.device)
    translation = translation.to(coords.dtype).to(coords.device)
    moved = coords @ rotation + translation
    if mask is not None:
        moved = moved * mask.to(moved.dtype).unsqueeze(-1)
    return moved


# ---- residue correspondence -------------------------------------------------


def residue_keys(chain_index, residue_index):
    """Stable per-residue identities as ``(chain, number)`` tuples.

    Positional equality is not correspondence: two copies of one chain in an
    assembly have identical sequences and different placements, and matching
    them by order or by sequence would silently pair the wrong atoms.
    """
    chains = torch.as_tensor(chain_index).reshape(-1).tolist()
    numbers = torch.as_tensor(residue_index).reshape(-1).tolist()
    if len(chains) != len(numbers):
        raise ValueError("chain and residue indices differ in length")
    return [(int(c), int(n)) for c, n in zip(chains, numbers)]


def assert_correspondence(structure, native):
    """Check that the two parses describe the same residues, in the same order.

    Keys first, sequence second. The sequence check is redundant when the keys
    agree, which is the point: it is an independent statement about the same
    correspondence rather than the correspondence itself.
    """
    length = int(structure.num_tokens)
    native_length = int(native["aatype"].reshape(-1).shape[0])
    if native_length != length:
        raise PreprocessingMismatch(
            f"{structure.sample_id}: the featurized crop has {length} residues and "
            f"the native parse {native_length}. The pilot trains on whole chains, so "
            "a length difference means the two are not the same selection"
        )
    featurized = atom37.sequence_from_aatype(structure.aatype.reshape(-1)[:length])
    parsed = atom37.sequence_from_aatype(native["aatype"].reshape(-1).long())
    if featurized != parsed:
        differing = sum(1 for a, b in zip(featurized, parsed) if a != b)
        raise PreprocessingMismatch(
            f"{structure.sample_id}: the two parses disagree at {differing} of "
            f"{length} residues, so the side-chain targets would be misaligned"
        )
    keys = residue_keys(
        native["chain_index"].reshape(-1), native["residue_index"].reshape(-1)
    )
    if len(set(keys)) != len(keys):
        duplicates = len(keys) - len(set(keys))
        raise PreprocessingMismatch(
            f"{structure.sample_id}: {duplicates} residues share a (chain, number) "
            "key, so the correspondence is not one-to-one. Assembly-aware keys are "
            "needed before this entry can be used"
        )
    return keys


# ---- masks ------------------------------------------------------------------


def _existence(aatype):
    from fampnn.data.data import get_rc_tensor

    rc = _rc()
    return get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, aatype.long())


def encoder_availability(aatype, seq_mask, backbone_atom_mask):
    """What the encoder may see: predicted backbone atoms, no side chains.

    ``backbone_atom_mask`` is ``[B, L, 37]`` over the *predicted* structure --
    which slots the backbone module actually produced. Every side-chain slot is
    forced to zero here regardless of what the native structure has, because the
    encoder is looking at a prediction that contains none.

    This is an encoder-input mask and nothing else. The supervised set comes
    from the native parse; deriving it from this one would supervise nothing.
    """
    rc = _rc()
    available = backbone_atom_mask.to(torch.float32).clone()
    available = available * _existence(aatype) * seq_mask.unsqueeze(-1)
    available[..., rc.non_bb_idxs] = 0.0
    return available


def supervision_masks(model, native_batch):
    """The local and physical side-chain masks, and the frames they rest on.

    ``local`` is the source objective's target set, ghosts included. ``physical``
    is the real-atom subset, which is what a placement or chemistry term may
    score. They differ by more than glycine: a quality-vetoed side chain is
    missing for ``local`` but its ghost slots survive, while ``physical`` drops
    the residue's real atoms entirely.
    """
    from pxf.train import step as train_step

    targets = train_step.frame_targets(model, native_batch, scn_mlm_mask=None)
    physical = targets["atom_mask"] * targets["frames_exist"].unsqueeze(-1)
    return dict(
        local_target=targets["local"],
        local_mask=targets["loss_mask"],
        physical_mask=physical,
        frames_exist=targets["frames_exist"],
        seq_unk_mask=targets["seq_unk_mask"],
    )


def mask_counts(masks):
    """Per-mask atom counts, and the ghost share of the local target set.

    Logged per run because a backbone gain driven by ghost slots -- predicting
    that an atom does not exist -- is different evidence from one driven by
    physical packing, and the totals are the cheapest way to see the split.
    """
    local = float(masks["local_mask"].sum())
    physical = float(masks["physical_mask"].sum())
    return dict(
        local_atoms=local,
        physical_atoms=physical,
        ghost_atoms=local - physical,
        ghost_fraction=(local - physical) / local if local else 0.0,
    )


# ---- the batch --------------------------------------------------------------


@dataclass
class JointRefinementBatch:
    """One structure, in both modules' conventions, with the masks kept apart."""

    sample_id: str
    # -- backbone side, on the flat atom axis --
    feature_dict: dict
    topology: object
    backbone_target: torch.Tensor  # [N_atom, 3]
    backbone_mask: torch.Tensor  # [N_atom] supervised N/CA/C/O
    # -- side-chain side, dense atom37 in the featurizer's frame --
    native_batch: dict  # x, aatype, seq_mask, missing_atom_mask, residue_index, chain_index
    local_target: torch.Tensor  # [B, L, 33, 3] backbone-local, detached
    local_mask: torch.Tensor  # [B, L, 33] ghosts included
    physical_mask: torch.Tensor  # [B, L, 33] real atoms only
    frames_exist: torch.Tensor  # [B, L]
    # -- provenance and reproducibility --
    residue_keys: list = field(default_factory=list)
    crop_indices: torch.Tensor = None
    split: str = None
    cluster: str = None
    alignment_rmsd: float = 0.0
    quality_policy: str = "fampnn_missing_atom_mask"
    counts: dict = field(default_factory=dict)

    @property
    def length(self):
        return int(self.native_batch["aatype"].shape[-1])

    @property
    def device(self):
        return self.backbone_target.device

    @property
    def backbone_mask_column(self):
        """``[N_atom, 1]``, for multiplying a coordinate tensor."""
        return self.backbone_mask.reshape(-1, 1).to(self.backbone_target.dtype)

    def to(self, device):
        moved = {
            name: value.to(device)
            for name, value in vars(self).items()
            if torch.is_tensor(value)
        }
        moved["feature_dict"] = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in self.feature_dict.items()
        }
        moved["native_batch"] = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in self.native_batch.items()
        }
        moved["topology"] = self.topology.to(device)
        return replace(self, **moved)

    def identity(self):
        """What a run record needs to say about this example."""
        return dict(
            sample_id=self.sample_id,
            length=self.length,
            split=self.split,
            cluster=self.cluster,
            alignment_rmsd=self.alignment_rmsd,
            quality_policy=self.quality_policy,
            **self.counts,
        )


def build_joint_batch(
    model,
    structure,
    native,
    *,
    split=None,
    cluster=None,
    tolerance=DEFAULT_TOLERANCE_ANGSTROM,
):
    """Assemble one :class:`JointRefinementBatch` from the two parses.

    ``structure`` is a :class:`pxf.backbone.driver.FeaturizedStructure`;
    ``native`` is FaMPNN's own parse of the same file
    (``process_single_pdb(load_feats_from_pdb(path))``), unbatched.

    Raises :class:`PreprocessingMismatch` when the two cannot be reconciled, so
    a caller can skip and record the reason.
    """
    from pxf.couple.pilot import backbone_supervision_mask

    keys = assert_correspondence(structure, native)
    rotation, translation, rmsd = recover_preprocessing_transform(
        structure, native, tolerance=tolerance
    )

    device = structure.aatype.device
    batched = {
        key: value.unsqueeze(0).to(device)
        for key, value in native.items()
        if torch.is_tensor(value)
        and key
        in ("x", "aatype", "seq_mask", "missing_atom_mask", "residue_index", "chain_index")
    }
    batched["aatype"] = batched["aatype"].long()
    present = 1.0 - batched["missing_atom_mask"]
    batched["x"] = align_native_to_features(
        batched["x"], rotation, translation, mask=present * _existence(batched["aatype"])
    )

    masks = supervision_masks(model, batched)
    backbone_mask = backbone_supervision_mask(
        structure.topology.atom_names,
        coordinate_mask=structure.label_dict.get("coordinate_mask"),
        device=device,
    )
    batch = JointRefinementBatch(
        sample_id=structure.sample_id,
        feature_dict=structure.feature_dict,
        topology=structure.topology,
        backbone_target=structure.backbone_target,
        backbone_mask=backbone_mask,
        native_batch=batched,
        local_target=masks["local_target"].detach(),
        local_mask=masks["local_mask"].detach(),
        physical_mask=masks["physical_mask"].detach(),
        frames_exist=masks["frames_exist"].detach(),
        residue_keys=keys,
        crop_indices=torch.arange(int(structure.num_tokens), device=device),
        split=split,
        cluster=cluster,
        alignment_rmsd=rmsd,
        counts=mask_counts(masks),
    )
    return batch
