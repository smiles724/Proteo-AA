"""Held-out packing evaluation for the coupling adapters.

Phase 1 optimizes ``L_SC``, which is FaMPNN's *diffusion* objective at randomly
drawn noise levels. A falling training curve says the adapter fit something; it
does not say the packing got better, and it says nothing about structures the
adapter never saw. The claim the staged plan actually makes is that PXDesign's
``a_token`` carries information that improves side-chain packing, and the only
way to test that is to pack held-out structures and score the result.

**The comparison is paired, and it isolates the adapter.** In phase 1 the
backbone proposal ``bb0`` is produced before ``A_BS`` is applied and ``A_SB`` is
off, so running one structure at one sigma_B with one seed, once with
``enable_bb_to_sc=True`` and once with it False, changes exactly one quantity:
the residual added to ``h_V``. The generated backbone is identical between the
two arms, so backbone error cancels out of the delta. Because the adapters are
zero-initialized, the disabled arm is not an approximation of the pretrained
system -- it *is* the pretrained system.

**The absolute numbers here are not comparable to**
``scripts/eval_protenix_sidechain.py``. That script packs onto a deposited
crystal backbone; this one packs onto a denoised proposal at sigma_B, which is a
harder target. Read the delta between arms, not the level. When you want to know
how much of the level is backbone error rather than packing error, the
native-backbone reference arm answers exactly that.

Scoring a proposal also needs care that the deposited-backbone case does not:
the two structures are not in a common frame, so side chains are transferred
through their own residue frames before being scored. See
:func:`place_on_native_backbone`, which is where that is done and why.

**sigma_B is swept, not sampled.** Both adapters are conditioned on log sigma_B,
so a single evaluation point would only license a claim at that point. The sweep
walks the same window the schedule draws from during training, and the report is
per-sigma as well as pooled -- an adapter that helps at low noise and hurts at
high noise is a real outcome, and pooling alone would hide it.
"""

import hashlib

import torch

from pxf import atom37
from pxf.eval.sidechain_metrics import NCAC

# Direction conventions, shared with scripts/eval_protenix_sidechain.py.
# tests/test_eval_couple.py pins these equal to that script's tables, so the two
# reports cannot drift into disagreeing about what "better" means.
LOWER_IS_BETTER = (
    "symmetry_rmsd",
    "bad_bond_fraction",
    "bond_mae",
    "rotamer_outlier_fraction_40deg",
)
HEADLINE = ("symmetry_rmsd", "rotamer_recovery", "chi_recovery_20deg", "lddt_sc_sc")
REGRESSION_TOLERANCE = 1e-4

# What the per-sigma and pooled tables print.
REPORT = {
    "rmsd": ("symmetry_rmsd",),
    "rotamer recovery": ("rotamer_recovery", "chi_recovery_20deg", "chi1_accuracy_20deg"),
    "lddt": ("lddt_sc_sc", "lddt_sc_env"),
    "covalent failures": ("bad_bond_fraction", "rotamer_outlier_fraction_40deg"),
}

ARMS = ("coupled", "uncoupled")


def target_seed(base, name, sigma, replicate=0):
    """A seed fixed by *what* is being evaluated, not by how the run is arranged.

    Two properties are needed and an earlier version only had the first.

    Within a process, both arms must draw the same packing noise, or the
    comparison measures the sampler rather than the adapter. Keying on the
    target and sigma rather than on loop position also survives
    ``--max-targets``.

    **Across processes the seed must also be identical**, and Python's ``hash``
    is salted per interpreter for ``str``, so the previous implementation
    returned different values in every run. Arms inside one job stayed paired,
    but two jobs -- a real run and its shuffled control, or a rerun of the same
    config -- silently drew different backbone noise and different packing
    trajectories, so their absolute levels were not comparable. A digest fixes
    that; ``blake2b`` is in the standard library and stable across versions and
    platforms.

    ``sigma`` is the actual noise level rather than its index, so a run with a
    different ``--n-sigma`` reuses the same seed wherever it evaluates the same
    sigma. ``replicate`` distinguishes repeated draws at one (target, sigma).
    """
    key = f"{int(base)}|{name}|{float(sigma):.12g}|{int(replicate)}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


def sweep_sigmas(schedule, count):
    """``count`` deterministic sigma_B values spanning the schedule's own window.

    Deterministic rather than drawn: the point of the sweep is that two runs are
    comparable, and quantiles of the training window cover it more evenly than
    ``count`` random draws would.
    """
    count = int(count)
    if count < 1:
        raise ValueError(f"--n-sigma must be at least 1, got {count}")
    if schedule.mode == "fixed":
        return [float(schedule.sigma)]
    if schedule.mode == "trajectory":
        # The discrete sigmas the sampler actually visits inside the window.
        # `window_steps()` is the inclusive (first, last) index pair, and the
        # trajectory is monotonic, so the slice between them is exactly the set
        # `sample()` draws from -- taken this way rather than recomputing the
        # mask, so the sweep cannot drift from the training distribution.
        first, last = schedule.window_steps()
        if first is None:
            raise ValueError(
                f"no trajectory step falls in [{schedule.sigma_min}, "
                f"{schedule.sigma_max}]; there is nothing to sweep"
            )
        window = torch.sort(schedule.trajectory()[first : last + 1].reshape(-1)).values
        if count == 1:
            return [float(window[window.numel() // 2])]
        picks = torch.linspace(0, window.numel() - 1, count).round().long()
        return [float(window[i]) for i in picks.tolist()]
    # loguniform: even in log space, which is how it is drawn.
    lo, hi = float(schedule.sigma_min), float(schedule.sigma_max)
    if count == 1:
        return [
            float(
                torch.exp(
                    torch.tensor(
                        (torch.log(torch.tensor(lo)) + torch.log(torch.tensor(hi))) / 2
                    )
                )
            )
        ]
    steps = torch.linspace(torch.log(torch.tensor(lo)), torch.log(torch.tensor(hi)), count)
    return [float(torch.exp(s)) for s in steps]


def restype_atom37_mask(aatype, rc, device=None):
    """``[L, 37]`` true where an atom37 slot exists for that residue type."""
    table = torch.as_tensor(rc.restype_atom37_mask, device=device or aatype.device)
    return table[aatype.reshape(-1).long()].bool()


def predicted_atom37(cycle, aatype, rc):
    """The cycle's packing as atom37 coordinates plus the mask of generated atoms.

    The backbone comes from the diffusion proposal (``bb0_dense``) and the side
    chains from the packer, written into the 33 non-backbone slots. FaMPNN emits
    every slot that exists for the residue type, so the generated mask is the
    residue-type mask -- there is no partial output to account for.
    """
    backbone = cycle.bb0_dense
    if backbone.dim() != 4:
        raise ValueError(f"bb0_dense must be [B, L, 37, 3], got {tuple(backbone.shape)}")
    sidechains = cycle.sidechains
    if sidechains is None:
        raise ValueError("the cycle ran no packing step, so there is nothing to score")
    slots = list(atom37.SIDECHAIN_SLOTS)
    if sidechains.shape[-2] != len(slots):
        raise ValueError(
            f"packed side chains have {sidechains.shape[-2]} slots, expected {len(slots)}"
        )
    pred = backbone.clone()
    pred[..., slots, :] = sidechains.to(pred.dtype)
    mask = restype_atom37_mask(aatype, rc, device=pred.device)
    return pred[0], mask


def place_on_native_backbone(pred37, native37, canonical):
    """Move each predicted side chain onto the native backbone, frame by frame.

    ``sidechain_metrics.score`` builds one set of residue frames from the
    reference and uses it for both structures, on the stated grounds that "the
    backbone is shared: FaMPNN does not move it". That holds for
    ``eval_protenix_sidechain.py``, which hands the packer the deposited
    backbone. **It is false here, twice over.** Globally, because the featurizer
    centers the structure while FaMPNN's parse of the same file does not, so the
    two are not even in the same coordinate frame; and locally, because the
    backbone being packed is a diffusion proposal rather than the deposited one.
    Scoring without correcting for that compares side chains against frames they
    were never built in, and produces ~20 A RMSD on a perfectly reasonable
    packing.

    Transferring each residue's side chain through its own backbone frame makes
    the shared-backbone assumption true by construction. What is measured is
    then side-chain conformation *relative to its own backbone* -- the standard
    side-chain accuracy quantity, invariant to global pose, and the one that
    isolates packing quality from backbone error. Backbone error does not vanish
    from the report; it is measured separately by :func:`backbone_rmsd`.
    """
    frames = canonical.frames
    slots = list(atom37.SIDECHAIN_SLOTS)
    ncac = list(NCAC)
    pred_bb, native_bb = pred37[:, ncac, :], native37[:, ncac, :]
    r_pred, t_pred = frames.build_frame(pred_bb[:, 0], pred_bb[:, 1], pred_bb[:, 2])
    r_native, t_native = frames.build_frame(
        native_bb[:, 0], native_bb[:, 1], native_bb[:, 2]
    )
    local = frames.to_local(pred37[:, slots, :], r_pred, t_pred)
    placed = native37.clone()
    placed[:, slots, :] = frames.to_global(local, r_native, t_native)
    return placed


def backbone_rmsd(pred37, native37, mask=None):
    """Kabsch-superposed backbone RMSD, in Angstroms.

    Reported alongside the packing metrics so the proposal's own quality is
    visible. A coupling that improves packing while the backbone drifts is a
    different result from one that improves both, and the packing metrics alone
    cannot tell them apart once side chains are scored in local frames.
    """
    slots = list(atom37.BACKBONE_SLOTS)
    pred = pred37[:, slots, :].reshape(-1, 3).double()
    native = native37[:, slots, :].reshape(-1, 3).double()
    if mask is not None:
        keep = mask[:, slots].reshape(-1).bool()
        pred, native = pred[keep], native[keep]
    if pred.shape[0] < 3:
        return float("nan")
    pred = pred - pred.mean(0, keepdim=True)
    native = native - native.mean(0, keepdim=True)
    u, _s, vh = torch.linalg.svd(pred.T @ native)
    d = torch.sign(torch.det(u @ vh))
    correction = torch.diag(torch.tensor([1.0, 1.0, float(d)], dtype=pred.dtype))
    rotation = u @ correction @ vh
    aligned = pred @ rotation
    return float(torch.sqrt(((aligned - native) ** 2).sum(-1).mean()))


def native_atom37(native, rc):
    """Reference coordinates and the mask of atoms actually present.

    ``missing_atom_mask`` is upstream's "should exist but is absent". For the
    AFDB subset it comes out all-zero by construction, but it is applied anyway:
    the same function has to be correct if the evaluation set is ever pointed at
    deposited structures.
    """
    coords = native["x"]
    coords = coords[0] if coords.dim() == 4 else coords
    aatype = native["aatype"].reshape(-1).long()
    exists = restype_atom37_mask(aatype, rc, device=coords.device)
    missing = native.get("missing_atom_mask")
    if missing is not None:
        missing = missing[0] if missing.dim() == 3 else missing
        exists = exists & ~missing.bool().to(exists.device)
    return coords, exists


def check_alignment(sample_id, native_aatype, structure_aatype):
    """Positional correspondence between the two independent parses of one file.

    ``pxf.eval.sidechain_metrics.score`` aligns prediction to reference by index,
    and the two tensors come from different readers -- the PXDesign featurizer
    and FaMPNN's parser. Length equality is necessary but not sufficient; the
    sequences have to match, which is what catches an assembly-versus-asymmetric-
    unit disagreement that happens to preserve the count.
    """
    native_aatype = native_aatype.reshape(-1).long()
    structure_aatype = structure_aatype.reshape(-1).long()
    if native_aatype.shape[0] != structure_aatype.shape[0]:
        raise ValueError(
            f"{sample_id}: featurized structure has {structure_aatype.shape[0]} "
            f"residues but the file parses to {native_aatype.shape[0]}; "
            "side-chain targets would be misaligned"
        )
    if not torch.equal(native_aatype.cpu(), structure_aatype.cpu()):
        differing = int((native_aatype.cpu() != structure_aatype.cpu()).sum())
        raise ValueError(
            f"{sample_id}: featurized sequence differs from the file's at "
            f"{differing} position(s); side-chain targets would be misaligned"
        )


def delta_table(arms, keys=None):
    """``{metric: (uncoupled, coupled, delta, better)}`` for one pooled summary.

    ``better`` is the signed improvement, so a caller can test one direction
    regardless of whether the metric counts up or down.
    """
    keys = keys or [key for group in REPORT.values() for key in group]
    base, tuned = arms["uncoupled"], arms["coupled"]
    out = {}
    for key in keys:
        if key not in base or key not in tuned:
            continue
        lo, hi = float(base[key]), float(tuned[key])
        delta = hi - lo
        out[key] = (lo, hi, delta, -delta if key in LOWER_IS_BETTER else delta)
    return out


def regressions(arms, *, headline=HEADLINE, tolerance=REGRESSION_TOLERANCE):
    """Headline metrics the coupling made worse. Empty means it did not hurt.

    Note what this does *not* assert: that the coupling helped. An adapter that
    changes nothing produces an empty list, which is the correct answer to "did
    it regress" and the wrong answer to "did it work". The caller reports both.
    """
    table = delta_table(arms, keys=headline)
    return [
        (key, lo, hi)
        for key, (lo, hi, _delta, better) in table.items()
        if better < -abs(tolerance)
    ]


def improvements(arms, *, headline=HEADLINE, tolerance=REGRESSION_TOLERANCE):
    """Headline metrics the coupling improved beyond float noise."""
    table = delta_table(arms, keys=headline)
    return [
        (key, lo, hi)
        for key, (lo, hi, _delta, better) in table.items()
        if better > abs(tolerance)
    ]


def donor_a_token(source, length):
    """Reshape another protein's ``a_token`` to this one's residue count.

    The shuffled control needs a donor from a *different* structure, and
    structures differ in length, so the donor has to be made to fit. Cropping
    when it is longer and tiling when it is shorter keeps the per-residue
    feature distribution intact -- the magnitudes and channel statistics the
    adapter sees are still real ``a_token`` values -- while destroying any
    correspondence to the recipient's residues, which is the whole point.

    What this cannot control for is that a tiled donor is periodic. Where the
    donor is much shorter than the recipient that periodicity is an artefact of
    the control rather than of the model, so pair donors of similar length when
    the comparison matters.
    """
    if source.dim() != 3:
        raise ValueError(f"a_token must be [B, L, C], got {tuple(source.shape)}")
    have = source.shape[-2]
    if have == length:
        return source
    if have > length:
        return source[..., :length, :]
    repeats = -(-length // have)  # ceil
    return source.repeat(1, repeats, 1)[..., :length, :]


# --- run manifest -----------------------------------------------------------

# Bumped whenever the seed derivation changes. Two runs with different scheme
# versions drew different noise and must not be pooled or compared as levels,
# however identical their configuration looks.
SEED_SCHEME = "blake2b-v1"

# Fields that must agree before results from two runs may be combined. Anything
# that changes what was computed belongs here; anything that only changes how it
# was scheduled (job id, node, wall time) deliberately does not.
COMPATIBILITY_KEYS = (
    "seed_scheme",
    "seed_base",
    "sigma_values",
    "pack_steps",
    "run_feedback",
    "codesign",
    "fampnn_weights",
    "checkpoint_sha256",
    "ema",
    "structures_fingerprint",
    "upstream",
)


def structures_fingerprint(paths):
    """A digest of the evaluated panel: which structures, in which order.

    Order matters because the shuffled control's donor is the previous target,
    so a reordered panel is a different experiment even with the same members.
    """
    digest = hashlib.blake2b(digest_size=16)
    for path in paths:
        digest.update(str(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def run_manifest(*, args, sigmas, checkpoint_path=None, structures=(), upstream=None):
    """Everything needed to say whether two runs measured the same thing.

    Recorded rather than inferred. A comparison across runs is only meaningful
    if the seed scheme, the sigma grid, the packing settings, the frozen
    components and the panel all match, and none of those are recoverable from
    the metrics file after the fact.
    """
    checkpoint_sha = None
    if checkpoint_path:
        from pxf.provenance import file_sha256

        checkpoint_sha = file_sha256(checkpoint_path)
    return {
        "seed_scheme": SEED_SCHEME,
        "seed_base": int(args.seed),
        "sigma_values": [float(s) for s in sigmas],
        "pack_steps": int(args.pack_steps),
        "run_feedback": bool(args.run_feedback),
        "codesign": bool(getattr(args, "codesign", False)),
        "fampnn_weights": args.fampnn_weights,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": checkpoint_sha,
        "ema": bool(getattr(args, "ema", False) and checkpoint_path),
        "a_token_source": (
            "shuffled-donor" if getattr(args, "shuffle_a_token", False) else "own"
        ),
        "n_structures": len(structures),
        "structures_fingerprint": structures_fingerprint(structures),
        "upstream": upstream,
    }


def incompatible_fields(left, right):
    """Which compatibility keys disagree. Empty means the runs may be compared.

    ``a_token_source`` is deliberately absent from the comparison: a real run
    and its shuffled control *should* differ there, and that pair is the whole
    point of the control.
    """
    out = []
    for key in COMPATIBILITY_KEYS:
        if left.get(key) != right.get(key):
            out.append((key, left.get(key), right.get(key)))
    return out


# --- paired uncertainty -----------------------------------------------------


def per_target_deltas(rows, metric, sigma=None):
    """``{target: coupled - uncoupled}`` from a run's per-target rows.

    The delta is formed *within* a target before anything is averaged, which is
    what makes the interval below a paired one: per-target packing difficulty
    varies far more than the adapter's effect does, and an unpaired interval
    would be dominated by it.
    """
    by_target = {}
    for row in rows:
        if sigma is not None and abs(float(row["sigma"]) - float(sigma)) > 1e-9:
            continue
        by_target.setdefault(row["target"], {})[row["arm"]] = float(row[metric])
    return {
        target: arms["coupled"] - arms["uncoupled"]
        for target, arms in by_target.items()
        if "coupled" in arms and "uncoupled" in arms
    }


def paired_bootstrap(values, *, n_resamples=10000, alpha=0.05, seed=0):
    """Percentile CI for a mean, resampling whole targets.

    Targets are the independent unit, not (target, sigma) pairs: one structure
    contributes a correlated row at every sigma, so resampling rows would
    understate the interval.
    """
    import torch as _t

    data = _t.tensor([float(v) for v in values], dtype=_t.float64)
    if data.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    generator = _t.Generator().manual_seed(int(seed))
    index = _t.randint(data.numel(), (int(n_resamples), data.numel()), generator=generator)
    means = data[index].mean(dim=1)
    low = float(means.quantile(alpha / 2))
    high = float(means.quantile(1 - alpha / 2))
    return float(data.mean()), low, high


def retention_interval(candidate, reference, *, n_resamples=10000, seed=0):
    """Bootstrap the *ratio* of two paired effects on the same targets.

    Resampling the shared target list once per replicate -- rather than each
    run independently -- keeps numerator and denominator paired, which is the
    only way the ratio has an interval worth quoting.

    Returns ``(ratio, low, high, n)``. A ratio whose denominator interval spans
    zero is meaningless however tight it looks, so the caller is expected to
    check the reference effect first.
    """
    import torch as _t

    shared = sorted(set(candidate) & set(reference))
    if not shared:
        return float("nan"), float("nan"), float("nan"), 0
    cand = _t.tensor([candidate[t] for t in shared], dtype=_t.float64)
    ref = _t.tensor([reference[t] for t in shared], dtype=_t.float64)
    generator = _t.Generator().manual_seed(int(seed))
    index = _t.randint(len(shared), (int(n_resamples), len(shared)), generator=generator)
    ratios = cand[index].mean(dim=1) / ref[index].mean(dim=1)
    finite = ratios[_t.isfinite(ratios)]
    point = float(cand.mean() / ref.mean()) if float(ref.mean()) != 0 else float("nan")
    if finite.numel() == 0:
        return point, float("nan"), float("nan"), len(shared)
    return point, float(finite.quantile(0.025)), float(finite.quantile(0.975)), len(shared)
