"""Backbone accuracy for the before/after comparison: bb0 versus bb1.

The SC -> BB pilot's question is whether one corrective event improves the
*backbone*, so the report needs backbone instruments rather than the side-chain
ones :mod:`pxf.eval.sidechain_metrics` provides. Three, because they fail
differently:

``rmsd``      superposition-dependent and dominated by the worst region, which
              is what a roadmap criterion stated in Angstroms is about.
``lddt``      superposition-free and local, so a correction that fixes local
              geometry while leaving a hinge misplaced still registers.
``tm_score``  length-normalized and topology-sensitive, so a small RMSD change
              on a large protein is not mistaken for a fold change.

**The residue correspondence is given, not searched.** ``bb0``, ``bb1`` and the
reference are the same chain with the same residues in the same order -- the
sequence is fixed and the evaluator checks it -- so there is no alignment
problem, only a superposition problem. :func:`tm_score` therefore implements
TM-score *with the given one-to-one alignment*: it searches superpositions the
way TM-score does (seeded fragments, iterative extension) but never re-aligns
residues, which is what TM-align adds and what would be wrong here.

The seed search is **strided** rather than exhaustive, which can leave the
reported value a little below the exhaustive optimum on long chains. That is
acceptable and stated because the quantity of interest is the *paired* delta
between two arms measured with the identical search, not the absolute level; a
search that is equally suboptimal for both does not bias the difference.
"""

import torch

from pxf import atom37

CA = 1
FRAME = (0, 1, 2)
LDDT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
LDDT_INCLUSION_RADIUS = 15.0
# TM-score's own constants (Zhang & Skolnick 2004).
TM_D0_MIN = 0.5
TM_SEED_MIN = 4
TM_ITERATIONS = 20


def _kabsch(pred, ref, weight):
    """Batched weighted superposition of ``pred`` onto ``ref``.

    ``pred``/``ref`` are ``[L, 3]`` and ``weight`` is ``[S, L]``; returns the
    ``[S, L]`` post-superposition distances. Batching over seeds is what keeps
    the fragment search cheap enough to run on every target.
    """
    w = weight.to(torch.float64)
    total = w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y = pred.to(torch.float64), ref.to(torch.float64)
    cx = (w[..., None] * x).sum(dim=-2) / total
    cy = (w[..., None] * y).sum(dim=-2) / total
    xc = x[None] - cx[:, None, :]
    yc = y[None] - cy[:, None, :]
    cov = torch.einsum("sl,sli,slj->sij", w, xc, yc)
    u, _s, vh = torch.linalg.svd(cov)
    sign = torch.sign(torch.det(torch.matmul(u, vh)))
    correction = torch.eye(3, dtype=torch.float64).expand(u.shape[0], 3, 3).clone()
    correction[:, 2, 2] = sign
    rotation = torch.matmul(torch.matmul(u, correction), vh)
    aligned = torch.einsum("sli,sij->slj", xc, rotation)
    return (aligned - yc).norm(dim=-1)


def superposed_rmsd(pred, ref, mask=None):
    """Kabsch-superposed RMSD over the masked atoms, in Angstroms."""
    pred, ref = pred.reshape(-1, 3), ref.reshape(-1, 3)
    if mask is not None:
        keep = mask.reshape(-1).bool()
        pred, ref = pred[keep], ref[keep]
    if pred.shape[0] < 3:
        return float("nan")
    weight = torch.ones(1, pred.shape[0])
    distances = _kabsch(pred, ref, weight)
    return float(distances.pow(2).mean().sqrt())


def tm_score(pred_ca, ref_ca, mask=None, *, max_seeds=256):
    """TM-score with the given residue correspondence.

    ``pred_ca``/``ref_ca`` are ``[L, 3]`` C-alpha coordinates of the same
    residues in the same order. ``d0`` follows the published normalization; the
    denominator is the number of scored residues, so two arms on the same target
    share it.
    """
    pred, ref = pred_ca.reshape(-1, 3), ref_ca.reshape(-1, 3)
    if mask is not None:
        keep = mask.reshape(-1).bool()
        pred, ref = pred[keep], ref[keep]
    length = pred.shape[0]
    if length < TM_SEED_MIN:
        return float("nan")
    d0 = max(1.24 * (length - 15) ** (1 / 3) - 1.8, TM_D0_MIN) if length > 15 else TM_D0_MIN

    # Seeded fragments: the full chain, then halves, quarters, ... down to four
    # residues. Strided so the seed count stays bounded; see the module note.
    seeds = []
    window = length
    while window >= TM_SEED_MIN:
        stride = max(1, -(-(length - window + 1) // max(1, max_seeds // 8)))
        for start in range(0, length - window + 1, stride):
            row = torch.zeros(length)
            row[start : start + window] = 1.0
            seeds.append(row)
        window //= 2
    weight = torch.stack(seeds)  # [S, L]

    best = 0.0
    for _ in range(TM_ITERATIONS):
        distances = _kabsch(pred, ref, weight)
        scores = (1.0 / (1.0 + (distances / d0) ** 2)).mean(dim=-1)
        best = max(best, float(scores.max()))
        # Iterative extension: keep the residues that superposed well and
        # re-superpose on them. The cutoff floor stops a seed collapsing below
        # the three points a superposition needs.
        cutoff = max(d0, 3.5)
        updated = (distances < cutoff).float()
        while True:
            short = updated.sum(dim=-1) < TM_SEED_MIN
            if not bool(short.any()):
                break
            cutoff += 0.5
            updated = torch.where(short[:, None], (distances < cutoff).float(), updated)
            if cutoff > 20.0:
                updated = torch.where(short[:, None], weight, updated)
                break
        if torch.equal(updated, weight):
            break
        weight = updated
    return best


def lddt(
    pred,
    ref,
    *,
    subject_residue,
    partner_pred=None,
    partner_ref=None,
    partner_residue=None,
    canonical=None,
    inclusion_radius=LDDT_INCLUSION_RADIUS,
    thresholds=LDDT_THRESHOLDS,
):
    """Superposition-free lDDT, through the canonical implementation.

    Delegates to ``pxdesign_train.sidechain.lddt.lddt_score`` so the definition
    -- inclusion radius, thresholds, same-residue exclusion -- is the one the
    side-chain numbers already use, rather than a second implementation that
    could drift from it.
    """
    from pxf.eval.canonical import load

    canonical = canonical or load()
    if partner_pred is None:
        partner_pred, partner_ref = pred, ref
        partner_residue = subject_residue
    score, pairs = canonical.lddt.lddt_score(
        pred.double(),
        ref.double(),
        partner_pred.double(),
        partner_ref.double(),
        subject_residue.long(),
        partner_residue.long(),
        inclusion_radius=inclusion_radius,
        thresholds=thresholds,
    )
    return float(score), int(pairs)


def backbone_report(pred37, native37, mask37, *, canonical=None, with_tm=True):
    """RMSD / lDDT / TM-score for one backbone against the reference.

    ``mask37`` is the ``[L, 37]`` mask of atoms present in *both* structures;
    residues are scored only where the atom they contribute exists on both
    sides, so bb0 and bb1 are measured over the same set.
    """
    backbone = list(atom37.BACKBONE_SLOTS)
    length = pred37.shape[0]
    residue = torch.arange(length)

    ca_ok = mask37[:, CA].bool()
    bb_ok = mask37[:, backbone].bool()
    out = dict(
        scored_residues=int(ca_ok.sum()),
        scored_backbone_atoms=int(bb_ok.sum()),
        ca_rmsd=superposed_rmsd(pred37[:, CA], native37[:, CA], ca_ok),
        backbone_rmsd=superposed_rmsd(
            pred37[:, backbone].reshape(-1, 3),
            native37[:, backbone].reshape(-1, 3),
            bb_ok.reshape(-1),
        ),
    )
    ca_index = torch.nonzero(ca_ok, as_tuple=True)[0]
    if ca_index.numel() >= TM_SEED_MIN:
        score, pairs = lddt(
            pred37[ca_index, CA],
            native37[ca_index, CA],
            subject_residue=residue[ca_index],
            canonical=canonical,
        )
        out["lddt_ca"], out["lddt_ca_pairs"] = score, pairs
        flat_pred = pred37[:, backbone][bb_ok]
        flat_native = native37[:, backbone][bb_ok]
        flat_residue = residue[:, None].expand(length, len(backbone))[bb_ok]
        score, pairs = lddt(
            flat_pred,
            flat_native,
            subject_residue=flat_residue,
            canonical=canonical,
        )
        out["lddt_backbone"], out["lddt_backbone_pairs"] = score, pairs
        if with_tm:
            out["tm_score"] = tm_score(pred37[ca_index, CA], native37[ca_index, CA])
    return out


# Which direction counts as better, shared with the side-chain conventions.
LOWER_IS_BETTER = ("ca_rmsd", "backbone_rmsd")
HEADLINE = ("ca_rmsd", "backbone_rmsd", "lddt_ca", "lddt_backbone", "tm_score")


def improvement(metric, before, after):
    """Signed improvement of ``after`` over ``before``, sign-corrected."""
    delta = float(after) - float(before)
    return -delta if metric in LOWER_IS_BETTER else delta
