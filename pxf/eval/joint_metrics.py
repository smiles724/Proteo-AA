"""Metric groups for the joint-refinement evaluator, kept apart on purpose.

The separation is the point. A side-chain number computed after transferring
conformations onto the native backbone says something about *conformation*; the
same number computed on the assembled prediction says something about
*placement*; neither is the other, and reporting one under the other's name is
how a packing regression hides. This module currently implements the backbone
group; the side-chain groups are declared but not yet implemented, and say so
rather than returning a plausible number.

Everything is scored on CPU in float64. ``backbone_metrics`` builds masks and
search tensors on the CPU, so feeding it GPU inputs mixes devices; and the
residual is the accept/reject signal, so it must not be dominated by the
arithmetic that produced it.
"""

import math

import torch

from pxf.eval import backbone_metrics
from pxf import atom37

# Bond references, frozen from the canonical geometry rather than fitted to
# whatever this run happens to produce. A dynamic threshold would widen to
# accommodate a model that was getting worse.
IDEAL_BOND = {
    "n_ca": 1.458,
    "ca_c": 1.525,
    "c_o": 1.231,
    "c_n": 1.329,  # the true adjacent peptide bond
}
BOND_TOLERANCE = 0.12  # Angstroms; a deviation past this counts as a failure

N, CA, C, O = (atom37.ATOM37.index(name) for name in ("N", "CA", "C", "O"))


def _cpu64(tensor):
    return tensor.detach().to("cpu", torch.float64)


def _null():
    """An undefined metric: null with a zero count, never a fabricated zero."""
    return None


def backbone_group(pred37, native37, mask37, *, canonical=None, with_tm=True):
    """Aligned backbone accuracy, through the established report.

    Denominators and atom/residue counts are preserved as ``backbone_report``
    defines them; this wrapper only fixes the device and dtype and adds the
    same-frame diagnostics beside the aligned figures.
    """
    pred37, native37 = _cpu64(pred37), _cpu64(native37)
    mask37 = mask37.detach().to("cpu")
    report = backbone_metrics.backbone_report(
        pred37.float(), native37.float(), mask37.float(),
        canonical=canonical, with_tm=with_tm,
    )
    out = {f"bb_{key}": value for key, value in report.items()}

    # Same-frame, unaligned. A diagnostic, not a headline: it moves with any
    # global pose difference the aligned figure removes by construction.
    backbone = list(atom37.BACKBONE_SLOTS)
    bb_ok = mask37[:, backbone].bool().reshape(-1)
    delta = (pred37[:, backbone].reshape(-1, 3) - native37[:, backbone].reshape(-1, 3))
    if int(bb_ok.sum()) > 0:
        squared = float((delta[bb_ok] ** 2).sum())
        out["bb_rmsd_unaligned"] = math.sqrt(squared / int(bb_ok.sum()))
        out["bb_sq_error_sum"] = squared
        out["bb_sq_error_count"] = int(bb_ok.sum())
    else:
        out["bb_rmsd_unaligned"] = _null()
        out["bb_sq_error_sum"] = 0.0
        out["bb_sq_error_count"] = 0
    return out


def _pair_distance(coords, mask, left, right):
    ok = mask[:, left].bool() & mask[:, right].bool()
    if not bool(ok.any()):
        return None, 0
    delta = coords[:, left][ok] - coords[:, right][ok]
    return delta.norm(dim=-1), int(ok.sum())


def backbone_geometry(pred37, mask37, *, residue_index=None, chain_index=None):
    """Bond-length and frame diagnostics on the PREDICTED backbone.

    Adjacency is taken from the original residue numbering, so a crop or a
    chain break does not manufacture a peptide bond between residues that were
    never neighbours -- which would report a failure the model did not commit.
    """
    pred37 = _cpu64(pred37)
    mask37 = mask37.detach().to("cpu")
    out = {}

    for label, (left, right) in (
        ("n_ca", (N, CA)),
        ("ca_c", (CA, C)),
        ("c_o", (C, O)),
    ):
        distances, count = _pair_distance(pred37, mask37, left, right)
        if distances is None:
            out[f"geom_{label}_mean"] = _null()
            out[f"geom_{label}_bad_rate"] = _null()
            out[f"geom_{label}_count"] = 0
            continue
        bad = (distances - IDEAL_BOND[label]).abs() > BOND_TOLERANCE
        out[f"geom_{label}_mean"] = float(distances.mean())
        out[f"geom_{label}_bad_rate"] = float(bad.double().mean())
        out[f"geom_{label}_bad_count"] = int(bad.sum())
        out[f"geom_{label}_count"] = count

    length = pred37.shape[0]
    if residue_index is None:
        residue_index = torch.arange(length)
    residue_index = residue_index.detach().to("cpu").reshape(-1)[:length].long()
    if chain_index is None:
        chain_index = torch.zeros(length, dtype=torch.long)
    chain_index = chain_index.detach().to("cpu").reshape(-1)[:length].long()

    # A real peptide bond needs consecutive numbering in the same chain.
    adjacent = (
        (residue_index[1:] - residue_index[:-1] == 1)
        & (chain_index[1:] == chain_index[:-1])
        & mask37[:-1, C].bool()
        & mask37[1:, N].bool()
    )
    if bool(adjacent.any()):
        distances = (pred37[1:, N][adjacent] - pred37[:-1, C][adjacent]).norm(dim=-1)
        bad = (distances - IDEAL_BOND["c_n"]).abs() > BOND_TOLERANCE
        out["geom_c_n_mean"] = float(distances.mean())
        out["geom_c_n_bad_rate"] = float(bad.double().mean())
        out["geom_c_n_bad_count"] = int(bad.sum())
        out["geom_c_n_count"] = int(adjacent.sum())
    else:
        out["geom_c_n_mean"] = _null()
        out["geom_c_n_bad_rate"] = _null()
        out["geom_c_n_bad_count"] = 0
        out["geom_c_n_count"] = 0

    frame_ok = mask37[:, N].bool() & mask37[:, CA].bool() & mask37[:, C].bool()
    finite = torch.isfinite(pred37[:, [N, CA, C]]).all(dim=-1).all(dim=-1)
    out["geom_valid_frame_rate"] = float((frame_ok & finite).double().mean())
    out["geom_valid_frame_count"] = int((frame_ok & finite).sum())
    out["geom_residues"] = int(length)
    return out


def model_failure(reason, **context):
    """A model that could not produce a scorable prediction.

    Recorded as a failure rather than an absent row. A model cannot improve its
    score by emitting fewer finite atoms, so the scored set never shrinks to
    accommodate one.
    """
    return dict(failure=True, failure_reason=str(reason), **context)


def prediction_is_scorable(pred37, supplied_mask):
    """Whether a prediction has finite coordinates where it claims atoms."""
    supplied = supplied_mask.detach().to("cpu").bool()
    coords = pred37.detach().to("cpu")
    if not bool(supplied.any()):
        return False, "the prediction supplies no backbone atoms"
    if not bool(torch.isfinite(coords[supplied]).all()):
        return False, "the prediction contains non-finite supplied coordinates"
    return True, None


# ---- declared, not yet implemented ------------------------------------------

NOT_IMPLEMENTED_GROUPS = (
    "sc_local",       # conformation after frame transfer
    "sc_global",      # placement on the predicted backbone
    "sc_environment",  # predicted-environment lDDT
    "sc_chemistry",   # clashes, bonds and chirality on the assembled prediction
)


def unimplemented_group(name):
    raise NotImplementedError(
        f"metric group {name!r} is declared by the evaluation plan but not "
        "implemented yet; it requires the common inference packer (milestone 3) "
        "and must not be approximated from training-time tensors"
    )
