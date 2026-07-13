"""Tests for the template-anchored side-chain init (Overleaf 0721/0712, para 221).

    y_{T,ij} = mu_ideal[a_i, j] + sigma_T * eps_ij,   eps ~ N(0, I)
    x_{T,ij} = F_hat_i y_{T,ij}

The thesis of the change: an ISOTROPIC Gaussian is rotation-invariant, so pushing
it through F_hat carries no backbone-orientation information. The ANISOTROPIC ideal
template does — and `test_global_init_carries_backbone_orientation` proves exactly
that, with the old Gaussian init as the negative control.

The ideal-template table (`pxdesign_train.sidechain.templates`) is owned by another
agent. If it is not importable, these tests run against a stub table that satisfies
the same contract (patched onto `init._ideal_template`, no sys.modules pollution),
and the table-consistency test is skipped.
"""
import inspect
import math
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from pxdesign_train.sidechain import init as init_mod
from pxdesign_train.sidechain.init import (
    DEFAULT_SIGMA_T,
    gaussian_init_local,
    template_init_local,
)
from pxdesign_train.sidechain.instantiate import MAX_SC, STD_AA_3, sidechain_mask

try:  # the real table, once the templates agent lands it
    from pxdesign_train.sidechain import templates as _templates  # noqa: F401

    HAVE_TEMPLATES = True
except Exception:  # pragma: no cover - depends on integration order
    HAVE_TEMPLATES = False


# --- stub ideal-template table (contract-compatible) -------------------------
def _stub_table():
    """[20, MAX_SC, 3] anisotropic coords + [20, MAX_SC] mask, plausible radii."""
    g = torch.Generator().manual_seed(1234)
    mask = sidechain_mask(STD_AA_3)                       # real atom counts
    dirs = torch.randn(len(STD_AA_3), MAX_SC, 3, generator=g)
    dirs = torch.nn.functional.normalize(dirs, dim=-1)
    radii = 1.5 + 0.9 * torch.arange(MAX_SC, dtype=torch.float32)  # grow out from CA
    coords = dirs * radii[None, :, None]
    return (coords * mask[..., None]).float(), mask


_STUB_COORDS, _STUB_MASK = _stub_table()


def _stub_ideal_template(type_idx: torch.Tensor):
    idx = type_idx.long()
    return _STUB_COORDS.to(idx.device)[idx], _STUB_MASK.to(idx.device)[idx]


@pytest.fixture(autouse=True)
def _template_table(monkeypatch):
    if not HAVE_TEMPLATES:
        monkeypatch.setattr(init_mod, "_ideal_template", _stub_ideal_template)


# --- helpers -----------------------------------------------------------------
def _random_rotation(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=g))
    q = q * torch.sign(torch.diagonal(r))[None, :]
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _kabsch(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Rotation R minimizing ||P_c @ R.T - Q_c||."""
    Pc = P - P.mean(0, keepdim=True)
    Qc = Q - Q.mean(0, keepdim=True)
    U, _, Vt = torch.linalg.svd(Pc.T @ Qc)
    d = torch.sign(torch.det(Vt.T @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, d.item()]))
    return Vt.T @ D @ U.T


def _angle_deg(R1: torch.Tensor, R2: torch.Tensor) -> float:
    cos = ((R1 @ R2.T).diagonal().sum() - 1.0) / 2.0
    return math.degrees(math.acos(float(cos.clamp(-1.0, 1.0))))


def _resid_rmsd(P: torch.Tensor, Q: torch.Tensor, R: torch.Tensor) -> float:
    """RMSD of P superimposed on Q by rotation R + optimal translation."""
    Pc = P - P.mean(0, keepdim=True)
    Qc = Q - Q.mean(0, keepdim=True)
    return float((Pc @ R.T - Qc).pow(2).sum(-1).mean().sqrt())


def _chain(n_res: int = 48, seed: int = 3):
    """Random residue types + their side-chain atom masks."""
    g = torch.Generator().manual_seed(seed)
    type_idx = torch.randint(0, len(STD_AA_3), (n_res,), generator=g)
    mask = sidechain_mask([STD_AA_3[int(i)] for i in type_idx])
    return type_idx, mask


def _cloud(local: torch.Tensor, mask: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Flatten valid local atoms and map them to global through a single rotation."""
    pts = local[mask]                                    # [K, 3]
    return pts @ R.T


# --- leakage guard -----------------------------------------------------------
def test_no_gt_argument():
    """template_init_local may see residue TYPE + atom mask, never GT coords."""
    params = set(inspect.signature(template_init_local).parameters)
    assert not (params & {"gt", "gt_coords", "x0", "true", "ground_truth", "sc_gt_local"})
    assert params == {"type_idx", "mask", "sigma_T", "generator"}


# --- shape / mask / determinism ---------------------------------------------
def test_shape_and_zeroing():
    type_idx, mask = _chain(n_res=12)
    y = template_init_local(type_idx, mask, generator=torch.Generator().manual_seed(0))
    assert y.shape == (12, MAX_SC, 3)
    assert y.dtype == torch.float32
    assert torch.count_nonzero(y[~mask]) == 0            # padded slots stay zero
    assert torch.count_nonzero(y[mask]) > 0              # valid slots are populated


def test_batched_leading_dims():
    type_idx, mask = _chain(n_res=7)
    B = 4
    tix = type_idx[None].expand(B, -1)
    m = mask[None].expand(B, -1, -1)
    y = template_init_local(tix, m, generator=torch.Generator().manual_seed(0))
    assert y.shape == (B, 7, MAX_SC, 3)
    # independent eps per row (this is what the per-sigma tiling relies on)
    assert not torch.allclose(y[0], y[1])


def test_deterministic_under_seed():
    type_idx, mask = _chain(n_res=9)
    a = template_init_local(type_idx, mask, generator=torch.Generator().manual_seed(7))
    b = template_init_local(type_idx, mask, generator=torch.Generator().manual_seed(7))
    assert torch.allclose(a, b)


def test_sigma_T_zero_reproduces_ideal_template():
    type_idx, mask = _chain(n_res=20)
    y = template_init_local(type_idx, mask, sigma_T=0.0)
    mu, tmask = init_mod._ideal_template(type_idx)
    expect = mu.float() * (mask & tmask)[..., None]
    assert torch.allclose(y, expect, atol=0)


def test_sigma_T_perturbs_but_preserves_template():
    """sigma_T must be small enough that the template survives it."""
    type_idx, mask = _chain(n_res=64)
    mu, tmask = init_mod._ideal_template(type_idx)
    valid = mask & tmask
    y = template_init_local(
        type_idx, mask, sigma_T=DEFAULT_SIGMA_T, generator=torch.Generator().manual_seed(2)
    )
    dev = (y - mu.float() * valid[..., None])[valid]
    assert abs(dev.std().item() - DEFAULT_SIGMA_T) < 0.05
    # perturbation is small relative to the template's own radial scale
    assert dev.norm(dim=-1).mean() < 0.25 * mu.float()[valid].norm(dim=-1).mean()


# --- THE KEY PROPERTY --------------------------------------------------------
def _orientation_recovery(init_fn, n_trials: int = 20):
    """Median error (deg) in recovering the relative backbone rotation from two
    INDEPENDENTLY-drawn global initializations, and the median residual RMSD when
    superimposing them by the TRUE relative rotation."""
    type_idx, mask = _chain(n_res=48)
    angles, resids, trans_only = [], [], []
    for k in range(n_trials):
        R1 = _random_rotation(100 + 2 * k)
        R2 = _random_rotation(101 + 2 * k)
        assert _angle_deg(R1, R2) > 20.0                 # the rotations really differ
        # independent draws — the model draws fresh eps for every sigma row
        y1 = init_fn(type_idx, mask, torch.Generator().manual_seed(1000 + k))
        y2 = init_fn(type_idx, mask, torch.Generator().manual_seed(5000 + k))
        X1, X2 = _cloud(y1, mask, R1), _cloud(y2, mask, R2)
        R_true = R2 @ R1.T                               # x2 = R_true x1 (up to eps)
        angles.append(_angle_deg(_kabsch(X1, X2), R_true))
        resids.append(_resid_rmsd(X1, X2, R_true))
        trans_only.append(_resid_rmsd(X1, X2, torch.eye(3)))
    med = lambda v: float(torch.tensor(v).median())
    return med(angles), med(resids), med(trans_only)


def test_global_init_carries_backbone_orientation():
    """Template init: F_hat y_T encodes the backbone orientation.

    Two different backbone rotations produce global clouds that are NOT
    superimposable by translation alone, and the relative rotation is recoverable
    from the clouds themselves.
    """
    ang, resid, trans_only = _orientation_recovery(
        lambda t, m, g: template_init_local(t, m, sigma_T=DEFAULT_SIGMA_T, generator=g)
    )
    # the true relative rotation IS recoverable from the two point clouds
    assert ang < 10.0, f"template init failed to encode orientation ({ang:.1f} deg)"
    # ... and once applied, it superimposes them (residual ~ sqrt(2)*sigma_T)
    assert resid < 3.0 * DEFAULT_SIGMA_T
    # ... whereas translation alone does NOT superimpose them
    assert trans_only > 4.0 * resid


def test_gaussian_init_is_orientation_blind():
    """Negative control: the isotropic Gaussian init is rotation-invariant, so the
    backbone rotation is NOT recoverable from the global cloud — the very defect
    paragraph 221 fixes."""
    ang, resid, trans_only = _orientation_recovery(
        lambda t, m, g: gaussian_init_local(m, sigma=1.0, generator=g)
    )
    # recovered rotation is essentially arbitrary (random 3D rotations sit ~90 deg away)
    assert ang > 45.0, f"gaussian init unexpectedly encoded orientation ({ang:.1f} deg)"
    # the true rotation does not superimpose the clouds any better than doing nothing
    assert resid > 1.5
    assert trans_only < 1.5 * resid


def test_template_beats_gaussian_on_orientation():
    """Direct A/B of the two inits on the property the change is about."""
    ang_t, _, _ = _orientation_recovery(
        lambda t, m, g: template_init_local(t, m, sigma_T=DEFAULT_SIGMA_T, generator=g)
    )
    ang_g, _, _ = _orientation_recovery(
        lambda t, m, g: gaussian_init_local(m, sigma=1.0, generator=g)
    )
    assert ang_t < 0.25 * ang_g


def test_large_sigma_T_destroys_the_orientation_signal():
    """Guard on the hyperparameter: a big sigma_T washes out the template anisotropy
    and degrades template init back toward the orientation-blind Gaussian."""
    ang_small, _, _ = _orientation_recovery(
        lambda t, m, g: template_init_local(t, m, sigma_T=DEFAULT_SIGMA_T, generator=g)
    )
    ang_big, _, _ = _orientation_recovery(
        lambda t, m, g: template_init_local(t, m, sigma_T=8.0, generator=g)
    )
    assert ang_big > 3.0 * ang_small


# --- config wiring -----------------------------------------------------------
def test_config_defaults():
    from pxdesign_train.configs.configs_train import training_configs

    sc = training_configs["sidechain"]
    assert sc["template_init"] is True          # paragraph 221 on by default
    assert sc["init_sigma_T"] == DEFAULT_SIGMA_T
    assert sc["init_sigma"] == 1.0              # old Gaussian knob preserved for A/B


# --- real table (skipped until the templates agent lands it) ------------------
@pytest.mark.skipif(not HAVE_TEMPLATES, reason="templates.py not present yet")
def test_real_table_matches_instantiate_layout():
    from pxdesign_train.sidechain.templates import (
        IDEAL_SC_LOCAL,
        IDEAL_SC_MASK,
        ideal_template,
    )

    assert IDEAL_SC_LOCAL.shape == (len(STD_AA_3), MAX_SC, 3)
    assert IDEAL_SC_MASK.shape == (len(STD_AA_3), MAX_SC)
    # column layout must agree with instantiate.sidechain_mask (atom counts/order)
    assert torch.equal(IDEAL_SC_MASK.cpu().bool(), sidechain_mask(STD_AA_3))
    # anisotropic: the template mean is far from the origin (unlike Gaussian noise)
    c, m = ideal_template(torch.arange(len(STD_AA_3)))
    assert c[m].norm(dim=-1).mean() > 1.0
