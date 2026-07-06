"""Tests for residue-local frame transforms."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from pxdesign_train.sidechain.frames import build_frame, to_local, to_global


def test_frame_roundtrip_identity():
    torch.manual_seed(0)
    n, ca, c = torch.randn(5, 3), torch.randn(5, 3), torch.randn(5, 3)
    R, t = build_frame(n, ca, c)
    x = torch.randn(5, 8, 3)  # 8 side-chain atoms per residue
    y = to_local(x, R, t)
    x2 = to_global(y, R, t)
    assert torch.allclose(x, x2, atol=1e-5)


def test_frame_orthonormal():
    n, ca, c = torch.randn(4, 3), torch.randn(4, 3), torch.randn(4, 3)
    R, _ = build_frame(n, ca, c)
    eye = R @ R.transpose(-1, -2)
    assert torch.allclose(eye, torch.eye(3).expand_as(eye), atol=1e-5)
    # det ~ +1 (right-handed)
    assert torch.allclose(torch.linalg.det(R), torch.ones(4), atol=1e-4)


def test_frame_origin_is_ca():
    n, ca, c = torch.randn(3, 3), torch.randn(3, 3), torch.randn(3, 3)
    _, t = build_frame(n, ca, c)
    assert torch.allclose(t, ca)


def test_batched_leading_dims():
    # [B, L, 3] backbone -> [B, L, 3, 3] frames, [B, L, A, 3] coords.
    n, ca, c = torch.randn(2, 4, 3), torch.randn(2, 4, 3), torch.randn(2, 4, 3)
    R, t = build_frame(n, ca, c)
    assert R.shape == (2, 4, 3, 3) and t.shape == (2, 4, 3)
    x = torch.randn(2, 4, 6, 3)
    assert torch.allclose(to_global(to_local(x, R, t), R, t), x, atol=1e-5)
