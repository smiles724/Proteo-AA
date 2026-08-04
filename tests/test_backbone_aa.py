"""Contract tests for the leakage-resistant AA representation."""

import importlib.util
import unittest
from pathlib import Path

import torch


_MODULE_PATH = Path(__file__).resolve().parents[1] / "pxdesign_train" / "backbone_aa.py"
_SPEC = importlib.util.spec_from_file_location("backbone_aa_standalone", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
BackboneGeometryEncoder = _MODULE.BackboneGeometryEncoder


def _chain(n_residue: int = 6, n_extra: int = 7):
    n_atom = 4 * n_residue + n_extra
    xyz = torch.randn(n_atom, 3)
    idx = torch.arange(4 * n_residue).reshape(n_residue, 4)
    for i in range(n_residue):
        x = 3.8 * i
        xyz[idx[i]] = torch.tensor(
            [[x - 1.2, 0.7, 0.1], [x, 0.0, 0.0],
             [x + 1.3, 0.2, 0.2], [x + 1.8, 1.1, 0.3]]
        )
    return xyz, idx


class BackboneGeometryEncoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.encoder = BackboneGeometryEncoder(
            c_out=32, c_hidden=48, n_blocks=2, sigma_dim=8
        ).eval()
        self.xyz, self.idx = _chain()
        self.chain = torch.zeros(self.idx.shape[0], dtype=torch.long)
        self.resid = torch.arange(self.idx.shape[0])

    def _encode(self, xyz):
        return self.encoder(
            xyz,
            self.idx,
            asym_id=self.chain,
            residue_index=self.resid,
            sigma=torch.tensor(0.4),
        )

    def test_shape_and_finite(self):
        out = self._encode(self.xyz)
        self.assertEqual(out.shape, (self.idx.shape[0], 32))
        self.assertTrue(torch.isfinite(out).all())

    def test_global_rigid_transform_invariant(self):
        q, _ = torch.linalg.qr(torch.randn(3, 3))
        if torch.linalg.det(q) < 0:
            q[:, 0] *= -1
        moved = self.xyz @ q + torch.tensor([12.0, -4.0, 8.5])
        torch.testing.assert_close(self._encode(self.xyz), self._encode(moved), atol=2e-5, rtol=2e-5)

    def test_ungathered_sidechain_rows_are_invisible(self):
        changed = self.xyz.clone()
        changed[4 * self.idx.shape[0]:] = 1000.0 * torch.randn_like(
            changed[4 * self.idx.shape[0]:]
        )
        torch.testing.assert_close(self._encode(self.xyz), self._encode(changed), atol=0, rtol=0)

    def test_spatial_contact_changes_query_representation(self):
        encoder = BackboneGeometryEncoder(
            c_out=16, c_hidden=24, n_blocks=1, sigma_dim=8, spatial_neighbors=1
        ).eval()
        moved = self.xyz.clone()
        last = self.idx[-1]
        shift = self.xyz[self.idx[0, 1]] + torch.tensor([0.0, 2.0, 0.0]) - moved[last[1]]
        moved[last] += shift
        kwargs = dict(
            backbone_atom_idx=self.idx, asym_id=self.chain,
            residue_index=self.resid, sigma=torch.tensor(0.4),
        )
        before = encoder(self.xyz, **kwargs)
        after = encoder(moved, **kwargs)
        self.assertFalse(torch.allclose(before[0], after[0]))

    def test_batch_and_noise_sample_broadcast(self):
        coords = self.xyz.expand(2, 3, *self.xyz.shape).clone()
        indices = self.idx.expand(2, *self.idx.shape)
        chain = self.chain.expand(2, -1)
        resid = self.resid.expand(2, -1)
        sigma = torch.tensor([[0.04, 0.4, 4.0], [0.08, 0.8, 8.0]])
        out = self.encoder(
            coords, indices, asym_id=chain, residue_index=resid, sigma=sigma
        )
        self.assertEqual(out.shape, (2, 3, self.idx.shape[0], 32))

    def test_encoder_and_coordinate_gradients_exist(self):
        encoder = BackboneGeometryEncoder(
            c_out=16, c_hidden=24, n_blocks=1, sigma_dim=8
        )
        xyz = self.xyz.clone().requires_grad_(True)
        loss = encoder(
            xyz, self.idx, asym_id=self.chain, residue_index=self.resid,
            sigma=torch.tensor(1.0),
        ).square().mean()
        loss.backward()
        self.assertIsNotNone(xyz.grad)
        self.assertGreater(float(xyz.grad.abs().sum()), 0.0)
        self.assertTrue(any(p.grad is not None for p in encoder.parameters()))


if __name__ == "__main__":
    unittest.main()
