import torch
from pxdesign_train.sidechain.coevolution import AResBSConcat


def test_aresbsconcat_zero_init_is_identity():
    m = AResBSConcat(c_atom=16)
    pooled = torch.randn(2, 5, 16)
    a_token = torch.randn(2, 5, 16)
    out = m(pooled, a_token)
    assert torch.allclose(out, pooled, atol=1e-6), "zero-init must be exact identity"


def test_aresbsconcat_a_token_reaches_output_when_armed():
    m = AResBSConcat(c_atom=16)
    torch.nn.init.normal_(m.mlp[-1].weight, std=0.1)  # arm the residual branch
    pooled = torch.randn(1, 3, 16)
    out_a = m(pooled, torch.zeros(1, 3, 16))
    out_b = m(pooled, torch.ones(1, 3, 16))
    assert not torch.allclose(out_a, out_b), "a_token must influence the output when armed"
