import torch
from pxdesign_train.sidechain.coevolution import AResBSConcat
from pxdesign_train.sidechain.module import SideChainModule


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


def _sc_inputs():
    B, L, A = 1, 3, 10
    h_res = torch.randn(B, L, 8)
    logits = torch.randn(B, L, 20)
    ids = torch.randint(1, 5, (B, L, A))
    mask = torch.ones(B, L, A, dtype=torch.bool)
    noisy = torch.randn(B, L, A, 3)
    ca = torch.randn(B, L, 3)
    return h_res, logits, ids, mask, noisy, ca


def test_a_bs_concat_off_matches_baseline():
    torch.manual_seed(0)
    base = SideChainModule(c_res=8, c_atom=16, n_type=20, a_bs_concat=False).eval()
    on = SideChainModule(c_res=8, c_atom=16, n_type=20, a_bs_concat=True).eval()
    on.load_state_dict(base.state_dict(), strict=False)  # shared weights; fusion is zero-init identity
    h, l, ids, m, noisy, ca = _sc_inputs()
    with torch.no_grad():
        y0, _ = base(h, l, ids, m, noisy, torch.ones(1), ca_coords=ca)
        y1, _ = on(h, l, ids, m, noisy, torch.ones(1), ca_coords=ca)
    assert torch.allclose(y0, y1, atol=1e-6), "zero-init a_bs_concat must match baseline"


def test_a_bs_concat_changes_output_when_armed():
    torch.manual_seed(0)
    on = SideChainModule(c_res=8, c_atom=16, n_type=20, a_bs_concat=True).eval()
    torch.nn.init.normal_(on.a_bs_concat_fusion.mlp[-1].weight, std=0.1)
    h, l, ids, m, noisy, ca = _sc_inputs()
    with torch.no_grad():
        y_on, _ = on(h, l, ids, m, noisy, torch.ones(1), ca_coords=ca)
        on.a_bs_concat = False
        y_off, _ = on(h, l, ids, m, noisy, torch.ones(1), ca_coords=ca)
    assert not torch.allclose(y_on, y_off, atol=1e-6), "armed a_bs_concat must change the output"
