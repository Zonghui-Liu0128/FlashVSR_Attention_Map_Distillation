import torch, math
from flashvsr_b1.attn.shadow_block_pool_attn import (
    block_mean_pool_3d, shadow_block_pool_attn,
)

def _set_seed(s=0):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def test_block_mean_pool_3d_shape_landscape():
    _set_seed()
    B, H, T, Hh, Ww, d = 2, 4, 22, 64, 120, 16
    x = torch.randn(B, H, T*Hh*Ww, d)
    out = block_mean_pool_3d(x, block_size=(2,8,8), grid_shape=(T, Hh, Ww))
    assert out.shape == (B, H, (T//2)*(Hh//8)*(Ww//8), d)
    assert out.shape[-2] == 11 * 8 * 15  # = 1320

def test_block_mean_pool_3d_shape_portrait():
    _set_seed()
    B, H, T, Hh, Ww, d = 2, 4, 22, 120, 64, 16
    x = torch.randn(B, H, T*Hh*Ww, d)
    out = block_mean_pool_3d(x, block_size=(2,8,8), grid_shape=(T, Hh, Ww))
    assert out.shape == (B, H, 11 * 15 * 8, d)

def test_block_mean_pool_3d_equals_explicit_mean():
    _set_seed()
    B, H, T, Hh, Ww, d = 1, 1, 2, 8, 8, 4
    x = torch.randn(B, H, T*Hh*Ww, d)
    out = block_mean_pool_3d(x, block_size=(2,8,8), grid_shape=(T, Hh, Ww))
    expected = x.mean(dim=2, keepdim=True)
    assert torch.allclose(out, expected, atol=1e-6)

def test_shadow_attention_shape_and_softmax():
    _set_seed()
    B, H, T, Hh, Ww, d = 1, 2, 22, 64, 120, 16
    Q = torch.randn(B, H, T*Hh*Ww, d)
    K = torch.randn(B, H, T*Hh*Ww, d)
    A = shadow_block_pool_attn(Q, K, block_size=(2,8,8),
                                grid_shape=(T, Hh, Ww), causal=True)
    assert A.shape == (B, H, 1320, 1320)
    row_sums = A.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

def test_shadow_attention_causal_zeros_future_columns():
    _set_seed()
    B, H, T, Hh, Ww, d = 1, 1, 4, 8, 8, 4
    Q = torch.randn(B, H, T*Hh*Ww, d)
    K = torch.randn(B, H, T*Hh*Ww, d)
    A = shadow_block_pool_attn(Q, K, block_size=(2,8,8),
                                grid_shape=(T, Hh, Ww), causal=True)
    assert A[0, 0, 0, 0] == 1.0
    assert A[0, 0, 0, 1] == 0.0

def test_shadow_attention_grad_flows_to_Q_and_K():
    _set_seed()
    # N_blk must be >= 2 AND the loss must depend on individual softmax
    # probabilities (not row sums, which are always 1.0 -> grad = 0).
    B, H, T, Hh, Ww, d = 1, 1, 4, 8, 8, 4   # N_blk = 2 * 1 * 1 = 2
    Q = torch.randn(B, H, T*Hh*Ww, d, requires_grad=True)
    K = torch.randn(B, H, T*Hh*Ww, d, requires_grad=True)
    A = shadow_block_pool_attn(Q, K, block_size=(2,8,8),
                                grid_shape=(T, Hh, Ww), causal=True)
    # Loss on a single probability (A[..., 1, 0]) - non-constant function of Q,K.
    A[..., 1, 0].sum().backward()
    assert Q.grad is not None and Q.grad.abs().sum() > 0
    assert K.grad is not None and K.grad.abs().sum() > 0
