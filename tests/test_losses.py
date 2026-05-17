import math
import pytest
import torch
from flashvsr_b1.losses.output_loss import L_output
from flashvsr_b1.losses.block_kl_loss import L_block
from flashvsr_b1.losses.attn_out_loss import L_attn_out

def test_L_output_zero_on_equal_tensors():
    x = torch.randn(2, 3, 4)
    assert L_output(x, x).item() < 1e-8

def test_L_output_huber_smoothness_near_zero():
    """For small diff, Huber ≈ 0.5 * (diff^2) / beta. β=0.1 → at diff=0.05, loss=0.5·0.0025/0.1=0.0125."""
    x = torch.zeros(1); y = torch.full((1,), 0.05)
    assert abs(L_output(x, y).item() - 0.0125) < 1e-4

def test_L_block_zero_on_identical_distributions():
    p = torch.rand(2, 4, 8, 8)
    p = p / p.sum(dim=-1, keepdim=True)
    loss = L_block(p, p.clone())
    assert loss.item() < 1e-5

def test_L_block_positive_on_different_distributions():
    p = torch.zeros(1, 1, 1, 4); p[..., 0] = 1.0          # concentrated at 0
    q = torch.zeros(1, 1, 1, 4); q[..., 3] = 1.0          # concentrated at 3
    loss = L_block(p, q)
    assert loss.item() > 1.0

def test_L_block_grad_flows_to_student_only():
    p = torch.rand(1, 1, 4, 4); p = p / p.sum(-1, keepdim=True); p = p.detach()
    q_logits = torch.randn(1, 1, 4, 4, requires_grad=True)
    q = q_logits.softmax(-1)
    L_block(p, q).backward()
    assert q_logits.grad is not None

def test_L_attn_out_zero_on_equal():
    h = torch.randn(2, 4, 8)
    assert L_attn_out(h, h.detach()).item() < 1e-8

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

@pytest.mark.skipif(not HAS_LPIPS, reason="lpips not installed")
def test_L_lpips_shape():
    from flashvsr_b1.losses.lpips_loss import L_lpips
    class IdentityDecoder:
        def __call__(self, x): return x[:, :3]
    net = lpips.LPIPS(net="vgg").eval()
    x_s = torch.randn(1, 16, 4, 32, 32)
    gt  = torch.randn(1, 3, 32, 32)
    loss = L_lpips(x_s, gt, IdentityDecoder(), net)
    assert loss.dim() == 0
