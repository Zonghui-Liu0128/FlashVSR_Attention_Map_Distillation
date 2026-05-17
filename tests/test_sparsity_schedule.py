import math

import torch

from flashvsr_b1.attn.sparsity_schedule import (
    cosine_sparsity_ramp,
    set_current_sparsity,
)


def test_ramp_init():
    assert cosine_sparsity_ramp(0, ramp_end_step=12000, init=0.85, target=0.90) == 0.85


def test_ramp_clamps_to_target():
    assert cosine_sparsity_ramp(15000, ramp_end_step=12000, target=0.90) == 0.90
    assert cosine_sparsity_ramp(12000, ramp_end_step=12000, target=0.90) == 0.90


def test_ramp_monotonic_increasing():
    vals = [
        cosine_sparsity_ramp(s, ramp_end_step=12000, target=0.90)
        for s in range(0, 12000, 200)
    ]
    for a, b in zip(vals, vals[1:]):
        assert b >= a - 1e-9


def test_ramp_midpoint():
    mid = cosine_sparsity_ramp(6000, ramp_end_step=12000, init=0.85, target=0.90)
    expected = 0.85 + (0.90 - 0.85) * 0.5 * (1 - math.cos(math.pi * 0.5))
    assert abs(mid - expected) < 1e-6


def test_set_current_sparsity_writes_to_marked_modules_only():
    class A(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.current_sparsity = 0.85

    class B(torch.nn.Module):
        pass

    root = torch.nn.Module()
    root.a = A()
    root.b = B()
    set_current_sparsity(root, 0.93)
    assert root.a.current_sparsity == 0.93
    assert not hasattr(root.b, "current_sparsity")
