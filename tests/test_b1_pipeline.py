from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn

from flashvsr_b1.models.wan_dit_b1 import SelfAttentionB1
from flashvsr_b1.pipelines.b1_pipeline import B1Pipeline


class DummySelfAttention(nn.Module):
    def __init__(self, dim=8, num_heads=2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)


def _fake_dit(num_blocks=30):
    blocks = [
        SimpleNamespace(self_attn=DummySelfAttention())
        for _ in range(num_blocks)
    ]
    return SimpleNamespace(blocks=blocks)


def _cfg(**overrides):
    values = {
        "teacher_ckpt": "/mock/teacher.safetensors",
        "student_ckpt": "/mock/student.safetensors",
        "tc_decoder_ckpt": None,
        "lq_proj_ckpt": None,
        "block_size": (2, 8, 8),
        "window_size": (2, 21, 21),
        "dim": 8,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_pipeline_replaces_self_attn_with_b1_variant():
    """After B1Pipeline.from_b1_config, every block.self_attn must be SelfAttentionB1."""
    fake_pipe = SimpleNamespace(dit=_fake_dit())

    with (
        patch(
            "flashvsr_b1.pipelines.b1_pipeline.WanVideoPipeline.from_pretrained",
            return_value=fake_pipe,
        ) as from_pretrained,
        patch("flashvsr_b1.pipelines.b1_pipeline.Causal_LQ4x_Proj", return_value=nn.Identity(), create=True),
        patch("flashvsr_b1.pipelines.b1_pipeline.build_tc_decoder", return_value=nn.Identity(), create=True),
        patch("flashvsr_b1.pipelines.b1_pipeline._build_lpips_net", return_value=MagicMock(), create=True),
    ):
        pipe = B1Pipeline.from_b1_config(_cfg())

    from_pretrained.assert_called_once()
    assert all(isinstance(block.self_attn, SelfAttentionB1) for block in pipe.dit.blocks)
    assert {block.self_attn.block_size for block in pipe.dit.blocks} == {(2, 8, 8)}


def test_pipeline_asserts_block_size_match():
    """If cfg block sizes differ between teacher and student parts, init must raise."""
    with pytest.raises(AssertionError, match="block_size"):
        B1Pipeline.from_b1_config(
            _cfg(teacher_block_size=(2, 8, 8), student_block_size=(1, 8, 8))
        )


def test_pipeline_distill_layers_default():
    """B1WanModel inside the pipeline has distill_layers == {4,9,14,19,24,29} by default."""
    fake_pipe = SimpleNamespace(dit=_fake_dit())

    with (
        patch(
            "flashvsr_b1.pipelines.b1_pipeline.WanVideoPipeline.from_pretrained",
            return_value=fake_pipe,
        ),
        patch("flashvsr_b1.pipelines.b1_pipeline.Causal_LQ4x_Proj", return_value=nn.Identity(), create=True),
        patch("flashvsr_b1.pipelines.b1_pipeline.build_tc_decoder", return_value=nn.Identity(), create=True),
        patch("flashvsr_b1.pipelines.b1_pipeline._build_lpips_net", return_value=None, create=True),
    ):
        pipe = B1Pipeline.from_b1_config(_cfg())

    assert pipe.dit.distill_layers == {4, 9, 14, 19, 24, 29}
