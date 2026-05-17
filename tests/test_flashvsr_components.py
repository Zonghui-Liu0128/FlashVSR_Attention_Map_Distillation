import torch
from flashvsr_b1.models.flashvsr_components import (
    FlashVSRTinyConfig, Causal_LQ4x_Proj, build_tc_decoder,
)

def test_tiny_config_defaults():
    c = FlashVSRTinyConfig()
    assert c.num_layers == 30
    assert c.num_heads == 12
    assert c.dim == 1536
    assert c.in_dim == 16
    assert c.out_dim == 16

def test_lq_proj_forward_shape():
    proj = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1)
    x = torch.randn(1, 3, 4, 64, 96)
    out = proj(x)
    assert out.shape[1] == 1536

def test_tc_decoder_builds_without_checkpoint():
    dec = build_tc_decoder(checkpoint_path=None)
    assert dec is not None
