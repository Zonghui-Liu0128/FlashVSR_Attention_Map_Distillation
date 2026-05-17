import torch
from flashvsr_b1.models.wan_dit_b1 import SelfAttentionB1, B1WanModel

def test_self_attention_default_attrs():
    sa = SelfAttentionB1(dim=1536, num_heads=12)
    assert sa.attn_mode == "BSA"
    assert sa.current_sparsity == 0.85
    assert sa.block_size == (2, 8, 8)
    assert sa.window_size == (2, 21, 21)
    assert sa.distill_export is False

def test_self_attention_forward_lswa_mode_no_aux():
    sa = SelfAttentionB1(dim=64, num_heads=4)
    sa.attn_mode = "LSWA"
    sa.window_size = (2, 3, 3)
    x = torch.randn(1, 4*4*4, 64)
    out = sa(x, freqs=None, f=4, h=4, w=4)
    assert isinstance(out, torch.Tensor)
    assert out.shape == x.shape

def test_self_attention_returns_aux_for_distill_layer_lswa():
    sa = SelfAttentionB1(dim=64, num_heads=4, distill_export=True)
    sa.attn_mode = "LSWA"
    sa.window_size = (2, 3, 3)
    x = torch.randn(1, 4*4*4, 64)
    out, aux = sa(x, freqs=None, f=4, h=4, w=4, return_aux=True)
    assert "h_out" in aux
    assert "A_blk" not in aux

def test_b1_wan_model_distill_layers_default():
    m = B1WanModel.__new__(B1WanModel)
    m._init_distill_layers_for_test()
    assert m.distill_layers == {4, 9, 14, 19, 24, 29}
