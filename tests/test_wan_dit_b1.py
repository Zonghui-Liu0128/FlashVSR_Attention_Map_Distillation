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


def test_b1_forward_threads_LQ_latents_to_block_loop():
    """Fix H: LR_latents must reach B1WanModel.forward as LQ_latents kwarg
    and be added per-block. Pin this contract so a future refactor doesn't
    accidentally regress to additive-before-patchify."""
    import torch
    import torch.nn as nn
    from flashvsr_b1.models.wan_dit_b1 import B1WanModel

    captured = {}

    class SpyB1(B1WanModel):
        def forward(self, x, timestep, context, LQ_latents=None, return_aux=False, **kwargs):
            captured["x_shape"] = tuple(x.shape)
            captured["LQ_latents_is_list"] = isinstance(LQ_latents, list)
            captured["LQ_latents_len"] = len(LQ_latents) if LQ_latents is not None else None
            if LQ_latents:
                captured["LQ_latents_0_ndim"] = LQ_latents[0].ndim
                captured["LQ_latents_0_last_dim"] = LQ_latents[0].shape[-1]
            if return_aux:
                return torch.zeros(1), {}
            return torch.zeros(1)

    spy = SpyB1.__new__(SpyB1)
    nn.Module.__init__(spy)
    spy.text_dim = 4096

    LR_latents = [torch.randn(1, 8, 1536)]
    z_t = torch.randn(1, 16, 1, 2, 4)
    t_star = torch.tensor(999)

    spy.b1_forward(LR_latents, z_t, t_star, return_aux=True)

    assert captured["x_shape"] == tuple(z_t.shape), (
        "b1_forward must pass z_t as x to forward(), NOT z_t + LR"
    )
    assert captured["LQ_latents_is_list"], (
        "LR_latents must reach forward() as a list (upstream contract)"
    )
    assert captured["LQ_latents_len"] == 1
    assert captured["LQ_latents_0_ndim"] == 3, (
        f"LQ_latents[0] must be 3D (B, N, dim); got ndim={captured['LQ_latents_0_ndim']}"
    )
    assert captured["LQ_latents_0_last_dim"] == 1536, (
        f"LQ_latents[0] last dim {captured['LQ_latents_0_last_dim']} != dim=1536"
    )


def test_b1_forward_rejects_tensor_LR_latents():
    """Defense-in-depth: if a caller forgets to wrap in list, b1_forward
    must raise - not silently broadcast."""
    import torch
    import torch.nn as nn
    import pytest
    from flashvsr_b1.models.wan_dit_b1 import B1WanModel

    class StubB1(B1WanModel):
        def forward(self, *args, **kwargs):
            return torch.zeros(1)

    stub = StubB1.__new__(StubB1)
    nn.Module.__init__(stub)
    stub.text_dim = 4096

    z_t = torch.randn(1, 16, 1, 2, 4)
    bad_LR_latent = torch.randn(1, 8, 1536)  # tensor, not list
    with pytest.raises(ValueError, match="list"):
        stub.b1_forward(bad_LR_latent, z_t, torch.tensor(999))


def test_b1_forward_handles_diffsynth_patchify_tensor_contract():
    """DiffSynth's vendored WanModel.patchify returns only a 5D tensor.

    B1WanModel.forward must adapt that contract by deriving the token grid and
    flattening to (B, N, dim) before entering the block loop.
    """
    model = B1WanModel(
        dim=12,
        in_dim=2,
        ffn_dim=24,
        out_dim=2,
        text_dim=4096,
        freq_dim=8,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=3,
        num_layers=0,
        has_image_input=False,
        distill_layers=[],
    )

    z_t = torch.randn(1, 2, 2, 4, 4)
    LR_latents = [torch.zeros(1, 8, 12)]

    out, aux = model.b1_forward(LR_latents, z_t, torch.tensor(999), return_aux=True)

    assert out.shape == z_t.shape
    assert aux == {}
