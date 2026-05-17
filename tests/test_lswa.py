import importlib.util
import sys
import types

import torch

from flashvsr_b1.attn.lswa import lswa_forward


REF_PATH = "/Users/zonghuiliu/Documents/Codex/VideoGen/FlashVSR_Attention_Map_Distillation/wan_video_dit.py"


def _load_ref_module():
    """Load the root wan_video_dit.py as a module without polluting sys.path."""
    spec = importlib.util.spec_from_file_location("ref_wan_video_dit", REF_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("ref_wan_video_dit", mod)
    old_utils = sys.modules.get("utils")
    shim = types.ModuleType("utils")
    shim.hash_state_dict_keys = lambda state_dict: state_dict
    sys.modules["utils"] = shim
    try:
        spec.loader.exec_module(mod)
    finally:
        if old_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = old_utils
    return mod


def test_lswa_output_shape_train_mode():
    torch.manual_seed(0)
    B, f, h, w, D = 1, 4, 16, 16, 64
    Q = torch.randn(B, f * h * w, D)
    K = torch.randn(B, f * h * w, D)
    V = torch.randn(B, f * h * w, D)
    out = lswa_forward(Q, K, V, window_size=(2, 5, 5), f=f, h=h, w=w)
    assert out.shape == (B, f * h * w, D)


def test_lswa_is_causal_in_time():
    """window_t=2: query frame t only sees K/V at frames t-1 and t."""
    torch.manual_seed(0)
    B, f, h, w, D = 1, 4, 4, 4, 16
    Q = torch.randn(B, f * h * w, D)
    K = torch.randn(B, f * h * w, D)
    V = torch.randn(B, f * h * w, D)
    out_a = lswa_forward(Q, K, V, window_size=(2, 3, 3), f=f, h=h, w=w)
    K2 = K.clone()
    V2 = V.clone()
    K2[:, 3 * h * w :, :] += 1.0
    V2[:, 3 * h * w :, :] += 1.0
    out_b = lswa_forward(Q, K2, V2, window_size=(2, 3, 3), f=f, h=h, w=w)
    assert torch.allclose(out_a[:, : 2 * h * w, :], out_b[:, : 2 * h * w, :], atol=1e-5)


def test_lswa_matches_reference_implementation():
    """
    Numerical parity with root wan_video_dit.py _lswa_forward.

    The reference helper takes projected dense tensors shaped [B, S, D] and
    returns (self.o(attn_out), last_k, last_v). The standalone helper returns
    only attn_out, so set self.o to Identity for an exact math comparison.
    """
    ref = _load_ref_module()

    torch.manual_seed(0)
    B, f, h, w = 1, 4, 8, 8
    D = 96
    num_heads = 12
    Q = torch.randn(B, f * h * w, D)
    K = torch.randn(B, f * h * w, D)
    V = torch.randn(B, f * h * w, D)

    sa_cls = ref.SelfAttention
    sa = sa_cls(
        dim=D,
        num_heads=num_heads,
        lswa_spatial_window=(5, 5),
        lswa_temporal_window=2,
    ).eval()
    sa.o = torch.nn.Identity()

    with torch.no_grad():
        out_ref, _, _ = sa._lswa_forward(
            Q,
            K,
            V,
            f=f,
            h=h,
            w=w,
            is_stream=False,
            pre_cache_k=None,
            pre_cache_v=None,
        )

    out_ours = lswa_forward(Q, K, V, window_size=(2, 5, 5), f=f, h=h, w=w, is_stream=False)

    assert out_ref.shape == out_ours.shape, (out_ref.shape, out_ours.shape)
    assert torch.allclose(out_ref, out_ours, atol=1e-5), (
        f"Max abs diff: {(out_ref - out_ours).abs().max().item():.6f}"
    )
