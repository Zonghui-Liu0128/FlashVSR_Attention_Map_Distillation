import pytest, torch
from flashvsr_b1.attn.bsa_kernel import topk_for, bsa_forward


def test_topk_for_85pct():
    assert topk_for(0.85, 1320) == 198


def test_topk_for_90pct():
    assert topk_for(0.90, 1320) == 132


def test_topk_for_95pct():
    assert topk_for(0.95, 1320) == 66


def test_topk_for_clamps_to_one():
    assert topk_for(0.999, 10) == 1


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="bsa_forward requires CUDA + block_sparse_attn lib")
def test_bsa_forward_shape():
    """block_sparse_attn library only accepts fp16/bf16. Production reaches
    bsa_forward inside autocast(bf16); mirror that explicitly here."""
    torch.manual_seed(0)
    B, T, H_lat, W_lat, D, H = 1, 4, 8, 8, 128, 4
    S = T * H_lat * W_lat
    Q = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)
    out = bsa_forward(Q, K, V,
                      block_size=(2,8,8), grid_shape=(T, H_lat, W_lat),
                      current_sparsity=0.85,
                      num_heads=H, local_window_mask=None)
    assert out.shape == (B, S, D)
    assert out.dtype == torch.bfloat16, (
        f"bsa_forward should preserve bf16, got {out.dtype}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="parity test requires CUDA")
def test_bsa_parity_with_root_implementation():
    """Shape parity (not numerical) between our bsa_forward and the root
    `SelfAttention._block_sparse_forward`.

    We deliberately layer an extra block-time causal mask on top of
    `generate_draft_block_mask` (see flashvsr_b1/attn/bsa_kernel.py line 184),
    so block-by-block we mask future-time blocks that the reference path
    does not. Numerical equality is therefore NOT expected and was never
    actually verified (the test failed before this fix with FileNotFoundError
    / ModuleNotFoundError / fp32-rejected). We assert shape parity only —
    this still catches grid_shape / block_size / partition regressions.
    """
    import importlib.util, sys, types
    from pathlib import Path
    REF_PATH = Path(__file__).resolve().parents[1] / "wan_video_dit.py"
    if not REF_PATH.exists():
        pytest.skip(f"wan_video_dit.py reference not present at {REF_PATH}")
    spec = importlib.util.spec_from_file_location("ref_wan_video_dit", REF_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("ref_wan_video_dit", mod)
    # wan_video_dit.py has `from .utils import hash_state_dict_keys` and falls
    # back to `from utils import hash_state_dict_keys` outside a package; we
    # only need the import to succeed, so stub it (same pattern as test_lswa.py).
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

    torch.manual_seed(0)
    B, f, h, w = 1, 22, 16, 16
    D = 96
    num_heads = 12
    # block_sparse_attn only accepts fp16/bf16; both bsa_forward and the
    # reference SelfAttention internally call it.
    Q = torch.randn(B, f*h*w, D, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(B, f*h*w, D, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(B, f*h*w, D, device="cuda", dtype=torch.bfloat16)

    sa = mod.SelfAttention(dim=D, num_heads=num_heads).eval().cuda().bfloat16()
    total_blocks = (f//2) * (h//8) * (w//8)
    topk = max(1, int(round(total_blocks * 0.15)))
    local_range = 9
    window_size = 2 * h * w // 128
    seqlen = f // 2
    # Reference forward derives kv_len from randomized kv_ratio; this direct
    # no-cache parity path only needs a concrete int. Validate on B200.
    kv_len = min(max(int(window_size * local_range), 1),
                 int(window_size * seqlen) - 1)

    with torch.no_grad():
        out_ref = sa._block_sparse_forward(
            Q, K, V, B, f, h, w, D,
            local_num=0, topk=topk,
            kv_len=kv_len,
            is_stream=False, pre_cache_k=None, pre_cache_v=None,
            local_range=local_range,
        )
        if isinstance(out_ref, tuple):
            out_ref = out_ref[0]
        # Both bsa_forward and the reference sparse path are compared at the
        # tensor output returned by their block-sparse implementations.
        out_ours = bsa_forward(Q, K, V,
                                block_size=(2,8,8),
                                grid_shape=(f, h, w),
                                current_sparsity=0.85,
                                num_heads=num_heads,
                                local_window_mask=None)

    assert out_ref.shape == out_ours.shape
