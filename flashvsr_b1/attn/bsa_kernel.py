from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import torch
from einops import rearrange


_BSA_MISSING_MSG = (
    "block_sparse_attn library required for BSA mode — install from FlashVSR repo"
)
_REF_MODULE_NAME = "flashvsr_b1_ref_wan_video_dit"
_REF_PATH = Path(__file__).resolve().parents[2] / "wan_video_dit.py"


def topk_for(sparsity: float, total_kv_blocks: int) -> int:
    """active = max(1, round(total * (1 - sparsity)))."""
    return max(1, int(round(total_kv_blocks * (1.0 - sparsity))))


@lru_cache(maxsize=1)
def _load_reference_module():
    spec = importlib.util.spec_from_file_location(_REF_MODULE_NAME, _REF_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load FlashVSR reference module at {_REF_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(_REF_MODULE_NAME, mod)
    spec.loader.exec_module(mod)
    return mod


def _partition_for_bsa(
    x: torch.Tensor,
    *,
    block_size: tuple[int, int, int],
    grid_shape: tuple[int, int, int],
) -> torch.Tensor:
    B, S, D = x.shape
    f, h, w = grid_shape
    bt, bh, bw = block_size
    if S != f * h * w:
        raise ValueError(
            f"Q/K/V sequence length {S} must equal grid_shape product {f * h * w}"
        )
    if f % bt != 0 or h % bh != 0 or w % bw != 0:
        raise ValueError(
            "grid_shape axes must be divisible by block_size; "
            f"got grid_shape={grid_shape}, block_size={block_size}"
        )

    x = x.view(B, f, h, w, D)
    x = x.view(B, f // bt, bt, h // bh, bh, w // bw, bw, D)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return x.view(-1, bt * bh * bw, D)


def _reverse_bsa_partition(
    x: torch.Tensor,
    *,
    block_size: tuple[int, int, int],
    grid_shape: tuple[int, int, int],
    batch_size: int,
) -> torch.Tensor:
    f, h, w = grid_shape
    bt, bh, bw = block_size
    D = x.shape[-1]

    x = x.view(batch_size, f // bt, h // bh, w // bw, bt, bh, bw, D)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(batch_size, f * h * w, D)


def _block_sparse_attention(
    block_sparse_attn_func,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    num_heads: int,
) -> torch.Tensor:
    seqlen = q.shape[1]
    seqlen_kv = k.shape[1]
    q = rearrange(q, "b s (n d) -> (b s) n d", n=num_heads)
    k = rearrange(k, "b s (n d) -> (b s) n d", n=num_heads)
    v = rearrange(v, "b s (n d) -> (b s) n d", n=num_heads)
    cu_seqlens_q = torch.tensor([0, seqlen], device=q.device, dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, seqlen_kv], device=q.device, dtype=torch.int32)
    head_mask_type = torch.tensor([1] * num_heads, device=q.device, dtype=torch.int32)
    x = block_sparse_attn_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        head_mask_type,
        None,
        attention_mask,
        seqlen,
        seqlen_kv,
        0.0,
        deterministic=False,
        softmax_scale=None,
        is_causal=False,
        exact_streaming=False,
        return_attn_probs=False,
    ).unsqueeze(0)
    return rearrange(x, "b s n d -> b s (n d)", n=num_heads)


def bsa_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, *,
                block_size: tuple[int, int, int],
                grid_shape: tuple[int, int, int],
                current_sparsity: float,
                num_heads: int,
                local_window_mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Wraps block_sparse_attn_func with causal block sparse mask derived from
    current_sparsity. Falls back gracefully (with a clear exception message) when
    block_sparse_attn is not importable — DO NOT silently switch to SDPA.
    """
    if Q.shape != K.shape or Q.shape != V.shape:
        raise ValueError("Q, K, and V must have identical shapes")
    if Q.ndim != 3:
        raise ValueError("Q, K, and V must have shape (B, S, D)")

    B, _, _ = Q.shape
    q_w = _partition_for_bsa(Q, block_size=block_size, grid_shape=grid_shape)
    k_w = _partition_for_bsa(K, block_size=block_size, grid_shape=grid_shape)
    v_w = _partition_for_bsa(V, block_size=block_size, grid_shape=grid_shape)

    try:
        from block_sparse_attn import block_sparse_attn_func
    except ImportError as exc:
        raise RuntimeError(_BSA_MISSING_MSG) from exc

    block_n = q_w.shape[0] // B
    block_s = q_w.shape[1]
    block_n_kv = k_w.shape[0] // B
    reorder_q = rearrange(
        q_w, "(b block_n) block_s d -> b (block_n block_s) d",
        block_n=block_n, block_s=block_s,
    )
    reorder_k = rearrange(
        k_w, "(b block_n) block_s d -> b (block_n block_s) d",
        block_n=block_n_kv, block_s=block_s,
    )
    reorder_v = rearrange(
        v_w, "(b block_n) block_s d -> b (block_n block_s) d",
        block_n=block_n_kv, block_s=block_s,
    )

    ref_mod = _load_reference_module()
    f, h, w = grid_shape
    bt, bh, bw = block_size
    total_kv_blocks = (f // bt) * (h // bh) * (w // bw)
    topk = topk_for(current_sparsity, total_kv_blocks)
    if local_window_mask is None:
        local_window_mask = ref_mod.build_local_block_mask_shifted_vec_normal_slide(
            h // bh, w // bw, 9, 9, include_self=True, device=K.device
        )

    seqlen = grid_shape[0] // block_size[0]
    attention_mask = ref_mod.generate_draft_block_mask(
        B,
        num_heads,
        seqlen,
        q_w,
        k_w,
        topk=topk,
        local_attn_mask=local_window_mask,
    )

    out = _block_sparse_attention(
        block_sparse_attn_func,
        reorder_q,
        reorder_k,
        reorder_v,
        attention_mask,
        num_heads=num_heads,
    )
    out = rearrange(
        out, "b (block_n block_s) d -> (b block_n) block_s d",
        block_n=block_n, block_s=block_s,
    )
    return _reverse_bsa_partition(
        out,
        block_size=block_size,
        grid_shape=grid_shape,
        batch_size=B,
    )
