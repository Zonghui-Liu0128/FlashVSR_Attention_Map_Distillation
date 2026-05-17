"""Local Sparse Window Attention ported from root ``wan_video_dit.py``.

The reference ``SelfAttention._lswa_forward`` consumes projected dense Q/K/V
tensors shaped ``[B, S, D]`` where ``S = f * h * w``. This standalone module
keeps that layout and returns only the LSWA attention output ``[B, S, D]``;
the caller remains responsible for any output projection.

The original class stores ``num_heads`` separately. This standalone helper
requires callers to pass it explicitly so model config, tests, and runtime use
the same head partitioning.
"""

from __future__ import annotations

import torch


_DEFAULT_QUERY_CHUNK_SIZE = 1024


@torch.no_grad()
def _lswa_offsets(window_size: tuple[int, int, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    _, h_w, w_w = window_size
    r_half, c_half = h_w // 2, w_w // 2
    dr = torch.arange(-r_half, h_w - r_half, device=device)
    dc = torch.arange(-c_half, w_w - c_half, device=device)
    rr, cc = torch.meshgrid(dr, dc, indexing="ij")
    return rr.reshape(-1), cc.reshape(-1)


def _local_spatial_attention(
    q_frame: torch.Tensor,
    k_context: list[torch.Tensor],
    v_context: list[torch.Tensor],
    *,
    window_size: tuple[int, int, int],
    num_heads: int,
    h: int,
    w: int,
) -> torch.Tensor:
    """Chunked LSWA attention for one query frame.

    q_frame/k_context/v_context are dense per-frame tensors with shape
    (B, H*W, D). The method materializes only one query chunk of local spatial
    neighborhoods at a time, matching the reference implementation.
    """
    B, L, D = q_frame.shape
    assert L == h * w, "Frame token length mismatch."
    assert D % num_heads == 0, "Embedding dim must be divisible by num_heads."
    head_dim = D // num_heads
    qh = q_frame.view(B, L, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    kh_frames = [
        k_frame.view(B, L, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        for k_frame in k_context
    ]
    vh_frames = [
        v_frame.view(B, L, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        for v_frame in v_context
    ]
    off_r, off_c = _lswa_offsets(window_size, q_frame.device)
    n_spatial = int(off_r.numel())
    chunk_size = max(1, int(_DEFAULT_QUERY_CHUNK_SIZE))
    out = qh.new_empty(B, num_heads, L, head_dim)
    scale = head_dim**-0.5

    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        pos = torch.arange(start, end, device=q_frame.device)
        row = pos // w
        col = pos % w
        neigh_r = row[:, None] + off_r[None, :]
        neigh_c = col[:, None] + off_c[None, :]
        valid = (neigh_r >= 0) & (neigh_r < h) & (neigh_c >= 0) & (neigh_c < w)
        neigh_idx = neigh_r.clamp(0, h - 1) * w + neigh_c.clamp(0, w - 1)
        neigh_idx = neigh_idx.to(torch.long)

        q_chunk = qh[:, :, start:end, :]
        k_chunks = []
        v_chunks = []
        valid_chunks = []
        for k_frame, v_frame in zip(kh_frames, vh_frames):
            gather_idx = neigh_idx.view(1, 1, end - start, n_spatial, 1).expand(
                B, num_heads, end - start, n_spatial, head_dim
            )
            k_expand = k_frame.unsqueeze(2).expand(B, num_heads, end - start, L, head_dim)
            v_expand = v_frame.unsqueeze(2).expand(B, num_heads, end - start, L, head_dim)
            k_chunks.append(torch.gather(k_expand, 3, gather_idx))
            v_chunks.append(torch.gather(v_expand, 3, gather_idx))
            valid_chunks.append(valid)

        k_local = torch.cat(k_chunks, dim=3)
        v_local = torch.cat(v_chunks, dim=3)
        valid_local = torch.cat(valid_chunks, dim=1)
        scores = (q_chunk.unsqueeze(3) * k_local).sum(dim=-1) * scale
        scores = scores.masked_fill(~valid_local.view(1, 1, end - start, -1), -torch.finfo(scores.dtype).max)
        probs = torch.softmax(scores, dim=-1)
        out[:, :, start:end, :] = (probs.unsqueeze(-1) * v_local).sum(dim=3)

    return out.permute(0, 2, 1, 3).contiguous().view(B, L, D)


def lswa_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    window_size: tuple[int, int, int],
    num_heads: int,
    f: int,
    h: int,
    w: int,
    is_stream: bool = False,
    pre_cache_k: torch.Tensor | None = None,
    pre_cache_v: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Q, K, V: [B, S, D] where S = f * h * w.
    window_size: (window_t, window_h, window_w), e.g. (2, 21, 21).
    Returns: [B, S, D] attention output. Strictly causal in time.
    No attention logits returned (LSWA is L_block-exempt per task_b1 §4.1).
    """
    del is_stream
    B = Q.shape[0]
    L = h * w
    assert Q.shape == K.shape == V.shape, "Q, K, V shapes must match."
    assert Q.shape[1] == f * L, "Sequence length mismatch with provided (f,h,w)."

    q_frames = Q.split(L, dim=1)
    k_frames = K.split(L, dim=1)
    v_frames = V.split(L, dim=1)

    out_frames = []
    context_k: list[torch.Tensor] = []
    context_v: list[torch.Tensor] = []
    if pre_cache_k is not None and pre_cache_v is not None:
        if pre_cache_k.shape[1] == L and pre_cache_v.shape[1] == L:
            context_k.append(pre_cache_k)
            context_v.append(pre_cache_v)

    temporal_window = window_size[0]
    for i in range(f):
        q_i, k_i, v_i = q_frames[i], k_frames[i], v_frames[i]
        context_k.append(k_i)
        context_v.append(v_i)
        if len(context_k) > temporal_window:
            context_k = context_k[-temporal_window:]
            context_v = context_v[-temporal_window:]

        out_i = _local_spatial_attention(
            q_i,
            context_k,
            context_v,
            window_size=window_size,
            num_heads=num_heads,
            h=h,
            w=w,
        )
        out_frames.append(out_i)

    x = torch.cat(out_frames, dim=1)
    assert x.shape == (B, f * L, Q.shape[-1])
    return x
