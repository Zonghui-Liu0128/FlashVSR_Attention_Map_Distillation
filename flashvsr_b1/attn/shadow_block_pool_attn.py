import math

import torch


def block_mean_pool_3d(x: torch.Tensor,
                       block_size: tuple[int, int, int],
                       grid_shape: tuple[int, int, int]) -> torch.Tensor:
    """x: [B, H, S, d] with S = T*Hh*Ww; returns [B, H, N_blk, d] via 3D mean-pool."""
    B, H, S, d = x.shape
    T, Hh, Ww = grid_shape
    bt, bh, bw = block_size

    if S != T * Hh * Ww:
        raise ValueError("x.shape[2] must equal T*Hh*Ww from grid_shape")
    if T % bt != 0 or Hh % bh != 0 or Ww % bw != 0:
        raise ValueError("grid_shape dimensions must be divisible by block_size")

    T_blk = T // bt
    H_blk = Hh // bh
    W_blk = Ww // bw

    x_grid = x.reshape(B, H, T, Hh, Ww, d)
    x_blocks = x_grid.reshape(B, H, T_blk, bt, H_blk, bh, W_blk, bw, d)
    x_pooled = x_blocks.mean(dim=(3, 5, 7))
    return x_pooled.reshape(B, H, T_blk * H_blk * W_blk, d)


def shadow_block_pool_attn(Q: torch.Tensor, K: torch.Tensor, *,
                            block_size: tuple[int, int, int],
                            grid_shape: tuple[int, int, int],
                            causal: bool = True) -> torch.Tensor:
    """Q, K: [B, H, S, d]; returns [B, H, N_blk, N_blk] softmax-normalized,
    causal=True puts -inf on future blocks before softmax."""
    Q_blk = block_mean_pool_3d(Q, block_size, grid_shape)
    K_blk = block_mean_pool_3d(K, block_size, grid_shape)

    d = Q_blk.shape[-1]
    s = torch.einsum("bhid,bhjd->bhij", Q_blk, K_blk) / math.sqrt(d)

    if causal:
        N_blk = s.shape[-1]
        future_mask = torch.ones(
            N_blk, N_blk, dtype=torch.bool, device=s.device
        ).triu(diagonal=1)
        s = s.masked_fill(future_mask, float("-inf"))

    return s.softmax(dim=-1)
