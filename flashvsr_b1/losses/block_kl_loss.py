import torch


def L_block(
    A_blk_t_detached: torch.Tensor, A_blk_s: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """KL(teacher || student) on softmax-normalized [B,H,Nq,Nk] block-pool attn maps.
    Inputs are already softmaxed (clamped to >=eps before log)."""
    p = A_blk_t_detached
    q = A_blk_s.clamp_min(eps)
    return (p * (p.clamp_min(eps).log() - q.log())).sum(-1).mean()
