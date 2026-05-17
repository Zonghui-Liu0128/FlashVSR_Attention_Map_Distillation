import torch
import torch.nn.functional as F


def L_attn_out(h_s: torch.Tensor, h_t_detached: torch.Tensor) -> torch.Tensor:
    """Huber on hidden state per shadow layer."""
    return F.smooth_l1_loss(h_s, h_t_detached, beta=0.1)
