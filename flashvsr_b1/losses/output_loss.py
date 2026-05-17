import torch
import torch.nn.functional as F


def L_output(x_s: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
    """Huber loss with beta=0.1."""
    return F.smooth_l1_loss(x_s, x_t, beta=0.1)
