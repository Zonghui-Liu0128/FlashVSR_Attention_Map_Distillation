import math

import torch.nn as nn


def cosine_sparsity_ramp(
    step: int,
    *,
    ramp_end_step: int,
    init: float = 0.85,
    target: float = 0.90,
) -> float:
    if step >= ramp_end_step:
        return target
    p = step / ramp_end_step
    return init + (target - init) * 0.5 * (1.0 - math.cos(math.pi * p))


def set_current_sparsity(model: nn.Module, rate: float) -> None:
    for m in model.modules():
        if hasattr(m, "current_sparsity"):
            m.current_sparsity = rate
