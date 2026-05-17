import math

from flashvsr_b1.attn.sparsity_schedule import cosine_sparsity_ramp


def lambda_at(step: int, *, total: int = 20000) -> dict:
    if step < 2000:
        return {"l1": 1.0, "l2": 0.5, "l3": 0.5, "l4": 0.1}
    if step < 15000:
        p = (step - 2000) / (15000 - 2000)
        l3 = 0.5 + (0.1 - 0.5) * 0.5 * (1.0 - math.cos(math.pi * p))
        return {"l1": 1.0, "l2": 0.5, "l3": l3, "l4": 0.1}
    return {"l1": 1.0, "l2": 1.0, "l3": 0.1, "l4": 0.05}


def sparsity_at(step: int, *, target: float, total: int = 20000) -> float:
    return cosine_sparsity_ramp(
        step, ramp_end_step=int(total * 0.6), init=0.85, target=target
    )
