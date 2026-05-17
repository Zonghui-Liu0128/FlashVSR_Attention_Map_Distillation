import json
import time
from pathlib import Path

import torch


def _evaluate_one_video(ckpt_path, sample, cfg, device):
    """Run one validation video and return per-video SR/sparsity metrics."""
    raise NotImplementedError(
        "Real implementation deferred to B200 -- needs torchmetrics + lpips + pyiqa"
    )


def _measure_fps(ckpt_path, cfg, device):
    """Measure steady-state FPS at 720p and 1080p on production hardware."""
    raise NotImplementedError("Real implementation deferred to B200")


def evaluate_checkpoint(
    ckpt_path: str, val_json: str, cfg: dict, device: str = "cuda"
) -> dict:
    """
    Returns dict with keys: psnr, ssim, lpips, dists, sparsity_rate,
    fps_720p, fps_1080p, peak_mem_gb.
    """
    val = json.loads(Path(val_json).read_text())
    samples = val.get("samples", [])
    if not samples:
        raise ValueError(f"No samples found in validation json: {val_json}")

    per_sample = []
    for sample in samples:
        per_sample.append(_evaluate_one_video(ckpt_path, sample, cfg, device))

    fps = _measure_fps(ckpt_path, cfg, device)
    keys = ["psnr", "ssim", "lpips", "dists", "sparsity_rate", "peak_mem_gb"]
    agg = {key: sum(metrics[key] for metrics in per_sample) / len(per_sample) for key in keys}
    agg.update(fps)
    return agg
