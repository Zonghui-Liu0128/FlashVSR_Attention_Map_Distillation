import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist


def make_run_dir(log_root: str, config_path: str) -> str:
    stem = Path(config_path).stem
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(log_root, f"{ts}_{stem}")


class MetricsLogger:
    # 960x720@93 -> LQ projector token grid (ceil(T/4), H/16, W/16).
    SEQLEN_PER_VIDEO = 24 * 45 * 60

    JSONL_FIELDS = [
        "step",
        "epoch",
        "L_total",
        "L_out",
        "L_lpips",
        "L_block",
        "L_attn_out",
        "lam1",
        "lam2",
        "lam3",
        "lam4",
        "current_sparsity",
        "lr",
        "step_time_sec",
        "tokens_per_sec",
        "tokens_per_hour",
        "videos_per_hour",
        "global_batch",
        "world_size",
        "gpu_mem_alloc_gb",
        "gpu_mem_peak_gb",
    ]

    def __init__(
        self,
        run_dir: str,
        *,
        global_batch: int,
        world_size: int,
        log_every_steps: int = 50,
        ema_span: int = 100,
    ):
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "ckpt"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "eval"), exist_ok=True)

        self.is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
        self.run_dir = run_dir
        self.global_batch = global_batch
        self.world_size = world_size
        self.log_every = log_every_steps
        self.ema_alpha = 2.0 / (ema_span + 1)
        self.ema = {}

        self._window_start = time.perf_counter()
        self._window_steps = 0
        self._last_log_step = 0

        if self.is_rank0:
            self.console_fp = open(os.path.join(run_dir, "log.txt"), "a", buffering=1)
            self.jsonl_fp = open(
                os.path.join(run_dir, "train_metrics.jsonl"), "a", buffering=1
            )
            csv_path = os.path.join(run_dir, "train_metrics.csv")
            csv_new = not os.path.exists(csv_path)
            self.csv_fp = open(csv_path, "a", newline="", buffering=1)
            self.csv_w = csv.DictWriter(self.csv_fp, fieldnames=self.JSONL_FIELDS)
            if csv_new:
                self.csv_w.writeheader()

    def _ema_update(self, key, val):
        self.ema[key] = (
            val
            if key not in self.ema
            else self.ema[key] + self.ema_alpha * (val - self.ema[key])
        )
        return self.ema[key]

    def step(self, step: int, *, loss_dict, lam, sparsity, lr, epoch=0):
        if step == 0 or step % self.log_every != 0:
            if step != 0:
                self._window_steps += 1
            return

        now = time.perf_counter()
        window_wall = max(now - self._window_start, 1e-6)
        window_steps = max(step - self._last_log_step, self._window_steps, 1)
        tokens_per_step = self.global_batch * self.SEQLEN_PER_VIDEO
        tokens_per_sec = tokens_per_step * window_steps / window_wall
        tokens_per_hour = tokens_per_sec * 3600.0
        videos_per_hour = tokens_per_hour / self.SEQLEN_PER_VIDEO
        step_time_sec = window_wall / window_steps

        if not self.is_rank0:
            self._window_start = now
            self._window_steps = 0
            self._last_log_step = step
            return

        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / 1024**3
            mem_peak = torch.cuda.max_memory_allocated() / 1024**3
            torch.cuda.reset_peak_memory_stats()
        else:
            mem_alloc = 0.0
            mem_peak = 0.0

        L_total = float(loss_dict.get("total", float("nan")))
        record = {
            "step": step,
            "epoch": epoch,
            "L_total": L_total,
            "L_out": float(loss_dict.get("out", 0.0)),
            "L_lpips": float(loss_dict.get("lpips", 0.0)),
            "L_block": float(loss_dict.get("block", 0.0)),
            "L_attn_out": float(loss_dict.get("attn_out", 0.0)),
            "lam1": float(lam["l1"]),
            "lam2": float(lam["l2"]),
            "lam3": float(lam["l3"]),
            "lam4": float(lam["l4"]),
            "current_sparsity": float(sparsity),
            "lr": float(lr),
            "step_time_sec": step_time_sec,
            "tokens_per_sec": tokens_per_sec,
            "tokens_per_hour": tokens_per_hour,
            "videos_per_hour": videos_per_hour,
            "global_batch": self.global_batch,
            "world_size": self.world_size,
            "gpu_mem_alloc_gb": mem_alloc,
            "gpu_mem_peak_gb": mem_peak,
        }
        self.jsonl_fp.write(json.dumps(record) + "\n")
        self.csv_w.writerow(record)

        ema_L = self._ema_update("L_total", L_total)
        msg = (
            f"[step {step:>6d}] L={ema_L:.4f} "
            f"(out={record['L_out']:.4f} lpips={record['L_lpips']:.4f} "
            f"blk={record['L_block']:.4f} hid={record['L_attn_out']:.4f}) | "
            f"sp={sparsity:.3f} lam3={lam['l3']:.3f} | "
            f"thr={tokens_per_sec/1e6:.2f}M tok/s "
            f"= {tokens_per_hour/1e9:.2f}G tok/h "
            f"~= {videos_per_hour:.1f} vid/h | "
            f"mem={mem_alloc:.1f}/{mem_peak:.1f}GB | "
            f"st={step_time_sec*1000:.0f}ms"
        )
        print(msg, flush=True)
        self.console_fp.write(msg + "\n")

        self._window_start = now
        self._window_steps = 0
        self._last_log_step = step

    def close(self):
        if self.is_rank0:
            for fp in (self.console_fp, self.jsonl_fp, self.csv_fp):
                fp.close()
