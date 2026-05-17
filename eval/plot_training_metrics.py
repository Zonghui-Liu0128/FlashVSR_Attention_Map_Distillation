import argparse
import json
import os


def load_jsonl(path):
    rows = [json.loads(l) for l in open(path)]
    try:
        import pandas as pd

        return pd.DataFrame(rows)
    except ModuleNotFoundError:
        return {key: [row[key] for row in rows] for key in rows[0]}


def ema(s, span):
    if hasattr(s, "ewm"):
        return s.ewm(span=span, adjust=False).mean()
    alpha = 2.0 / (span + 1)
    vals = []
    cur = None
    for val in s:
        cur = val if cur is None else cur + alpha * (val - cur)
        vals.append(cur)
    return vals


def _median(s):
    if hasattr(s, "median"):
        return s.median()
    vals = sorted(s)
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def _scale(s, factor):
    if hasattr(s, "__truediv__"):
        try:
            return s / factor
        except TypeError:
            pass
    return [v / factor for v in s]


def _mul(s, factor):
    if hasattr(s, "__mul__"):
        try:
            out = s * factor
            if not isinstance(out, list) or len(out) == len(s):
                return out
        except TypeError:
            pass
    return [v * factor for v in s]


def plot(run_dir, *, ema_span=100, out_name="loss_throughput.png"):
    import matplotlib.pyplot as plt

    df = load_jsonl(os.path.join(run_dir, "train_metrics.jsonl"))
    x = df["step"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f"Training metrics - {os.path.basename(run_dir.rstrip('/'))}")

    ax = axes[0, 0]
    for key, color in zip(
        ["L_total", "L_out", "L_lpips", "L_block", "L_attn_out"],
        ["k", "C0", "C1", "C2", "C3"],
    ):
        ax.plot(x, df[key], color=color, alpha=0.25, lw=0.8)
        ax.plot(x, ema(df[key], ema_span), color=color, lw=1.6, label=key)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title(f"Loss curves (EMA span={ema_span})")

    ax = axes[0, 1]
    ax.plot(x, df["current_sparsity"], "C0-", label="sparsity")
    ax2 = ax.twinx()
    ax2.plot(x, df["lam3"], "C3--", label="lam3 (block)")
    ax.set_xlabel("step")
    ax.set_ylabel("sparsity")
    ax2.set_ylabel("lam3")
    ax.set_title("Sparsity ramp + lam3 schedule")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    tokens_m = _scale(df["tokens_per_sec"], 1e6)
    ax.plot(x, tokens_m, "C0-", alpha=0.3, lw=0.8)
    ax.plot(x, ema(tokens_m, ema_span), "C0-", lw=1.6, label="EMA")
    ax.axhline(_median(df["tokens_per_sec"]) / 1e6, color="gray", ls=":", label="median")
    ax.set_xlabel("step")
    ax.set_ylabel("M tokens / s")
    ax.set_title("Token throughput")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(x, df["videos_per_hour"], "C2-", alpha=0.3, lw=0.8)
    ax.plot(x, ema(df["videos_per_hour"], ema_span), "C2-", lw=1.6)
    ax.axhline(_median(df["videos_per_hour"]), color="gray", ls=":")
    ax.set_xlabel("step")
    ax.set_ylabel("videos / hour")
    ax.set_title("Video throughput")
    ax.grid(alpha=0.3)

    ax = axes[2, 0]
    step_ms = _mul(df["step_time_sec"], 1000)
    ax.plot(x, step_ms, "C4-", alpha=0.3, lw=0.8)
    ax.plot(x, ema(step_ms, ema_span), "C4-", lw=1.6)
    ax.set_xlabel("step")
    ax.set_ylabel("step time (ms)")
    ax.set_title("Step time")
    ax.grid(alpha=0.3)

    ax = axes[2, 1]
    ax.plot(x, df["gpu_mem_alloc_gb"], "C5-", lw=1.2, label="allocated")
    ax.plot(x, df["gpu_mem_peak_gb"], "C6-", lw=1.2, label="peak (per window)")
    ax.set_xlabel("step")
    ax.set_ylabel("GB")
    ax.set_title("GPU memory")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(run_dir, out_name)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ema_span", type=int, default=100)
    ap.add_argument("--out_name", default="loss_throughput.png")
    args = ap.parse_args()
    plot(args.run_dir, ema_span=args.ema_span, out_name=args.out_name)
