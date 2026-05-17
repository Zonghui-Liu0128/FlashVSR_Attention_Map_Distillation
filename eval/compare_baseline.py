from pathlib import Path


def _fmt_metric(metrics, key, fmt):
    value = metrics.get(key)
    if value is None:
        return "-"
    return format(value, fmt)


def build_comparison_table(run_results: list[dict], out_md_path: str) -> None:
    """Each run_results entry: {"label": "FlashVSR Tiny", "metrics": {...}}."""
    cols = [
        "Method",
        "Sparsity",
        "PSNR↑",
        "SSIM↑",
        "LPIPS↓",
        "DISTS↓",
        "FPS@720p",
        "FPS@1080p",
    ]
    rows = []
    for result in run_results:
        metrics = result["metrics"]
        rows.append(
            [
                result["label"],
                _fmt_metric(metrics, "sparsity_rate", ".2f"),
                _fmt_metric(metrics, "psnr", ".2f"),
                _fmt_metric(metrics, "ssim", ".4f"),
                _fmt_metric(metrics, "lpips", ".4f"),
                _fmt_metric(metrics, "dists", ".4f"),
                _fmt_metric(metrics, "fps_720p", ".1f"),
                _fmt_metric(metrics, "fps_1080p", ".1f"),
            ]
        )

    md_lines = []
    md_lines.append("| " + " | ".join(cols) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
        md_lines.append("| " + " | ".join(row) + " |")
    Path(out_md_path).write_text("\n".join(md_lines) + "\n")
