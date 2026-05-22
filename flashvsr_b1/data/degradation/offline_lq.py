"""Offline CSV GT -> degraded LQ video generation."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_METADATA_CSV = "data/metadata_wxh_960x720.csv"
DEFAULT_LQ_OUTPUT_DIR = (
    "/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/"
    "vsr_datasets/animal_videos/videos_960x720/lq"
)


def resolve_lq_output_path(gt_path: str, output_dir: str | Path = DEFAULT_LQ_OUTPUT_DIR) -> Path:
    return Path(output_dir) / Path(gt_path).name


def tensor_to_uint8_rgb(video: torch.Tensor) -> np.ndarray:
    """Convert a TCHW or CTHW RGB float tensor in [0, 1] to THWC uint8."""
    if video.ndim != 4:
        raise ValueError(f"Expected 4D RGB video tensor, got shape={tuple(video.shape)}")
    if video.shape[0] == 3:
        video = video.permute(1, 2, 3, 0)
    elif video.shape[1] == 3:
        video = video.permute(0, 2, 3, 1)
    else:
        raise ValueError(f"Expected TCHW or CTHW RGB video tensor, got shape={tuple(video.shape)}")
    arr = video.detach().cpu().float().clamp(0.0, 1.0).mul(255.0).round().byte().numpy()
    return arr


def write_rgb_video_cv2(video_rgb_uint8: np.ndarray, output_path: str | Path, *, fps: float, codec: str = "mp4v") -> None:
    if video_rgb_uint8.ndim != 4 or video_rgb_uint8.shape[-1] != 3:
        raise ValueError(f"Expected THWC RGB uint8 video, got shape={video_rgb_uint8.shape}")
    if video_rgb_uint8.dtype != np.uint8:
        raise ValueError(f"Expected uint8 video, got dtype={video_rgb_uint8.dtype}")

    import cv2

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(video_rgb_uint8.shape[1]), int(video_rgb_uint8.shape[2])
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")
    try:
        for frame_rgb in video_rgb_uint8:
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _load_data_config(config_path: str | Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)


def _seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def degrade_csv_to_lq(
    opt: dict[str, Any],
    *,
    metadata_csv_path: str | Path = DEFAULT_METADATA_CSV,
    output_dir: str | Path = DEFAULT_LQ_OUTPUT_DIR,
    overwrite: bool = False,
    max_videos: int = 0,
    seed: int | None = None,
    codec: str = "mp4v",
) -> list[Path]:
    """Apply the current online degradation pipeline to each CSV GT video."""
    from .basic_vsr_dataset_hw_crop import BasicVSRDataset_hw_crop

    _seed_everything(seed)
    run_opt = dict(opt)
    run_opt.update(
        {
            "datapath_config_method": "metadata_csv",
            "metadata_csv_path": str(metadata_csv_path),
            "shuffle_samples": False,
            "data_repeat": 1,
            "return_degradation_stages": False,
            "save_degradation_stages": False,
        }
    )
    run_opt["rebuild_sample_json"] = bool(run_opt.get("rebuild_sample_json", True))

    dataset = BasicVSRDataset_hw_crop(run_opt)
    written: list[Path] = []
    limit = len(dataset) if int(max_videos) <= 0 else min(len(dataset), int(max_videos))
    for idx in range(limit):
        with torch.no_grad():
            item = dataset[idx]
        sample_meta = item["sample_meta"]
        output_path = resolve_lq_output_path(sample_meta["path"], output_dir)
        if output_path.exists() and not overwrite:
            written.append(output_path)
            continue
        fps = float(sample_meta.get("fps") or 30.0)
        lq_rgb = tensor_to_uint8_rgb(item["aigc_input"])
        write_rgb_video_cv2(lq_rgb, output_path, fps=fps, codec=codec)
        written.append(output_path)
    return written


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Degrade CSV GT videos and write LQ mp4 files.")
    parser.add_argument("--config", default="flashvsr_b1/configs/data_b1.yaml")
    parser.add_argument("--metadata-csv", default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_LQ_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--codec", default="mp4v")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    opt = _load_data_config(args.config)
    written = degrade_csv_to_lq(
        opt,
        metadata_csv_path=args.metadata_csv,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        max_videos=args.max_videos,
        seed=args.seed,
        codec=args.codec,
    )
    for path in written:
        print(path)
    print(f"[offline_lq] processed={len(written)} output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
