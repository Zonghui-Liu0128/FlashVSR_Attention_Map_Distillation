"""Standalone offline GT -> degraded LQ video generation.

This module intentionally does not import FlashVSR training packages.  It
contains the Real-ESRGAN-style degradation pieces needed to preprocess GT mp4
videos listed by metadata CSV into LQ mp4 files.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import rgb_to_grayscale

np.seterr(divide="ignore", invalid="ignore")


DEFAULT_CONFIG_PATH = Path("Prepare/degradation_config_960x720.yaml")
DEFAULT_METADATA_CSV = Path("data/metadata_wxh_960x720.csv")
DEFAULT_LQ_OUTPUT_DIR = Path(
    "/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/"
    "vsr_datasets/animal_videos/videos_960x720/lq"
)


@dataclass(frozen=True)
class VideoPlan:
    row_index: int
    gt_path: Path
    lq_path: Path
    fps: float
    frame_num: int
    height: int | None
    width: int | None


@dataclass(frozen=True)
class DegradationResult:
    lq_up: torch.Tensor
    lr_native: torch.Tensor | None
    meta: dict[str, Any]


def _as_int(value: Any, default: int | None = None) -> int | None:
    if value in {None, ""}:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value in {None, ""}:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _row_value(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = row.get(name)
        if value not in {None, ""}:
            return value
    lower = {str(k).strip().lower(): v for k, v in row.items()}
    for name in names:
        value = lower.get(name.strip().lower())
        if value not in {None, ""}:
            return value
    return None


def _load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        import yaml

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config must be a mapping: {path}")
        return dict(data)
    except ModuleNotFoundError:
        from omegaconf import OmegaConf

        data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        if not isinstance(data, dict):
            raise ValueError(f"Config must be a mapping: {path}")
        return dict(data)


def load_degradation_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    return _load_yaml(config_path)


def resolve_lq_output_path(gt_path: str | Path, output_dir: str | Path = DEFAULT_LQ_OUTPUT_DIR) -> Path:
    return Path(output_dir) / Path(gt_path).name


def iter_metadata_rows(metadata_csv_path: str | Path) -> list[dict[str, Any]]:
    with Path(metadata_csv_path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_lq_plan(
    metadata_csv_path: str | Path = DEFAULT_METADATA_CSV,
    output_dir: str | Path = DEFAULT_LQ_OUTPUT_DIR,
    *,
    max_videos: int = 0,
    shard_count: int = 1,
    shard_index: int = 0,
) -> list[VideoPlan]:
    shard_count = int(shard_count)
    shard_index = int(shard_index)
    if shard_count <= 0:
        raise ValueError(f"shard_count must be positive, got {shard_count}")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count}), got {shard_index}")

    rows = iter_metadata_rows(metadata_csv_path)
    if max_videos and int(max_videos) > 0:
        rows = rows[: int(max_videos)]

    plan: list[VideoPlan] = []
    for row_index, row in enumerate(rows):
        if row_index % shard_count != shard_index:
            continue
        gt = Path(str(_row_value(row, "Path", "path", "video_path") or "").strip())
        if not str(gt):
            raise ValueError(f"Metadata row is missing Path: {row}")
        fps = _as_float(_row_value(row, "FPS", "fps"), 30.0) or 30.0
        frame_num = _as_int(_row_value(row, "Frame", "Frames", "frames", "total_frames"), 0) or 0
        plan.append(
            VideoPlan(
                row_index=row_index,
                gt_path=gt,
                lq_path=resolve_lq_output_path(gt, output_dir),
                fps=float(fps),
                frame_num=int(frame_num),
                height=_as_int(_row_value(row, "Height", "height", "H", "h"), None),
                width=_as_int(_row_value(row, "Width", "width", "W", "w"), None),
            )
        )
    return plan


def validate_gt_paths(
    metadata_csv_path: str | Path = DEFAULT_METADATA_CSV,
    *,
    max_videos: int = 0,
    shard_count: int = 1,
    shard_index: int = 0,
) -> list[Path]:
    return [
        item.gt_path
        for item in build_lq_plan(
            metadata_csv_path,
            max_videos=max_videos,
            shard_count=shard_count,
            shard_index=shard_index,
        )
        if not item.gt_path.exists()
    ]


def choose_output_fps(metadata_fps: float, output_fps: float | None = None) -> float:
    if output_fps is not None:
        output_fps = float(output_fps)
        if output_fps <= 0:
            raise ValueError(f"output_fps must be positive, got {output_fps}")
        return output_fps
    metadata_fps = float(metadata_fps)
    if metadata_fps <= 0:
        return 30.0
    return metadata_fps


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


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
    return video.detach().cpu().float().clamp(0.0, 1.0).mul(255.0).round().byte().numpy()


def read_rgb_video_cv2(path: str | Path, *, frame_num: int, strict_decode: bool = True) -> torch.Tensor:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Open video failed: {path}")
    frames: list[np.ndarray] = []
    try:
        for frame_idx in range(int(frame_num)):
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                if strict_decode or not frames:
                    raise RuntimeError(f"Decode failed: path={path}, frame={frame_idx}")
                frame_bgr = frames[-1][:, :, ::-1].copy()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    finally:
        cap.release()
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    arr = np.transpose(arr, (0, 3, 1, 2))
    return torch.from_numpy(arr).contiguous()


def write_rgb_video_cv2(video_rgb_uint8: np.ndarray, output_path: str | Path, *, fps: float, codec: str = "mp4v") -> None:
    import cv2

    if video_rgb_uint8.ndim != 4 or video_rgb_uint8.shape[-1] != 3:
        raise ValueError(f"Expected THWC RGB uint8 video, got shape={video_rgb_uint8.shape}")
    if video_rgb_uint8.dtype != np.uint8:
        raise ValueError(f"Expected uint8 video, got dtype={video_rgb_uint8.dtype}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(video_rgb_uint8.shape[1]), int(video_rgb_uint8.shape[2])
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*codec), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")
    try:
        for frame_rgb in video_rgb_uint8:
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def mesh_grid(kernel_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ax = np.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((kernel_size * kernel_size, 1)), yy.reshape(kernel_size * kernel_size, 1))).reshape(
        kernel_size, kernel_size, 2
    )
    return xy, xx, yy


def sigma_matrix2(sig_x: float, sig_y: float, theta: float) -> np.ndarray:
    d_matrix = np.array([[sig_x**2, 0], [0, sig_y**2]])
    u_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))


def bivariate_gaussian(kernel_size: int, sig_x: float, sig_y: float, theta: float, *, isotropic: bool) -> np.ndarray:
    grid, _, _ = mesh_grid(kernel_size)
    sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]]) if isotropic else sigma_matrix2(sig_x, sig_y, theta)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))
    return kernel / np.sum(kernel)


def bivariate_generalized_gaussian(
    kernel_size: int,
    sig_x: float,
    sig_y: float,
    theta: float,
    beta: float,
    *,
    isotropic: bool,
) -> np.ndarray:
    grid, _, _ = mesh_grid(kernel_size)
    sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]]) if isotropic else sigma_matrix2(sig_x, sig_y, theta)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta))
    return kernel / np.sum(kernel)


def bivariate_plateau(
    kernel_size: int,
    sig_x: float,
    sig_y: float,
    theta: float,
    beta: float,
    *,
    isotropic: bool,
) -> np.ndarray:
    grid, _, _ = mesh_grid(kernel_size)
    sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]]) if isotropic else sigma_matrix2(sig_x, sig_y, theta)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.reciprocal(np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta) + 1)
    return kernel / np.sum(kernel)


def _random_beta(beta_range: list[float]) -> float:
    return float(np.random.uniform(beta_range[0], 1) if np.random.uniform() < 0.5 else np.random.uniform(1, beta_range[1]))


def random_mixed_kernel(
    kernel_list: list[str],
    kernel_prob: list[float],
    kernel_size: int,
    sigma_range: list[float],
    rotation_range: list[float],
    betag_range: list[float],
    betap_range: list[float],
) -> tuple[np.ndarray, str]:
    kernel_type = random.choices(kernel_list, kernel_prob, k=1)[0]
    sig_x = float(np.random.uniform(sigma_range[0], sigma_range[1]))
    sig_y = float(np.random.uniform(sigma_range[0], sigma_range[1]))
    theta = float(np.random.uniform(rotation_range[0], rotation_range[1]))

    if kernel_type == "iso":
        return bivariate_gaussian(kernel_size, sig_x, sig_y, theta, isotropic=True), kernel_type
    if kernel_type == "aniso":
        return bivariate_gaussian(kernel_size, sig_x, sig_y, theta, isotropic=False), kernel_type
    if kernel_type == "generalized_iso":
        return bivariate_generalized_gaussian(
            kernel_size, sig_x, sig_y, theta, _random_beta(betag_range), isotropic=True
        ), kernel_type
    if kernel_type == "generalized_aniso":
        return bivariate_generalized_gaussian(
            kernel_size, sig_x, sig_y, theta, _random_beta(betag_range), isotropic=False
        ), kernel_type
    if kernel_type == "plateau_iso":
        return bivariate_plateau(kernel_size, sig_x, sig_y, theta, _random_beta(betap_range), isotropic=True), kernel_type
    if kernel_type == "plateau_aniso":
        return bivariate_plateau(kernel_size, sig_x, sig_y, theta, _random_beta(betap_range), isotropic=False), kernel_type
    raise ValueError(f"Unsupported kernel_type={kernel_type}")


def circular_lowpass_kernel(cutoff: float, kernel_size: int, pad_to: int = 0) -> np.ndarray:
    try:
        from scipy import special
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing scipy. Install it before running offline degradation: python -m pip install scipy") from exc

    if kernel_size % 2 != 1:
        raise ValueError("Kernel size must be odd")
    kernel = np.fromfunction(
        lambda x, y: cutoff
        * special.j1(cutoff * np.sqrt((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2))
        / (2 * np.pi * np.sqrt((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2)),
        [kernel_size, kernel_size],
    )
    kernel[(kernel_size - 1) // 2, (kernel_size - 1) // 2] = cutoff**2 / (4 * np.pi)
    kernel = kernel / np.sum(kernel)
    if pad_to > kernel_size:
        pad_size = (pad_to - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
    return kernel


def filter2d(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    k = kernel.size(-1)
    if k % 2 != 1:
        raise ValueError("Kernel size must be odd")
    b, c, h, w = img.size()
    img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode="reflect")
    ph, pw = img.size()[-2:]
    img = img.reshape(b * c, 1, ph, pw)
    kernel = kernel.view(1, 1, k, k).to(device=img.device, dtype=img.dtype)
    return F.conv2d(img, kernel, padding=0).view(b, c, h, w)


def _interpolate(x: torch.Tensor, *, size: tuple[int, int] | None = None, scale_factor: float | None = None, mode: str) -> torch.Tensor:
    kwargs: dict[str, Any] = {"mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(x, size=size, scale_factor=scale_factor, **kwargs)


def generate_gaussian_noise_pt(img: torch.Tensor, sigma: torch.Tensor, gray_noise: torch.Tensor) -> torch.Tensor:
    b, _, h, w = img.size()
    sigma = sigma.view(b, 1, 1, 1)
    gray_noise = gray_noise.view(b, 1, 1, 1)
    noise = torch.randn(*img.size(), dtype=img.dtype, device=img.device) * sigma / 255.0
    if torch.sum(gray_noise) > 0:
        noise_gray = torch.randn(b, 1, h, w, dtype=img.dtype, device=img.device) * sigma / 255.0
        noise = noise * (1 - gray_noise) + noise_gray * gray_noise
    return noise


def random_add_gaussian_noise_pt(
    img: torch.Tensor,
    *,
    sigma_range: list[float],
    gray_prob: float,
    clip: bool = True,
    video_range: float = 0.01,
) -> torch.Tensor:
    sigma_sample = np.random.rand() * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
    low = max(0.0, sigma_sample - video_range)
    high = max(low, sigma_sample + video_range)
    sigma = torch.rand(img.size(0), dtype=img.dtype, device=img.device) * (high - low) + low
    gray_noise = (torch.rand(1, dtype=img.dtype, device=img.device).repeat(img.size(0)) < gray_prob).float()
    out = img + generate_gaussian_noise_pt(img, sigma, gray_noise)
    return torch.clamp(out, 0, 1) if clip else out


def generate_poisson_noise_pt(img: torch.Tensor, scale: torch.Tensor, gray_noise: torch.Tensor) -> torch.Tensor:
    b, _, h, w = img.size()
    gray_noise = gray_noise.view(b, 1, 1, 1)
    if torch.sum(gray_noise) > 0:
        img_gray = rgb_to_grayscale(img, num_output_channels=1)
        img_gray = torch.clamp((img_gray * 255.0).round(), 0, 255) / 255.0
        vals = img_gray.new_tensor([2 ** np.ceil(np.log2(max(1, len(torch.unique(img_gray[i]))))) for i in range(b)]).view(
            b, 1, 1, 1
        )
        noise_gray = torch.poisson(img_gray * vals) / vals - img_gray
        noise_gray = noise_gray.expand(b, 3, h, w)
    else:
        noise_gray = torch.zeros_like(img)

    rounded = torch.clamp((img * 255.0).round(), 0, 255) / 255.0
    vals = rounded.new_tensor([2 ** np.ceil(np.log2(max(1, len(torch.unique(rounded[i]))))) for i in range(b)]).view(
        b, 1, 1, 1
    )
    noise = torch.poisson(rounded * vals) / vals - rounded
    noise = noise * (1 - gray_noise) + noise_gray * gray_noise
    return noise * scale.view(b, 1, 1, 1)


def random_add_poisson_noise_pt(
    img: torch.Tensor,
    *,
    scale_range: list[float],
    gray_prob: float,
    clip: bool = True,
) -> torch.Tensor:
    scale_sample = np.random.rand() * (scale_range[1] - scale_range[0]) + scale_range[0]
    scale = torch.full((img.size(0),), float(scale_sample), dtype=img.dtype, device=img.device)
    gray_noise = (torch.rand(1, dtype=img.dtype, device=img.device).repeat(img.size(0)) < gray_prob).float()
    out = img + generate_poisson_noise_pt(img, scale, gray_noise)
    return torch.clamp(out, 0, 1) if clip else out


class OfflineDegrader:
    def __init__(self, cfg: Mapping[str, Any]):
        self.cfg = dict(cfg)
        seed_everything(_as_int(self.cfg.get("seed"), None))
        self.scl_factor = float(self.cfg.get("scl_factor", 4.0))
        self.input_bits = int(self.cfg.get("input_bits", 10))
        self.max_value = 2**self.input_bits - 1
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]

    def _get(self, key: str, default: Any) -> Any:
        return self.cfg[key] if key in self.cfg else default

    def _sample_interpolate_mode(self, mode_probs: Any = None) -> str:
        if mode_probs is None:
            return random.choice(["area", "bilinear", "bicubic"])
        if isinstance(mode_probs, Mapping):
            names = list(mode_probs.keys())
            probs = [float(mode_probs[k]) for k in names]
        elif isinstance(mode_probs, list):
            names = ["area", "bilinear", "bicubic"] if len(mode_probs) == 3 else []
            probs = [float(x) for x in mode_probs]
        else:
            raise TypeError(f"Invalid interpolate mode probabilities: {mode_probs}")
        if not names or sum(probs) <= 0:
            raise ValueError(f"Invalid interpolate mode probabilities: {mode_probs}")
        return random.choices(names, weights=probs, k=1)[0]

    def _sample_resize(self, resize_prob: list[float], resize_range: list[float]) -> tuple[str, float, str]:
        updown_type = random.choices(["up", "down", "keep"], resize_prob, k=1)[0]
        if updown_type == "up":
            scale = float(np.random.uniform(1.0, float(resize_range[1])))
        elif updown_type == "down":
            scale = float(np.random.uniform(float(resize_range[0]), 1.0))
        else:
            scale = 1.0
        return updown_type, scale, self._sample_interpolate_mode(None)

    def _sample_kernel(
        self,
        kernel_list: list[str],
        kernel_prob: list[float],
        blur_sigma: list[float],
        betag_range: list[float],
        betap_range: list[float],
        sinc_prob: float,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < float(sinc_prob):
            omega_c = float(np.random.uniform(np.pi / 3, np.pi) if kernel_size < 13 else np.random.uniform(np.pi / 5, np.pi))
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            kernel_type = "sinc"
            meta = {"kernel_type": kernel_type, "kernel_size": kernel_size, "omega_c": omega_c}
        else:
            kernel, kernel_type = random_mixed_kernel(
                list(kernel_list),
                list(kernel_prob),
                kernel_size,
                list(blur_sigma),
                [-math.pi, math.pi],
                list(betag_range),
                list(betap_range),
            )
            meta = {"kernel_type": kernel_type, "kernel_size": kernel_size}
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
        return torch.FloatTensor(kernel), meta

    def _maybe_add_noise(
        self,
        out: torch.Tensor,
        *,
        noise_prob: float,
        gaussian_noise_prob: float,
        noise_range: list[float],
        poisson_scale_range: list[float],
        gray_noise_prob: float,
    ) -> tuple[torch.Tensor, str]:
        if np.random.uniform() >= float(noise_prob):
            return out, "none"
        if np.random.uniform() < float(gaussian_noise_prob):
            return (
                random_add_gaussian_noise_pt(out, sigma_range=list(noise_range), gray_prob=float(gray_noise_prob)),
                "gaussian",
            )
        return (
            random_add_poisson_noise_pt(out, scale_range=list(poisson_scale_range), gray_prob=float(gray_noise_prob)),
            "poisson",
        )

    def _final_down_up(
        self,
        out: torch.Tensor,
        gt: torch.Tensor,
        sinc_kernel: torch.Tensor | None,
        *,
        mode_probs: Any,
        upsample_mode: str,
        final_sinc_stage: str,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        ori_h, ori_w = gt.size()[2:4]
        mode = self._sample_interpolate_mode(mode_probs)
        if sinc_kernel is not None and final_sinc_stage == "pre_downsample":
            out = filter2d(out, sinc_kernel)
        lr = _interpolate(out, size=(ori_h // int(self.scl_factor), ori_w // int(self.scl_factor)), mode=mode)
        if sinc_kernel is not None and final_sinc_stage == "post_lr":
            lr = filter2d(lr, sinc_kernel)
        elif sinc_kernel is not None and final_sinc_stage not in {"pre_downsample", "post_lr", "none"}:
            raise ValueError(f"Unsupported final_sinc_stage={final_sinc_stage}")
        lr = torch.clamp((lr * self.max_value).round(), 0, self.max_value) / self.max_value
        lq_up = _interpolate(lr, size=(ori_h, ori_w), mode=upsample_mode) if self.scl_factor != 1 else lr
        return lr, torch.clamp(lq_up, 0.0, 1.0), mode

    def apply(self, gt: torch.Tensor, *, return_native_lr: bool = False) -> DegradationResult:
        if gt.ndim != 4 or gt.shape[1] != 3:
            raise ValueError(f"Expected TCHW RGB tensor, got shape={tuple(gt.shape)}")
        out = gt.float().clamp(0.0, 1.0).contiguous()
        meta: dict[str, Any] = {}

        kernel1, kernel1_meta = self._sample_kernel(
            self._get("kernel_list", ["iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_iso", "plateau_aniso"]),
            self._get("kernel_prob", [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]),
            self._get("blur_sigma", [0.2, 3]),
            self._get("betag_range", [0.5, 4]),
            self._get("betap_range", [1, 2]),
            float(self._get("sinc_prob", 0.1)),
        )
        kernel2, kernel2_meta = self._sample_kernel(
            self._get("kernel_list2", ["iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_iso", "plateau_aniso"]),
            self._get("kernel_prob2", [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]),
            self._get("blur_sigma2", [0.2, 1.5]),
            self._get("betag_range2", [0.5, 4]),
            self._get("betap_range2", [1, 2]),
            float(self._get("sinc_prob2", 0.1)),
        )
        meta["kernel1"] = kernel1_meta
        meta["kernel2"] = kernel2_meta

        final_sinc_prob = float(self._get("final_sinc_prob", 0.3))
        final_sinc_stage = str(self._get("final_sinc_stage", "post_lr"))
        if np.random.uniform() < final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = float(np.random.uniform(np.pi / 3, np.pi))
            sinc_kernel = torch.FloatTensor(circular_lowpass_kernel(omega_c, kernel_size, pad_to=21))
            meta["final_sinc"] = {"enabled": True, "kernel_size": kernel_size, "omega_c": omega_c, "stage": final_sinc_stage}
        else:
            sinc_kernel = None
            meta["final_sinc"] = {"enabled": False, "stage": final_sinc_stage}

        scale1 = 1.0
        scale2 = 1.0
        if bool(self._get("degradation_1", True)):
            blur_enabled = np.random.uniform() < float(self._get("first_blur_prob", 1.0))
            if blur_enabled:
                out = filter2d(out, kernel1)
            resize_type, scale1, resize_mode = self._sample_resize(self._get("resize_prob", [0.1, 0.2, 0.7]), self._get("resize_range", [0.15, 1.5]))
            out = _interpolate(out, scale_factor=scale1, mode=resize_mode)
            out, noise_type = self._maybe_add_noise(
                out,
                noise_prob=float(self._get("noise_prob", 1.0)),
                gaussian_noise_prob=float(self._get("gaussian_noise_prob", 0.5)),
                noise_range=self._get("noise_range", [0.5, 20]),
                poisson_scale_range=self._get("poisson_scale_range", [0.015, 2]),
                gray_noise_prob=float(self._get("gray_noise_prob", 0.9)),
            )
            meta["degradation_1"] = {
                "enabled": True,
                "blur_enabled": bool(blur_enabled),
                "resize_type": resize_type,
                "resize_scale": float(scale1),
                "resize_mode": resize_mode,
                "noise_type": noise_type,
            }
        else:
            meta["degradation_1"] = {"enabled": False}

        if bool(self._get("degradation_2", True)):
            blur_enabled = np.random.uniform() < float(self._get("second_blur_prob", 0.8))
            if blur_enabled:
                out = filter2d(out, kernel2)
            resize_type, scale2, resize_mode = self._sample_resize(
                self._get("resize_prob2", [0.3, 0.4, 0.3]), self._get("resize_range2", [0.3, 1.2])
            )
            out = _interpolate(out, scale_factor=scale2, mode=resize_mode)
            out, noise_type = self._maybe_add_noise(
                out,
                noise_prob=float(self._get("noise_prob2", 1.0)),
                gaussian_noise_prob=float(self._get("gaussian_noise_prob2", 0.5)),
                noise_range=self._get("noise_range2", [0.5, 20]),
                poisson_scale_range=self._get("poisson_scale_range2", [0.015, 2]),
                gray_noise_prob=float(self._get("gray_noise_prob2", 0.9)),
            )
            meta["degradation_2"] = {
                "enabled": True,
                "blur_enabled": bool(blur_enabled),
                "resize_type": resize_type,
                "resize_scale": float(scale2),
                "resize_mode": resize_mode,
                "noise_type": noise_type,
            }
        else:
            meta["degradation_2"] = {"enabled": False}

        meta["effective_magnification"] = float(self.scl_factor / max(1e-8, float(scale1) * float(scale2)))
        lr, lq_up, resize_mode = self._final_down_up(
            out,
            gt,
            sinc_kernel if bool(self._get("degradation_3", True)) else None,
            mode_probs=self._get("final_downsample_mode_probs", None),
            upsample_mode=str(self._get("final_upsample_mode", "bilinear")),
            final_sinc_stage=final_sinc_stage,
        )
        meta["degradation_3"] = {"enabled": bool(self._get("degradation_3", True)), "final_resize_mode": resize_mode}
        return DegradationResult(lq_up=lq_up, lr_native=lr if return_native_lr else None, meta=meta)


def degrade_csv_to_lq(
    cfg: Mapping[str, Any],
    *,
    metadata_csv_path: str | Path = DEFAULT_METADATA_CSV,
    output_dir: str | Path = DEFAULT_LQ_OUTPUT_DIR,
    overwrite: bool = False,
    max_videos: int = 0,
    seed: int | None = None,
    codec: str = "mp4v",
    save_native_lr: bool = False,
    strict_paths: bool = True,
    output_fps: float | None = None,
    shard_count: int = 1,
    shard_index: int = 0,
) -> list[Path]:
    seed_everything(seed)
    missing = validate_gt_paths(
        metadata_csv_path,
        max_videos=max_videos,
        shard_count=shard_count,
        shard_index=shard_index,
    )
    if missing and strict_paths:
        lines = "\n".join(str(p) for p in missing[:20])
        raise FileNotFoundError(f"Missing GT videos ({len(missing)}):\n{lines}")

    plan = build_lq_plan(
        metadata_csv_path,
        output_dir,
        max_videos=max_videos,
        shard_count=shard_count,
        shard_index=shard_index,
    )
    degrader = OfflineDegrader(cfg)
    written: list[Path] = []
    configured_frames = _as_int(cfg.get("frame_num"), None)
    target_h = _as_int(cfg.get("crop_height"), None)
    target_w = _as_int(cfg.get("crop_width"), None)
    for item in plan:
        if item.lq_path.exists() and not overwrite:
            written.append(item.lq_path)
            continue
        if seed is not None:
            seed_everything(int(seed) + int(item.row_index))
        frame_num = int(configured_frames or item.frame_num)
        gt = read_rgb_video_cv2(item.gt_path, frame_num=frame_num)
        if target_h is not None and gt.shape[-2] != target_h:
            raise RuntimeError(f"Height mismatch for {item.gt_path}: got {gt.shape[-2]}, expected {target_h}")
        if target_w is not None and gt.shape[-1] != target_w:
            raise RuntimeError(f"Width mismatch for {item.gt_path}: got {gt.shape[-1]}, expected {target_w}")
        result = degrader.apply(gt, return_native_lr=save_native_lr)
        out_tensor = result.lr_native if save_native_lr else result.lq_up
        if out_tensor is None:
            raise RuntimeError("Internal error: native LR output was requested but not returned")
        write_rgb_video_cv2(
            tensor_to_uint8_rgb(out_tensor),
            item.lq_path,
            fps=choose_output_fps(item.fps, output_fps=output_fps),
            codec=codec,
        )
        written.append(item.lq_path)
    return written


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone offline GT -> degraded LQ mp4 generation.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--metadata-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1, help="Total number of modulo shards for parallel preprocessing.")
    parser.add_argument("--shard-index", type=int, default=0, help="Current modulo shard index in [0, shard-count).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--output-fps", type=float, default=None, help="Override output mp4 FPS. Defaults to config output_fps, then metadata FPS.")
    parser.add_argument("--save-native-lr", action="store_true", help="Write native 1/scl_factor LQ instead of upsampled model-input LQ.")
    parser.add_argument("--no-strict-paths", action="store_true", help="Skip missing GT paths instead of failing fast.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = load_degradation_config(args.config)
    metadata_csv = Path(args.metadata_csv or cfg.get("metadata_csv_path", DEFAULT_METADATA_CSV))
    output_dir = Path(args.output_dir or cfg.get("lq_output_dir", DEFAULT_LQ_OUTPUT_DIR))
    output_fps = args.output_fps if args.output_fps is not None else _as_float(cfg.get("output_fps"), None)
    written = degrade_csv_to_lq(
        cfg,
        metadata_csv_path=metadata_csv,
        output_dir=output_dir,
        overwrite=args.overwrite,
        max_videos=args.max_videos,
        seed=args.seed,
        codec=args.codec,
        save_native_lr=args.save_native_lr,
        strict_paths=not args.no_strict_paths,
        output_fps=output_fps,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    for path in written:
        print(path)
    print(f"[offline_degradation] processed={len(written)} output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
