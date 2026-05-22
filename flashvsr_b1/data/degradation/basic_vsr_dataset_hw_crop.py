"""Metadata-aware BasicVSRDataset_hw_crop for FlashVSR/LSWA smoke training.

This implementation keeps the training-facing return interface compatible with the
existing stage3 pipeline:

    read_input  -> HQ/GT tensor, RGB, [F, 3, H, W], float32 in [0, 1]
    aigc_input  -> degraded LQ tensor upsampled to GT size, same shape
    xymap_pc    -> zeros grid placeholder, [F, 2, H, W]
    data_name   -> stable sample id

Supported path sources:
  * `metadata_json`: scenes -> fixed-frame clips -> resolution-aware crops.
  * `metadata_csv`: one preprocessed full-frame GT clip per CSV row, no center
    crop or sliding crop planning before online degradation.
Both paths write/read a frozen sample.json before training.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import cv2
import numpy as np
import torch
# import torch.nn.functional as F
from torchvision.utils import save_image

from ..sample_index import (
    build_sample_records_from_csv,
    build_sample_records_from_metadata,
    validate_sample_index_contract,
)
from .degradations import (
    USMSharp,
    circular_lowpass_kernel,
    filter2D,
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt,
    random_mixed_kernels,
)
from .dataset_common_utils import *
from .dataset_common_utils import SuitPatchPrepareRect


class BasicVSRDataset_hw_crop(torch.utils.data.Dataset):
    def __init__(self, opt: Dict[str, Any]):
        self.opt = opt
        self.train_mode = opt.get("train_mode", 0)
        self.crop_patch_size = opt.get("crop_patch_size", 256)
        self.crop_height = opt.get("crop_height", None)
        self.crop_width = opt.get("crop_width", None)
        self.actual_crop_height = int(self.crop_height if self.crop_height is not None else self.crop_patch_size)
        self.actual_crop_width = int(self.crop_width if self.crop_width is not None else self.crop_patch_size)
        self.frame_num = int(opt.get("frame_num", 45))
        self.data_repeat = int(opt.get("data_repeat", 1))
        self.scl_factor = float(opt.get("scl_factor", 1.0))
        self.input_bits = int(opt.get("input_bits", 10))
        self.max_value = 2**self.input_bits - 1

        # video reading / metadata options
        self.strict_decode = bool(opt.get("strict_decode", True))
        self.max_retry = int(opt.get("max_retry", 5))
        self.shuffle_samples = bool(opt.get("shuffle_samples", True))
        self.return_degradation_stages = bool(opt.get("return_degradation_stages", False))
        self.allow_frame_truncation = bool(opt.get("allow_frame_truncation", False))

        # color/gain options; default disabled for clean GT
        self.random_color_prob = opt.get("random_color_prob", 0)
        self.hue_range = opt.get("hue_range", 0.03)
        self.saturation_range = opt.get("saturation_range", 0.3)
        self.brightness_range = opt.get("brightness_range", 0.3)
        self.random_gain_ = opt.get("random_gain_", False)
        self.usm_sharper_ = opt.get("usm_sharper_", False)
        gain_range = opt.get("gain_range", [0.8, 1.2])
        gamma_range = opt.get("gamma_range", [0.5, 1.5])
        p_gain = opt.get("p_gain", 0.3)
        p_dark_gain = opt.get("p_dark_gain", 0.1)
        p_gamma = opt.get("p_gamma", 0.3)
        dark_gain_range = opt.get("dark_gain_range", [0.05, 0.15])
        self.random_gain = RandomBrightnessGamma(
            gain_range=(gain_range[0], gain_range[1]),
            gamma_range=(gamma_range[0], gamma_range[1]),
            p_gain=p_gain,
            p_gamma=p_gamma,
            p_dark_gain=p_dark_gain,
            dark_gain_range=(dark_gain_range[0], dark_gain_range[1]),
        )
        self.random_color = RandomColorize(self.hue_range, self.saturation_range, self.brightness_range)
        self.random_color_for_face = RandomColorize(0, self.saturation_range, self.brightness_range)
        self.usm_sharper = USMSharp()

        # degradation parameters
        self.blur_kernel_size = opt.get("blur_kernel_size", 21)
        self.kernel_list = opt.get("kernel_list", ["iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_iso", "plateau_aniso"])
        self.kernel_prob = opt.get("kernel_prob", [0.45, 0.25, 0.12, 0.03, 0.12, 0.03])
        self.blur_sigma = opt.get("blur_sigma", [0.2, 3])
        self.betag_range = opt.get("betag_range", [0.5, 4])
        self.betap_range = opt.get("betap_range", [1, 2])
        self.sinc_prob = opt.get("sinc_prob", 0.1)

        self.resize_prob = opt.get("resize_prob", [0.1, 0.2, 0.7])
        self.resize_range = opt.get("resize_range", [0.15, 1.5])
        self.gaussian_noise_prob = opt.get("gaussian_noise_prob", 0.5)
        self.noise_range = opt.get("noise_range", [0.5, 20])
        self.poisson_scale_range = opt.get("poisson_scale_range", [0.015, 2])
        self.gray_noise_prob = opt.get("gray_noise_prob", 0.9)

        self.second_blur_prob = opt.get("second_blur_prob", 0.8)
        self.blur_kernel_size2 = opt.get("blur_kernel_size2", 21)
        self.kernel_list2 = opt.get("kernel_list2", ["iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_iso", "plateau_aniso"])
        self.kernel_prob2 = opt.get("kernel_prob2", [0.45, 0.25, 0.12, 0.03, 0.12, 0.03])
        self.blur_sigma2 = opt.get("blur_sigma2", [0.2, 1.5])
        self.betag_range2 = opt.get("betag_range2", [0.5, 4])
        self.betap_range2 = opt.get("betap_range2", [1, 2])
        self.sinc_prob2 = opt.get("sinc_prob2", 0.1)

        self.resize_prob2 = opt.get("resize_prob2", [0.3, 0.4, 0.3])
        self.resize_range2 = opt.get("resize_range2", [0.3, 1.2])
        self.gaussian_noise_prob2 = opt.get("gaussian_noise_prob2", 0.5)
        self.noise_range2 = opt.get("noise_range2", [0.5, 20])
        self.poisson_scale_range2 = opt.get("poisson_scale_range2", [0.015, 2])
        self.gray_noise_prob2 = opt.get("gray_noise_prob2", 0.9)
        self.final_sinc_prob = opt.get("final_sinc_prob", 0.3)

        # Backward-compatible optimized degradation controls. Defaults reproduce
        # the original behavior unless explicitly overridden by YAML/presets.
        self.first_blur_prob = float(opt.get("first_blur_prob", 1.0))
        self.noise_prob = float(opt.get("noise_prob", 1.0))
        self.noise_prob2 = float(opt.get("noise_prob2", 1.0))
        self.final_downsample_mode_probs = opt.get("final_downsample_mode_probs", None)
        self.final_upsample_mode = opt.get("final_upsample_mode", "bilinear")
        self.final_sinc_stage = opt.get("final_sinc_stage", "post_lr")
        self.max_effective_magnification_for_kd = opt.get("max_effective_magnification_for_kd", None)
        self.degradation_mix = self._normalize_degradation_mix(opt.get("degradation_mix", None))

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1

        # visualization controls.  Accept both legacy and typo variants.
        self.save_image_ = opt.get("save_image_", opt.get("save_images_", False))
        self.save_degradation_stages = bool(opt.get("save_degradation_stages", False))
        self.vis_output_dir = opt.get("vis_output_dir", "debug/flashvsr_degradation")
        self.vis_num_samples = int(opt.get("vis_num_samples", 0))
        self.vis_frame_indices = list(opt.get("vis_frame_indices", [0, self.frame_num // 2, self.frame_num - 1]))
        self._vis_saved = 0

        datapath_config_method = opt.get("datapath_config_method", "metadata_json")
        if datapath_config_method not in {"metadata_json", "metadata_csv"}:
            raise ValueError(
                "BasicVSRDataset_hw_crop supports datapath_config_method "
                f"metadata_json or metadata_csv, got {datapath_config_method!r}."
            )

        metadata_path = opt["metadata_csv_path"] if datapath_config_method == "metadata_csv" else opt["metadata_json_path"]
        sample_json_path = opt.get("sample_json_path", None)
        if sample_json_path is None:
            if datapath_config_method == "metadata_csv":
                sample_json_path = str(Path(metadata_path).resolve().with_suffix(".sample.json"))
            else:
                sample_json_path = str(Path(metadata_path).resolve().parent / "sample.json")
        self.sample_json_path = sample_json_path
        if opt.get("rebuild_sample_json", False) or not os.path.exists(sample_json_path):
            if datapath_config_method == "metadata_csv":
                sample_index = build_sample_records_from_csv(opt)
            else:
                sample_index = build_sample_records_from_metadata(opt)
            Path(sample_json_path).parent.mkdir(parents=True, exist_ok=True)
            tmp_sample_json_path = f"{sample_json_path}.tmp.{os.getpid()}"
            with open(tmp_sample_json_path, "w", encoding="utf-8") as f:
                json.dump(sample_index, f, ensure_ascii=False, indent=2)
            os.replace(tmp_sample_json_path, sample_json_path)
            print(f"[BasicVSRDataset_hw_crop] wrote sample index: {sample_json_path}")
            print(f"[BasicVSRDataset_hw_crop] stats: {sample_index.get('stats', {})}")
        with open(sample_json_path, "r", encoding="utf-8") as f:
            sample_index = json.load(f)
        validate_sample_index_contract(
            sample_index,
            {
                "frame_num": self.frame_num,
                "crop_width": self.actual_crop_width,
                "crop_height": self.actual_crop_height,
                "allow_frame_truncation": self.allow_frame_truncation,
            },
            sample_json_path=sample_json_path,
        )
        samples = sample_index.get("samples", [])
        if not samples:
            raise RuntimeError(f"No samples found in sample_json_path={sample_json_path}")
        max_samples = int(opt.get("max_samples", 0))
        if max_samples > 0:
            samples = samples[:max_samples]
        if self.shuffle_samples:
            random.shuffle(samples)
        self.imgs = samples * self.data_repeat
        print(f"[BasicVSRDataset_hw_crop] Number of metadata samples: {len(samples)}; repeated: {len(self.imgs)}")

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Retry a few random samples in case a single mp4 has an intermittent decode failure.
        last_error: Optional[Exception] = None
        for retry in range(max(1, self.max_retry)):
            sample = self.imgs[index if retry == 0 else random.randrange(len(self.imgs))]
            try:
                return self._get_item_from_metadata_sample(sample)
            except Exception as exc:
                last_error = exc
                if self.strict_decode:
                    raise
        raise RuntimeError(f"Failed to load sample after {self.max_retry} retries; last error: {last_error}")

    def _read_video_clip_cv2(self, path: str, start_frame: int, frame_num: int) -> torch.Tensor:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Open video failed: {path}")
        frames: List[np.ndarray] = []
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
            for i in range(frame_num):
                ret, frame = cap.read()
                if not ret or frame is None:
                    if self.strict_decode or not frames:
                        raise RuntimeError(f"Decode failed: path={path}, frame={start_frame + i}")
                    frame = frames[-1].copy()
                frames.append(frame)
        finally:
            cap.release()
        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # [F,H,W,BGR]
        arr = np.transpose(arr, (0, 3, 1, 2))  # [F,3,H,W], BGR
        return torch.from_numpy(arr)

    def _get_item_from_metadata_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        path = sample["path"]
        clip_start = int(sample["clip_start"])
        sample_frame_num = int(sample.get("frame_num", self.frame_num))
        frame_num = sample_frame_num
        if self.allow_frame_truncation and sample_frame_num >= self.frame_num:
            frame_num = self.frame_num
        crop_x = int(sample["crop_x"])
        crop_y = int(sample["crop_y"])
        crop_w = int(sample.get("crop_width", self.actual_crop_width))
        crop_h = int(sample.get("crop_height", self.actual_crop_height))
        sample_id = str(sample.get("sample_id", Path(path).stem))

        clean_bgr_full = self._read_video_clip_cv2(path, clip_start, frame_num)
        orientation_transform = str(sample.get("orientation_transform", "none"))
        if orientation_transform == "rotate_90_clockwise":
            clean_bgr_full = torch.rot90(clean_bgr_full, k=-1, dims=(-2, -1)).contiguous()
        elif orientation_transform not in {"none", ""}:
            raise RuntimeError(f"Unsupported orientation_transform={orientation_transform} for sample={sample_id}")
        _, _, full_h, full_w = clean_bgr_full.shape
        if crop_y < 0 or crop_x < 0 or crop_y + crop_h > full_h or crop_x + crop_w > full_w:
            raise RuntimeError(
                f"Invalid crop for {sample_id}: crop=({crop_x},{crop_y},{crop_w},{crop_h}), "
                f"video_hw=({full_h},{full_w})"
            )
        clean_bgr = clean_bgr_full[:, :, crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
        clean_rgb = clean_bgr[:, [2, 1, 0], :, :].contiguous()

        assert clean_rgb.shape[0] == self.frame_num, f"Frame num mismatch: {clean_rgb.shape[0]} vs {self.frame_num}, sample={sample_id}"
        assert clean_rgb.shape[2] == self.actual_crop_height, f"Height mismatch: {clean_rgb.shape[2]} vs {self.actual_crop_height}, sample={sample_id}"
        assert clean_rgb.shape[3] == self.actual_crop_width, f"Width mismatch: {clean_rgb.shape[3]} vs {self.actual_crop_width}, sample={sample_id}"

        gt = clean_rgb
        if self.random_gain_:
            gt = self.random_gain(gt)
        if np.random.uniform() < self.random_color_prob:
            if "face" in path:
                gt = self.random_color_for_face(gt)
            else:
                gt = self.random_color(gt)
        if self.usm_sharper_:
            gt = self.usm_sharper(gt)
        gt = torch.clamp(gt, 0.0, 1.0)

        aigc_input, stages, degradation_meta = self._apply_degradation(gt, return_intermediate=(self.return_degradation_stages or self.save_degradation_stages))
        grid = torch.zeros(self.frame_num, 2, self.actual_crop_height, self.actual_crop_width, dtype=gt.dtype)

        if self.save_degradation_stages and self._vis_saved < self.vis_num_samples:
            self._save_degradation_debug(sample_id, gt, aigc_input, stages)
            self._vis_saved += 1

        ret_dict: Dict[str, Any] = {
            "read_input": gt,
            "aigc_input": aigc_input,
            "xymap_pc": grid,
            "data_name": sample_id,
            "sample_meta": sample,
            "degradation_meta": degradation_meta,
        }
        if self.return_degradation_stages:
            ret_dict["degradation_stages"] = stages
        return ret_dict

    def _sample_kernel(self, kernel_list, kernel_prob, blur_sigma, betag_range, betap_range, sinc_prob):
        kernel_size = random.choice(self.kernel_range)
        meta = {"kernel_size": kernel_size}
        if np.random.uniform() < sinc_prob:
            omega_c = np.random.uniform(np.pi / 3, np.pi) if kernel_size < 13 else np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            meta.update({"kernel_type": "sinc", "omega_c": float(omega_c)})
        else:
            kernel = random_mixed_kernels(kernel_list, kernel_prob, kernel_size, blur_sigma, blur_sigma, [-math.pi, math.pi], betag_range, betap_range, noise_range=None)
            meta.update({"kernel_type": "mixed"})
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
        return torch.FloatTensor(kernel), meta

    def _normalize_degradation_mix(self, mix: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Normalize optional per-sample degradation mixture config.

        Expected item schema after stage3 expansion:
            {"name": "camera_animal_teacher_safe_v0", "prob": 0.2, "overrides": {...}}

        The method is deliberately permissive for YAML use: if overrides is
        missing, an empty override dict is used, which means the base opt is the
        concrete degradation policy for that branch.
        """
        if not mix:
            return []
        if not isinstance(mix, list):
            raise TypeError(f"degradation_mix must be a list, got {type(mix)}")

        normalized: List[Dict[str, Any]] = []
        total = 0.0
        for i, item in enumerate(mix):
            if not isinstance(item, Mapping):
                raise TypeError(f"degradation_mix[{i}] must be a dict, got {type(item)}")
            prob = float(item.get("prob", 0.0))
            if prob <= 0:
                raise ValueError(f"degradation_mix[{i}] has invalid prob={prob}")
            name = str(item.get("name", f"mix_{i:02d}"))
            overrides = copy.deepcopy(item.get("overrides", {}))
            if not isinstance(overrides, dict):
                raise TypeError(f"degradation_mix[{i}].overrides must be a dict, got {type(overrides)}")
            normalized.append({"name": name, "prob": prob, "overrides": overrides})
            total += prob

        if total <= 0:
            raise ValueError("degradation_mix probability sum must be positive")
        for item in normalized:
            item["prob"] = float(item["prob"] / total)
        return normalized

    def _select_degradation_opt(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Return active degradation opt and metadata for this sample.

        No mix configured => returns self.opt unchanged and a disabled mix meta.
        Mix configured    => samples one branch, merges branch overrides on top
                            of base opt, and returns the selected branch meta.
        """
        if not self.degradation_mix:
            return self.opt, {"enabled": False}

        weights = [float(item["prob"]) for item in self.degradation_mix]
        item = random.choices(self.degradation_mix, weights=weights, k=1)[0]
        active_opt = copy.deepcopy(self.opt)
        active_opt.pop("degradation_mix", None)
        active_opt.update(copy.deepcopy(item.get("overrides", {})))
        active_opt["degradation_preset_name"] = item.get("name", active_opt.get("degradation_preset_name", "mixed"))
        return active_opt, {
            "enabled": True,
            "selected_name": item.get("name"),
            "selected_prob": float(item.get("prob", 0.0)),
        }

    def _get_degradation_value(self, opt: Dict[str, Any], key: str, default: Any) -> Any:
        return opt[key] if key in opt else default

    def _sample_interpolate_mode(self, mode_probs: Optional[Any] = None) -> str:
        """Sample interpolation mode with backward-compatible uniform default."""
        if mode_probs is None:
            return random.choice(["area", "bilinear", "bicubic"])
        if isinstance(mode_probs, Mapping):
            names = list(mode_probs.keys())
            probs = [float(mode_probs[k]) for k in names]
        elif isinstance(mode_probs, list):
            if len(mode_probs) == 3 and all(isinstance(x, (int, float)) for x in mode_probs):
                names = ["area", "bilinear", "bicubic"]
                probs = [float(x) for x in mode_probs]
            else:
                names, probs = [], []
                for item in mode_probs:
                    if isinstance(item, Mapping):
                        names.append(str(item["mode"]))
                        probs.append(float(item.get("prob", 0.0)))
                    elif isinstance(item, (list, tuple)) and len(item) == 2:
                        names.append(str(item[0]))
                        probs.append(float(item[1]))
                    else:
                        raise ValueError(f"Invalid final_downsample_mode_probs item: {item}")
        else:
            raise TypeError(f"final_downsample_mode_probs has invalid type: {type(mode_probs)}")

        valid_modes = {"area", "bilinear", "bicubic", "nearest"}
        for name in names:
            if name not in valid_modes:
                raise ValueError(f"Unsupported interpolate mode={name}; valid={sorted(valid_modes)}")
        if not names or sum(probs) <= 0:
            raise ValueError(f"Invalid interpolate mode probabilities: {mode_probs}")
        return random.choices(names, weights=probs, k=1)[0]

    def _sample_resize(self, resize_prob: List[float], resize_range: List[float]) -> tuple[str, float, str]:
        updown_type = random.choices(["up", "down", "keep"], resize_prob, k=1)[0]
        if updown_type == "up":
            scale = float(np.random.uniform(1.0, float(resize_range[1])))
        elif updown_type == "down":
            scale = float(np.random.uniform(float(resize_range[0]), 1.0))
        else:
            scale = 1.0
        mode = self._sample_interpolate_mode(None)
        return updown_type, scale, mode

    def _maybe_add_noise(
        self,
        out: torch.Tensor,
        *,
        noise_prob: float,
        gaussian_noise_prob: float,
        noise_range: List[float],
        poisson_scale_range: List[float],
        gray_noise_prob: float,
    ) -> tuple[torch.Tensor, str]:
        if np.random.uniform() >= float(noise_prob):
            return out, "none"
        if np.random.uniform() < float(gaussian_noise_prob):
            out = random_add_gaussian_noise_pt(out, sigma_range=noise_range, clip=True, rounds=False, gray_prob=gray_noise_prob)
            return out, "gaussian"
        out = random_add_poisson_noise_pt(out, scale_range=poisson_scale_range, gray_prob=gray_noise_prob, clip=True, rounds=False)
        return out, "poisson"

    def _final_down_up(
        self,
        out: torch.Tensor,
        gt: torch.Tensor,
        sinc_kernel: Optional[torch.Tensor] = None,
        *,
        mode_probs: Optional[Any] = None,
        upsample_mode: str = "bilinear",
        final_sinc_stage: str = "post_lr",
    ):
        ori_h, ori_w = gt.size()[2:4]
        mode = self._sample_interpolate_mode(mode_probs)
        final_sinc_stage = str(final_sinc_stage or "post_lr")

        if sinc_kernel is not None and final_sinc_stage == "pre_downsample":
            out = filter2D(out, sinc_kernel)

        lr = torch.nn.functional.interpolate(out, size=(ori_h // int(self.scl_factor), ori_w // int(self.scl_factor)), mode=mode)

        if sinc_kernel is not None and final_sinc_stage == "post_lr":
            lr = filter2D(lr, sinc_kernel)
        elif sinc_kernel is not None and final_sinc_stage not in {"pre_downsample", "post_lr", "none"}:
            raise ValueError(f"Unsupported final_sinc_stage={final_sinc_stage}; use pre_downsample, post_lr, or none")

        lr = torch.clamp((lr * self.max_value).round(), 0, self.max_value) / self.max_value
        if self.scl_factor != 1:
            aigc_input = torch.nn.functional.interpolate(lr, size=(ori_h, ori_w), mode=upsample_mode)
        else:
            aigc_input = lr
        return lr, torch.clamp(aigc_input, 0.0, 1.0), mode

    def _apply_degradation(self, gt: torch.Tensor, return_intermediate: bool = False):
        """Apply Real-ESRGAN-style degradation and optionally return stage tensors.

        Optimized additions are backward-compatible:
          * first_blur_prob defaults to 1.0, matching original always-blur stage 1.
          * noise_prob/noise_prob2 default to 1.0, matching original always-noise.
          * final_downsample_mode_probs defaults to uniform random choice.
          * degradation_mix is optional and only selects per-sample overrides.
        """
        stages: Dict[str, torch.Tensor] = {}
        meta: Dict[str, Any] = {}
        active_opt, mix_meta = self._select_degradation_opt()
        meta["degradation_mix"] = mix_meta
        if "degradation_preset_name" in active_opt:
            meta["active_degradation_preset"] = active_opt.get("degradation_preset_name")

        # Read active hyperparameters. self.* are used as old-behavior defaults.
        kernel_list = self._get_degradation_value(active_opt, "kernel_list", self.kernel_list)
        kernel_prob = self._get_degradation_value(active_opt, "kernel_prob", self.kernel_prob)
        blur_sigma = self._get_degradation_value(active_opt, "blur_sigma", self.blur_sigma)
        betag_range = self._get_degradation_value(active_opt, "betag_range", self.betag_range)
        betap_range = self._get_degradation_value(active_opt, "betap_range", self.betap_range)
        sinc_prob = float(self._get_degradation_value(active_opt, "sinc_prob", self.sinc_prob))

        kernel_list2 = self._get_degradation_value(active_opt, "kernel_list2", self.kernel_list2)
        kernel_prob2 = self._get_degradation_value(active_opt, "kernel_prob2", self.kernel_prob2)
        blur_sigma2 = self._get_degradation_value(active_opt, "blur_sigma2", self.blur_sigma2)
        betag_range2 = self._get_degradation_value(active_opt, "betag_range2", self.betag_range2)
        betap_range2 = self._get_degradation_value(active_opt, "betap_range2", self.betap_range2)
        sinc_prob2 = float(self._get_degradation_value(active_opt, "sinc_prob2", self.sinc_prob2))

        first_blur_prob = float(self._get_degradation_value(active_opt, "first_blur_prob", self.first_blur_prob))
        resize_prob = self._get_degradation_value(active_opt, "resize_prob", self.resize_prob)
        resize_range = self._get_degradation_value(active_opt, "resize_range", self.resize_range)
        noise_prob = float(self._get_degradation_value(active_opt, "noise_prob", self.noise_prob))
        gaussian_noise_prob = float(self._get_degradation_value(active_opt, "gaussian_noise_prob", self.gaussian_noise_prob))
        noise_range = self._get_degradation_value(active_opt, "noise_range", self.noise_range)
        poisson_scale_range = self._get_degradation_value(active_opt, "poisson_scale_range", self.poisson_scale_range)
        gray_noise_prob = float(self._get_degradation_value(active_opt, "gray_noise_prob", self.gray_noise_prob))

        second_blur_prob = float(self._get_degradation_value(active_opt, "second_blur_prob", self.second_blur_prob))
        resize_prob2 = self._get_degradation_value(active_opt, "resize_prob2", self.resize_prob2)
        resize_range2 = self._get_degradation_value(active_opt, "resize_range2", self.resize_range2)
        noise_prob2 = float(self._get_degradation_value(active_opt, "noise_prob2", self.noise_prob2))
        gaussian_noise_prob2 = float(self._get_degradation_value(active_opt, "gaussian_noise_prob2", self.gaussian_noise_prob2))
        noise_range2 = self._get_degradation_value(active_opt, "noise_range2", self.noise_range2)
        poisson_scale_range2 = self._get_degradation_value(active_opt, "poisson_scale_range2", self.poisson_scale_range2)
        gray_noise_prob2 = float(self._get_degradation_value(active_opt, "gray_noise_prob2", self.gray_noise_prob2))

        final_sinc_prob = float(self._get_degradation_value(active_opt, "final_sinc_prob", self.final_sinc_prob))
        final_downsample_mode_probs = self._get_degradation_value(active_opt, "final_downsample_mode_probs", self.final_downsample_mode_probs)
        final_upsample_mode = str(self._get_degradation_value(active_opt, "final_upsample_mode", self.final_upsample_mode))
        final_sinc_stage = str(self._get_degradation_value(active_opt, "final_sinc_stage", self.final_sinc_stage))
        max_effective_mag = self._get_degradation_value(active_opt, "max_effective_magnification_for_kd", self.max_effective_magnification_for_kd)

        kernel, kernel_meta = self._sample_kernel(kernel_list, kernel_prob, blur_sigma, betag_range, betap_range, sinc_prob)
        kernel2, kernel2_meta = self._sample_kernel(kernel_list2, kernel_prob2, blur_sigma2, betag_range2, betap_range2, sinc_prob2)
        meta["kernel1"] = kernel_meta
        meta["kernel2"] = kernel2_meta

        if np.random.uniform() < final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = torch.FloatTensor(circular_lowpass_kernel(omega_c, kernel_size, pad_to=21))
            meta["final_sinc"] = {"enabled": True, "kernel_size": kernel_size, "omega_c": float(omega_c), "stage": final_sinc_stage}
        else:
            sinc_kernel = None
            meta["final_sinc"] = {"enabled": False, "stage": final_sinc_stage}

        out = gt.clone()
        scale1 = 1.0
        scale2 = 1.0

        if active_opt.get("degradation_1", True):
            blur1_enabled = np.random.uniform() < first_blur_prob
            if blur1_enabled:
                out = filter2D(out, kernel)
            updown_type, scale1, mode = self._sample_resize(resize_prob, resize_range)
            out = torch.nn.functional.interpolate(out, scale_factor=scale1, mode=mode)
            out, noise_type = self._maybe_add_noise(
                out,
                noise_prob=noise_prob,
                gaussian_noise_prob=gaussian_noise_prob,
                noise_range=noise_range,
                poisson_scale_range=poisson_scale_range,
                gray_noise_prob=gray_noise_prob,
            )
            meta["degradation_1"] = {
                "enabled": True,
                "blur1_enabled": bool(blur1_enabled),
                "first_blur_prob": float(first_blur_prob),
                "resize_type": updown_type,
                "resize_scale": float(scale1),
                "resize_mode": mode,
                "noise_prob": float(noise_prob),
                "noise_type": noise_type,
            }
            if return_intermediate:
                stages["degradation_1"] = torch.clamp(out.detach().clone(), 0.0, 1.0)
        else:
            meta["degradation_1"] = {"enabled": False}
            if return_intermediate:
                stages["degradation_1"] = out.detach().clone()

        if active_opt.get("degradation_2", True):
            blur2_enabled = np.random.uniform() < second_blur_prob
            if blur2_enabled:
                out = filter2D(out, kernel2)
            updown_type, scale2, mode = self._sample_resize(resize_prob2, resize_range2)
            out = torch.nn.functional.interpolate(out, scale_factor=scale2, mode=mode)
            out, noise_type = self._maybe_add_noise(
                out,
                noise_prob=noise_prob2,
                gaussian_noise_prob=gaussian_noise_prob2,
                noise_range=noise_range2,
                poisson_scale_range=poisson_scale_range2,
                gray_noise_prob=gray_noise_prob2,
            )
            meta["degradation_2"] = {
                "enabled": True,
                "blur2_enabled": bool(blur2_enabled),
                "second_blur_prob": float(second_blur_prob),
                "resize_type": updown_type,
                "resize_scale": float(scale2),
                "resize_mode": mode,
                "noise_prob": float(noise_prob2),
                "noise_type": noise_type,
            }
            if return_intermediate:
                stages["degradation_2"] = torch.clamp(out.detach().clone(), 0.0, 1.0)
        else:
            meta["degradation_2"] = {"enabled": False}
            if return_intermediate:
                stages["degradation_2"] = torch.clamp(out.detach().clone(), 0.0, 1.0)

        effective_mag = float(self.scl_factor / max(1e-8, float(scale1) * float(scale2)))
        meta["effective_magnification"] = effective_mag
        if max_effective_mag is not None:
            max_effective_mag = float(max_effective_mag)
            meta["kd_gate"] = {
                "max_effective_magnification": max_effective_mag,
                "effective_magnification_pass": bool(effective_mag <= max_effective_mag),
            }

        if active_opt.get("degradation_3", True):
            lr, aigc_input, mode = self._final_down_up(
                out,
                gt,
                sinc_kernel=sinc_kernel,
                mode_probs=final_downsample_mode_probs,
                upsample_mode=final_upsample_mode,
                final_sinc_stage=final_sinc_stage,
            )
            meta["degradation_3"] = {
                "enabled": True,
                "final_resize_mode": mode,
                "final_upsample_mode": final_upsample_mode,
            }
        else:
            # Still generate the model-facing LQ by scale-factor down/up, without final sinc.
            lr, aigc_input, mode = self._final_down_up(
                out,
                gt,
                sinc_kernel=None,
                mode_probs=final_downsample_mode_probs,
                upsample_mode=final_upsample_mode,
                final_sinc_stage=final_sinc_stage,
            )
            meta["degradation_3"] = {
                "enabled": False,
                "fallback_down_up_mode": mode,
                "final_upsample_mode": final_upsample_mode,
            }

        if return_intermediate:
            stages["lr_native"] = lr.detach().clone()
            stages["aigc_input"] = aigc_input.detach().clone()
        return aigc_input, stages, meta

    def _select_vis_frames(self, tensor: torch.Tensor) -> torch.Tensor:
        idxs = []
        for idx in self.vis_frame_indices:
            idx = int(idx)
            if 0 <= idx < tensor.shape[0]:
                idxs.append(idx)
        if not idxs:
            idxs = [0, tensor.shape[0] // 2, tensor.shape[0] - 1]
        return tensor[idxs]

    def _save_grid_resized(self, tensor: torch.Tensor, path: str, target_hw: Optional[tuple[int, int]] = None, nrow: int = 4) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        x = self._select_vis_frames(tensor.detach().cpu().float())
        if target_hw is not None and x.shape[-2:] != target_hw:
            x = torch.nn.functional.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        save_image(torch.clamp(x, 0.0, 1.0), path, nrow=nrow)

    def _save_degradation_debug(self, sample_id: str, gt: torch.Tensor, aigc_input: torch.Tensor, stages: Dict[str, torch.Tensor]) -> None:
        out_dir = Path(self.vis_output_dir) / sample_id
        target_hw = (gt.shape[-2], gt.shape[-1])
        self._save_grid_resized(gt, str(out_dir / "gt_grid.png"), target_hw=target_hw)
        self._save_grid_resized(aigc_input, str(out_dir / "lq_up_grid.png"), target_hw=target_hw)
        if "degradation_1" in stages:
            self._save_grid_resized(stages["degradation_1"], str(out_dir / "deg1_up_for_view_grid.png"), target_hw=target_hw)
        if "degradation_2" in stages:
            self._save_grid_resized(stages["degradation_2"], str(out_dir / "deg2_up_for_view_grid.png"), target_hw=target_hw)
        if "lr_native" in stages:
            self._save_grid_resized(stages["lr_native"], str(out_dir / "lr_native_grid.png"), target_hw=None)

    def save_grid(self, images, path, nrow=4):
        if self.save_image_:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            save_image(images, path, nrow=nrow)
