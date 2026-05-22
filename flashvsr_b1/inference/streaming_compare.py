from __future__ import annotations

import importlib.util
import math
import os
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


@dataclass(frozen=True)
class CanvasSpec:
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    pad_left: int = 0
    pad_top: int = 0
    pad_right: int = 0
    pad_bottom: int = 0

    @property
    def output_crop_box(self) -> tuple[int, int, int, int]:
        return (
            self.pad_left,
            self.pad_top,
            self.pad_left + self.source_width,
            self.pad_top + self.source_height,
        )


@dataclass(frozen=True)
class FrameSelection:
    indices: list[int]
    model_frames: int
    effective_output_frames: int


def natural_key(path: str | Path) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"([0-9]+)", Path(path).name)
    ]


def parse_window_size(value: str | Sequence[int]) -> tuple[int, int, int]:
    parts = re.split(r"[xX, ]+", value.strip()) if isinstance(value, str) else list(value)
    parts = [p for p in parts if p != ""]
    if len(parts) != 3:
        raise ValueError(f"window_size must contain 3 integers, got {value!r}")
    window = tuple(int(p) for p in parts)
    if any(v <= 0 for v in window):
        raise ValueError(f"window_size values must be positive, got {window}")
    return window


def canvas_for_model_input(
    *,
    width: int,
    height: int,
    multiple: int = 128,
    mode: str = "pad",
    scale: float = 1.0,
) -> CanvasSpec:
    if mode != "pad":
        raise ValueError("This simplified inference path only supports padding model-input LQ.")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    width = int(round(width * scale))
    height = int(round(height * scale))
    target_width = int(math.ceil(width / multiple) * multiple)
    target_height = int(math.ceil(height / multiple) * multiple)
    return CanvasSpec(
        source_width=width,
        source_height=height,
        target_width=target_width,
        target_height=target_height,
        pad_right=target_width - width,
        pad_bottom=target_height - height,
    )


def largest_8n1_leq(n: int) -> int:
    return 0 if n < 1 else ((int(n) - 1) // 8) * 8 + 1


def select_streaming_frame_count(
    total_frames: int,
    *,
    tail_padding: int = 4,
    max_frames: int = 0,
) -> FrameSelection:
    usable = min(total_frames, int(max_frames)) if int(max_frames) > 0 else total_frames
    if usable <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    indices = list(range(usable)) + [usable - 1] * int(tail_padding)
    model_frames = largest_8n1_leq(len(indices))
    if model_frames < 25:
        raise ValueError(f"FlashVSR streaming needs at least 25 model frames, got {model_frames}")
    return FrameSelection(
        indices=indices[:model_frames],
        model_frames=model_frames,
        effective_output_frames=model_frames - int(tail_padding),
    )


def discover_inputs(path: str | Path, *, max_videos: int = 0) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(root)
    if any(p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in root.iterdir()):
        return [root]
    items = [
        p
        for p in root.iterdir()
        if not p.name.startswith(".")
        and (
            (p.is_file() and p.suffix.lower() in VIDEO_EXTS)
            or (p.is_dir() and any(c.is_file() and c.suffix.lower() in IMAGE_EXTS for c in p.iterdir()))
        )
    ]
    items.sort(key=natural_key)
    return items[: int(max_videos)] if int(max_videos) > 0 else items


def build_output_path(*, save_root: str | Path, input_path: str | Path, model_type: str, seed: int) -> Path:
    input_path = Path(input_path)
    stem = input_path.stem if input_path.suffix else input_path.name
    return Path(save_root) / model_type / f"{stem}_{model_type}_seed{int(seed)}.mp4"


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }[name.lower()]


def add_flashvsr_to_path(flashvsr_root: str | None) -> None:
    root = flashvsr_root or os.environ.get("FLASHVSR_ROOT")
    if not root:
        return
    root_path = Path(root).expanduser().resolve()
    for path in (root_path / "examples" / "WanVSR", root_path):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _pil_to_tensor(img: Image.Image, *, dtype: torch.dtype, device: str) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    tensor = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1).div(255.0).mul(2.0).sub(1.0)
    return tensor.to(dtype)


def _list_images(folder: Path) -> list[Path]:
    images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    images.sort(key=natural_key)
    return images


def _video_frame_count(reader, meta: dict[str, Any]) -> int:
    nframes = meta.get("nframes")
    if isinstance(nframes, int) and 0 < nframes < 10**9:
        return nframes
    try:
        return int(reader.count_frames())
    except Exception:
        idx = 0
        while True:
            try:
                reader.get_data(idx)
            except Exception:
                return idx
            idx += 1


def prepare_lq_video(
    path: str | Path,
    *,
    dtype: torch.dtype,
    device: str = "cuda",
    multiple: int = 128,
    max_frames: int = 0,
    scale: float = 1.0,
) -> tuple[torch.Tensor, int, int, int, int, CanvasSpec, int]:
    import imageio

    path = Path(path)
    if path.is_dir():
        frame_paths = _list_images(path)
        if not frame_paths:
            raise FileNotFoundError(f"No image frames in {path}")
        with Image.open(frame_paths[0]) as first:
            width, height = first.size
        fps = 30
        selection = select_streaming_frame_count(len(frame_paths), max_frames=max_frames)
        frames = []
        for idx in selection.indices:
            with Image.open(frame_paths[idx]) as img:
                frames.append(img.convert("RGB").copy())
    elif path.is_file() and path.suffix.lower() in VIDEO_EXTS:
        reader = imageio.get_reader(path)
        try:
            meta = reader.get_meta_data()
            fps_value = meta.get("fps", 30)
            fps = int(round(fps_value)) if isinstance(fps_value, (int, float)) else 30
            first = Image.fromarray(reader.get_data(0)).convert("RGB")
            width, height = first.size
            selection = select_streaming_frame_count(
                _video_frame_count(reader, meta),
                max_frames=max_frames,
            )
            frames = [Image.fromarray(reader.get_data(idx)).convert("RGB") for idx in selection.indices]
        finally:
            try:
                reader.close()
            except Exception:
                pass
    else:
        raise ValueError(f"Unsupported input: {path}")

    canvas = canvas_for_model_input(width=width, height=height, multiple=multiple, scale=scale)
    if scale != 1.0:
        frames = [
            img.resize((canvas.source_width, canvas.source_height), Image.BICUBIC)
            for img in frames
        ]
    tensors = [_pil_to_tensor(img, dtype=dtype, device=device) for img in frames]
    video = torch.stack(tensors, 0).permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    if canvas.pad_right or canvas.pad_bottom:
        video = F.pad(video, (0, canvas.pad_right, 0, canvas.pad_bottom), mode="replicate")
    return (
        video,
        canvas.target_height,
        canvas.target_width,
        selection.model_frames,
        fps,
        canvas,
        selection.effective_output_frames,
    )


def tensor_to_pil(frames: torch.Tensor, *, canvas: CanvasSpec, keep_frames: int) -> list[Image.Image]:
    from einops import rearrange

    arr = rearrange(frames, "c t h w -> t h w c")
    arr = ((arr.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    crop = canvas.output_crop_box
    return [Image.fromarray(frame).crop(crop) for frame in arr[:keep_frames]]


def save_video(frames: list[Image.Image], path: str | Path, *, fps: int, quality: int = 6) -> None:
    import imageio
    from tqdm import tqdm

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, quality=quality)
    try:
        for frame in tqdm(frames, desc=f"Saving {path.name}"):
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()


def _strip_prefixes(key: str) -> str:
    prefixes = ("student.", "teacher.", "module.", "model.", "dit.", "denoising_model.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
                break
    return key


def normalize_dit_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for raw_key, value in state_dict.items():
        key = _strip_prefixes(raw_key).replace(".o_proj.", ".o.")
        if ".qkv_proj." not in key:
            out[key] = value
            continue
        base = key.replace(".qkv_proj.", ".")
        q, k, v = value.chunk(3, dim=0)
        if base.endswith("weight"):
            out[base.replace(".weight", ".q.weight")] = q
            out[base.replace(".weight", ".k.weight")] = k
            out[base.replace(".weight", ".v.weight")] = v
        elif base.endswith("bias"):
            out[base.replace(".bias", ".q.bias")] = q
            out[base.replace(".bias", ".k.bias")] = k
            out[base.replace(".bias", ".v.bias")] = v
    return out


def _load_state(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    ckpt = torch.load(str(path), map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("student", "state_dict", "model", "module", "dit"):
            if isinstance(ckpt.get(key), dict):
                return ckpt[key]
    return ckpt


def _load_local_wan_module():
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "wan_video_dit.py"
    spec = importlib.util.spec_from_file_location("flashvsr_b1_lswa_wan_video_dit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    old_utils = sys.modules.get("utils")
    shim = types.ModuleType("utils")
    shim.hash_state_dict_keys = lambda state_dict: state_dict
    sys.modules["utils"] = shim
    try:
        spec.loader.exec_module(module)
    finally:
        if old_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = old_utils
    return module


def replace_dit_with_lswa(pipe, *, window_size: tuple[int, int, int], student_ckpt: str | None = None) -> None:
    source = pipe.dit
    first_block = source.blocks[0]
    patch_size = tuple(int(v) for v in source.patch_size)
    local_wan = _load_local_wan_module()
    lswa_dit = local_wan.WanModel(
        dim=int(source.dim),
        in_dim=int(source.patch_embedding.in_channels),
        ffn_dim=int(first_block.ffn[0].out_features),
        out_dim=int(source.head.head.out_features) // math.prod(patch_size),
        text_dim=int(source.text_embedding[0].in_features),
        freq_dim=int(source.freq_dim),
        eps=1e-6,
        patch_size=patch_size,
        num_heads=int(first_block.num_heads),
        num_layers=len(source.blocks),
        has_image_input=bool(getattr(source, "has_image_input", False)),
        lswa_spatial_window=(int(window_size[1]), int(window_size[2])),
        lswa_temporal_window=int(window_size[0]),
    )
    lswa_dit.load_state_dict(source.state_dict(), strict=False)
    if student_ckpt:
        state = normalize_dit_state_dict(_load_state(student_ckpt))
        result = lswa_dit.load_state_dict(state, strict=False)
        print(f"[LSWA] loaded student_ckpt={student_ckpt} missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")
    pipe.dit = lswa_dit
