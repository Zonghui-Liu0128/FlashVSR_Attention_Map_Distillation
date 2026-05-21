from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
DEFAULT_TEST_PATH = (
    "/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/"
    "vsr_datasets/animal_videos/videos_960x720/lq/test"
)
DEFAULT_FLASHVSR_CKPT_DIR = (
    "/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/"
    "FlashVSR-v1.1"
)


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
    crop_left: int = 0
    crop_top: int = 0

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


@dataclass
class PreparedInput:
    path: Path
    video: torch.Tensor
    height: int
    width: int
    model_frames: int
    effective_output_frames: int
    fps: int
    canvas: CanvasSpec


def natural_key(path: str | Path) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"([0-9]+)", Path(path).name)
    ]


def parse_window_size(value: str | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, str):
        parts = [part for part in re.split(r"[xX, ]+", value.strip()) if part]
    else:
        parts = list(value)
    if len(parts) != 3:
        raise ValueError(f"window_size must contain 3 integers, got {value!r}")
    window = tuple(int(part) for part in parts)
    if any(v <= 0 for v in window):
        raise ValueError(f"window_size values must be positive, got {window}")
    return window


def largest_8n1_leq(n: int) -> int:
    return 0 if n < 1 else ((int(n) - 1) // 8) * 8 + 1


def select_streaming_frame_count(
    total_frames: int,
    *,
    tail_padding: int = 4,
    max_frames: int = 0,
) -> FrameSelection:
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    usable = min(total_frames, int(max_frames)) if int(max_frames) > 0 else total_frames
    indices = list(range(usable)) + [usable - 1] * int(tail_padding)
    model_frames = largest_8n1_leq(len(indices))
    if model_frames < 25:
        raise ValueError(
            "FlashVSR streaming inference needs at least 25 model frames after "
            f"tail padding; got total_frames={total_frames}, model_frames={model_frames}"
        )
    indices = indices[:model_frames]
    return FrameSelection(
        indices=indices,
        model_frames=model_frames,
        effective_output_frames=max(0, model_frames - int(tail_padding)),
    )


def canvas_for_model_input(
    *,
    width: int,
    height: int,
    multiple: int = 128,
    mode: str = "pad",
) -> CanvasSpec:
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid input size {width}x{height}")
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    mode = mode.lower()
    if mode == "none":
        return CanvasSpec(width, height, width, height)
    if mode == "pad":
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
    if mode == "center_crop":
        target_width = int(width // multiple * multiple)
        target_height = int(height // multiple * multiple)
        if target_width <= 0 or target_height <= 0:
            raise ValueError(
                f"Cannot crop {width}x{height} to positive multiple={multiple}"
            )
        crop_left = (width - target_width) // 2
        crop_top = (height - target_height) // 2
        return CanvasSpec(
            source_width=target_width,
            source_height=target_height,
            target_width=target_width,
            target_height=target_height,
            crop_left=crop_left,
            crop_top=crop_top,
        )
    raise ValueError(f"Unknown canvas mode {mode!r}; use pad, center_crop, or none")


def discover_inputs(path: str | Path, *, max_videos: int = 0) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(root)

    direct_images = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if direct_images:
        return [root]

    items = [
        p
        for p in root.iterdir()
        if not p.name.startswith(".")
        and (
            (p.is_file() and p.suffix.lower() in VIDEO_EXTS)
            or (p.is_dir() and any(c.suffix.lower() in IMAGE_EXTS for c in p.iterdir() if c.is_file()))
        )
    ]
    items.sort(key=natural_key)
    if int(max_videos) > 0:
        items = items[: int(max_videos)]
    return items


def build_output_path(
    *,
    save_root: str | Path,
    input_path: str | Path,
    model_type: str,
    seed: int,
) -> Path:
    input_path = Path(input_path)
    model_type = model_type.upper()
    stem = input_path.stem if input_path.suffix else input_path.name
    return Path(save_root) / model_type / f"{stem}_{model_type}_seed{int(seed)}.mp4"


def _strip_known_prefix(key: str) -> str:
    changed = True
    prefixes = (
        "student.",
        "teacher.",
        "module.",
        "model.",
        "dit.",
        "denoising_model.",
    )
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
                break
    return key


def normalize_dit_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for raw_key, value in state_dict.items():
        key = _strip_known_prefix(raw_key)
        key = key.replace(".o_proj.", ".o.")
        if ".qkv_proj." not in key:
            normalized[key] = value
            continue

        qkv_key = key.replace(".qkv_proj.", ".")
        if qkv_key.endswith("weight"):
            q, k, v = value.chunk(3, dim=0)
            normalized[qkv_key.replace(".weight", ".q.weight")] = q
            normalized[qkv_key.replace(".weight", ".k.weight")] = k
            normalized[qkv_key.replace(".weight", ".v.weight")] = v
        elif qkv_key.endswith("bias"):
            q, k, v = value.chunk(3, dim=0)
            normalized[qkv_key.replace(".bias", ".q.bias")] = q
            normalized[qkv_key.replace(".bias", ".k.bias")] = k
            normalized[qkv_key.replace(".bias", ".v.bias")] = v
        else:
            normalized[key] = value
    return normalized


def _load_raw_state(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    ckpt = torch.load(str(path), map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("student", "state_dict", "model", "module", "dit"):
            value = ckpt.get(key)
            if isinstance(value, dict):
                return value
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint payload in {path}: {type(ckpt)!r}")
    return ckpt


def _dtype_from_name(name: str) -> torch.dtype:
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}") from exc


def _add_flashvsr_paths(flashvsr_root: str | Path | None) -> Path | None:
    if flashvsr_root is None:
        env_root = os.environ.get("FLASHVSR_ROOT")
        flashvsr_root = env_root if env_root else None
    if flashvsr_root is None:
        return None
    root = Path(flashvsr_root).expanduser().resolve()
    examples = root / "examples" / "WanVSR"
    for candidate in (examples, root):
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    return root


def _import_local_wan_module():
    import importlib.util
    import types

    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "wan_video_dit.py"
    spec = importlib.util.spec_from_file_location("flashvsr_b1_local_wan_video_dit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load local WanModel from {path}")
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


def _build_local_wan_from(source: torch.nn.Module, *, model_type: str, window_size: tuple[int, int, int]):
    local_wan = _import_local_wan_module()
    first_block = source.blocks[0]
    ffn = first_block.ffn
    patch_size = tuple(int(v) for v in source.patch_size)
    head_out = int(source.head.head.out_features)
    out_dim = head_out // math.prod(patch_size)
    lswa_spatial = (window_size[1], window_size[2]) if model_type.upper() == "LSWA" else None
    model = local_wan.WanModel(
        dim=int(source.dim),
        in_dim=int(source.patch_embedding.in_channels),
        ffn_dim=int(ffn[0].out_features),
        out_dim=out_dim,
        text_dim=int(source.text_embedding[0].in_features),
        freq_dim=int(source.freq_dim),
        eps=1e-6,
        patch_size=patch_size,
        num_heads=int(first_block.num_heads),
        num_layers=len(source.blocks),
        has_image_input=bool(getattr(source, "has_image_input", False)),
        lswa_spatial_window=lswa_spatial,
        lswa_temporal_window=int(window_size[0]),
    )
    missing, unexpected = model.load_state_dict(source.state_dict(), strict=False)
    if unexpected:
        print(f"[warn] local WanModel source load unexpected_keys={len(unexpected)}")
    if missing:
        print(f"[warn] local WanModel source load missing_keys={len(missing)}")
    return model


def _maybe_replace_dit_for_mode(pipe, args) -> None:
    model_type = args.model_type.upper()
    model_weight = Path(args.model_weight)
    needs_local = model_type == "LSWA" or model_weight.suffix == ".pt" or args.force_local_wan
    if not needs_local:
        return

    source = pipe.dit
    device = next(source.parameters()).device
    dtype = next(source.parameters()).dtype
    local_model = _build_local_wan_from(
        source,
        model_type=model_type,
        window_size=parse_window_size(args.window_size),
    ).to(device=device, dtype=dtype)

    raw_state = _load_raw_state(model_weight)
    normalized = normalize_dit_state_dict(raw_state)
    load_result = local_model.load_state_dict(normalized, strict=False)
    print(
        "[load] local WanModel "
        f"missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}"
    )
    pipe.dit = local_model


def _resolve_prompt_tensor(flashvsr_root: Path | None, prompt_tensor: str | None) -> Path | None:
    if prompt_tensor:
        return Path(prompt_tensor)
    if flashvsr_root is not None:
        candidate = flashvsr_root / "examples" / "WanVSR" / "prompt_tensor" / "posi_prompt.pth"
        if candidate.exists():
            return candidate
    return None


def init_flashvsr_pipeline(args):
    flashvsr_root = _add_flashvsr_paths(args.flashvsr_root)
    try:
        from diffsynth import ModelManager, FlashVSRTinyLongPipeline, FlashVSRTinyPipeline
        from utils.TCDecoder import build_tcdecoder
        from utils.utils import Causal_LQ4x_Proj
    except Exception as exc:
        raise RuntimeError(
            "Unable to import official FlashVSR. Set FLASHVSR_ROOT or pass "
            "--flashvsr-root /path/to/OpenImagingLab/FlashVSR."
        ) from exc

    dtype = _dtype_from_name(args.dtype)
    model_weight = Path(args.model_weight)
    if model_weight.suffix == ".pt":
        base_model_weight = (
            Path(args.base_model_weight)
            if args.base_model_weight
            else Path(DEFAULT_FLASHVSR_CKPT_DIR) / "diffusion_pytorch_model_streaming_dmd.safetensors"
        )
    else:
        base_model_weight = model_weight

    print(f"[load] base diffusion={base_model_weight}")
    manager = ModelManager(torch_dtype=dtype, device="cpu")
    manager.load_models([str(base_model_weight)])
    pipeline_cls = FlashVSRTinyLongPipeline if args.pipeline == "long" else FlashVSRTinyPipeline
    pipe = pipeline_cls.from_model_manager(manager, device=args.device)
    _maybe_replace_dit_for_mode(pipe, args)

    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(
        in_dim=3,
        out_dim=1536,
        layer_num=1,
    ).to(args.device, dtype=dtype)
    if args.lq_proj_ckpt:
        print(f"[load] LQ_proj_in={args.lq_proj_ckpt}")
        pipe.denoising_model().LQ_proj_in.load_state_dict(
            torch.load(args.lq_proj_ckpt, map_location="cpu"),
            strict=True,
        )
    pipe.denoising_model().LQ_proj_in.to(args.device)

    pipe.TCDecoder = build_tcdecoder(
        new_channels=[512, 256, 128, 128],
        new_latent_channels=16 + 768,
    )
    if args.tc_decoder_ckpt:
        print(f"[load] TCDecoder={args.tc_decoder_ckpt}")
        load_result = pipe.TCDecoder.load_state_dict(
            torch.load(args.tc_decoder_ckpt, map_location="cpu"),
            strict=False,
        )
        print(f"[load] TCDecoder missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}")

    pipe.to(args.device)
    if not args.disable_vram_management:
        pipe.enable_vram_management(num_persistent_param_in_dit=None)

    prompt_tensor = _resolve_prompt_tensor(flashvsr_root, args.prompt_tensor)
    context_tensor = None
    if prompt_tensor is not None and prompt_tensor.exists():
        context_tensor = torch.load(prompt_tensor, map_location=args.device)
    pipe.init_cross_kv(context_tensor=context_tensor)
    pipe.load_models_to_device(["dit", "vae"])
    return pipe


def _is_video(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS


def _list_images(path: Path) -> list[Path]:
    images = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    images.sort(key=natural_key)
    return images


def _read_video_frame_count(reader, meta: dict[str, Any]) -> int:
    nframes = meta.get("nframes")
    if isinstance(nframes, int) and nframes > 0 and nframes < 10**9:
        return nframes
    try:
        return int(reader.count_frames())
    except Exception:
        count = 0
        while True:
            try:
                reader.get_data(count)
            except Exception:
                return count
            count += 1


def _pil_to_tensor(img: Image.Image, *, dtype: torch.dtype, device: str) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    tensor = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1).div(255.0).mul(2.0).sub(1.0)
    return tensor.to(dtype=dtype)


def _scale_then_center_crop(img: Image.Image, *, scale: float, multiple: int) -> tuple[Image.Image, CanvasSpec]:
    width, height = img.size
    scaled_width = int(round(width * scale))
    scaled_height = int(round(height * scale))
    target_width = scaled_width // multiple * multiple
    target_height = scaled_height // multiple * multiple
    if target_width <= 0 or target_height <= 0:
        raise ValueError(
            f"Scaled input {scaled_width}x{scaled_height} is too small for multiple={multiple}"
        )
    up = img.resize((scaled_width, scaled_height), Image.BICUBIC)
    left = (scaled_width - target_width) // 2
    top = (scaled_height - target_height) // 2
    cropped = up.crop((left, top, left + target_width, top + target_height))
    return cropped, CanvasSpec(
        source_width=target_width,
        source_height=target_height,
        target_width=target_width,
        target_height=target_height,
        crop_left=left,
        crop_top=top,
    )


def _prepare_pil_frames(
    frames: Iterable[Image.Image],
    *,
    input_mode: str,
    canvas_mode: str,
    scale: float,
    multiple: int,
    dtype: torch.dtype,
    device: str,
) -> tuple[torch.Tensor, CanvasSpec]:
    tensors: list[torch.Tensor] = []
    canvas: CanvasSpec | None = None
    for img in frames:
        if input_mode == "native_lr":
            img, frame_canvas = _scale_then_center_crop(img, scale=scale, multiple=multiple)
        else:
            if canvas is None:
                frame_canvas = canvas_for_model_input(
                    width=img.size[0],
                    height=img.size[1],
                    multiple=multiple,
                    mode=canvas_mode,
                )
            else:
                frame_canvas = canvas
            if frame_canvas.crop_left or frame_canvas.crop_top:
                img = img.crop(
                    (
                        frame_canvas.crop_left,
                        frame_canvas.crop_top,
                        frame_canvas.crop_left + frame_canvas.target_width,
                        frame_canvas.crop_top + frame_canvas.target_height,
                    )
                )
        canvas = frame_canvas
        tensors.append(_pil_to_tensor(img, dtype=dtype, device=device))

    if canvas is None or not tensors:
        raise RuntimeError("No frames were prepared")
    video = torch.stack(tensors, dim=0).permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    if input_mode != "native_lr" and (canvas.pad_right or canvas.pad_bottom):
        video = F.pad(
            video,
            (canvas.pad_left, canvas.pad_right, canvas.pad_top, canvas.pad_bottom),
            mode="replicate",
        )
    return video, canvas


def prepare_input_tensor(path: str | Path, args) -> PreparedInput:
    import imageio

    path = Path(path)
    dtype = _dtype_from_name(args.dtype)
    input_mode = args.input_mode
    if input_mode not in {"model_input", "native_lr"}:
        raise ValueError(f"Unsupported input_mode={input_mode!r}")

    if path.is_dir():
        image_paths = _list_images(path)
        if not image_paths:
            raise FileNotFoundError(f"No images found in {path}")
        total = len(image_paths)
        with Image.open(image_paths[0]) as first:
            source_width, source_height = first.size
        fps = int(args.fps)
        selection = select_streaming_frame_count(
            total,
            tail_padding=args.tail_padding,
            max_frames=args.max_frames,
        )
        selected_frames = []
        for idx in selection.indices:
            with Image.open(image_paths[idx]) as img:
                selected_frames.append(img.convert("RGB").copy())
    elif _is_video(path):
        reader = imageio.get_reader(path)
        try:
            meta = {}
            try:
                meta = reader.get_meta_data()
            except Exception:
                pass
            fps_value = meta.get("fps", args.fps)
            fps = int(round(fps_value)) if isinstance(fps_value, (int, float)) else int(args.fps)
            first = Image.fromarray(reader.get_data(0)).convert("RGB")
            source_width, source_height = first.size
            total = _read_video_frame_count(reader, meta)
            selection = select_streaming_frame_count(
                total,
                tail_padding=args.tail_padding,
                max_frames=args.max_frames,
            )
            selected_frames = [
                Image.fromarray(reader.get_data(idx)).convert("RGB")
                for idx in selection.indices
            ]
        finally:
            try:
                reader.close()
            except Exception:
                pass
    else:
        raise ValueError(f"Unsupported input path: {path}")

    video, canvas = _prepare_pil_frames(
        selected_frames,
        input_mode=input_mode,
        canvas_mode=args.canvas_mode,
        scale=float(args.scale),
        multiple=int(args.multiple),
        dtype=dtype,
        device=args.device,
    )
    print(
        f"[input] {path.name}: source={source_width}x{source_height} "
        f"frames={total} -> model={canvas.target_width}x{canvas.target_height} "
        f"model_frames={selection.model_frames} output_frames={selection.effective_output_frames}"
    )
    return PreparedInput(
        path=path,
        video=video,
        height=canvas.target_height,
        width=canvas.target_width,
        model_frames=selection.model_frames,
        effective_output_frames=selection.effective_output_frames,
        fps=fps,
        canvas=canvas,
    )


def tensor_to_pil_frames(frames: torch.Tensor) -> list[Image.Image]:
    from einops import rearrange

    frames = rearrange(frames, "c t h w -> t h w c")
    arr = ((frames.float() + 1.0) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    return [Image.fromarray(frame) for frame in arr]


def save_video(
    frames: list[Image.Image],
    save_path: str | Path,
    *,
    fps: int,
    quality: int,
) -> None:
    import imageio
    from tqdm import tqdm

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(save_path), fps=fps, quality=quality)
    try:
        for frame in tqdm(frames, desc=f"Saving {save_path.name}"):
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()


def run_inference(args) -> list[dict[str, Any]]:
    from tqdm import tqdm

    inputs = discover_inputs(args.test_path, max_videos=args.max_videos)
    if not inputs:
        raise FileNotFoundError(f"No input videos or frame folders found under {args.test_path}")
    if args.dry_run:
        for path in inputs:
            print(path)
        return []

    pipe = init_flashvsr_pipeline(args)
    manifest: list[dict[str, Any]] = []
    for path in tqdm(inputs, desc="Inputs"):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
        prepared = prepare_input_tensor(path, args)
        topk_ratio = float(args.sparse_ratio) * 768 * 1280 / (prepared.height * prepared.width)
        start = time.perf_counter()
        with torch.no_grad():
            output = pipe(
                prompt="",
                negative_prompt="",
                cfg_scale=1.0,
                num_inference_steps=1,
                seed=int(args.seed),
                LQ_video=prepared.video,
                num_frames=prepared.model_frames,
                height=prepared.height,
                width=prepared.width,
                is_full_block=False,
                if_buffer=True,
                topk_ratio=topk_ratio,
                kv_ratio=float(args.kv_ratio),
                local_range=int(args.local_range),
                color_fix=not args.no_color_fix,
            )
        elapsed = time.perf_counter() - start
        pil_frames = tensor_to_pil_frames(output)
        if prepared.canvas.output_crop_box != (0, 0, prepared.width, prepared.height):
            pil_frames = [frame.crop(prepared.canvas.output_crop_box) for frame in pil_frames]
        pil_frames = pil_frames[: prepared.effective_output_frames]
        out_path = build_output_path(
            save_root=args.save_root,
            input_path=path,
            model_type=args.model_type,
            seed=args.seed,
        )
        save_video(pil_frames, out_path, fps=prepared.fps, quality=int(args.quality))
        peak_mem_gb = (
            torch.cuda.max_memory_allocated() / 1024**3
            if torch.cuda.is_available()
            else 0.0
        )
        record = {
            "input": str(path),
            "output": str(out_path),
            "model_type": args.model_type.upper(),
            "model_weight": str(args.model_weight),
            "frames": len(pil_frames),
            "fps": prepared.fps,
            "elapsed_sec": elapsed,
            "throughput_fps": len(pil_frames) / elapsed if elapsed > 0 else None,
            "peak_mem_gb": peak_mem_gb,
            "canvas": asdict(prepared.canvas),
        }
        manifest.append(record)
        print(
            f"[done] {path.name}: output={out_path} frames={len(pil_frames)} "
            f"time={elapsed:.2f}s fps={record['throughput_fps']:.3f} peak_mem={peak_mem_gb:.2f}GB"
        )

    manifest_path = Path(args.save_root) / args.model_type.upper() / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"[manifest] {manifest_path}")
    return manifest


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FlashVSR v1.1-style streaming inference for BSA/LSWA comparison."
    )
    parser.add_argument("--model-type", choices=["BSA", "LSWA"], default="BSA")
    parser.add_argument("--test-path", default=DEFAULT_TEST_PATH)
    parser.add_argument(
        "--model-weight",
        default=str(Path(DEFAULT_FLASHVSR_CKPT_DIR) / "diffusion_pytorch_model_streaming_dmd.safetensors"),
        help="Official diffusion safetensors or B1 .pt checkpoint. For .pt, base model is loaded first.",
    )
    parser.add_argument("--base-model-weight", default=None)
    parser.add_argument("--lq-proj-ckpt", default=str(Path(DEFAULT_FLASHVSR_CKPT_DIR) / "LQ_proj_in.ckpt"))
    parser.add_argument("--tc-decoder-ckpt", default=str(Path(DEFAULT_FLASHVSR_CKPT_DIR) / "TCDecoder.ckpt"))
    parser.add_argument("--save-root", default="log/streaming_compare")
    parser.add_argument("--flashvsr-root", default=os.environ.get("FLASHVSR_ROOT"))
    parser.add_argument("--prompt-tensor", default=None)
    parser.add_argument("--window-size", default="2,21,21")
    parser.add_argument("--pipeline", choices=["tiny", "long"], default="long")
    parser.add_argument("--input-mode", choices=["model_input", "native_lr"], default="model_input")
    parser.add_argument("--canvas-mode", choices=["pad", "center_crop", "none"], default="pad")
    parser.add_argument("--multiple", type=int, default=128)
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--tail-padding", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--sparse-ratio", type=float, default=2.0)
    parser.add_argument("--kv-ratio", type=float, default=3.0)
    parser.add_argument("--local-range", type=int, default=11)
    parser.add_argument("--quality", type=int, default=6)
    parser.add_argument("--no-color-fix", action="store_true")
    parser.add_argument("--disable-vram-management", action="store_true")
    parser.add_argument("--force-local-wan", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    args.model_type = args.model_type.upper()
    parse_window_size(args.window_size)
    run_inference(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
