from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from flashvsr_b1.models.flashvsr_components import (
    Causal_LQ4x_Proj,
    build_tc_decoder,
    load_flashvsr_tiny_checkpoint,
)
from flashvsr_b1.models.wan_dit_b1 import B1WanModel


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DIFFSYNTH_ROOT = _REPO_ROOT / "DiffSynth-Studio"

if _DIFFSYNTH_ROOT.exists() and str(_DIFFSYNTH_ROOT) not in sys.path:
    sys.path.insert(0, str(_DIFFSYNTH_ROOT))

try:
    from diffsynth.core import ModelConfig
    from diffsynth.pipelines.wan_video import WanVideoPipeline
except Exception as exc:  # pragma: no cover - exercised only when DiffSynth deps are absent.
    ModelConfig = None

    class WanVideoPipeline:  # type: ignore[no-redef]
        _import_error = exc

        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise RuntimeError("DiffSynth WanVideoPipeline is unavailable") from WanVideoPipeline._import_error


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _as_tuple(value: Any) -> tuple:
    if value is None:
        return ()
    return tuple(value)


def _build_lpips_net():
    try:
        import lpips
    except (ImportError, ModuleNotFoundError):
        return None
    return lpips.LPIPS(net="vgg").eval()


def _pretrained_kwargs_from_cfg(cfg: Any) -> dict[str, Any]:
    kwargs = dict(_cfg_get(cfg, "pretrained_kwargs", {}) or {})
    if "torch_dtype" not in kwargs and _cfg_get(cfg, "torch_dtype", None) is not None:
        kwargs["torch_dtype"] = _cfg_get(cfg, "torch_dtype")
    if "device" not in kwargs and _cfg_get(cfg, "device", None) is not None:
        kwargs["device"] = _cfg_get(cfg, "device")
    if "redirect_common_files" not in kwargs:
        kwargs["redirect_common_files"] = _cfg_get(cfg, "redirect_common_files", False)
    if "tokenizer_config" not in kwargs:
        kwargs["tokenizer_config"] = _cfg_get(cfg, "tokenizer_config", None)
    if "audio_processor_config" not in kwargs:
        kwargs["audio_processor_config"] = _cfg_get(cfg, "audio_processor_config", None)

    model_configs = _cfg_get(cfg, "model_configs", None)
    if model_configs is None:
        student_ckpt = _cfg_get(cfg, "student_ckpt", None)
        teacher_ckpt = _cfg_get(cfg, "teacher_ckpt", None)
        dit_ckpt = student_ckpt or teacher_ckpt
        if dit_ckpt is not None and ModelConfig is not None:
            model_configs = [ModelConfig(path=str(dit_ckpt), skip_download=True)]
    if model_configs is not None and "model_configs" not in kwargs:
        kwargs["model_configs"] = model_configs
    return kwargs


class B1Pipeline(WanVideoPipeline):
    """
    DiffSynth Wan video pipeline with the loaded Wan DiT replaced by B1WanModel.

    The parent loader still owns checkpoint materialization. This class only wraps the
    resulting `self.dit`, then attaches FlashVSR-side LQ projection, temporal decoder,
    and optional LPIPS modules required by B1 training.
    """

    @classmethod
    def from_b1_config(cls, cfg) -> "B1Pipeline":
        block_size = _as_tuple(_cfg_get(cfg, "block_size", (2, 8, 8)))
        teacher_block_size = _as_tuple(_cfg_get(cfg, "teacher_block_size", block_size))
        student_block_size = _as_tuple(_cfg_get(cfg, "student_block_size", block_size))
        assert teacher_block_size == student_block_size, (
            f"teacher/student block_size mismatch: {teacher_block_size} != {student_block_size}"
        )

        window_size = _as_tuple(_cfg_get(cfg, "window_size", (2, 21, 21)))
        distill_layers = _cfg_get(cfg, "distill_layers", None)
        attn_mode = _cfg_get(cfg, "attn_mode", "BSA")

        base_pipe = WanVideoPipeline.from_pretrained(**_pretrained_kwargs_from_cfg(cfg))
        pipe = cls.__new__(cls)
        pipe.__dict__.update(getattr(base_pipe, "__dict__", {}))
        pipe.teacher_ckpt = _cfg_get(cfg, "teacher_ckpt", None)
        pipe.student_ckpt = _cfg_get(cfg, "student_ckpt", None)

        if getattr(pipe, "dit", None) is None:
            raise ValueError("Parent WanVideoPipeline.from_pretrained did not populate `dit`.")
        pipe.dit = B1WanModel.from_wan_model(
            pipe.dit,
            block_size=student_block_size,
            window_size=window_size,
            distill_layers=distill_layers,
            attn_mode=attn_mode,
        )

        dim = int(_cfg_get(cfg, "dim", 1536))
        pipe.lq_proj = Causal_LQ4x_Proj(in_dim=3, out_dim=dim, layer_num=1)
        lq_proj_ckpt = _cfg_get(cfg, "lq_proj_ckpt", None)
        if lq_proj_ckpt:
            load_flashvsr_tiny_checkpoint(pipe.lq_proj, str(lq_proj_ckpt), strict=False)

        pipe.tc_decoder = build_tc_decoder(_cfg_get(cfg, "tc_decoder_ckpt", None))
        pipe.lpips_net = _build_lpips_net()
        return pipe
