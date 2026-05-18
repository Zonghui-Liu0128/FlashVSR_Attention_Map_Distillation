from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import torch

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

    class WanVideoPipeline(torch.nn.Module):  # type: ignore[no-redef]
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


def _iter_modules(model):
    if hasattr(model, "modules") and "_modules" in getattr(model, "__dict__", {}):
        yield from model.modules()
        return
    for block in getattr(model, "blocks", []):
        attn = getattr(block, "self_attn", None)
        if attn is not None:
            yield attn


def _iter_parameters(model):
    if hasattr(model, "parameters") and "_parameters" in getattr(model, "__dict__", {}):
        yield from model.parameters()


def _module_device(model) -> torch.device:
    for p in _iter_parameters(model):
        return p.device
    return torch.device("cpu")


def _build_lpips_net():
    try:
        import lpips
    except (ImportError, ModuleNotFoundError):
        return None
    net = lpips.LPIPS(net="vgg").eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


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
        torch.nn.Module.__init__(pipe)
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
        pipe.student = pipe.dit

        teacher_ckpt = _cfg_get(cfg, "teacher_ckpt", None)
        student_ckpt = _cfg_get(cfg, "student_ckpt", None)
        if teacher_ckpt and teacher_ckpt != student_ckpt:
            teacher_kwargs = dict(_pretrained_kwargs_from_cfg(cfg))
            if ModelConfig is not None:
                teacher_kwargs["model_configs"] = [
                    ModelConfig(path=str(teacher_ckpt), skip_download=True)
                ]
            teacher_pipe = WanVideoPipeline.from_pretrained(**teacher_kwargs)
            teacher_source = getattr(teacher_pipe, "dit", None)
            if teacher_source is None:
                raise ValueError("Teacher WanVideoPipeline.from_pretrained did not populate `dit`.")
            if teacher_source is pipe.dit:
                teacher_source = copy.deepcopy(pipe.dit)
            teacher_dit = B1WanModel.from_wan_model(
                teacher_source,
                block_size=student_block_size,
                window_size=window_size,
                distill_layers=distill_layers,
                attn_mode="BSA",
            )
        else:
            teacher_dit = copy.deepcopy(pipe.dit)
            teacher_dit._init_distill_layers_for_test()
            for layer_idx, block in enumerate(teacher_dit.blocks):
                block.self_attn.distill_export = layer_idx in teacher_dit.distill_layers

        for module in _iter_modules(teacher_dit):
            if hasattr(module, "current_sparsity"):
                module.current_sparsity = 0.85
            if hasattr(module, "attn_mode"):
                module.attn_mode = "BSA"
        if hasattr(teacher_dit, "eval") and "_modules" in getattr(teacher_dit, "__dict__", {}):
            teacher_dit.eval()
        for p in _iter_parameters(teacher_dit):
            p.requires_grad_(False)
        pipe.teacher = teacher_dit
        pipe.teacher_dit = teacher_dit

        dim = int(_cfg_get(cfg, "dim", 1536))
        pipe.lq_proj = Causal_LQ4x_Proj(in_dim=3, out_dim=dim, layer_num=1)
        lq_proj_ckpt = _cfg_get(cfg, "lq_proj_ckpt", None)
        if lq_proj_ckpt:
            load_flashvsr_tiny_checkpoint(pipe.lq_proj, str(lq_proj_ckpt), strict=False)

        pipe.tc_decoder = build_tc_decoder(_cfg_get(cfg, "tc_decoder_ckpt", None))
        pipe.lpips_net = _build_lpips_net()
        pipe.cfg_single_step_t = int(_cfg_get(cfg, "single_step_t", 999))
        device = _module_device(pipe.dit)
        pipe.lq_proj.to(device)
        if hasattr(pipe.tc_decoder, "to"):
            pipe.tc_decoder.to(device)
        if pipe.lpips_net is not None and hasattr(pipe.lpips_net, "to"):
            pipe.lpips_net.to(device)
        return pipe

    def dit_device(self):
        return _module_device(self.dit)

    def prepare_batch(self, batch):
        """Take a dataset batch and build the one-step B1 training tensors."""
        device = self.dit_device()
        lr_rgb = batch["lr"].to(device)
        hr_rgb = batch["hr"].to(device)
        self.lq_proj.to(device)
        lr_latent = self.lq_proj(lr_rgb)
        if lr_latent.ndim == 3:
            latent_shape = batch.get("latent_shape", None)
            if latent_shape is None:
                raise ValueError("batch must include latent_shape when lq_proj returns flattened output")
            if isinstance(latent_shape, torch.Tensor):
                latent_shape = latent_shape.tolist()
            if isinstance(latent_shape, (list, tuple)) and latent_shape and isinstance(latent_shape[0], (list, tuple)):
                latent_shape = latent_shape[0]
            t_lat, h_lat, w_lat = (int(v) for v in latent_shape)
            b, c, n = lr_latent.shape
            if n != t_lat * h_lat * w_lat:
                raise ValueError(
                    f"lq_proj token count {n} does not match latent_shape {(t_lat, h_lat, w_lat)}"
                )
            lr_latent = lr_latent.view(b, c, t_lat, h_lat, w_lat)
        z_t = torch.randn_like(lr_latent)
        t_star = torch.tensor(self.cfg_single_step_t, device=lr_latent.device)
        return lr_latent, z_t, t_star, hr_rgb
