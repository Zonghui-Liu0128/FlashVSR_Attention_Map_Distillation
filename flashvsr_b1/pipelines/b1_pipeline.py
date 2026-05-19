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


def _require_bcthw_rgb(video: torch.Tensor, name: str) -> torch.Tensor:
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError(
            f"{name} must be BCTHW RGB video matching FlashVSR input contract; "
            f"got shape={tuple(video.shape)}"
        )
    return video.contiguous()


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
        """Build one-step B1 training tensors per upstream FlashVSR contract.

        Returns:
            LR_latents (list[Tensor]): per-block residual condition; one element
                of shape (B, N_tok, cfg.dim=1536) token-last for layer_num=1.
                Added inside DiT block loop, see wan_video_dit.py:862-864.
            z_t (Tensor): noisy VAE-latent input of shape
                (B, dit.in_dim=16, T_lat, H_lat, W_lat), pre-patchify. Patched
                inside B1WanModel.forward.
            t_star (Tensor): single-step diffusion timestep scalar.
            hr_rgb (Tensor): ground-truth HR pixels for L_lpips.
        """
        device = self.dit_device()
        lr_rgb = _require_bcthw_rgb(batch["lr"].to(device), "batch['lr']")
        hr_rgb = _require_bcthw_rgb(batch["hr"].to(device), "batch['hr']")
        self.lq_proj.to(device)

        token_grid = batch.get("latent_shape", None)
        if token_grid is None:
            raise ValueError(
                "prepare_batch requires batch['latent_shape'] (post-patch token grid)"
            )
        if isinstance(token_grid, torch.Tensor):
            token_grid = token_grid.tolist()
        if isinstance(token_grid, (list, tuple)) and token_grid and isinstance(token_grid[0], (list, tuple)):
            token_grid = token_grid[0]
        token_grid = tuple(int(v) for v in token_grid)

        # Step 1: project LR pixels to DiT-inner-dim tokens.
        # Causal_LQ4x_Proj returns (B, 1536, N) for layer_num=1 or
        # (B, 1536, layer_num, N) for layer_num>1. Normalize to upstream list contract.
        # FlashVSR v1.1 feeds LQ_proj_in a four-frame tail buffer. Those frames
        # are not training targets; they make projector cache semantics align
        # with Wan VAE/DiT's 85 effective frames -> 22 latent-frame contract.
        lr_rgb_for_lq_proj = torch.cat(
            [lr_rgb, lr_rgb[:, :, -1:, :, :].repeat(1, 1, 4, 1, 1)],
            dim=2,
        ).contiguous()
        lr_tokens = self.lq_proj(lr_rgb_for_lq_proj)
        if lr_tokens.ndim == 3:
            # layer_num=1: (B, 1536, N) -> list[(B, N, 1536)]
            LR_latents = [lr_tokens.transpose(1, 2).contiguous()]
        elif lr_tokens.ndim == 4:
            # layer_num>1: (B, 1536, layer_num, N) -> list[(B, N, 1536)]
            LR_latents = [
                lr_tokens[:, :, i, :].transpose(1, 2).contiguous()
                for i in range(lr_tokens.shape[2])
            ]
        else:
            raise ValueError(
                f"lq_proj returned unexpected ndim {lr_tokens.ndim} (shape "
                f"{tuple(lr_tokens.shape)}); expected 3D or 4D - see "
                f"Causal_LQ4x_Proj.forward in flashvsr_components.py:135-140"
            )
        expected_tokens = token_grid[0] * token_grid[1] * token_grid[2]
        for i, lr_tok in enumerate(LR_latents):
            if lr_tok.shape[1] != expected_tokens:
                raise ValueError(
                    f"LR_latents[{i}] token count {lr_tok.shape[1]} != DiT token "
                    f"count {expected_tokens}; lr_rgb={tuple(lr_rgb.shape)}, "
                    f"lr_rgb_for_lq_proj={tuple(lr_rgb_for_lq_proj.shape)}, "
                    f"token_grid={token_grid}"
                )

        # Step 2: build noisy 16-channel VAE-latent z_t at the pre-patch shape.
        # batch["latent_shape"] is the POST-patch token grid (e.g. (22,64,120)
        # for landscape) per dataset_b1.py:32 + Fix B clarification. Multiply by
        # dit.patch_size to recover the pre-patch latent shape.
        patch_size = tuple(int(p) for p in getattr(self.dit, "patch_size", (1, 2, 2)))
        if len(patch_size) != 3 or len(token_grid) != 3:
            raise ValueError(
                f"patch_size and token_grid must both be 3-tuples; got "
                f"patch_size={patch_size}, token_grid={token_grid}"
            )
        pre_patch_latent_shape = tuple(g * p for g, p in zip(token_grid, patch_size))

        in_dim = int(getattr(self.dit, "in_dim", 16))
        B = lr_rgb.shape[0]
        dit_dtype = next((p.dtype for p in self.dit.parameters() if p is not None), lr_rgb.dtype)
        z_t = torch.randn(
            (B, in_dim, *pre_patch_latent_shape),
            device=device,
            dtype=dit_dtype,
        )

        t_star = torch.tensor(self.cfg_single_step_t, device=device)
        return LR_latents, z_t, t_star, hr_rgb
