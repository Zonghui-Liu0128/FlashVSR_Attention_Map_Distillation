from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from flashvsr_b1.attn.bsa_kernel import bsa_forward
from flashvsr_b1.attn.lswa import lswa_forward
from flashvsr_b1.attn.shadow_block_pool_attn import shadow_block_pool_attn


_DEFAULT_DISTILL_LAYERS = {4, 9, 14, 19, 24, 29}
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DIFFSYNTH_ROOT = _REPO_ROOT / "DiffSynth-Studio"


def _load_wan_video_dit():
    try:
        if _DIFFSYNTH_ROOT.exists():
            sys.path.insert(0, str(_DIFFSYNTH_ROOT))
        from diffsynth.models import wan_video_dit

        return wan_video_dit
    except ModuleNotFoundError:
        import importlib.util

        package = sys.modules.setdefault("diffsynth", types.ModuleType("diffsynth"))
        models = sys.modules.setdefault("diffsynth.models", types.ModuleType("diffsynth.models"))
        core = sys.modules.setdefault("diffsynth.core", types.ModuleType("diffsynth.core"))
        gradient = sys.modules.setdefault(
            "diffsynth.core.gradient", types.ModuleType("diffsynth.core.gradient")
        )
        camera = sys.modules.setdefault(
            "diffsynth.models.wan_video_camera_controller",
            types.ModuleType("diffsynth.models.wan_video_camera_controller"),
        )
        wantodance = sys.modules.setdefault(
            "diffsynth.models.wantodance",
            types.ModuleType("diffsynth.models.wantodance"),
        )
        package.models = models
        package.core = core
        core.gradient = gradient

        if not hasattr(gradient, "gradient_checkpoint_forward"):
            gradient.gradient_checkpoint_forward = (
                lambda module, _checkpointing, _offload, *args: module(*args)
            )
        if not hasattr(camera, "SimpleAdapter"):
            camera.SimpleAdapter = _UnavailableAdapter
        if not hasattr(wantodance, "WanToDanceRotaryEmbedding"):
            wantodance.WanToDanceRotaryEmbedding = _UnavailableAdapter
        if not hasattr(wantodance, "WanToDanceMusicEncoderLayer"):
            wantodance.WanToDanceMusicEncoderLayer = _UnavailableAdapter

        path = _DIFFSYNTH_ROOT / "diffsynth" / "models" / "wan_video_dit.py"
        spec = importlib.util.spec_from_file_location("diffsynth.models.wan_video_dit", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load DiffSynth Wan model from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["diffsynth.models.wan_video_dit"] = module
        spec.loader.exec_module(module)
        models.wan_video_dit = module
        return module


class _UnavailableAdapter(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()
        raise RuntimeError("Optional DiffSynth dependency is unavailable in this environment")


wan_video_dit = _load_wan_video_dit()


class SelfAttentionB1(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        *,
        block_size=(2, 8, 8),
        window_size=(2, 21, 21),
        distill_export=False,
    ):
        super().__init__()
        self.attn_mode = "BSA"
        self.current_sparsity = 0.85
        self.block_size = block_size
        self.window_size = window_size
        self.distill_export = distill_export
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        self.qkv_proj = nn.Linear(dim, 3 * dim)
        self.o_proj = nn.Linear(dim, dim)
        self.norm_q = wan_video_dit.RMSNorm(dim, eps=1e-6)
        self.norm_k = wan_video_dit.RMSNorm(dim, eps=1e-6)

        self.register_load_state_dict_pre_hook(self._upgrade_legacy_state_dict)

    def _upgrade_legacy_state_dict(
        self,
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        q_w = state_dict.pop(prefix + "q.weight", None)
        k_w = state_dict.pop(prefix + "k.weight", None)
        v_w = state_dict.pop(prefix + "v.weight", None)
        if q_w is not None and k_w is not None and v_w is not None:
            state_dict.setdefault(prefix + "qkv_proj.weight", torch.cat([q_w, k_w, v_w], dim=0))

        q_b = state_dict.pop(prefix + "q.bias", None)
        k_b = state_dict.pop(prefix + "k.bias", None)
        v_b = state_dict.pop(prefix + "v.bias", None)
        if q_b is not None and k_b is not None and v_b is not None:
            state_dict.setdefault(prefix + "qkv_proj.bias", torch.cat([q_b, k_b, v_b], dim=0))

        o_w = state_dict.pop(prefix + "o.weight", None)
        o_b = state_dict.pop(prefix + "o.bias", None)
        if o_w is not None:
            state_dict.setdefault(prefix + "o_proj.weight", o_w)
        if o_b is not None:
            state_dict.setdefault(prefix + "o_proj.bias", o_b)

    def copy_from_wan_self_attention(self, source: nn.Module) -> None:
        if all(hasattr(source, name) for name in ("q", "k", "v")):
            with torch.no_grad():
                self.qkv_proj.weight.copy_(
                    torch.cat([source.q.weight, source.k.weight, source.v.weight], dim=0)
                )
                self.qkv_proj.bias.copy_(
                    torch.cat([source.q.bias, source.k.bias, source.v.bias], dim=0)
                )
        elif hasattr(source, "qkv_proj"):
            self.qkv_proj.load_state_dict(source.qkv_proj.state_dict())

        if hasattr(source, "o"):
            self.o_proj.load_state_dict(source.o.state_dict())
        elif hasattr(source, "o_proj"):
            self.o_proj.load_state_dict(source.o_proj.state_dict())

        if hasattr(source, "norm_q"):
            self.norm_q.load_state_dict(source.norm_q.state_dict())
        if hasattr(source, "norm_k"):
            self.norm_k.load_state_dict(source.norm_k.state_dict())

    def _project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        return self.norm_q(q), self.norm_k(k), v

    def _as_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        return x.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

    def forward(self, x, freqs, *, return_aux=False, f=None, h=None, w=None, **kwargs):
        q, k, v = self._project_qkv(x)
        if freqs is not None:
            q = wan_video_dit.rope_apply(q, freqs, self.num_heads)
            k = wan_video_dit.rope_apply(k, freqs, self.num_heads)

        if self.attn_mode == "BSA":
            attn_out = bsa_forward(
                q,
                k,
                v,
                block_size=self.block_size,
                grid_shape=(f, h, w),
                current_sparsity=self.current_sparsity,
                num_heads=self.num_heads,
                local_window_mask=None,
            )
        else:
            attn_out = lswa_forward(
                q,
                k,
                v,
                window_size=self.window_size,
                num_heads=self.num_heads,
                f=f,
                h=h,
                w=w,
                is_stream=False,
            )

        out = self.o_proj(attn_out)
        if return_aux and self.distill_export:
            aux = {"h_out": out}
            if self.attn_mode == "BSA":
                aux["A_blk"] = shadow_block_pool_attn(
                    self._as_heads(q),
                    self._as_heads(k),
                    block_size=self.block_size,
                    grid_shape=(f, h, w),
                    causal=True,
                )
            return out, aux
        return out


class B1WanModel(wan_video_dit.WanModel):
    def __init__(self, *args, distill_layers=None, **kwargs):
        self.distill_layers = set(_DEFAULT_DISTILL_LAYERS if distill_layers is None else distill_layers)
        super().__init__(*args, **kwargs)
        self._replace_self_attention_modules()

    @classmethod
    def from_wan_model(
        cls,
        wan_model,
        *,
        block_size=(2, 8, 8),
        window_size=(2, 21, 21),
        distill_layers=None,
        attn_mode="BSA",
    ):
        b1_model = cls.__new__(cls)
        torch.nn.Module.__init__(b1_model)
        b1_model.__dict__.update(getattr(wan_model, "__dict__", {}))
        b1_model.distill_layers = set(
            _DEFAULT_DISTILL_LAYERS if distill_layers is None else distill_layers
        )
        b1_model._replace_self_attention_modules(
            block_size=block_size,
            window_size=window_size,
            attn_mode=attn_mode,
        )
        return b1_model

    def _init_distill_layers_for_test(self):
        self.distill_layers = set(_DEFAULT_DISTILL_LAYERS)

    def b1_forward(self, LR_latent, z_t, t_star, return_aux: bool = False):
        """B1 one-step forward with LR latent conditioning and fixed timestep."""
        B = z_t.shape[0]
        device = z_t.device
        x = z_t + LR_latent
        if not torch.is_tensor(t_star):
            t_star = torch.tensor(t_star, device=device)
        if t_star.ndim == 0:
            timestep = t_star.expand(B).to(device)
        else:
            timestep = t_star.to(device)
        text_ctx_dim = getattr(self, "text_dim", 4096)
        context = torch.zeros(B, 1, text_ctx_dim, device=device, dtype=x.dtype)
        return self.forward(
            x,
            timestep,
            context,
            return_aux=return_aux,
        )

    def _replace_self_attention_modules(
        self,
        *,
        block_size=(2, 8, 8),
        window_size=(2, 21, 21),
        attn_mode="BSA",
    ) -> None:
        for layer_idx, block in enumerate(self.blocks):
            old_attn = block.self_attn
            new_attn = SelfAttentionB1(
                old_attn.dim,
                old_attn.num_heads,
                block_size=block_size,
                window_size=window_size,
                distill_export=layer_idx in self.distill_layers,
            )
            new_attn.attn_mode = attn_mode
            new_attn.copy_from_wan_self_attention(old_attn)
            block.self_attn = new_attn

    def _forward_block_b1(self, block, x, context, t_mod, freqs, *, f, h, w, return_aux):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )

        input_x = wan_video_dit.modulate(block.norm1(x), shift_msa, scale_msa)
        attn_result = block.self_attn(
            input_x,
            freqs,
            return_aux=return_aux,
            f=f,
            h=h,
            w=w,
        )
        if isinstance(attn_result, tuple):
            attn_out, aux = attn_result
        else:
            attn_out, aux = attn_result, None
        x = block.gate(x, gate_msa, attn_out)
        x = x + block.cross_attn(block.norm3(x), context)
        input_x = wan_video_dit.modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(input_x))
        return x, aux

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        return_aux: bool = False,
        **kwargs,
    ):
        del use_gradient_checkpointing, use_gradient_checkpointing_offload, kwargs
        t = self.time_embedding(
            wan_video_dit.sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype)
        )
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if self.has_image_input:
            x = torch.cat([x, y], dim=1)
            clip_embedding = self.img_emb(clip_feature)
            context = torch.cat([clip_embedding, context], dim=1)

        x, (f, h, w) = self.patchify(x)
        freqs = torch.cat(
            [
                self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1).to(x.device)

        layer_aux: dict[str, dict[int, torch.Tensor]] = {}
        for layer_idx, block in enumerate(self.blocks):
            x, aux = self._forward_block_b1(
                block,
                x,
                context,
                t_mod,
                freqs,
                f=f,
                h=h,
                w=w,
                return_aux=return_aux,
            )
            if aux is None:
                continue
            # Spec (task_b1.md line 415): aux["h_out"][layer_idx], aux["A_blk"][layer_idx].
            # Outer key is the metric name, inner key is the distill layer index.
            for key, value in aux.items():
                layer_aux.setdefault(key, {})[layer_idx] = value

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        if return_aux:
            return x, layer_aux
        return x
