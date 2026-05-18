"""FlashVSR component ports used by the B1 training path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


@dataclass(frozen=True)
class FlashVSRTinyConfig:
    patch_size: tuple[int, int, int] = (1, 2, 2)
    dim: int = 1536
    ffn_dim: int = 8960
    num_heads: int = 12
    num_layers: int = 30
    in_dim: int = 16
    out_dim: int = 16

    @classmethod
    def default(cls) -> "FlashVSRTinyConfig":
        return cls()


class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class CausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding, mode="replicate")
        return super().forward(x)


class PixelShuffle3d(nn.Module):
    def __init__(self, ff, hh, ww):
        super().__init__()
        self.ff = ff
        self.hh = hh
        self.ww = ww

    def forward(self, x):
        return rearrange(
            x,
            "b c (f ff) (h hh) (w ww) -> b (c ff hh ww) f h w",
            ff=self.ff,
            hh=self.hh,
            ww=self.ww,
        )


class Causal_LQ4x_Proj(nn.Module):
    def __init__(self, in_dim=3, out_dim=1536, layer_num=1):
        super().__init__()
        self.ff = 1
        self.hh = 16
        self.ww = 16
        self.hidden_dim1 = 2048
        self.hidden_dim2 = 3072
        self.layer_num = int(layer_num)
        self.pixel_shuffle = PixelShuffle3d(self.ff, self.hh, self.ww)
        self.conv1 = CausalConv3d(
            in_dim * self.ff * self.hh * self.ww,
            self.hidden_dim1,
            (4, 3, 3),
            stride=(2, 1, 1),
            padding=(1, 1, 1),
        )
        self.norm1 = RMS_norm(self.hidden_dim1, images=False)
        self.act1 = nn.SiLU()
        self.conv2 = CausalConv3d(self.hidden_dim1, self.hidden_dim2, (4, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1))
        self.norm2 = RMS_norm(self.hidden_dim2, images=False)
        self.act2 = nn.SiLU()
        self.linear_layers = nn.ModuleList([nn.Linear(self.hidden_dim2, out_dim) for _ in range(self.layer_num)])
        self.clear_cache()

    def clear_cache(self):
        self.cache = {"conv1": None, "conv2": None}
        self.clip_idx = 0

    def _forward_features(self, video):
        self.clear_cache()
        t = video.shape[2]
        iter_ = 1 + (t - 1) // 4
        first_frame = video[:, :, :1, :, :].repeat(1, 1, 3, 1, 1)
        video = torch.cat([first_frame, video], dim=2)
        out_x = []
        for i in range(iter_):
            x = self.pixel_shuffle(video[:, :, i * 4 : (i + 1) * 4, :, :])
            cache1_x = x[:, :, -2:, :, :].clone()
            x = self.conv1(x, self.cache["conv1"])
            self.cache["conv1"] = cache1_x
            x = self.act1(self.norm1(x))
            cache2_x = x[:, :, -2:, :, :].clone()
            if i == 0 and iter_ > 1:
                self.cache["conv2"] = cache2_x
                continue
            x = self.conv2(x, self.cache["conv2"])
            self.cache["conv2"] = cache2_x
            x = self.act2(self.norm2(x))
            out_x.append(x)
        out_x = torch.cat(out_x, dim=2)
        return rearrange(out_x, "b c f h w -> b (f h w) c")

    def forward(self, x):
        out_x = self._forward_features(x)
        projected = [layer(out_x) for layer in self.linear_layers]
        if len(projected) == 1:
            return rearrange(projected[0], "b n c -> b c n")
        return torch.stack([rearrange(y, "b n c -> b c n") for y in projected], dim=2)


class IdentityConv2d(nn.Conv2d):
    def __init__(self, channels, kernel_size=3, bias=False):
        padding = kernel_size // 2
        super().__init__(channels, channels, kernel_size, padding=padding, bias=bias)
        with torch.no_grad():
            nn.init.dirac_(self.weight)
            if self.bias is not None:
                self.bias.zero_()


def _conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(_conv(n_in * 2, n_out), nn.ReLU(inplace=True), _conv(n_out, n_out), nn.ReLU(inplace=True), _conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):
        nt, c, h, w = x.shape
        x = self.conv(x)
        return x.reshape(-1, c, h, w)


def _apply_model_with_memblocks(model, x, mem):
    assert x.ndim == 5
    n, t, c, h, w = x.shape
    out = []
    work_queue = [(xt, 0) for xt in x.reshape(n, t * c, h, w).chunk(t, dim=1)]
    while work_queue:
        xt, i = work_queue.pop(0)
        if i == len(model):
            out.append(xt)
            continue
        block = model[i]
        if isinstance(block, MemBlock):
            past = xt * 0 if mem[i] is None else mem[i]
            xt_new = block(xt, past)
            mem[i] = xt.detach()
            work_queue.insert(0, (xt_new, i + 1))
        elif isinstance(block, TGrow):
            xt = block(xt)
            _, c_, h_, w_ = xt.shape
            for xt_next in reversed(xt.view(n, block.stride * c_, h_, w_).chunk(block.stride, 1)):
                work_queue.insert(0, (xt_next, i + 1))
        else:
            work_queue.insert(0, (block(xt), i + 1))
    return torch.stack(out, 1), mem


class TAEHV(nn.Module):
    image_channels = 3

    def __init__(self, channels=(512, 256, 128, 128), latent_channels=16):
        super().__init__()
        self.latent_channels = int(latent_channels)
        n_f = list(channels)
        self.frames_to_trim = 3
        base_decoder = nn.Sequential(
            Clamp(), _conv(self.latent_channels, n_f[0]), nn.ReLU(inplace=True),
            MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]),
            nn.Upsample(scale_factor=2), TGrow(n_f[0], 1), _conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]),
            nn.Upsample(scale_factor=2), TGrow(n_f[1], 2), _conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]),
            nn.Upsample(scale_factor=2), TGrow(n_f[2], 2), _conv(n_f[2], n_f[3], bias=False),
            nn.ReLU(inplace=True), _conv(n_f[3], self.image_channels),
        )
        self.decoder = self._apply_identity_deepen(base_decoder)
        self.clean_mem()

    @staticmethod
    def _apply_identity_deepen(decoder):
        new_layers = []
        for block in decoder:
            new_layers.append(block)
            if isinstance(block, nn.ReLU):
                channels = None
                prev = new_layers[-2] if len(new_layers) >= 2 else None
                if isinstance(prev, nn.Conv2d):
                    channels = prev.out_channels
                elif isinstance(prev, MemBlock):
                    channels = prev.conv[-1].out_channels
                if channels is not None:
                    new_layers.append(IdentityConv2d(channels, kernel_size=3, bias=False))
                    new_layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*new_layers)

    def decode_video(self, x, parallel=False, show_progress_bar=False, cond=None):
        trim_flag = self.mem[-8] is None if len(self.mem) >= 8 else self.mem[-1] is None
        if cond is not None:
            if cond.shape[2] % 4 != 0:
                pad_t = 4 - cond.shape[2] % 4
                first_frame = cond[:, :, :1, :, :].repeat(1, 1, pad_t, 1, 1)
                cond = torch.cat([first_frame, cond], dim=2)
            cond = rearrange(cond, "b c (f ff) (h hh) (w ww) -> b f (c ff hh ww) h w", ff=4, hh=8, ww=8)
            if cond.shape[1] > x.shape[1]:
                cond = cond[:, -x.shape[1] :]
            x = torch.cat([cond, x], dim=2)
        x, self.mem = _apply_model_with_memblocks(self.decoder, x, self.mem)
        return x[:, self.frames_to_trim :] if trim_flag else x

    def forward(self, x, parallel=False, show_progress_bar=False, cond=None):
        return self.decode_video(x, parallel=parallel, show_progress_bar=show_progress_bar, cond=cond)

    def clean_mem(self):
        self.mem = [None] * len(self.decoder)


class _TCDecoderStub(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


def _load_state_dict_file(path: str | Path, *, map_location: str = "cpu") -> dict[str, Any]:
    path = Path(path)
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ModuleNotFoundError as exc:
            raise RuntimeError("Missing dependency `safetensors` for FlashVSR checkpoint loading.") from exc
        return load_file(str(path), device=map_location)
    state = torch.load(str(path), map_location=map_location)
    if isinstance(state, dict):
        for key in ("model", "state_dict", "module"):
            nested = state.get(key)
            if isinstance(nested, dict):
                return nested
    return state


def build_tc_decoder(checkpoint_path: str | None = None) -> nn.Module:
    if checkpoint_path is None:
        return _TCDecoderStub()
    decoder = TAEHV(channels=(512, 256, 128, 128), latent_channels=16).to(device="cpu")
    load_flashvsr_tiny_checkpoint(decoder, checkpoint_path, strict=False)
    decoder.train()
    decoder.clean_mem()
    return decoder


def load_flashvsr_tiny_checkpoint(model: nn.Module, path: str, *, strict: bool = True) -> dict:
    state = _load_state_dict_file(path)
    if not strict:
        own_state = model.state_dict()
        normalized = {}
        for key, value in state.items():
            candidates = [key]
            for prefix in ("module.", "model.", "student.", "teacher.", "dit.", "denoising_model."):
                if key.startswith(prefix):
                    candidates.append(key[len(prefix):])
            for candidate in candidates:
                if candidate in own_state and tuple(own_state[candidate].shape) == tuple(value.shape):
                    normalized[candidate] = value
                    break
        state = normalized
    load_result = model.load_state_dict(state, strict=strict)
    return {
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }
