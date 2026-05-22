#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashvsr_b1.inference.streaming_compare import (
    add_flashvsr_to_path,
    build_output_path,
    discover_inputs,
    dtype_from_name,
    normalize_dit_state_dict,
    parse_window_size,
    prepare_lq_video,
    replace_dit_with_lswa,
    save_video,
    tensor_to_pil,
)


def init_pipeline(args):
    add_flashvsr_to_path(args.flashvsr_root)
    from diffsynth import FlashVSRTinyPipeline, ModelManager
    from utils.TCDecoder import build_tcdecoder
    from utils.utils import Causal_LQ4x_Proj

    dtype = dtype_from_name(args.dtype)
    mm = ModelManager(torch_dtype=dtype, device="cpu")
    mm.load_models([args.base_model_weight])
    pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=args.device)
    replace_dit_with_lswa(
        pipe,
        window_size=parse_window_size(args.window_size),
        student_ckpt=args.student_ckpt,
    )
    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(args.device, dtype=dtype)
    pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(args.lq_proj_ckpt, map_location="cpu"), strict=True)
    pipe.TCDecoder = build_tcdecoder(new_channels=[512, 256, 128, 128], new_latent_channels=16 + 768)
    pipe.TCDecoder.load_state_dict(torch.load(args.tc_decoder_ckpt, map_location="cpu"), strict=False)
    pipe.to(args.device)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv()
    pipe.load_models_to_device(["dit", "vae"])
    return pipe


def run_one(pipe, path: Path, args) -> Path:
    dtype = dtype_from_name(args.dtype)
    LQ, th, tw, F, fps, canvas, keep_frames = prepare_lq_video(
        path,
        dtype=dtype,
        device=args.device,
        multiple=args.multiple,
        max_frames=args.max_frames,
        scale=args.scale,
    )
    video = pipe(
        prompt="",
        negative_prompt="",
        cfg_scale=1.0,
        num_inference_steps=1,
        seed=args.seed,
        LQ_video=LQ,
        num_frames=F,
        height=th,
        width=tw,
        is_full_block=False,
        if_buffer=True,
        topk_ratio=args.sparse_ratio * 768 * 1280 / (th * tw),
        kv_ratio=args.kv_ratio,
        local_range=args.local_range,
        color_fix=not args.no_color_fix,
    )
    label = "LSWA_student" if args.student_ckpt else "LSWA_direct"
    frames = tensor_to_pil(video, canvas=canvas, keep_frames=keep_frames)
    out_path = build_output_path(save_root=args.save_root, input_path=path, model_type=label, seed=args.seed)
    save_video(frames, out_path, fps=fps, quality=args.quality)
    return out_path


def build_parser():
    p = argparse.ArgumentParser(description="FlashVSR LSWA inference, with optional trained student checkpoint.")
    p.add_argument("--input", required=True)
    p.add_argument("--save-root", required=True)
    p.add_argument("--flashvsr-root", default=None)
    p.add_argument("--base-model-weight", required=True)
    p.add_argument("--student-ckpt", default="")
    p.add_argument("--lq-proj-ckpt", required=True)
    p.add_argument("--tc-decoder-ckpt", required=True)
    p.add_argument("--window-size", default="2,21,21")
    p.add_argument("--max-videos", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--multiple", type=int, default=128)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--sparse-ratio", type=float, default=2.0)
    p.add_argument("--kv-ratio", type=float, default=3.0)
    p.add_argument("--local-range", type=int, default=11)
    p.add_argument("--quality", type=int, default=6)
    p.add_argument("--no-color-fix", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    args.student_ckpt = args.student_ckpt or None
    _ = normalize_dit_state_dict  # keep the checkpoint conversion dependency explicit in this script.
    pipe = init_pipeline(args)
    for path in tqdm(discover_inputs(args.input, max_videos=args.max_videos), desc="LSWA"):
        print(run_one(pipe, path, args))


if __name__ == "__main__":
    main()
