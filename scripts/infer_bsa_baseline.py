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
    create_flashvsr_inference_pipeline,
    DECODER_ARG_CHOICES,
    decoder_model_type,
    discover_inputs,
    dtype_from_name,
    finalize_flashvsr_inference_pipeline,
    normalize_decoder_name,
    prepare_lq_video,
    save_video,
    tensor_to_pil,
)


def init_pipeline(args):
    add_flashvsr_to_path(args.flashvsr_root)
    dtype = dtype_from_name(args.dtype)
    pipe = create_flashvsr_inference_pipeline(
        decoder=args.decoder,
        model_weight=args.model_weight,
        wan_vae_ckpt=args.wan_vae_ckpt,
        tc_decoder_ckpt=args.tc_decoder_ckpt,
        dtype=dtype,
        device=args.device,
    )
    return finalize_flashvsr_inference_pipeline(
        pipe,
        decoder=args.decoder,
        lq_proj_ckpt=args.lq_proj_ckpt,
        dtype=dtype,
        device=args.device,
    )


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
    frames = tensor_to_pil(video, canvas=canvas, keep_frames=keep_frames)
    out_path = build_output_path(
        save_root=args.save_root,
        input_path=path,
        model_type=decoder_model_type("BSA_baseline", args.decoder),
        seed=args.seed,
    )
    save_video(frames, out_path, fps=fps, quality=args.quality)
    return out_path


def build_parser():
    p = argparse.ArgumentParser(description="Official FlashVSR BSA baseline inference.")
    p.add_argument("--input", required=True)
    p.add_argument("--save-root", required=True)
    p.add_argument("--flashvsr-root", default=None)
    p.add_argument("--model-weight", required=True)
    p.add_argument("--lq-proj-ckpt", required=True)
    p.add_argument("--tc-decoder-ckpt", default="")
    p.add_argument("--wan-vae-ckpt", default="")
    p.add_argument("--decoder", default="tcdecoder", type=normalize_decoder_name, choices=DECODER_ARG_CHOICES)
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
    pipe = init_pipeline(args)
    for path in tqdm(discover_inputs(args.input, max_videos=args.max_videos), desc="BSA baseline"):
        print(run_one(pipe, path, args))


if __name__ == "__main__":
    main()
