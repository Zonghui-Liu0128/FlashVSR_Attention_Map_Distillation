# B1 Streaming Inference Entry

## Scope

Implemented a FlashVSR v1.1-style comparison inference entry for the 960x720 offline LQ test set.

## Files

- `flashvsr_b1/inference/streaming_compare.py`
  - Small shared helpers only: input discovery, `scale` + padding, FlashVSR frame contract, video saving, LSWA DiT replacement, and B1 `.pt` qkv checkpoint key conversion.
- `scripts/infer_bsa_baseline.py`
  - Thin FlashVSR v1.1 Tiny BSA baseline script, matching the official example style.
- `scripts/infer_lswa.py`
  - Thin LSWA script. Without `--student-ckpt`, it runs open BSA weights through the LSWA attention implementation as a direct baseline. With `--student-ckpt`, it loads the trained LSWA student.
- `scripts/40_run_b1_infer_three_cases.sh`
  - B200 launcher that serially runs BSA baseline, LSWA direct baseline, and optional LSWA student.
- `tests/test_streaming_compare_infer.py`
  - Covers helper behavior, script shape, `scale`, and checkpoint key conversion.

## B200 Usage

Official BSA v1.1 baseline:

```bash
FLASHVSR_ROOT=/path/to/OpenImagingLab/FlashVSR \
TEST_PATH=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/vsr_datasets/animal_videos/videos_960x720/lq/test \
SAVE_ROOT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/vsr_outputs/b1_compare \
RUN_LSWA_DIRECT=0 \
RUN_LSWA_STUDENT=0 \
scripts/40_run_b1_infer_three_cases.sh
```

All three cases. `SCALE=1.0` is the B1 training/inference default for the preprocessed 960x720 model-input LQ videos. Use `SCALE=4.0` only when testing native low-resolution LQ inputs with open FlashVSR-style 4x upscaling.

```bash
FLASHVSR_ROOT=/path/to/OpenImagingLab/FlashVSR \
LSWA_STUDENT_CKPT=/path/to/log/.../ckpt/latest.pt \
WINDOW_SIZE=2,21,21 \
SCALE=1.0 \
SAVE_ROOT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/vsr_outputs/b1_compare \
scripts/40_run_b1_infer_three_cases.sh
```

The baseline model is FlashVSR v1.1 Tiny (`diffusion_pytorch_model_streaming_dmd.safetensors`, `LQ_proj_in.ckpt`, `TCDecoder.ckpt`). `BASE_MODEL_WEIGHT` instantiates the open Tiny model before LSWA replacement and optional student checkpoint loading.

## Verification

```bash
pytest tests/test_streaming_compare_infer.py tests/test_scripts.py -q
python -m py_compile scripts/infer_bsa_baseline.py scripts/infer_lswa.py flashvsr_b1/inference/streaming_compare.py
bash -n scripts/40_run_b1_infer_three_cases.sh
```
