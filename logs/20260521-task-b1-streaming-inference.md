# B1 Streaming Inference Entry

## Scope

Implemented a FlashVSR v1.1-style comparison inference entry for the 960x720 offline LQ test set.

## Files

- `flashvsr_b1/inference/streaming_compare.py`
  - CLI for BSA / LSWA streaming inference.
  - Supports test path discovery, model-input 960x720 padding to 128 multiples, frame padding to FlashVSR's `8n+1` contract, B1 `.pt` qkv checkpoint key conversion, and manifest output.
  - BSA uses the official FlashVSR pipeline by default.
  - LSWA or B1 `.pt` checkpoints replace the official DiT with the repo-root `wan_video_dit.py` WanModel, preserving official pipeline streaming/decode behavior.
- `scripts/40_infer_streaming_compare.sh`
  - B200 launcher with environment-variable knobs for `MODEL_TYPE`, `TEST_PATH`, `MODEL_WEIGHT`, `BASE_MODEL_WEIGHT`, `SAVE_ROOT`, and `WINDOW_SIZE`.
- `tests/test_streaming_compare_infer.py`
  - Covers CLI helper behavior and checkpoint key conversion.

## B200 Usage

Official BSA v1.1 baseline:

```bash
FLASHVSR_ROOT=/path/to/OpenImagingLab/FlashVSR \
MODEL_TYPE=BSA \
TEST_PATH=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/vsr_datasets/animal_videos/videos_960x720/lq/test \
SAVE_ROOT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/vsr_outputs/b1_compare \
scripts/40_infer_streaming_compare.sh
```

LSWA trained checkpoint:

```bash
FLASHVSR_ROOT=/path/to/OpenImagingLab/FlashVSR \
MODEL_TYPE=LSWA \
MODEL_WEIGHT=/path/to/log/.../ckpt/latest.pt \
WINDOW_SIZE=2,21,21 \
SAVE_ROOT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/vsr_outputs/b1_compare \
scripts/40_infer_streaming_compare.sh
```

If `MODEL_WEIGHT` is a B1 `.pt` checkpoint, `BASE_MODEL_WEIGHT` is used to instantiate the official FlashVSR v1.1 model before loading the student weights.

## Verification

```bash
pytest tests/test_streaming_compare_infer.py tests/test_scripts.py -q
python -m flashvsr_b1.inference.streaming_compare --dry-run --test-path <tmpdir> --max-videos 1
python -m compileall flashvsr_b1/inference
```
