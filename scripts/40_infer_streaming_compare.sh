#!/bin/bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/FlashVSR_Attention_Map_Distillation}"
cd "$PROJECT_ROOT"

MODEL_TYPE=${MODEL_TYPE:-BSA}
TEST_PATH=${TEST_PATH:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/vsr_datasets/animal_videos/videos_960x720/lq/test}
FLASHVSR_CKPT_DIR=${FLASHVSR_CKPT_DIR:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/FlashVSR-v1.1}
MODEL_WEIGHT=${MODEL_WEIGHT:-${FLASHVSR_CKPT_DIR}/diffusion_pytorch_model_streaming_dmd.safetensors}
BASE_MODEL_WEIGHT=${BASE_MODEL_WEIGHT:-${FLASHVSR_CKPT_DIR}/diffusion_pytorch_model_streaming_dmd.safetensors}
LQ_PROJ_CKPT=${LQ_PROJ_CKPT:-${FLASHVSR_CKPT_DIR}/LQ_proj_in.ckpt}
TC_DECODER_CKPT=${TC_DECODER_CKPT:-${FLASHVSR_CKPT_DIR}/TCDecoder.ckpt}
SAVE_ROOT=${SAVE_ROOT:-log/streaming_compare}
WINDOW_SIZE=${WINDOW_SIZE:-2,21,21}
PIPELINE=${PIPELINE:-long}
INPUT_MODE=${INPUT_MODE:-model_input}
CANVAS_MODE=${CANVAS_MODE:-pad}
MULTIPLE=${MULTIPLE:-128}
MAX_VIDEOS=${MAX_VIDEOS:-0}
MAX_FRAMES=${MAX_FRAMES:-0}
SEED=${SEED:-0}
DTYPE=${DTYPE:-bf16}
SPARSE_RATIO=${SPARSE_RATIO:-2.0}
KV_RATIO=${KV_RATIO:-3.0}
LOCAL_RANGE=${LOCAL_RANGE:-11}
QUALITY=${QUALITY:-6}

python -m flashvsr_b1.inference.streaming_compare \
  --model-type "$MODEL_TYPE" \
  --test-path "$TEST_PATH" \
  --model-weight "$MODEL_WEIGHT" \
  --base-model-weight "$BASE_MODEL_WEIGHT" \
  --lq-proj-ckpt "$LQ_PROJ_CKPT" \
  --tc-decoder-ckpt "$TC_DECODER_CKPT" \
  --save-root "$SAVE_ROOT" \
  --window-size "$WINDOW_SIZE" \
  --pipeline "$PIPELINE" \
  --input-mode "$INPUT_MODE" \
  --canvas-mode "$CANVAS_MODE" \
  --multiple "$MULTIPLE" \
  --max-videos "$MAX_VIDEOS" \
  --max-frames "$MAX_FRAMES" \
  --seed "$SEED" \
  --dtype "$DTYPE" \
  --sparse-ratio "$SPARSE_RATIO" \
  --kv-ratio "$KV_RATIO" \
  --local-range "$LOCAL_RANGE" \
  --quality "$QUALITY" \
  "$@"
