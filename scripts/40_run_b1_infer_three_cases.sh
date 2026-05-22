#!/bin/bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/FlashVSR_Attention_Map_Distillation}"
cd "$PROJECT_ROOT"

TEST_PATH=${TEST_PATH:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/vsr_datasets/animal_videos/videos_960x720/lq/test} # 测试集地址
FLASHVSR_ROOT=${FLASHVSR_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/FlashVSR}
FLASHVSR_CKPT_DIR=${FLASHVSR_CKPT_DIR:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/FlashVSR-v1.1}
BASE_MODEL_WEIGHT=${BASE_MODEL_WEIGHT:-${FLASHVSR_CKPT_DIR}/diffusion_pytorch_model_streaming_dmd.safetensors}
LQ_PROJ_CKPT=${LQ_PROJ_CKPT:-${FLASHVSR_CKPT_DIR}/LQ_proj_in.ckpt}
TC_DECODER_CKPT=${TC_DECODER_CKPT:-${FLASHVSR_CKPT_DIR}/TCDecoder.ckpt}
LSWA_STUDENT_CKPT=${LSWA_STUDENT_CKPT:-}
SAVE_ROOT=${SAVE_ROOT:-outputs/baseline_flashvsr_v1.1_tiny/animal_test}    # predicated hq保存地址
WINDOW_SIZE=${WINDOW_SIZE:-2,21,21}
MAX_VIDEOS=${MAX_VIDEOS:-0}
MAX_FRAMES=${MAX_FRAMES:-0}
SEED=${SEED:-0}
DTYPE=${DTYPE:-bf16}
SCALE=${SCALE:-1.0} # 4x超分 or 单倍超分

# 推理模式
RUN_BSA_BASELINE=${RUN_BSA_BASELINE:-1}
RUN_LSWA_DIRECT=${RUN_LSWA_DIRECT:-0}
RUN_LSWA_STUDENT=${RUN_LSWA_STUDENT:-0}

COMMON_ARGS=(
  --input "$TEST_PATH"
  --save-root "$SAVE_ROOT"
  --flashvsr-root "$FLASHVSR_ROOT"
  --lq-proj-ckpt "$LQ_PROJ_CKPT"
  --tc-decoder-ckpt "$TC_DECODER_CKPT"
  --max-videos "$MAX_VIDEOS"
  --max-frames "$MAX_FRAMES"
  --seed "$SEED"
  --dtype "$DTYPE"
  --scale "$SCALE"
)

if [[ "$RUN_BSA_BASELINE" == "1" ]]; then
  python scripts/infer_bsa_baseline.py \
    "${COMMON_ARGS[@]}" \
    --model-weight "$BASE_MODEL_WEIGHT"
fi

if [[ "$RUN_LSWA_DIRECT" == "1" ]]; then
  python scripts/infer_lswa.py \
    "${COMMON_ARGS[@]}" \
    --base-model-weight "$BASE_MODEL_WEIGHT" \
    --window-size "$WINDOW_SIZE"
fi

if [[ "$RUN_LSWA_STUDENT" == "1" ]]; then
  if [[ -z "$LSWA_STUDENT_CKPT" ]]; then
    echo "LSWA_STUDENT_CKPT is empty; skip LSWA student inference."
  else
    python scripts/infer_lswa.py \
      "${COMMON_ARGS[@]}" \
      --base-model-weight "$BASE_MODEL_WEIGHT" \
      --student-ckpt "$LSWA_STUDENT_CKPT" \
      --window-size "$WINDOW_SIZE"
  fi
fi
