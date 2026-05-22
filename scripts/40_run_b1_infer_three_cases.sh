#!/bin/bash
set -euo pipefail

echo "B200 FlashVSR v1.1 Tiny / B1 inference"
echo "Run order: BSA baseline -> LSWA direct baseline -> LSWA trained student"

# =========================
# B200 paths. Edit here only.
# =========================
export PROJECT_ROOT="${PROJECT_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/FlashVSR_Attention_Map_Distillation}"
export FLASHVSR_ROOT="${FLASHVSR_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/FlashVSR}"
export TEST_PATH="${TEST_PATH:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/vsr_datasets/animal_videos/videos_960x720/lq/test}"
export FLASHVSR_CKPT_DIR="${FLASHVSR_CKPT_DIR:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/FlashVSR-v1.1}"

# FlashVSR v1.1 Tiny baseline weights.
export BASE_MODEL_WEIGHT="${BASE_MODEL_WEIGHT:-${FLASHVSR_CKPT_DIR}/diffusion_pytorch_model_streaming_dmd.safetensors}"
export LQ_PROJ_CKPT="${LQ_PROJ_CKPT:-${FLASHVSR_CKPT_DIR}/LQ_proj_in.ckpt}"
export TC_DECODER_CKPT="${TC_DECODER_CKPT:-${FLASHVSR_CKPT_DIR}/TCDecoder.ckpt}"

# Your trained LSWA student checkpoint. Leave empty to skip Case 3.
export LSWA_STUDENT_CKPT="${LSWA_STUDENT_CKPT:-}"

# Outputs and runtime knobs.
export SAVE_ROOT="${SAVE_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/vsr_outputs/b1_infer_three_cases}"
export WINDOW_SIZE="${WINDOW_SIZE:-2,21,21}"
export SCALE="${SCALE:-1.0}"          # B1 1x inference default. Use 4.0 only for native small-LQ 4x FlashVSR tests.
export MAX_VIDEOS="${MAX_VIDEOS:-0}"  # 0 means all videos.
export MAX_FRAMES="${MAX_FRAMES:-0}"  # 0 means full video; test set supports up to 93 frames.
export SEED="${SEED:-0}"
export DTYPE="${DTYPE:-bf16}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Per-case switches.
export RUN_BSA_BASELINE="${RUN_BSA_BASELINE:-1}"
export RUN_LSWA_DIRECT="${RUN_LSWA_DIRECT:-1}"
export RUN_LSWA_STUDENT="${RUN_LSWA_STUDENT:-1}"

check_dir() {
  if [[ ! -d "$1" ]]; then
    echo "[ERROR] Missing directory: $1" >&2
    exit 1
  fi
}

check_file() {
  if [[ ! -f "$1" ]]; then
    echo "[ERROR] Missing file: $1" >&2
    exit 1
  fi
}

check_dir "$PROJECT_ROOT"
check_dir "$FLASHVSR_ROOT"
check_dir "$TEST_PATH"
check_file "$BASE_MODEL_WEIGHT"
check_file "$LQ_PROJ_CKPT"
check_file "$TC_DECODER_CKPT"

mkdir -p "$SAVE_ROOT"
cd "$PROJECT_ROOT"

echo "[CONFIG] PROJECT_ROOT=$PROJECT_ROOT"
echo "[CONFIG] FLASHVSR_ROOT=$FLASHVSR_ROOT"
echo "[CONFIG] TEST_PATH=$TEST_PATH"
echo "[CONFIG] BASE_MODEL_WEIGHT=$BASE_MODEL_WEIGHT"
echo "[CONFIG] LQ_PROJ_CKPT=$LQ_PROJ_CKPT"
echo "[CONFIG] TC_DECODER_CKPT=$TC_DECODER_CKPT"
echo "[CONFIG] LSWA_STUDENT_CKPT=${LSWA_STUDENT_CKPT:-<empty>}"
echo "[CONFIG] SAVE_ROOT=$SAVE_ROOT"
echo "[CONFIG] SCALE=$SCALE WINDOW_SIZE=$WINDOW_SIZE MAX_VIDEOS=$MAX_VIDEOS MAX_FRAMES=$MAX_FRAMES"

if [[ "$RUN_BSA_BASELINE" == "1" ]]; then
  echo "========== Case 1/3: BSA baseline (official FlashVSR v1.1 Tiny) =========="
  python scripts/infer_bsa_baseline.py \
    --input "$TEST_PATH" \
    --save-root "$SAVE_ROOT" \
    --flashvsr-root "$FLASHVSR_ROOT" \
    --model-weight "$BASE_MODEL_WEIGHT" \
    --lq-proj-ckpt "$LQ_PROJ_CKPT" \
    --tc-decoder-ckpt "$TC_DECODER_CKPT" \
    --scale "$SCALE" \
    --max-videos "$MAX_VIDEOS" \
    --max-frames "$MAX_FRAMES" \
    --seed "$SEED" \
    --dtype "$DTYPE"
else
  echo "[SKIP] Case 1/3: BSA baseline"
fi

if [[ "$RUN_LSWA_DIRECT" == "1" ]]; then
  echo "========== Case 2/3: LSWA direct baseline (open BSA weights + LSWA attention) =========="
  python scripts/infer_lswa.py \
    --input "$TEST_PATH" \
    --save-root "$SAVE_ROOT" \
    --flashvsr-root "$FLASHVSR_ROOT" \
    --base-model-weight "$BASE_MODEL_WEIGHT" \
    --lq-proj-ckpt "$LQ_PROJ_CKPT" \
    --tc-decoder-ckpt "$TC_DECODER_CKPT" \
    --window-size "$WINDOW_SIZE" \
    --scale "$SCALE" \
    --max-videos "$MAX_VIDEOS" \
    --max-frames "$MAX_FRAMES" \
    --seed "$SEED" \
    --dtype "$DTYPE"
else
  echo "[SKIP] Case 2/3: LSWA direct baseline"
fi

if [[ "$RUN_LSWA_STUDENT" == "1" ]]; then
  echo "========== Case 3/3: LSWA trained student =========="
  if [[ -z "$LSWA_STUDENT_CKPT" ]]; then
    echo "[SKIP] LSWA_STUDENT_CKPT is empty. Set it to your trained latest.pt/step_xxx.pt."
  else
    check_file "$LSWA_STUDENT_CKPT"
    python scripts/infer_lswa.py \
      --input "$TEST_PATH" \
      --save-root "$SAVE_ROOT" \
      --flashvsr-root "$FLASHVSR_ROOT" \
      --base-model-weight "$BASE_MODEL_WEIGHT" \
      --student-ckpt "$LSWA_STUDENT_CKPT" \
      --lq-proj-ckpt "$LQ_PROJ_CKPT" \
      --tc-decoder-ckpt "$TC_DECODER_CKPT" \
      --window-size "$WINDOW_SIZE" \
      --scale "$SCALE" \
      --max-videos "$MAX_VIDEOS" \
      --max-frames "$MAX_FRAMES" \
      --seed "$SEED" \
      --dtype "$DTYPE"
  fi
else
  echo "[SKIP] Case 3/3: LSWA trained student"
fi

echo "[DONE] Outputs are under: $SAVE_ROOT"
