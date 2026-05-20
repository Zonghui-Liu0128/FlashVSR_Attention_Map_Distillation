#!/bin/bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/FlashVSR_Attention_Map_Distillation}"
cd "$PROJECT_ROOT"

CONFIG=${1:-flashvsr_b1/configs/b1_bsa90.yaml}
if [[ $# -gt 0 ]]; then
  shift
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export FLASHVSR_MEM_TRACE=1
export FLASHVSR_NVTX=1
export FLASHVSR_MEM_TRACE_MAX_ITEMS="${FLASHVSR_MEM_TRACE_MAX_ITEMS:-8}"

PROFILE_DIR="${PROFILE_DIR:-log/b200_memory_profile}"
mkdir -p "$PROFILE_DIR"

DMON_PID=""
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi dmon -s pucmt -d 1 -o DT > "$PROFILE_DIR/nvidia-smi-dmon.log" &
  DMON_PID=$!
fi

cleanup() {
  if [[ -n "$DMON_PID" ]]; then
    kill "$DMON_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

CMD=(bash scripts/11_dry_run_16.sh "$CONFIG" "$@")
if [[ "${USE_NSYS:-1}" == "1" ]] && command -v nsys >/dev/null 2>&1; then
  nsys profile \
    --force-overwrite=true \
    --trace=cuda,nvtx,cudnn,cublas \
    -o "$PROFILE_DIR/nsys_b200_memory" \
    "${CMD[@]}"
else
  "${CMD[@]}"
fi
