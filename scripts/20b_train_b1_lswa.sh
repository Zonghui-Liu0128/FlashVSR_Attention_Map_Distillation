#!/bin/bash
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/FlashVSR_Attention_Map_Distillation}"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NPROC_PER_NODE=${NPROC_PER_NODE:-8} \
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  -m flashvsr_b1.train.trainer_b1 \
  --config flashvsr_b1/configs/b1_lswa.yaml \
  "$@"
