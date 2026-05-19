#!/bin/bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CONFIG=${1:-flashvsr_b1/configs/b1_bsa90.yaml}
if [[ $# -gt 0 ]]; then
  shift
fi

EXTRA_FLAGS=(
  train.total_steps=${TOTAL_STEPS:-2}
  data.max_samples=${MAX_SAMPLES:-16}
  data.shuffle_samples=false
  data.num_workers=${NUM_WORKERS:-0}
  data.max_retry=1
  logging.log_every_steps=1
  logging.ckpt_every_steps=0
  logging.save_final=false
  eval.every_steps=0
)

python -m flashvsr_b1.train.trainer_b1 --config "$CONFIG" "${EXTRA_FLAGS[@]}" "$@"
