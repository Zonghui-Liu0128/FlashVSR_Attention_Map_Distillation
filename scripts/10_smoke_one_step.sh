#!/bin/bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CONFIG=${1:-flashvsr_b1/configs/b1_bsa90.yaml}
EXTRA_FLAGS="--train.total_steps=20 --logging.log_every_steps=2 --logging.ckpt_every_steps=10"

python -m flashvsr_b1.train.trainer_b1 --config "$CONFIG" $EXTRA_FLAGS
