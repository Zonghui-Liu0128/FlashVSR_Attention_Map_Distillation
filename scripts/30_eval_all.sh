#!/bin/bash
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/FlashVSR_Attention_Map_Distillation}"
cd "$PROJECT_ROOT"

# Default: discover all three runs under log/*_b1_*. Pass explicit paths to override.
RUNS=${@:-"$(ls -d log/*_b1_bsa90 log/*_b1_lswa log/*_b1_bsa95 2>/dev/null)"}

if [ -z "$RUNS" ]; then
  echo "No run directories found under log/. Provide them explicitly." >&2
  exit 1
fi

for run in $RUNS; do
  echo "Evaluating $run"
  python -m eval.eval_sr \
    --ckpt "$run/ckpt/latest.pt" \
    --val_json "${VAL_JSON:-/path/to/val_samples.json}" \
    --out_json "$run/eval/final_metrics.json"
done

python -m eval.compare_baseline \
  --runs "$RUNS" \
  --out docs/final_report.md

echo "Wrote docs/final_report.md"
