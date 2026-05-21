#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

METADATA_CSV="${1:-data/metadata_wxh_960x720.csv}"
OUTPUT_DIR="${2:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/vsr_datasets/animal_videos/videos_960x720/lq}"
CONFIG="${3:-Prepare/degradation_config_960x720.yaml}"
WORKERS="${4:-}"
OUTPUT_FPS="${5:-}"
SEED="${6:-42}"
OVERWRITE="${OVERWRITE:-0}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
MAX_VIDEOS="${MAX_VIDEOS:-0}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/_degrade_logs/$(date +%Y%m%d-%H%M%S)}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [[ -z "${WORKERS}" ]]; then
  WORKERS="$(python - "${METADATA_CSV}" "${MAX_VIDEOS}" <<'PY'
import csv
import os
import sys

metadata_csv, max_videos = sys.argv[1], int(sys.argv[2])
with open(metadata_csv, newline="", encoding="utf-8-sig") as f:
    total = sum(1 for _ in csv.DictReader(f))
if max_videos > 0:
    total = min(total, max_videos)
cpu = os.cpu_count() or 1
print(max(1, min(8, cpu, total if total > 0 else 1)))
PY
)"
fi

if [[ "${WORKERS}" -lt 1 ]]; then
  echo "[error] WORKERS must be >= 1, got ${WORKERS}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "[config] metadata=${METADATA_CSV}"
echo "[config] output_dir=${OUTPUT_DIR}"
echo "[config] config=${CONFIG}"
echo "[config] workers=${WORKERS}"
echo "[config] seed=${SEED}"
echo "[config] max_videos=${MAX_VIDEOS}"
echo "[config] overwrite=${OVERWRITE}"
echo "[config] log_dir=${LOG_DIR}"
if [[ -n "${OUTPUT_FPS}" ]]; then
  echo "[config] output_fps=${OUTPUT_FPS}"
fi

if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
  python - "${METADATA_CSV}" "${OUTPUT_DIR}" "${MAX_VIDEOS}" <<'PY'
import csv
import os
import sys
from pathlib import Path

metadata_csv, output_dir, max_videos = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
with open(metadata_csv, newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
if max_videos > 0:
    rows = rows[:max_videos]

missing_gt = []
existing_lq = 0
for row in rows:
    gt_path = Path(row["Path"])
    if not gt_path.exists():
        missing_gt.append(str(gt_path))
    if (output_dir / gt_path.name).exists():
        existing_lq += 1

print(f"[preflight] rows={len(rows)} existing_lq={existing_lq} missing_gt={len(missing_gt)}")
if missing_gt:
    print("[preflight] first missing GT paths:")
    for path in missing_gt[:20]:
        print(f"  {path}")
    raise SystemExit(2)
PY
fi

common_args=(
  --config "${CONFIG}"
  --metadata-csv "${METADATA_CSV}"
  --output-dir "${OUTPUT_DIR}"
  --seed "${SEED}"
  --shard-count "${WORKERS}"
)

if [[ "${MAX_VIDEOS}" -gt 0 ]]; then
  common_args+=(--max-videos "${MAX_VIDEOS}")
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  common_args+=(--overwrite)
fi

if [[ -n "${OUTPUT_FPS}" ]]; then
  common_args+=(--output-fps "${OUTPUT_FPS}")
fi

pids=()
for shard in $(seq 0 $((WORKERS - 1))); do
  log_file="${LOG_DIR}/shard_${shard}_of_${WORKERS}.log"
  echo "[launch] shard=${shard}/${WORKERS} log=${log_file}"
  python -m Prepare.offline_degradation "${common_args[@]}" --shard-index "${shard}" >"${log_file}" 2>&1 &
  pids+=("$!")
done

status=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  if wait "${pid}"; then
    echo "[done] shard=${idx}/${WORKERS}"
  else
    echo "[fail] shard=${idx}/${WORKERS}; see ${LOG_DIR}/shard_${idx}_of_${WORKERS}.log" >&2
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

python - "${METADATA_CSV}" "${OUTPUT_DIR}" "${MAX_VIDEOS}" <<'PY'
import csv
import os
import sys
from pathlib import Path

import cv2

metadata_csv, output_dir, max_videos = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
with open(metadata_csv, newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
if max_videos > 0:
    rows = rows[:max_videos]

missing = []
bad = []
total_bytes = 0
for row in rows:
    out_path = output_dir / Path(row["Path"]).name
    if not out_path.exists():
        missing.append(str(out_path))
        continue
    total_bytes += os.path.getsize(out_path)
    cap = cv2.VideoCapture(str(out_path))
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    expected_w = int(row.get("Width") or row.get("width") or 0)
    expected_h = int(row.get("Height") or row.get("height") or 0)
    expected_frames = int(float(row.get("Frame") or row.get("Frames") or row.get("frames") or 0))
    if (width, height, frames) != (expected_w, expected_h, expected_frames):
        bad.append((str(out_path), width, height, frames, fps))

print(f"[verify] rows={len(rows)} missing_lq={len(missing)} bad_shape={len(bad)} total_bytes={total_bytes}")
if missing[:20]:
    print("[verify] first missing LQ paths:")
    for path in missing[:20]:
        print(f"  {path}")
if bad[:20]:
    print("[verify] first bad outputs:")
    for item in bad[:20]:
        print(f"  {item}")
if missing or bad:
    raise SystemExit(3)
PY

echo "[ok] all offline degradation shards completed"
