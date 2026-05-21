#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

METADATA_CSV="${1:-data/metadata_wxh_960x720.csv}"
OUTPUT_DIR="${2:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/vsr_datasets/animal_videos/videos_960x720/lq}"
CONFIG="${3:-Prepare/degradation_config_960x720.yaml}"

python - "${METADATA_CSV}" <<'PY'
import csv
import os
import sys

metadata_csv = sys.argv[1]
missing = []
with open(metadata_csv, newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))[:3]

print("[preflight] first3 metadata rows:")
for idx, row in enumerate(rows):
    path = row["Path"]
    exists = os.path.exists(path)
    print(f"  {idx}: exists={exists} path={path}")
    if not exists:
        missing.append(path)

if missing:
    print("[preflight] missing GT videos; mount or fix metadata paths before running degradation.")
    raise SystemExit(2)
PY

python -m Prepare.offline_degradation \
  --config "${CONFIG}" \
  --metadata-csv "${METADATA_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-videos 3 \
  --seed 42 \
  --overwrite

python - "${METADATA_CSV}" "${OUTPUT_DIR}" <<'PY'
import csv
import os
import sys
from pathlib import Path

import cv2

metadata_csv, output_dir = sys.argv[1], Path(sys.argv[2])
with open(metadata_csv, newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))[:3]

print("[verify] written LQ videos:")
for row in rows:
    out_path = output_dir / Path(row["Path"]).name
    if not out_path.exists():
        raise SystemExit(f"missing output: {out_path}")
    cap = cv2.VideoCapture(str(out_path))
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    size = os.path.getsize(out_path)
    print(f"  {out_path} | {width}x{height}, frames={frames}, fps={fps:.3f}, bytes={size}")
PY
