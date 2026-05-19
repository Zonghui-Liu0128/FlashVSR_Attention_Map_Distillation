# B200 20-step smoke decode failure diagnostics

This note is for the internal B200 machine. It collects evidence for:

```text
RuntimeError: Decode failed: path=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/datasets_vsr/aigcvsr_dataset/8w_video/EDD-Video_V1130_purchase_v1/part-0097/86b29fe0-1517-4f9a-a882-2362f0caa59c.mp4, frame=330
```

The current hypothesis is that `frame_num: 85` limits each training clip to 85 frames, but the clip is sampled from a longer source mp4. Therefore `frame=330` is an absolute source-video frame number, not the 330th frame inside one training sample.

Please run the commands below from the repository root on the internal machine, then commit/push the updated `内网B200 单卡20步报错.txt` back to GitHub.

## One-shot collection

This appends a structured diagnostics section to `内网B200 单卡20步报错.txt`.

```bash
python - <<'PY'
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import traceback
from pathlib import Path

OUT = Path("内网B200 单卡20步报错.txt")
DATA_CFG_PATH = Path("flashvsr_b1/configs/data_b1.yaml")
BAD_PATH = "/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/datasets_vsr/aigcvsr_dataset/8w_video/EDD-Video_V1130_purchase_v1/part-0097/86b29fe0-1517-4f9a-a882-2362f0caa59c.mp4"
BAD_FRAME = 330


def append(text: str = "") -> None:
    with OUT.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def section(title: str) -> None:
    append("\n" + "=" * 88)
    append(title)
    append("=" * 88)


def run_cmd(args: list[str]) -> None:
    append("$ " + " ".join(args))
    try:
        proc = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        append(proc.stdout.rstrip() or "<no output>")
        append(f"[exit_code] {proc.returncode}")
    except Exception:
        append(traceback.format_exc().rstrip())


def load_data_cfg() -> dict:
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(OmegaConf.load(DATA_CFG_PATH), resolve=True)
    except Exception:
        append("[WARN] failed to load data_b1.yaml with OmegaConf:")
        append(traceback.format_exc().rstrip())
        return {}


section(f"B200 decode diagnostics appended at {_dt.datetime.now().isoformat(timespec='seconds')}")
append(f"repo={Path.cwd()}")
append(f"bad_path={BAD_PATH}")
append(f"bad_frame={BAD_FRAME}")

section("Git and Python environment")
run_cmd(["git", "rev-parse", "--short", "HEAD"])
run_cmd(["git", "status", "--short"])
run_cmd(["python", "-V"])

try:
    import cv2

    append(f"cv2.__version__={cv2.__version__}")
except Exception:
    append("[ERROR] cannot import cv2")
    append(traceback.format_exc().rstrip())
    raise

section("Runtime data config")
data_cfg = load_data_cfg()
for key in [
    "metadata_json_path",
    "sample_json_path",
    "rebuild_sample_json",
    "read_resolution_with_cv2",
    "strict_decode",
    "allow_decode_fail_pad",
    "max_retry",
    "frame_num",
    "temporal_stride",
    "include_tail_clip",
    "shuffle_samples",
    "max_samples",
]:
    append(f"{key}: {data_cfg.get(key)!r}")

sample_json_path = Path(str(data_cfg.get("sample_json_path", "")))
metadata_json_path = Path(str(data_cfg.get("metadata_json_path", "")))
append(f"sample_json_exists={sample_json_path.exists()} path={sample_json_path}")
append(f"metadata_json_exists={metadata_json_path.exists()} path={metadata_json_path}")

section("train_samples.json header")
sample_index = {}
samples = []
if sample_json_path.exists():
    with sample_json_path.open("r", encoding="utf-8") as f:
        sample_index = json.load(f)
    append("config=" + json.dumps(sample_index.get("config"), ensure_ascii=False, indent=2))
    append("stats=" + json.dumps(sample_index.get("stats"), ensure_ascii=False, indent=2))
    samples = sample_index.get("samples", [])
    append(f"num_samples={len(samples)}")
else:
    append("[ERROR] sample_json_path does not exist")

section("Samples from the failing mp4")
same_path = [s for s in samples if s.get("path") == BAD_PATH]
cover_bad_frame = [
    (i, s)
    for i, s in enumerate(samples)
    if s.get("path") == BAD_PATH
    and int(s.get("clip_start", -1)) <= BAD_FRAME < int(s.get("clip_start", -1)) + int(s.get("frame_num", data_cfg.get("frame_num", 85)))
]
append(f"samples_with_same_path={len(same_path)}")
append(f"samples_covering_bad_frame={len(cover_bad_frame)}")
for i, s in cover_bad_frame[:20]:
    keep = {
        k: s.get(k)
        for k in [
            "sample_id",
            "video_id",
            "clip_start",
            "clip_end",
            "frame_num",
            "scene_start",
            "scene_end",
            "scene_frames",
            "total_frames",
            "fps",
            "crop_x",
            "crop_y",
            "crop_width",
            "crop_height",
            "orientation_transform",
        ]
    }
    append(f"[sample_index={i}] " + json.dumps(keep, ensure_ascii=False, sort_keys=True))

section("Metadata entry for failing mp4")
if metadata_json_path.exists():
    try:
        with metadata_json_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        videos = metadata.get("videos", metadata) if isinstance(metadata, dict) else metadata
        key = Path(BAD_PATH).name
        info = videos.get(key) if isinstance(videos, dict) else None
        if info is None and isinstance(videos, list):
            for item in videos:
                if isinstance(item, dict) and item.get("path") == BAD_PATH:
                    info = item
                    break
        append(json.dumps(info, ensure_ascii=False, indent=2) if info is not None else "<not found>")
    except Exception:
        append(traceback.format_exc().rstrip())
else:
    append("<metadata_json_path missing>")

section("OpenCV probe around failing frame")
cap = cv2.VideoCapture(BAD_PATH)
append(f"opened={cap.isOpened()}")
if cap.isOpened():
    try:
        append(f"backend={cap.getBackendName()}")
    except Exception:
        pass
    props = {
        "CAP_PROP_FRAME_COUNT": cap.get(cv2.CAP_PROP_FRAME_COUNT),
        "CAP_PROP_FPS": cap.get(cv2.CAP_PROP_FPS),
        "CAP_PROP_FRAME_WIDTH": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        "CAP_PROP_FRAME_HEIGHT": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
    }
    append(json.dumps(props, indent=2))
    frames_to_check = sorted(set([0, 1, BAD_FRAME - 3, BAD_FRAME - 2, BAD_FRAME - 1, BAD_FRAME, BAD_FRAME + 1, BAD_FRAME + 2]))
    for frame_id in frames_to_check:
        if frame_id < 0:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ret, frame = cap.read()
        shape = None if frame is None else tuple(frame.shape)
        pos_after = cap.get(cv2.CAP_PROP_POS_FRAMES)
        append(f"seek_read frame={frame_id} ret={ret} shape={shape} pos_after={pos_after}")
cap.release()

section("ffprobe, if available")
run_cmd([
    "ffprobe",
    "-v",
    "error",
    "-select_streams",
    "v:0",
    "-count_frames",
    "-show_entries",
    "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,duration,nb_frames,nb_read_frames",
    "-of",
    "json",
    BAD_PATH,
])

section("Decode each matching 85-frame clip")
def decode_clip(start: int, frame_num: int) -> tuple[bool, list[int]]:
    c = cv2.VideoCapture(BAD_PATH)
    failed = []
    try:
        if not c.isOpened():
            return False, [-1]
        c.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        for offset in range(int(frame_num)):
            ret, frame = c.read()
            if not ret or frame is None:
                failed.append(int(start) + offset)
                break
        return len(failed) == 0, failed
    finally:
        c.release()

for i, s in cover_bad_frame[:20]:
    start = int(s.get("clip_start"))
    frame_num = int(s.get("frame_num", data_cfg.get("frame_num", 85)))
    ok, failed = decode_clip(start, frame_num)
    append(f"[sample_index={i}] clip_start={start} frame_num={frame_num} ok={ok} failed={failed}")

section("Interpretation checklist")
append("If seek_read frame=330 ret=False and clip decode fails at 330, root cause is a bad/unreadable source frame or stale sample window.")
append("If seek_read frame=330 ret=True but clip decode fails only in DataLoader, suspect OpenCV seek/worker/network filesystem instability.")
append("If sample config frame_num is not 85 or source_metadata_json points to an old dataset, suspect stale train_samples.json.")
append("If metadata total_frames/scene_end extends beyond the decodable frame count, clean or rebuild the sample index with decode validation.")
append("END_B200_DECODE_DIAGNOSTICS")
PY
```

## Minimal manual checks

If the one-shot script is too verbose, run these smaller checks instead and paste their output into `内网B200 单卡20步报错.txt`.

```bash
python - <<'PY'
import json
bad = "/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/datasets_vsr/aigcvsr_dataset/8w_video/EDD-Video_V1130_purchase_v1/part-0097/86b29fe0-1517-4f9a-a882-2362f0caa59c.mp4"
p = "/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/FlashVSR_LSWA_B200-main/data/animal_videos/stage1/train_samples.json"
data = json.load(open(p))
for i, s in enumerate(data["samples"]):
    start = int(s.get("clip_start", -1))
    frame_num = int(s.get("frame_num", 85))
    if s.get("path") == bad and start <= 330 < start + frame_num:
        print(i, s)
PY
```

```bash
python - <<'PY'
import cv2
path = "/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/datasets_vsr/aigcvsr_dataset/8w_video/EDD-Video_V1130_purchase_v1/part-0097/86b29fe0-1517-4f9a-a882-2362f0caa59c.mp4"
cap = cv2.VideoCapture(path)
print("opened", cap.isOpened())
print("frame_count", cap.get(cv2.CAP_PROP_FRAME_COUNT), "fps", cap.get(cv2.CAP_PROP_FPS))
for frame_id in [327, 328, 329, 330, 331, 332]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    print(frame_id, ret, None if frame is None else frame.shape, "pos_after", cap.get(cv2.CAP_PROP_POS_FRAMES))
cap.release()
PY
```

## Expected next decision

- If frame 330 is not decodable, fix the data index by removing this video/window and preferably validating clip decodability when building `train_samples.json`.
- If frame 330 is decodable in isolation but not in DataLoader, rerun with `data.num_workers=0` and collect whether the failure disappears. That points to worker/seek/filesystem instability rather than a deterministic bad frame.
- If `train_samples.json` was generated with a different config than `flashvsr_b1/configs/data_b1.yaml`, rebuild or replace the sample index before training.
