"""Build and validate FlashVSR LSWA sample indexes from scene metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


def _as_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _dedupe_crops(crops: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out: list[dict[str, Any]] = []
    for crop in crops:
        key = (int(crop["crop_x"]), int(crop["crop_y"]), int(crop["crop_width"]), int(crop["crop_height"]))
        if key in seen:
            continue
        seen.add(key)
        crop = dict(crop)
        crop["crop_id"] = len(out)
        out.append(crop)
    return out


def _axis_positions(length: int, crop: int, stride: int) -> list[int]:
    if length < crop:
        return []
    if length == crop:
        return [0]
    positions = list(range(0, length - crop + 1, max(1, stride)))
    if positions[-1] != length - crop:
        positions.append(length - crop)
    return positions


def get_video_info_cv2(path: str) -> dict[str, Any]:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency `opencv-python-headless`; install it in the target env.") from exc

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video resolution from cv2: {path}, H={height}, W={width}")
    return {
        "height": height,
        "width": width,
        "fps_cv2": fps,
        "total_frames_cv2": frame_count,
    }


def plan_spatial_crops(
    *,
    width: int,
    height: int,
    crop_width: int,
    crop_height: int,
    center_crop_max_width: int = 2304,
    center_crop_max_height: int = 1536,
    multi_crop_scale_threshold: float = 1.35,
    sliding_crop_scale_threshold: float = 2.2,
    spatial_crop_overlap: float = 0.25,
    max_spatial_crops_per_clip: int = 5,
) -> list[dict[str, Any]]:
    """Return center/corner/sliding crops without resizing the source video."""
    if width < crop_width or height < crop_height:
        return []

    center = {
        "crop_x": (width - crop_width) // 2,
        "crop_y": (height - crop_height) // 2,
        "crop_width": crop_width,
        "crop_height": crop_height,
        "crop_policy": "center",
    }
    if width <= center_crop_max_width and height <= center_crop_max_height:
        return _dedupe_crops([center])

    scale_min = min(width / float(crop_width), height / float(crop_height))
    if scale_min < multi_crop_scale_threshold:
        return _dedupe_crops([center])

    if scale_min < sliding_crop_scale_threshold:
        candidates = [
            {**center, "crop_policy": "center_corners:center"},
            {"crop_x": 0, "crop_y": 0, "crop_width": crop_width, "crop_height": crop_height, "crop_policy": "center_corners:top_left"},
            {"crop_x": width - crop_width, "crop_y": 0, "crop_width": crop_width, "crop_height": crop_height, "crop_policy": "center_corners:top_right"},
            {"crop_x": 0, "crop_y": height - crop_height, "crop_width": crop_width, "crop_height": crop_height, "crop_policy": "center_corners:bottom_left"},
            {"crop_x": width - crop_width, "crop_y": height - crop_height, "crop_width": crop_width, "crop_height": crop_height, "crop_policy": "center_corners:bottom_right"},
        ]
        return _dedupe_crops(candidates)[:max_spatial_crops_per_clip]

    stride_x = int(round(crop_width * (1.0 - spatial_crop_overlap)))
    stride_y = int(round(crop_height * (1.0 - spatial_crop_overlap)))
    candidates = [{**center, "crop_policy": "sliding:center"}]
    for y in _axis_positions(height, crop_height, stride_y):
        for x in _axis_positions(width, crop_width, stride_x):
            candidates.append({
                "crop_x": int(x),
                "crop_y": int(y),
                "crop_width": crop_width,
                "crop_height": crop_height,
                "crop_policy": "sliding",
            })
    return _dedupe_crops(candidates)[:max_spatial_crops_per_clip]


def plan_clip_starts(
    *,
    scene_start: int,
    scene_end: int,
    frame_num: int,
    temporal_stride: int,
    include_tail_clip: bool = True,
    max_clips_per_scene: int = 0,
) -> list[int]:
    """Return clip starts inside [scene_start, scene_end), never crossing cuts."""
    if scene_end - scene_start < frame_num:
        return []
    last_start = scene_end - frame_num
    starts = list(range(scene_start, last_start + 1, max(1, temporal_stride)))
    if include_tail_clip and starts and starts[-1] != last_start:
        starts.append(last_start)
    if not starts:
        starts = [scene_start]
    if max_clips_per_scene and len(starts) > max_clips_per_scene:
        if max_clips_per_scene == 1:
            starts = [starts[0]]
        else:
            keep = sorted(set(round(i * (len(starts) - 1) / (max_clips_per_scene - 1)) for i in range(max_clips_per_scene)))
            starts = [starts[i] for i in keep]
    return starts


def _get_metadata_resolution(video_info: dict[str, Any]) -> tuple[int | None, int | None]:
    h = video_info.get("height", video_info.get("H", video_info.get("h")))
    w = video_info.get("width", video_info.get("W", video_info.get("w")))
    return _as_int(h), _as_int(w)


def normalize_orientation_for_crop(*, width: int, height: int, crop_width: int, crop_height: int) -> tuple[int, int, str]:
    """Return crop-planning W/H plus a transform for portrait sources.

    Portrait 1080x1920 animal videos are useful for this training target.  They
    cannot directly provide a 1920-wide crop, so the dataset rotates them to
    landscape before applying the normal crop planner.
    """
    if width >= crop_width and height >= crop_height:
        return int(width), int(height), "none"
    if height >= crop_width and width >= crop_height:
        return int(height), int(width), "rotate_90_clockwise"
    return int(width), int(height), "none"


def _iter_metadata_videos(metadata: Any) -> Iterable[tuple[str, dict[str, Any]]]:
    videos = metadata.get("videos", metadata) if isinstance(metadata, dict) else metadata
    if isinstance(videos, dict):
        for video_key, info in videos.items():
            if isinstance(info, dict):
                yield str(video_key), info
        return
    if isinstance(videos, list):
        for i, info in enumerate(videos):
            if not isinstance(info, dict):
                continue
            video_key = info.get("video_id") or info.get("id") or info.get("name") or info.get("path") or f"video_{i:06d}"
            yield str(video_key), info


def _iter_scenes(video_info: dict[str, Any], total_frames: int | None) -> list[dict[str, Any]]:
    scenes = video_info.get("scenes")
    if isinstance(scenes, list) and scenes:
        return [scene for scene in scenes if isinstance(scene, dict)]
    if total_frames and total_frames > 0:
        return [{"scene_id": 0, "start_frame": 0, "end_frame": int(total_frames), "n_frames": int(total_frames)}]
    return []


def build_sample_records_from_metadata(opt: dict[str, Any]) -> dict[str, Any]:
    """Build a sample.json dict matching the internal B200 training schema."""
    metadata_json_path = opt["metadata_json_path"]
    crop_height = int(opt.get("crop_height", opt.get("crop_patch_size", 768)))
    crop_width = int(opt.get("crop_width", opt.get("crop_patch_size", 1280)))
    frame_num = int(opt.get("frame_num", 89))
    temporal_stride = int(opt.get("temporal_stride", frame_num))
    include_tail_clip = bool(opt.get("include_tail_clip", True))
    max_clips_per_scene = int(opt.get("max_clips_per_scene", 0))
    center_crop_max_width = int(opt.get("center_crop_max_width", 2304))
    center_crop_max_height = int(opt.get("center_crop_max_height", 1536))
    multi_crop_scale_threshold = float(opt.get("multi_crop_scale_threshold", 1.35))
    sliding_crop_scale_threshold = float(opt.get("sliding_crop_scale_threshold", 2.2))
    spatial_crop_overlap = float(opt.get("spatial_crop_overlap", 0.25))
    max_spatial_crops_per_clip = int(opt.get("max_spatial_crops_per_clip", 5))
    strict_path_exists = bool(opt.get("strict_path_exists", True))
    read_resolution_with_cv2 = bool(opt.get("read_resolution_with_cv2", True))
    min_source_width = _as_int(opt.get("min_source_width"), None)
    min_source_height = _as_int(opt.get("min_source_height"), None)
    max_source_width = _as_int(opt.get("max_source_width"), None)
    max_source_height = _as_int(opt.get("max_source_height"), None)

    with open(metadata_json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    stats = {
        "videos_seen": 0,
        "videos_used": 0,
        "scenes_seen": 0,
        "scenes_used": 0,
        "clips_built": 0,
        "samples_built": 0,
        "dropped_path_missing": 0,
        "dropped_open_failed": 0,
        "dropped_short_scene": 0,
        "dropped_small_resolution": 0,
        "dropped_resolution_filtered": 0,
    }
    samples: list[dict[str, Any]] = []

    for video_key, info in _iter_metadata_videos(metadata):
        stats["videos_seen"] += 1
        path = str(info.get("path", ""))
        if not path or (strict_path_exists and not os.path.exists(path)):
            stats["dropped_path_missing"] += 1
            continue

        fps = _as_float(info.get("fps"), None)
        total_frames = _as_int(info.get("total_frames"), None)
        height, width = _get_metadata_resolution(info)

        if read_resolution_with_cv2 or height is None or width is None:
            try:
                probe = get_video_info_cv2(path)
            except Exception:
                stats["dropped_open_failed"] += 1
                continue
            height = int(probe["height"])
            width = int(probe["width"])
            if fps is None or fps <= 0:
                fps = _as_float(probe.get("fps_cv2"), fps)
            if total_frames is None or total_frames <= 0:
                total_frames = _as_int(probe.get("total_frames_cv2"), total_frames)

        if height is None or width is None:
            stats["dropped_small_resolution"] += 1
            continue
        source_height = int(height)
        source_width = int(width)
        width, height, orientation_transform = normalize_orientation_for_crop(
            width=source_width,
            height=source_height,
            crop_width=crop_width,
            crop_height=crop_height,
        )
        if height < crop_height or width < crop_width:
            stats["dropped_small_resolution"] += 1
            continue
        if (
            (min_source_width is not None and width < min_source_width)
            or (min_source_height is not None and height < min_source_height)
            or (max_source_width is not None and width > max_source_width)
            or (max_source_height is not None and height > max_source_height)
        ):
            stats["dropped_resolution_filtered"] += 1
            continue

        crops = plan_spatial_crops(
            width=width,
            height=height,
            crop_width=crop_width,
            crop_height=crop_height,
            center_crop_max_width=center_crop_max_width,
            center_crop_max_height=center_crop_max_height,
            multi_crop_scale_threshold=multi_crop_scale_threshold,
            sliding_crop_scale_threshold=sliding_crop_scale_threshold,
            spatial_crop_overlap=spatial_crop_overlap,
            max_spatial_crops_per_clip=max_spatial_crops_per_clip,
        )
        if not crops:
            stats["dropped_small_resolution"] += 1
            continue

        video_used = False
        for scene in _iter_scenes(info, total_frames):
            stats["scenes_seen"] += 1
            scene_id = _as_int(scene.get("scene_id"), 0)
            scene_start = _as_int(scene.get("start_frame", scene.get("scene_start")), 0)
            scene_end = _as_int(scene.get("end_frame", scene.get("scene_end")), None)
            scene_frames = _as_int(scene.get("n_frames", scene.get("scene_frames")), None)
            if scene_end is None:
                scene_end = scene_start + scene_frames if scene_frames is not None else total_frames
            if scene_start is None or scene_end is None or scene_end <= scene_start:
                continue
            if scene_frames is None:
                scene_frames = scene_end - scene_start
            if scene_frames < frame_num:
                stats["dropped_short_scene"] += 1
                continue

            starts = plan_clip_starts(
                scene_start=scene_start,
                scene_end=scene_end,
                frame_num=frame_num,
                temporal_stride=temporal_stride,
                include_tail_clip=include_tail_clip,
                max_clips_per_scene=max_clips_per_scene,
            )
            stats["scenes_used"] += 1
            stats["clips_built"] += len(starts)
            video_used = True
            for clip_id, clip_start in enumerate(starts):
                for crop in crops:
                    sample_id = f"{Path(video_key).stem}_s{int(scene_id):03d}_c{clip_id:04d}_p{int(crop['crop_id']):03d}"
                    samples.append({
                        "sample_id": sample_id,
                        "video_id": video_key,
                        "path": path,
                        "fps": fps,
                        "total_frames": total_frames,
                        "source_height": int(source_height),
                        "source_width": int(source_width),
                        "orientation_transform": orientation_transform,
                        "height": int(height),
                        "width": int(width),
                        "scene_id": scene_id,
                        "scene_start": int(scene_start),
                        "scene_end": int(scene_end),
                        "scene_frames": int(scene_frames),
                        "clip_id": int(clip_id),
                        "clip_start": int(clip_start),
                        "clip_end": int(clip_start + frame_num),
                        "frame_num": int(frame_num),
                        **crop,
                    })
        if video_used:
            stats["videos_used"] += 1

    stats["samples_built"] = len(samples)
    return {
        "source_metadata_json": str(metadata_json_path),
        "config": {
            "crop_height": crop_height,
            "crop_width": crop_width,
            "frame_num": frame_num,
            "temporal_stride": temporal_stride,
            "include_tail_clip": include_tail_clip,
            "max_clips_per_scene": max_clips_per_scene,
            "center_crop_max_width": center_crop_max_width,
            "center_crop_max_height": center_crop_max_height,
            "multi_crop_scale_threshold": multi_crop_scale_threshold,
            "sliding_crop_scale_threshold": sliding_crop_scale_threshold,
            "spatial_crop_overlap": spatial_crop_overlap,
            "max_spatial_crops_per_clip": max_spatial_crops_per_clip,
            "strict_path_exists": strict_path_exists,
            "read_resolution_with_cv2": read_resolution_with_cv2,
            "min_source_width": min_source_width,
            "min_source_height": min_source_height,
            "max_source_width": max_source_width,
            "max_source_height": max_source_height,
        },
        "stats": stats,
        "samples": samples,
    }


def write_sample_index(index: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
