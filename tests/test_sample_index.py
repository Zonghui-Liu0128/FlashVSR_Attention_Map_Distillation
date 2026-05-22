import json

import pytest

from flashvsr_b1.data import sample_index


def test_build_sample_records_from_csv_uses_full_video_as_training_target(tmp_path, monkeypatch):
    video_path = tmp_path / "000000_960x720_93f.mp4"
    video_path.write_bytes(b"placeholder")
    csv_path = tmp_path / "metadata_wxh_960x720.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Path,Height,Width,Frame,FPS,Duration",
                f"{video_path},720,960,93,93.0,1.0",
            ]
        ),
        encoding="utf-8",
    )

    def fail_if_spatial_crop_planner_is_used(**kwargs):
        raise AssertionError(f"CSV metadata must not plan center/sliding crops: {kwargs}")

    monkeypatch.setattr(sample_index, "plan_spatial_crops", fail_if_spatial_crop_planner_is_used)

    index = sample_index.build_sample_records_from_csv(
        {
            "metadata_csv_path": str(csv_path),
            "strict_path_exists": True,
            "crop_height": 720,
            "crop_width": 960,
            "frame_num": 93,
        }
    )

    assert index["stats"]["videos_seen"] == 1
    assert index["stats"]["videos_used"] == 1
    assert index["stats"]["samples_built"] == 1
    sample = index["samples"][0]
    assert sample["path"] == str(video_path)
    assert sample["height"] == 720
    assert sample["width"] == 960
    assert sample["frame_num"] == 93
    assert sample["crop_x"] == 0
    assert sample["crop_y"] == 0
    assert sample["crop_height"] == 720
    assert sample["crop_width"] == 960
    assert sample["crop_policy"] == "full_frame_csv"
    assert sample["clip_start"] == 0
    assert sample["clip_end"] == 93


def test_build_sample_records_drops_undecodable_clip_when_validation_enabled(tmp_path, monkeypatch):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"placeholder")
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "videos": {
                    video_path.name: {
                        "path": str(video_path),
                        "fps": 30.0,
                        "total_frames": 100,
                        "height": 1024,
                        "width": 1920,
                        "scenes": [
                            {
                                "scene_id": 0,
                                "start_frame": 0,
                                "end_frame": 100,
                                "n_frames": 100,
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    checked = []

    def fake_clip_decodable(path, start_frame, frame_num):
        checked.append((path, start_frame, frame_num))
        return start_frame == 0

    monkeypatch.setattr(sample_index, "is_clip_decodable_cv2", fake_clip_decodable, raising=False)

    index = sample_index.build_sample_records_from_metadata(
        {
            "metadata_json_path": str(metadata_path),
            "read_resolution_with_cv2": False,
            "strict_path_exists": True,
            "crop_height": 1024,
            "crop_width": 1920,
            "frame_num": 85,
            "temporal_stride": 85,
            "include_tail_clip": True,
            "validate_clip_decode": True,
        }
    )

    assert checked == [(str(video_path), 0, 85), (str(video_path), 15, 85)]
    assert index["stats"]["clips_built"] == 1
    assert index["stats"]["dropped_decode_failed"] == 1
    assert [sample["clip_start"] for sample in index["samples"]] == [0]


def test_validate_sample_index_contract_rejects_stale_frame_and_crop_shape():
    stale_index = {
        "samples": [
            {
                "sample_id": "old_sample",
                "frame_num": 41,
                "crop_width": 960,
                "crop_height": 512,
            }
        ]
    }

    with pytest.raises(RuntimeError, match="stale sample index.*frame_num.*41.*85.*crop_width.*960.*1920.*crop_height.*512.*1024"):
        sample_index.validate_sample_index_contract(
            stale_index,
            {
                "frame_num": 85,
                "crop_width": 1920,
                "crop_height": 1024,
            },
            sample_json_path="/tmp/train_samples.json",
        )


def test_validate_sample_index_contract_allows_longer_clips_when_truncating_frames():
    index_built_for_85_frames = {
        "samples": [
            {
                "sample_id": "clip_000",
                "frame_num": 85,
                "crop_width": 1920,
                "crop_height": 1024,
            }
        ]
    }

    sample_index.validate_sample_index_contract(
        index_built_for_85_frames,
        {
            "frame_num": 45,
            "crop_width": 1920,
            "crop_height": 1024,
            "allow_frame_truncation": True,
        },
        sample_json_path="/tmp/train_samples_b1_85_1024x1920_validated.json",
    )
