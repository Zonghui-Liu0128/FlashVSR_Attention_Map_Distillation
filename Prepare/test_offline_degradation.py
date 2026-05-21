from pathlib import Path

import torch

from Prepare.offline_degradation import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_LQ_OUTPUT_DIR,
    DEFAULT_METADATA_CSV,
    OfflineDegrader,
    build_lq_plan,
    choose_output_fps,
    load_degradation_config,
    resolve_lq_output_path,
    tensor_to_uint8_rgb,
    validate_gt_paths,
)


def test_defaults_are_prepare_local_and_target_lq_dir():
    assert DEFAULT_CONFIG_PATH == Path("Prepare/degradation_config_960x720.yaml")
    assert DEFAULT_METADATA_CSV == Path("data/metadata_wxh_960x720.csv")
    assert (
        DEFAULT_LQ_OUTPUT_DIR
        == Path(
            "/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/"
            "vsr_datasets/animal_videos/videos_960x720/lq"
        )
    )


def test_standalone_module_does_not_import_flashvsr_training_code():
    source = Path("Prepare/offline_degradation.py").read_text(encoding="utf-8")

    assert "flashvsr_b1" not in source


def test_resolve_lq_output_path_preserves_gt_filename():
    gt_path = "/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/vsr_datasets/animal_videos/videos_960x720/gt/000000_960x720_93f.mp4"

    assert resolve_lq_output_path(gt_path, "/tmp/lq") == Path("/tmp/lq/000000_960x720_93f.mp4")


def test_tensor_to_uint8_rgb_converts_tchw_float_video():
    video = torch.tensor(
        [
            [
                [[0.0, 0.5]],
                [[1.0, 0.25]],
                [[0.25, 1.0]],
            ]
        ]
    )

    out = tensor_to_uint8_rgb(video)

    assert out.shape == (1, 1, 2, 3)
    assert out.dtype.name == "uint8"
    assert out.tolist() == [[[[0, 255, 64], [128, 64, 255]]]]


def test_build_lq_plan_uses_metadata_order_and_lq_output_dir(tmp_path):
    gt0 = tmp_path / "gt0.mp4"
    gt1 = tmp_path / "gt1.mp4"
    gt0.write_bytes(b"placeholder")
    gt1.write_bytes(b"placeholder")
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "Path,Height,Width,Frame,FPS,Duration\n"
        f"{gt0},720,960,93,93.0,1.0\n"
        f"{gt1},720,960,93,24.0,3.875\n",
        encoding="utf-8",
    )

    plan = build_lq_plan(metadata, tmp_path / "lq", max_videos=1)

    assert len(plan) == 1
    assert plan[0].gt_path == gt0
    assert plan[0].lq_path == tmp_path / "lq" / "gt0.mp4"
    assert plan[0].fps == 93.0
    assert plan[0].frame_num == 93


def test_choose_output_fps_prefers_explicit_override_over_metadata():
    assert choose_output_fps(metadata_fps=93.0, output_fps=30.0) == 30.0


def test_choose_output_fps_uses_metadata_when_override_is_missing():
    assert choose_output_fps(metadata_fps=93.0, output_fps=None) == 93.0


def test_validate_gt_paths_reports_missing_entries(tmp_path):
    existing = tmp_path / "existing.mp4"
    missing = tmp_path / "missing.mp4"
    existing.write_bytes(b"placeholder")
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "Path,Height,Width,Frame,FPS,Duration\n"
        f"{existing},720,960,93,93.0,1.0\n"
        f"{missing},720,960,93,93.0,1.0\n",
        encoding="utf-8",
    )

    missing_paths = validate_gt_paths(metadata, max_videos=2)

    assert missing_paths == [missing]


def test_offline_degrader_returns_upscaled_lq_and_native_lr_shapes():
    cfg = load_degradation_config(DEFAULT_CONFIG_PATH)
    cfg.update(
        {
            "scl_factor": 4.0,
            "input_bits": 10,
            "degradation_1": False,
            "degradation_2": False,
            "degradation_3": True,
            "final_sinc_prob": 0.0,
            "final_downsample_mode_probs": {"area": 1.0},
            "final_upsample_mode": "bilinear",
            "seed": 123,
        }
    )
    video = torch.linspace(0.0, 1.0, steps=2 * 3 * 16 * 20).reshape(2, 3, 16, 20)

    result = OfflineDegrader(cfg).apply(video, return_native_lr=True)

    assert result.lq_up.shape == video.shape
    assert result.lr_native is not None
    assert result.lr_native.shape == (2, 3, 4, 5)
    assert result.lq_up.amin().item() >= 0.0
    assert result.lq_up.amax().item() <= 1.0
