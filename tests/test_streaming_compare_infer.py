from pathlib import Path

import torch

from flashvsr_b1.inference.streaming_compare import (
    build_output_path,
    canvas_for_model_input,
    discover_inputs,
    normalize_dit_state_dict,
    parse_window_size,
    select_streaming_frame_count,
)


def test_parse_window_size_accepts_cli_forms():
    assert parse_window_size("2,21,21") == (2, 21, 21)
    assert parse_window_size("2x17x17") == (2, 17, 17)
    assert parse_window_size((2, 9, 9)) == (2, 9, 9)


def test_canvas_for_960x720_pads_to_block_safe_multiple():
    spec = canvas_for_model_input(width=960, height=720, multiple=128, mode="pad")

    assert (spec.target_width, spec.target_height) == (1024, 768)
    assert (spec.pad_left, spec.pad_top, spec.pad_right, spec.pad_bottom) == (0, 0, 64, 48)
    assert spec.output_crop_box == (0, 0, 960, 720)


def test_canvas_scale_4_maps_native_lq_to_4x_hq_canvas():
    spec = canvas_for_model_input(width=240, height=180, multiple=128, mode="pad", scale=4.0)

    assert (spec.source_width, spec.source_height) == (960, 720)
    assert (spec.target_width, spec.target_height) == (1024, 768)
    assert spec.output_crop_box == (0, 0, 960, 720)


def test_select_streaming_frame_count_matches_flashvsr_8n1_contract():
    selection = select_streaming_frame_count(total_frames=93, tail_padding=4)

    assert selection.model_frames == 97
    assert selection.effective_output_frames == 93
    assert selection.indices[:3] == [0, 1, 2]
    assert selection.indices[-4:] == [92, 92, 92, 92]


def test_discover_inputs_sorts_videos_and_limits(tmp_path):
    for name in ["b.mp4", "a.mp4", "c.txt", "d.MOV"]:
        (tmp_path / name).write_text("x")

    assert [p.name for p in discover_inputs(tmp_path, max_videos=2)] == ["a.mp4", "b.mp4"]


def test_normalize_dit_state_dict_splits_b1_qkv_keys():
    raw = {
        "student.blocks.0.self_attn.qkv_proj.weight": torch.arange(24).reshape(6, 4),
        "student.blocks.0.self_attn.qkv_proj.bias": torch.arange(6),
        "student.blocks.0.self_attn.o_proj.weight": torch.ones(2, 2),
        "student.blocks.0.self_attn.o_proj.bias": torch.zeros(2),
        "student.patch_embedding.weight": torch.ones(1, 1, 1, 1, 1),
    }

    state = normalize_dit_state_dict(raw)

    assert "blocks.0.self_attn.q.weight" in state
    assert "blocks.0.self_attn.k.weight" in state
    assert "blocks.0.self_attn.v.weight" in state
    assert "blocks.0.self_attn.o.weight" in state
    assert "patch_embedding.weight" in state
    assert torch.equal(state["blocks.0.self_attn.q.bias"], torch.tensor([0, 1]))
    assert torch.equal(state["blocks.0.self_attn.k.bias"], torch.tensor([2, 3]))
    assert torch.equal(state["blocks.0.self_attn.v.bias"], torch.tensor([4, 5]))


def test_build_output_path_includes_model_and_seed(tmp_path):
    out = build_output_path(
        save_root=tmp_path,
        input_path=Path("/data/lq/cat_960x720_93f.mp4"),
        model_type="LSWA",
        seed=7,
    )

    assert out == tmp_path / "LSWA" / "cat_960x720_93f_LSWA_seed7.mp4"


def test_bash_launcher_exposes_b200_runtime_knobs():
    script = Path("scripts/40_run_b1_infer_three_cases.sh").read_text()

    assert "RUN_BSA_BASELINE=${RUN_BSA_BASELINE:-1}" in script
    assert "RUN_LSWA_DIRECT=${RUN_LSWA_DIRECT:-1}" in script
    assert "RUN_LSWA_STUDENT=${RUN_LSWA_STUDENT:-1}" in script
    assert "SCALE=${SCALE:-1.0}" in script
    assert "TEST_PATH=${TEST_PATH:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/vsr_datasets/animal_videos/videos_960x720/lq/test}" in script
    assert "infer_bsa_baseline.py" in script
    assert "infer_lswa.py" in script
    assert "--scale" in script
    assert "--window-size" in script
    assert "--student-ckpt" in script
    assert "--save-root" in script


def test_bsa_python_script_is_official_baseline_only():
    script = Path("scripts/infer_bsa_baseline.py").read_text()

    assert "FlashVSRTinyPipeline" in script
    assert "FlashVSRTinyLongPipeline" not in script
    assert "sys.path.insert" in script
    assert "replace_dit_with_lswa" not in script
    assert "--scale" in script
    assert "--model-weight" in script
    assert "--lq-proj-ckpt" in script
    assert "--tc-decoder-ckpt" in script


def test_lswa_python_script_supports_direct_and_student_modes():
    script = Path("scripts/infer_lswa.py").read_text()

    assert "replace_dit_with_lswa" in script
    assert "--student-ckpt" in script
    assert "--window-size" in script
    assert "--scale" in script
    assert "FlashVSRTinyPipeline" in script
    assert "FlashVSRTinyLongPipeline" not in script
    assert "sys.path.insert" in script
    assert "normalize_dit_state_dict" in script
