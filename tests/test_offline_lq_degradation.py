from pathlib import Path

import torch

from flashvsr_b1.data.degradation.offline_lq import (
    DEFAULT_LQ_OUTPUT_DIR,
    resolve_lq_output_path,
    tensor_to_uint8_rgb,
)


def test_default_lq_output_dir_matches_internal_960x720_dataset():
    assert (
        DEFAULT_LQ_OUTPUT_DIR
        == "/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/vsr_datasets/animal_videos/videos_960x720/lq"
    )


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
