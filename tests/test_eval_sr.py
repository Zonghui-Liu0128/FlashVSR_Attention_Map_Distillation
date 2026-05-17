import json
from unittest.mock import patch

from eval.eval_sr import evaluate_checkpoint


def test_eval_returns_required_metric_keys(tmp_path, monkeypatch):
    """Stub per-video work so checkpoint evaluation only tests aggregation."""
    val_json = tmp_path / "val.json"
    val_json.write_text(
        json.dumps(
            {
                "samples": [
                    {"data_name": "v1.mp4", "crop_height": 1024, "crop_width": 1920},
                    {"data_name": "v2.mp4", "crop_height": 1920, "crop_width": 1024},
                ]
            }
        )
    )

    fake_per_video = {
        "psnr": 30.0,
        "ssim": 0.9,
        "lpips": 0.1,
        "dists": 0.1,
        "sparsity_rate": 0.85,
        "peak_mem_gb": 5.0,
    }
    fake_fps = {"fps_720p": 30.0, "fps_1080p": 15.0}
    with patch("eval.eval_sr._evaluate_one_video", return_value=fake_per_video), patch(
        "eval.eval_sr._measure_fps", return_value=fake_fps
    ):
        result = evaluate_checkpoint(
            "fake_ckpt.pt", str(val_json), cfg={"single_step_t": 999}, device="cpu"
        )

    for k in [
        "psnr",
        "ssim",
        "lpips",
        "dists",
        "sparsity_rate",
        "fps_720p",
        "fps_1080p",
        "peak_mem_gb",
    ]:
        assert k in result
    assert result["psnr"] == 30.0
