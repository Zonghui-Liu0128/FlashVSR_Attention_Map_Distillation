import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


def _make_trainer():
    """Build a B1Trainer instance via __new__ + manual attribute injection,
    bypassing the real parent __init__ which requires DiffSynth + ckpts."""
    from flashvsr_b1.train.trainer_b1 import B1Trainer

    trainer = B1Trainer.__new__(B1Trainer)
    trainer.cfg = SimpleNamespace(
        target_sparsity=0.90,
        distill_layers=[4, 9, 14, 19, 24, 29],
        train=SimpleNamespace(grad_clip=1.0),
    )
    trainer.config_path = "flashvsr_b1/configs/b1_bsa90.yaml"
    trainer._epoch = 0
    # Set teacher / student / decoders / lpips as MagicMocks. compute_loss path tests
    # only exercise the assembly logic, not real tensor flow.
    return trainer


def test_compute_loss_assembles_all_four_terms_for_bsa():
    trainer = _make_trainer()
    trainer.teacher = MagicMock()
    trainer.student = MagicMock()
    trainer.student.attn_mode = "BSA"
    trainer.vae_decoder = MagicMock()
    trainer.lpips_net = MagicMock()

    # teacher returns (x_t, aux_t) where aux_t has h_out and A_blk per layer.
    h_t = {l: torch.zeros(1, 8, 4) for l in [4, 9, 14, 19, 24, 29]}
    A_t = {l: torch.zeros(1, 1, 4, 4) for l in [4, 9, 14, 19, 24, 29]}
    trainer.teacher.b1_forward.return_value = (torch.zeros(1, 4, 4), {"h_out": h_t, "A_blk": A_t})

    h_s = {
        l: torch.zeros(1, 8, 4, requires_grad=True)
        for l in [4, 9, 14, 19, 24, 29]
    }
    A_s = {
        l: torch.zeros(1, 1, 4, 4).softmax(-1) for l in [4, 9, 14, 19, 24, 29]
    }
    trainer.student.b1_forward.return_value = (
        torch.zeros(1, 4, 4, requires_grad=True),
        {"h_out": h_s, "A_blk": A_s},
    )

    # vae_decoder returns rgb; lpips_net returns scalar
    trainer.vae_decoder.return_value = torch.zeros(1, 3, 16, 16)
    trainer.lpips_net.return_value = torch.tensor(0.1)

    # Mock prepare_batch and any helpers
    trainer.prepare_batch = MagicMock(
        return_value=(
            [torch.zeros(1, 8, 4)],  # LR_latents
            torch.zeros(1, 4, 4),  # z_t
            torch.tensor(999),  # t_star
            torch.zeros(1, 3, 64, 64),  # gt_hr
        )
    )

    L, ld = trainer.compute_loss({}, step=0)
    for k in ("out", "lpips", "block", "attn_out"):
        assert k in ld


def test_compute_loss_skips_block_for_lswa():
    trainer = _make_trainer()
    trainer.teacher = MagicMock()
    trainer.student = MagicMock()
    trainer.student.attn_mode = "LSWA"
    trainer.vae_decoder = MagicMock()
    trainer.lpips_net = MagicMock()

    h_t = {l: torch.zeros(1, 8, 4) for l in [4, 9, 14, 19, 24, 29]}
    trainer.teacher.b1_forward.return_value = (torch.zeros(1, 4, 4), {"h_out": h_t})

    h_s = {
        l: torch.zeros(1, 8, 4, requires_grad=True)
        for l in [4, 9, 14, 19, 24, 29]
    }
    trainer.student.b1_forward.return_value = (
        torch.zeros(1, 4, 4, requires_grad=True),
        {"h_out": h_s},
    )

    trainer.vae_decoder.return_value = torch.zeros(1, 3, 16, 16)
    trainer.lpips_net.return_value = torch.tensor(0.1)

    trainer.prepare_batch = MagicMock(
        return_value=(
            [torch.zeros(1, 8, 4)],
            torch.zeros(1, 4, 4),
            torch.tensor(999),
            torch.zeros(1, 3, 64, 64),
        )
    )

    L, ld = trainer.compute_loss({}, step=0)
    assert "block" not in ld
    assert "attn_out" in ld


def test_compute_loss_set_current_sparsity_called_for_bsa_only():
    """set_current_sparsity must be invoked when student.attn_mode == 'BSA'.
    Must NOT be invoked when attn_mode == 'LSWA'."""
    from flashvsr_b1.attn import sparsity_schedule as ss

    with patch.object(ss, "set_current_sparsity") as mock_set:
        # BSA call
        trainer_bsa = _make_trainer()
        trainer_bsa.teacher = MagicMock()
        trainer_bsa.student = MagicMock()
        trainer_bsa.student.attn_mode = "BSA"
        trainer_bsa.vae_decoder = MagicMock()
        trainer_bsa.lpips_net = MagicMock()
        h_t = {l: torch.zeros(1, 8, 4) for l in [4, 9, 14, 19, 24, 29]}
        A_t = {l: torch.zeros(1, 1, 4, 4) for l in [4, 9, 14, 19, 24, 29]}
        trainer_bsa.teacher.b1_forward.return_value = (
            torch.zeros(1, 4, 4),
            {"h_out": h_t, "A_blk": A_t},
        )
        h_s = {
            l: torch.zeros(1, 8, 4, requires_grad=True)
            for l in [4, 9, 14, 19, 24, 29]
        }
        A_s = {
            l: torch.zeros(1, 1, 4, 4).softmax(-1)
            for l in [4, 9, 14, 19, 24, 29]
        }
        trainer_bsa.student.b1_forward.return_value = (
            torch.zeros(1, 4, 4, requires_grad=True),
            {"h_out": h_s, "A_blk": A_s},
        )
        trainer_bsa.vae_decoder.return_value = torch.zeros(1, 3, 16, 16)
        trainer_bsa.lpips_net.return_value = torch.tensor(0.1)
        trainer_bsa.prepare_batch = MagicMock(
            return_value=(
                [torch.zeros(1, 8, 4)],
                torch.zeros(1, 4, 4),
                torch.tensor(999),
                torch.zeros(1, 3, 64, 64),
            )
        )
        trainer_bsa.compute_loss({}, step=0)
        assert mock_set.call_count == 1

    with patch.object(ss, "set_current_sparsity") as mock_set:
        # LSWA call
        trainer_lswa = _make_trainer()
        trainer_lswa.teacher = MagicMock()
        trainer_lswa.student = MagicMock()
        trainer_lswa.student.attn_mode = "LSWA"
        trainer_lswa.vae_decoder = MagicMock()
        trainer_lswa.lpips_net = MagicMock()
        h_t = {l: torch.zeros(1, 8, 4) for l in [4, 9, 14, 19, 24, 29]}
        trainer_lswa.teacher.b1_forward.return_value = (torch.zeros(1, 4, 4), {"h_out": h_t})
        h_s = {
            l: torch.zeros(1, 8, 4, requires_grad=True)
            for l in [4, 9, 14, 19, 24, 29]
        }
        trainer_lswa.student.b1_forward.return_value = (
            torch.zeros(1, 4, 4, requires_grad=True),
            {"h_out": h_s},
        )
        trainer_lswa.vae_decoder.return_value = torch.zeros(1, 3, 16, 16)
        trainer_lswa.lpips_net.return_value = torch.tensor(0.1)
        trainer_lswa.prepare_batch = MagicMock(
            return_value=(
                [torch.zeros(1, 8, 4)],
                torch.zeros(1, 4, 4),
                torch.tensor(999),
                torch.zeros(1, 3, 64, 64),
            )
        )
        trainer_lswa.compute_loss({}, step=0)
        assert mock_set.call_count == 0


def test_build_dataloader_threads_runtime_data_overrides(tmp_path, monkeypatch):
    from flashvsr_b1.train import trainer_b1

    data_cfg_path = tmp_path / "data.yaml"
    data_cfg_path.write_text(
        "\n".join(
            [
                "metadata_json_path: /dataset/scenes.json",
                "sample_json_path: /dataset/train_samples.json",
                "max_samples: 0",
                "shuffle_samples: true",
            ]
        )
    )
    captured = {}

    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self, opt):
            captured["opt"] = dict(opt)
            self.bucket_index = ["landscape", "landscape", "portrait", "portrait"]

        def __len__(self):
            return len(self.bucket_index)

        def __getitem__(self, index):
            return index

    fake_dataset_module = types.ModuleType("flashvsr_b1.data.dataset_b1")
    fake_dataset_module.DatasetB1 = FakeDataset
    monkeypatch.setitem(sys.modules, "flashvsr_b1.data.dataset_b1", fake_dataset_module)

    cfg = SimpleNamespace(
        data=SimpleNamespace(
            cfg=str(data_cfg_path),
            max_samples=16,
            shuffle_samples=False,
            num_workers=0,
        ),
        train=SimpleNamespace(per_rank_batch=1, seed=123),
    )

    trainer_b1.build_dataloader(cfg)

    assert captured["opt"]["max_samples"] == 16
    assert captured["opt"]["shuffle_samples"] is False


def test_train_main_can_skip_checkpoints_for_dry_run(tmp_path, monkeypatch):
    from flashvsr_b1.train import trainer_b1

    config_path = tmp_path / "dry_run.yaml"
    config_path.write_text(
        "\n".join(
            [
                "train:",
                "  total_steps: 1",
                "  seed: 123",
                "logging:",
                "  ckpt_every_steps: 0",
                "  save_final: false",
                "eval:",
                "  every_steps: 0",
                "data:",
                "  cfg: unused.yaml",
            ]
        )
    )
    captured = {"steps": [], "save_steps": [], "closed": False}

    class FakeMetrics:
        def close(self):
            captured["closed"] = True

    class FakeTrainer:
        def __init__(self, cfg, config_path):
            self.student = torch.nn.Linear(1, 1)
            self.metrics = FakeMetrics()

        def training_step(self, batch, step):
            captured["steps"].append(step)

        def save_checkpoint(self, step):
            captured["save_steps"].append(step)

    monkeypatch.setattr(trainer_b1, "B1Trainer", FakeTrainer)
    monkeypatch.setattr(
        trainer_b1,
        "build_optimizer_and_scheduler",
        lambda model, cfg: (object(), None),
    )
    monkeypatch.setattr(trainer_b1, "build_dataloader", lambda cfg: [{"batch": 0}])

    trainer_b1.train_main(str(config_path))

    assert captured["steps"] == [0]
    assert captured["save_steps"] == []
    assert captured["closed"] is True
