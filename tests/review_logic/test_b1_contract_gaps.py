from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn


def test_trainer_accepts_b1wanmodel_layer_aux_contract():
    """B1WanModel.forward returns aux keyed by layer index.

    B1Trainer.compute_loss must either consume that shape directly or normalize
    it before indexing losses by "h_out" / "A_blk".
    """
    from flashvsr_b1.train.trainer_b1 import B1Trainer

    layers = [4, 9, 14, 19, 24, 29]

    class FakeB1WanModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_mode = "BSA"
            self.current_sparsity = 0.85
            self.proj = nn.Linear(4, 4)

        def b1_forward(self, LR_latent, z_t, t_star, return_aux=False):
            out = self.proj(LR_latent + z_t)
            if not return_aux:
                return out
            attn = torch.full((1, 1, 2, 2), 0.5, dtype=out.dtype, device=out.device)
            return out, {layer: {"h_out": out, "A_blk": attn} for layer in layers}

    trainer = B1Trainer.__new__(B1Trainer)
    nn.Module.__init__(trainer)
    trainer.cfg = SimpleNamespace(
        target_sparsity=0.90,
        distill_layers=layers,
        attn_mode="BSA",
    )
    trainer.teacher = FakeB1WanModel().eval()
    trainer.student = FakeB1WanModel().train()
    trainer.vae_decoder = lambda x: torch.zeros(1, 3, 4, 4, dtype=x.dtype, device=x.device)
    trainer.lpips_net = lambda pred, target: torch.zeros(1, dtype=pred.dtype, device=pred.device)

    def prepare_batch(batch):
        LR_latent = torch.randn(1, 2, 4)
        z_t = torch.randn_like(LR_latent)
        gt_hr = torch.zeros(1, 3, 4, 4)
        return LR_latent, z_t, torch.tensor(999), gt_hr

    trainer.prepare_batch = prepare_batch

    loss, loss_dict = trainer.compute_loss({}, step=0)
    assert torch.isfinite(loss)
    assert {"out", "lpips", "attn_out", "block"} <= set(loss_dict)


def test_bsa_block_size_is_compatible_with_flashvsr_patchified_grid():
    """The spec's block_size=(2,8,8) must divide the actual token grid used by BSA."""
    from flashvsr_b1.models.flashvsr_components import FlashVSRTinyConfig

    latent_grid = (22, 64, 120)
    patch_size = FlashVSRTinyConfig.default().patch_size
    token_grid = tuple(size // patch for size, patch in zip(latent_grid, patch_size))
    block_size = (2, 8, 8)

    assert all(size % block == 0 for size, block in zip(token_grid, block_size)), (
        f"BSA sees token_grid={token_grid} after patch_size={patch_size}, "
        f"which is incompatible with block_size={block_size}."
    )


def test_prepare_batch_outputs_channels_accepted_by_wan_patch_embedding():
    """prepare_batch should not feed C_lq=1536 tensors into a Wan DiT expecting 16 channels."""
    from flashvsr_b1.models.flashvsr_components import FlashVSRTinyConfig
    from flashvsr_b1.pipelines.b1_pipeline import B1Pipeline

    cfg = FlashVSRTinyConfig.default()

    class FakeLQProj(nn.Module):
        def forward(self, lr_rgb):
            return torch.zeros(lr_rgb.shape[0], cfg.dim, 1)

    class FakeDit(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_dim = cfg.in_dim

    pipe = B1Pipeline.__new__(B1Pipeline)
    pipe.dit = FakeDit()
    pipe.lq_proj = FakeLQProj()
    pipe.cfg_single_step_t = 999

    batch = {
        "lr": torch.zeros(1, 3, 1, 16, 16),
        "hr": torch.zeros(1, 3, 1, 16, 16),
        "latent_shape": (1, 1, 1),
    }
    lr_latent, z_t, _, _ = B1Pipeline.prepare_batch(pipe, batch)

    assert lr_latent.shape[1] == pipe.dit.in_dim, (
        f"prepare_batch produced {lr_latent.shape[1]} channels and z_t {z_t.shape[1]} "
        f"channels, but Wan patch_embedding expects in_dim={pipe.dit.in_dim}."
    )


def test_bucket_sampler_keeps_bucket_choice_synchronized_across_ranks():
    """At the same DDP step, all ranks must receive the same aspect bucket."""
    from flashvsr_b1.data.bucket_sampler import AspectRatioBucketSampler

    class FakeDataset:
        bucket_index = ["landscape"] * 8 + ["portrait"] * 8

        def __len__(self):
            return len(self.bucket_index)

    dataset = FakeDataset()
    per_rank_batches = []
    for rank in range(2):
        sampler = AspectRatioBucketSampler(
            dataset,
            num_replicas=2,
            rank=rank,
            batch_size=4,
            seed=0,
            drop_last=True,
        )
        indices = list(iter(sampler))
        per_rank_batches.append(
            [
                {dataset.bucket_index[idx] for idx in indices[start : start + 4]}.pop()
                for start in range(0, len(indices), 4)
            ]
        )

    for step, (rank0_bucket, rank1_bucket) in enumerate(zip(*per_rank_batches)):
        assert rank0_bucket == rank1_bucket, (
            f"DDP step {step} uses different buckets across ranks: "
            f"rank0={rank0_bucket}, rank1={rank1_bucket}."
        )
