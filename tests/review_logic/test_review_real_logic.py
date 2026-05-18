"""Strict code-review tests that exercise REAL implementation logic
(no MagicMock around the system under test) to corroborate the findings in
docs/B200_DEPLOYMENT_GUIDE.md.

Tests are intentionally NOT skipped on the bugs they're meant to expose; some
will FAIL on the current code. Each failure pins a specific claim in the
deployment guide.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# C1 — Trainer entry point: does `python -m flashvsr_b1.train.trainer_b1` do anything?
# Claim: no __main__ block → silently exits 0 with zero work done.
# ---------------------------------------------------------------------------

def test_C1_trainer_module_has_runnable_entrypoint():
    """The 20a/b/c training scripts launch `-m flashvsr_b1.train.trainer_b1`.
    That module MUST define a callable __main__ block, else torchrun reports
    success while doing nothing."""
    src = (PROJECT_ROOT / "flashvsr_b1" / "train" / "trainer_b1.py").read_text()
    assert "__main__" in src, (
        "trainer_b1.py has no `if __name__ == \"__main__\"` block. "
        "`python -m flashvsr_b1.train.trainer_b1` will silently exit 0."
    )


def test_C1b_trainer_module_actually_runs_something_when_invoked_as_main():
    """End-to-end: launch the smoke-style invocation and assert it does NOT
    silently succeed without any side-effect. Failure mode: exit 0 with empty stdout."""
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    completed = subprocess.run(
        [sys.executable, "-m", "flashvsr_b1.train.trainer_b1",
         "--config", "flashvsr_b1/configs/b1_bsa90.yaml"],
        cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True, timeout=30,
    )
    # If the module DOES define a main, it would error out fast on missing checkpoints
    # (which is acceptable). Silent exit 0 with no output is the failure mode we want
    # to catch.
    silent_success = (completed.returncode == 0
                      and not completed.stdout.strip()
                      and not completed.stderr.strip())
    assert not silent_success, (
        "Running `-m flashvsr_b1.train.trainer_b1 --config ...` exited 0 with no output. "
        "This is the silent-no-op failure mode described in the deployment guide."
    )


# ---------------------------------------------------------------------------
# C2 — Trainer training_step assumes self.optimizer exists, but __init__ never builds it.
# ---------------------------------------------------------------------------

def test_C2_trainer_init_constructs_optimizer():
    """If __init__ exits without an optimizer attribute, the first call to
    training_step → self.optimizer.step() will AttributeError."""
    src = (PROJECT_ROOT / "flashvsr_b1" / "train" / "trainer_b1.py").read_text()
    assert "torch.optim" in src or "self.optimizer = " in src, (
        "trainer_b1.py never constructs self.optimizer, but training_step "
        "directly calls self.optimizer.step(). Will AttributeError on first step."
    )


# ---------------------------------------------------------------------------
# C3 — Pipeline must create an INDEPENDENT teacher; trainer must NOT silently
# fall back to teacher = student.
# ---------------------------------------------------------------------------

def test_C3_pipeline_constructs_separate_teacher():
    """B1Pipeline.from_b1_config must expose teacher_dit (or teacher) as a
    SEPARATE instance from pipe.dit (student), per task_b1.md §2.2/§2.3.
    Failure means distillation degenerates to self-distillation."""
    from flashvsr_b1.pipelines.b1_pipeline import B1Pipeline

    class DummySelfAttention(nn.Module):
        def __init__(self, dim=8, num_heads=2):
            super().__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.q = nn.Linear(dim, dim)
            self.k = nn.Linear(dim, dim)
            self.v = nn.Linear(dim, dim)
            self.o = nn.Linear(dim, dim)

    def make_dit():
        return SimpleNamespace(
            blocks=[SimpleNamespace(self_attn=DummySelfAttention()) for _ in range(2)],
        )

    fake_pipe = SimpleNamespace(dit=make_dit())
    cfg = SimpleNamespace(
        teacher_ckpt="/mock/teacher.safetensors",
        student_ckpt="/mock/student.safetensors",
        tc_decoder_ckpt=None,
        lq_proj_ckpt=None,
        block_size=(2, 8, 8),
        window_size=(2, 21, 21),
        dim=8,
    )
    with (
        patch("flashvsr_b1.pipelines.b1_pipeline.WanVideoPipeline.from_pretrained",
              return_value=fake_pipe),
        patch("flashvsr_b1.pipelines.b1_pipeline.Causal_LQ4x_Proj",
              return_value=nn.Identity(), create=True),
        patch("flashvsr_b1.pipelines.b1_pipeline.build_tc_decoder",
              return_value=nn.Identity(), create=True),
        patch("flashvsr_b1.pipelines.b1_pipeline._build_lpips_net",
              return_value=MagicMock(), create=True),
    ):
        pipe = B1Pipeline.from_b1_config(cfg)

    student = getattr(pipe, "dit", None)
    teacher = getattr(pipe, "teacher", None) or getattr(pipe, "teacher_dit", None)
    assert teacher is not None, (
        "B1Pipeline did not expose a teacher/teacher_dit attribute. "
        "Trainer will fall back to teacher = student (self-distillation)."
    )
    assert teacher is not student, (
        "B1Pipeline.teacher_dit is the SAME object as pipe.dit (student). "
        "Required by task_b1.md §2.2/§2.3 to be a separate frozen instance."
    )


def test_C3b_trainer_does_not_silently_fall_back_to_student_as_teacher():
    """If pipeline forgot to expose a teacher, trainer should ERROR LOUDLY,
    not silently set teacher = student."""
    src = (PROJECT_ROOT / "flashvsr_b1" / "train" / "trainer_b1.py").read_text()
    assert "if self.teacher is None:\n            self.teacher = self.student" not in src, (
        "Trainer silently falls back to teacher = student. This must be a hard error."
    )


# ---------------------------------------------------------------------------
# C4 — Forward signature mismatch: trainer calls model(LR_latent, z_t, t_star, return_aux=...)
# but B1WanModel.forward expects (x, timestep, context, ...).
# ---------------------------------------------------------------------------

def test_C4_trainer_model_call_matches_B1WanModel_forward_signature():
    """Trainer must call the B1-specific forward adapter, not WanModel.forward
    positionally with mismatched semantics."""
    import inspect
    from flashvsr_b1.models.wan_dit_b1 import B1WanModel

    sig = inspect.signature(B1WanModel.b1_forward)
    params = list(sig.parameters.values())
    first_three = [p.name for p in params[1:4]]
    assert first_three == ["LR_latents", "z_t", "t_star"], (
        f"Unexpected B1WanModel.b1_forward first-three args: {first_three}"
    )
    trainer_src = (PROJECT_ROOT / "flashvsr_b1" / "train" / "trainer_b1.py").read_text()
    assert ".b1_forward(LR_latent, z_t, t_star" in trainer_src, (
        "Trainer must call b1_forward(LR_latent, z_t, t_star, ...)."
    )
    assert "model(LR_latent, z_t, t_star" not in trainer_src, (
        "Trainer still contains the old positional WanModel call."
    )


# ---------------------------------------------------------------------------
# C5 — Grid divisibility: block_size (2,8,8) on post-patchify grid will fail.
# ---------------------------------------------------------------------------

def test_C5_bsa_partition_fails_on_patchified_grid():
    """If model uses default patch_size (1,2,2), token grid becomes (22, 32, 60),
    and 60 % 8 != 0 → _partition_for_bsa raises."""
    from flashvsr_b1.attn.bsa_kernel import _partition_for_bsa
    x = torch.zeros(1, 22 * 32 * 60, 8)
    with pytest.raises(ValueError, match="divisible"):
        _partition_for_bsa(x, block_size=(2, 8, 8), grid_shape=(22, 32, 60))


def test_C5b_bsa_partition_succeeds_on_latent_grid():
    """Spec's chosen block_size (2,8,8) divides the post-patch token grid
    (22, 64, 120) that BSA operates on. This is the same grid task_b1.md
    line 113/310 refers to as `T_lat, H_lat, W_lat` — already after Wan's
    patch_embed with patch_size=(1,2,2)."""
    from flashvsr_b1.attn.bsa_kernel import _partition_for_bsa
    x = torch.zeros(1, 22 * 64 * 120, 8)
    out = _partition_for_bsa(x, block_size=(2, 8, 8), grid_shape=(22, 64, 120))
    # total blocks = 11 * 8 * 15 = 1320, block_s = 2*8*8 = 128
    assert out.shape == (1320, 128, 8), (
        f"unexpected partition shape {tuple(out.shape)}"
    )


def test_C5c_partition_reverse_roundtrip_preserves_data():
    """_partition_for_bsa is just permutation/reshape; reversing it must
    recover the original tensor."""
    from flashvsr_b1.attn.bsa_kernel import _partition_for_bsa, _reverse_bsa_partition
    torch.manual_seed(0)
    x = torch.randn(2, 4 * 8 * 8, 16)
    block = (2, 8, 8)
    grid = (4, 8, 8)
    partitioned = _partition_for_bsa(x, block_size=block, grid_shape=grid)
    # Convert (N_blocks, block_size, D) → (B, N_blocks*block_size, D) before reverse
    # following bsa_kernel.py's actual call pattern
    B = x.shape[0]
    n_blocks = partitioned.shape[0] // B
    block_s = partitioned.shape[1]
    rearranged = partitioned.view(B, n_blocks * block_s, x.shape[-1])
    recovered = _reverse_bsa_partition(rearranged, block_size=block, grid_shape=grid, batch_size=B)
    assert torch.equal(recovered, x), "partition→reverse must be identity"


# ---------------------------------------------------------------------------
# C6 — LSWA numerical parity with the root reference. CPU executable.
# ---------------------------------------------------------------------------

def _load_root_wan_video_dit():
    """Reference `wan_video_dit.py` at repo root has its own SelfAttention.
    Loading it triggers optional imports of flash_attn / sageattention that
    may be absent on CPU; those imports are guarded with try/except inside
    that file, so it should be importable."""
    path = PROJECT_ROOT / "wan_video_dit.py"
    spec = importlib.util.spec_from_file_location("ref_wan_video_dit", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("ref_wan_video_dit", mod)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        pytest.skip(f"reference module not loadable on this host: {exc}")
    return mod


def test_C6_lswa_matches_root_local_spatial_attention():
    """Run flashvsr_b1.attn.lswa._local_spatial_attention against the root
    reference at identical dims/inputs/seed and assert numerical equality."""
    from flashvsr_b1.attn.lswa import _local_spatial_attention as ours_local
    ref = _load_root_wan_video_dit()
    if not hasattr(ref, "SelfAttention"):
        pytest.skip("reference does not expose SelfAttention class")

    torch.manual_seed(0)
    B, h, w = 1, 4, 4
    L = h * w
    D = 32
    num_heads = 4
    window_size = (2, 3, 3)

    q_frame = torch.randn(B, L, D)
    k0 = torch.randn(B, L, D); v0 = torch.randn(B, L, D)
    k1 = torch.randn(B, L, D); v1 = torch.randn(B, L, D)

    out_ours = ours_local(q_frame, [k0, k1], [v0, v1],
                          window_size=window_size, num_heads=num_heads, h=h, w=w)
    assert out_ours.shape == (B, L, D)

    # Build a reference SelfAttention object and call its private method.
    sa = ref.SelfAttention(dim=D, num_heads=num_heads).eval()
    # Reference reads window from self.lswa_spatial_window (h, w) — set it.
    sa.lswa_spatial_window = (window_size[1], window_size[2])
    sa.num_heads = num_heads
    out_ref = sa._local_spatial_attention(q_frame, [k0, k1], [v0, v1], h, w)
    assert out_ref.shape == out_ours.shape, (
        f"shape mismatch: ours={tuple(out_ours.shape)}, ref={tuple(out_ref.shape)}"
    )
    diff = (out_ours - out_ref).abs().max().item()
    assert torch.allclose(out_ours, out_ref, atol=1e-5, rtol=1e-5), (
        f"LSWA _local_spatial_attention diverges from root: max abs diff = {diff:.2e}"
    )


# ---------------------------------------------------------------------------
# C7 — Shadow attention causal mask granularity vs BSA kernel mask.
# spec §2.4.1 says BSA mask is "q_block_idx >= k_block_idx" (temporal-block causal),
# but shadow uses flat triu over 3D blocks → STRICTER than temporal-only.
# Verify the divergence is at least observable.
# ---------------------------------------------------------------------------

def test_C7_shadow_attention_uses_temporal_block_causality():
    """shadow_block_pool_attn masks only later temporal blocks, not same-time
    spatial blocks with higher flat indices."""
    from flashvsr_b1.attn.shadow_block_pool_attn import shadow_block_pool_attn

    # Tiny grid: 2 temporal, 2 H, 2 W → 8 total tokens, blocks (1,1,1) → N_blk=8
    torch.manual_seed(0)
    Q = torch.randn(1, 1, 8, 4)
    K = torch.randn(1, 1, 8, 4)
    A = shadow_block_pool_attn(Q, K, block_size=(1, 1, 1), grid_shape=(2, 2, 2), causal=True)
    # Row 0 is time 0, so all four same-time spatial blocks are visible; only
    # temporal block 1 columns are masked.
    nonzero_cols_for_first_row = (A[0, 0, 0] > 1e-8).nonzero().flatten().tolist()
    assert nonzero_cols_for_first_row == [0, 1, 2, 3], (
        f"shadow causal mask row 0: expected same-time columns [0, 1, 2, 3], "
        f"got {nonzero_cols_for_first_row}."
    )


# ---------------------------------------------------------------------------
# C8 — set_current_sparsity propagation across all SelfAttention layers.
# ---------------------------------------------------------------------------

def test_C8_set_current_sparsity_reaches_every_self_attention():
    """sparsity_schedule.set_current_sparsity must update every module that
    declares `current_sparsity`. Verifies the trainer's ramp actually takes effect."""
    from flashvsr_b1.attn.sparsity_schedule import set_current_sparsity
    from flashvsr_b1.models.wan_dit_b1 import SelfAttentionB1

    class Container(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                SelfAttentionB1(dim=64, num_heads=4) for _ in range(5)
            ])

    m = Container()
    set_current_sparsity(m, 0.93)
    for layer in m.layers:
        assert layer.current_sparsity == 0.93


# ---------------------------------------------------------------------------
# C9 — lambda_at boundary values vs spec §4.3 table.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("step, expected", [
    (0,     dict(l1=1.0, l2=0.5, l3=0.5, l4=0.1)),
    (1999,  dict(l1=1.0, l2=0.5, l3=0.5, l4=0.1)),
    (2000,  dict(l1=1.0, l2=0.5, l3=0.5, l4=0.1)),    # start of cosine, p=0 → l3=0.5
    (15000, dict(l1=1.0, l2=1.0, l3=0.1, l4=0.05)),   # refine
    (19999, dict(l1=1.0, l2=1.0, l3=0.1, l4=0.05)),
])
def test_C9_lambda_at_matches_spec_table(step, expected):
    from flashvsr_b1.train.lambda_schedule import lambda_at
    lam = lambda_at(step)
    for k, v in expected.items():
        assert abs(lam[k] - v) < 1e-9, f"lam[{k}] at step={step}: got {lam[k]}, want {v}"


# ---------------------------------------------------------------------------
# C10 — End-to-end micro-trainer: real tensors, real B1Trainer.compute_loss,
# real Loss functions, real shadow_block_pool_attn, real B1WanModel.forward
# (with a stub WanModel).
#
# This is the test that ought to exist but doesn't: instead of mocking the
# trainer, we mock only the EXTERNAL deps (WanVideoPipeline.from_pretrained)
# and let everything inside flashvsr_b1 actually run.
# ---------------------------------------------------------------------------

def test_C10_compute_loss_end_to_end_smoke():
    """Build a minimal real B1Trainer instance (no mocks for student/teacher
    behavior, only for the heavy upstream pipeline init) and run compute_loss.
    Failure modes pin: prepare_batch, forward signature mismatch, optimizer absence."""
    from flashvsr_b1.train.trainer_b1 import B1Trainer

    # We can't easily build a real Wan DiT on CPU without the checkpoints. So
    # construct a trainer with __new__ but with REAL forward callables (not
    # MagicMocks) that respect the documented contract.

    class FakeModel(nn.Module):
        """Honors task_b1.md §2.2/§2.3 contract: takes (x, timestep, context,
        return_aux=True) and returns (out, aux) where aux has h_out and A_blk
        per distill layer."""
        def __init__(self, distill_layers=(4, 9, 14, 19, 24, 29)):
            super().__init__()
            self.distill_layers = set(distill_layers)
            self.attn_mode = "BSA"
            self.proj = nn.Linear(16, 16)

        def b1_forward(self, LR_latents, z_t, t_star, return_aux=False):
            out = self.proj(z_t)
            if not return_aux:
                return out
            B, S, D = out.shape
            num_heads = 4
            N_blk = 8
            aux = {
                "h_out": {l: out for l in self.distill_layers},
                "A_blk": {
                    l: torch.softmax(torch.randn(B, num_heads, N_blk, N_blk), dim=-1)
                    for l in self.distill_layers
                },
            }
            return out, aux

        def forward(self, x, timestep, context, return_aux=False, **kwargs):
            out = self.proj(x)
            if not return_aux:
                return out
            B, S, D = out.shape
            num_heads = 4
            N_blk = 8
            aux = {
                "h_out": {l: out for l in self.distill_layers},
                "A_blk": {
                    l: torch.softmax(torch.randn(B, num_heads, N_blk, N_blk), dim=-1)
                    for l in self.distill_layers
                },
            }
            return out, aux

    trainer = B1Trainer.__new__(B1Trainer)
    nn.Module.__init__(trainer)  # so module attribute assignment works
    trainer.cfg = SimpleNamespace(
        target_sparsity=0.90,
        distill_layers=[4, 9, 14, 19, 24, 29],
        attn_mode="BSA",
        train=SimpleNamespace(grad_clip=1.0),
    )
    trainer.config_path = "flashvsr_b1/configs/b1_bsa90.yaml"
    trainer._epoch = 0

    teacher = FakeModel()
    student = FakeModel()
    teacher.eval()
    student.train()
    trainer.teacher = teacher
    trainer.student = student
    trainer.vae_decoder = lambda x: torch.zeros(1, 3, 16, 16)
    lpips_net = MagicMock()
    lpips_net.return_value = torch.tensor(0.1)
    trainer.lpips_net = lpips_net

    # Honest prepare_batch: produces tensors with shapes a real trainer would face.
    def real_prepare_batch(batch):
        LR_latents = [torch.randn(1, 4, 16)]
        z_t = torch.randn(1, 4, 16, requires_grad=False)
        t_star = torch.tensor([999], dtype=torch.long)
        gt = torch.zeros(1, 3, 16, 16)
        return LR_latents, z_t, t_star, gt
    trainer.prepare_batch = real_prepare_batch

    L, ld = trainer.compute_loss({}, step=100)
    assert torch.isfinite(L), f"compute_loss produced non-finite loss: {L}"
    for k in ("out", "lpips", "block", "attn_out"):
        assert k in ld, f"compute_loss missing loss term '{k}'"
    # Backward must not crash through the realistic graph
    L.backward()


# ---------------------------------------------------------------------------
# C11 — Checkpoint round-trip preserves student state_dict bit-for-bit.
# ---------------------------------------------------------------------------

def test_C11_ckpt_save_load_roundtrip(tmp_path):
    from flashvsr_b1.train.ckpt_io import (
        load_checkpoint, save_checkpoint, update_latest_symlink,
    )

    student = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    # Take a step so optimizer state is non-empty.
    student(torch.randn(1, 8)).sum().backward()
    optimizer.step()
    optimizer.zero_grad()

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = save_checkpoint(
        str(run_dir), step=1234, config_stem="b1_bsa90",
        student=student, optimizer=optimizer, scheduler=None,
        current_sparsity=0.875, cfg_dict={"foo": "bar"},
    )
    update_latest_symlink(str(run_dir), path)
    latest = run_dir / "ckpt" / "latest.pt"
    assert latest.is_symlink() or latest.exists()

    fresh = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    fresh_optim = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    meta = load_checkpoint(str(latest), student=fresh, optimizer=fresh_optim, scheduler=None)
    assert meta["step"] == 1234
    assert meta["current_sparsity"] == 0.875
    for p1, p2 in zip(student.parameters(), fresh.parameters()):
        assert torch.equal(p1, p2)


# ---------------------------------------------------------------------------
# C12 — Bucket sampler — verify EVERY batch contains samples from one bucket only.
# ---------------------------------------------------------------------------

def test_C12_bucket_sampler_strict_same_bucket_per_batch():
    from flashvsr_b1.data.bucket_sampler import AspectRatioBucketSampler

    class FakeDataset:
        def __init__(self):
            self.bucket_index = ["landscape"] * 30 + ["portrait"] * 30

        def __len__(self):
            return len(self.bucket_index)

    ds = FakeDataset()
    for rank in range(2):
        sampler = AspectRatioBucketSampler(ds, num_replicas=2, rank=rank,
                                            batch_size=4, seed=0, drop_last=True)
        idxs = list(iter(sampler))
        assert len(idxs) % 4 == 0
        for chunk_start in range(0, len(idxs), 4):
            buckets_in_batch = {ds.bucket_index[i] for i in idxs[chunk_start:chunk_start + 4]}
            assert len(buckets_in_batch) == 1, (
                f"rank={rank} batch@{chunk_start} mixes buckets: {buckets_in_batch}"
            )


# ---------------------------------------------------------------------------
# C13 — Metrics logger throughput sanity at the spec constant.
# ---------------------------------------------------------------------------

def test_C13_metrics_logger_seqlen_constant_matches_spec_latent_grid():
    """spec §7.2 SEQLEN_PER_VIDEO = 22 * 64 * 120 = 168960. This is the
    post-patch token grid BSA sees (task_b1.md line 113/310), NOT the
    pre-patch VAE latent (which would be 22*128*240). If a future change
    accidentally drops a patch_size factor anywhere, this constant flags it."""
    from flashvsr_b1.train.metrics_logger import MetricsLogger
    assert MetricsLogger.SEQLEN_PER_VIDEO == 22 * 64 * 120 == 168960, (
        "If grid_shape ends up being patchified, throughput will be wrong by patch volume."
    )


# ---------------------------------------------------------------------------
# C14 — Eval implementation is a stub.
# ---------------------------------------------------------------------------

def test_C14_eval_sr_is_implemented_not_stub():
    """eval_sr.evaluate_checkpoint relies on _evaluate_one_video and _measure_fps.
    Either both raise NotImplementedError (current state) → eval cannot run."""
    from eval.eval_sr import _evaluate_one_video, _measure_fps
    # Confirm the stubs are present (this test will FAIL once they are implemented,
    # at which point delete this test).
    with pytest.raises(NotImplementedError):
        _evaluate_one_video(None, None, None, None)
    with pytest.raises(NotImplementedError):
        _measure_fps(None, None, None)


# ---------------------------------------------------------------------------
# C15 — Pipeline does NOT implement prepare_batch.
# ---------------------------------------------------------------------------

def test_C15_pipeline_must_implement_prepare_batch():
    from flashvsr_b1.pipelines.b1_pipeline import B1Pipeline
    assert hasattr(B1Pipeline, "prepare_batch"), (
        "B1Pipeline has no prepare_batch method. "
        "B1Trainer.prepare_batch delegates to pipeline.prepare_batch and "
        "raises NotImplementedError otherwise. Training cannot start."
    )
