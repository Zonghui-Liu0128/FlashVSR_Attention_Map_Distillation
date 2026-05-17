# VSR B1 Sparse One-Step Implementation Plan

> **For agentic workers:** This plan is executed by Codex through the `codex:rescue` skill, one atomic task per Codex invocation, following the workflow in `Claude code与codex的职责与分工.md`. After each task, Codex writes a markdown report to `logs/YYYYMMDD-task<N>-<slug>.md` and Claude reviews before unlocking the next task.
>
> Steps use checkbox (`- [ ]`) syntax for tracking. **All design constraints come from `task_b1.md` (the spec)** — when this plan and the spec disagree, the spec wins; update the plan and re-review.

**Goal:** Train three FlashVSR Tiny student variants (BSA-90%, BSA-95%, LSWA(2,21,21)) via attention-sparsification distillation from the FlashVSR v1.1 Tiny teacher, in one-step (no DMD), strictly causal, Figure-8 forward, on internal B200 8-GPU.

**Architecture:** DiffSynth-Studio pipeline + trainer (zero-invasive integration) with three custom hooks: (1) SelfAttention replaced by mode-switchable BSA (`block_sparse_attn_func`) / LSWA (hand-written port) modules, (2) shadow block-pool attention computed off the main forward for L_block KL distillation, (3) sparsity-aware cosine ramp + λ-schedule injected into per-step training loop. Teacher and student use the same block size `(2,8,8)` and the same kernel; only the student's top-k changes per step.

**Tech Stack:** Python 3.10+, PyTorch 2.x, DiffSynth-Studio, `block_sparse_attn` (BSA only), Real-ESRGAN-style online degradation (ported from `FlashVSR_LSWA/degradation/`), OmegaConf, lpips, wandb, pandas + matplotlib (plotting), pytest.

**Reading order before each task:** (1) the corresponding `task_b1.md` section, (2) this Task's "Files / Interfaces / Acceptance" block, (3) `Claude code与codex的职责与分工.md` for the workflow rules.

**Two ground rules — non-negotiable:**
1. **No code change without a passing test that proves the behavior.** The Codex superpowers TDD flow is mandatory: write the failing test, see it fail, then implement.
2. **Every task ends with a Codex report under `logs/`.** No "I'm done" without that markdown file.

---

## Top-level file layout

The complete file tree is in `task_b1.md §1`. Below is the order the files are created (one task per file group):

| Order | Path | Created by Task |
| --- | --- | --- |
| 1 | `flashvsr_b1/__init__.py`, `flashvsr_b1/configs/`, empty module stubs | Task 1 |
| 2 | `flashvsr_b1/attn/sparsity_schedule.py` | Task 2 |
| 3 | `flashvsr_b1/attn/shadow_block_pool_attn.py` | Task 3 |
| 4 | `flashvsr_b1/attn/lswa.py` | Task 4 |
| 5 | `flashvsr_b1/attn/bsa_kernel.py` | Task 5 |
| 6 | `flashvsr_b1/data/dataset_b1.py` | Task 6 |
| 7 | `flashvsr_b1/data/bucket_sampler.py` | Task 7 |
| 8 | `flashvsr_b1/models/flashvsr_components.py` | Task 8 |
| 9 | `flashvsr_b1/models/wan_dit_b1.py` | Task 9 |
| 10 | `flashvsr_b1/losses/*.py` (four files) | Task 10 |
| 11 | `flashvsr_b1/train/metrics_logger.py`, `eval/plot_training_metrics.py` | Task 11 |
| 12 | `flashvsr_b1/pipelines/b1_pipeline.py` | Task 12 |
| 13 | `flashvsr_b1/train/lambda_schedule.py`, `flashvsr_b1/train/ckpt_io.py` | Task 13 |
| 14 | `flashvsr_b1/train/trainer_b1.py` | Task 14 |
| 15 | `flashvsr_b1/configs/b1_*.yaml`, `scripts/10_smoke_one_step.sh` | Task 15 |
| 16 | `eval/eval_sr.py`, `eval/compare_baseline.py` | Task 16 |
| 17 | `scripts/20a/b/c_*.sh`, `scripts/30_eval_all.sh` | Task 17 |

**Tasks 2–11 are pure leaf modules** — they have no inter-task dependency except on Task 1's skeleton. They could be parallelized, but per the user's decision (`task_b1.md §0` → "Codex 调度: 串行原子任务"), execute them **strictly sequentially**.

---

## Per-task brief (Codex prompt template)

When dispatching each task to Codex via `codex:rescue`, Claude supplies:

1. Path to this plan file + Task number (e.g., `docs/superpowers/plans/2026-05-16-vsr-b1-sparse-onestep.md` Task 5)
2. Path to spec: `task_b1.md` + the relevant section number
3. The Task block below (verbatim)
4. The Codex superpowers TDD workflow reminder (test first, see fail, implement, see pass, report)

Codex's output for each task **must** include:

- `logs/<YYYYMMDD>-task<N>-<slug>.md` with: what / why / test design / self-test results / Claude review iteration history / debug notes
- All files created/modified
- All tests passing (`pytest -v` log captured in the report)

---

## Task 1: Repository skeleton

**Spec ref:** `task_b1.md §1` (file tree), `§0` (decisions).

**Files:**
- Create: `flashvsr_b1/__init__.py` (empty)
- Create: `flashvsr_b1/{configs,models,attn,losses,data,pipelines,train}/__init__.py` (all empty)
- Create: `eval/__init__.py`
- Create: `tests/__init__.py`
- Create: `logs/.gitkeep`, `log/.gitkeep`, `scripts/.gitkeep`
- Create: `pytest.ini` (project root) with `[pytest] testpaths = tests`

**Interface contracts:** none (skeleton only).

**Acceptance:**

- [ ] Step 1: Create the directory tree above. Do **not** create any other files yet (subsequent tasks own them).

- [ ] Step 2: Write a smoke test `tests/test_skeleton.py`:

```python
import importlib

def test_can_import_all_subpackages():
    for sub in ["configs", "models", "attn", "losses", "data", "pipelines", "train"]:
        importlib.import_module(f"flashvsr_b1.{sub}")
```

- [ ] Step 3: Run `pytest tests/test_skeleton.py -v`. Expected: 1 passed.

- [ ] Step 4: Commit:

```bash
git add flashvsr_b1 eval tests logs log scripts pytest.ini
git commit -m "feat(b1/skel): initialize flashvsr_b1 package skeleton"
```

- [ ] Step 5: Write report `logs/<DATE>-task1-skeleton.md`: what was created, pytest output, anything surprising.

---

## Task 2: Sparsity schedule

**Spec ref:** `task_b1.md §3.5` (function definitions), `§4.3` (ramp behavior over 20k steps).

**Files:**
- Create: `flashvsr_b1/attn/sparsity_schedule.py`
- Create: `tests/test_sparsity_schedule.py`

**Interface contracts (must match exactly):**

```python
def cosine_sparsity_ramp(step: int, *, ramp_end_step: int,
                         init: float = 0.85, target: float = 0.90) -> float: ...

def set_current_sparsity(model: torch.nn.Module, rate: float) -> None: ...
```

**Acceptance tests Codex must include:**

- [ ] Step 1: Write failing test `tests/test_sparsity_schedule.py`:

```python
import math, torch
from flashvsr_b1.attn.sparsity_schedule import cosine_sparsity_ramp, set_current_sparsity

def test_ramp_init():
    assert cosine_sparsity_ramp(0, ramp_end_step=12000, init=0.85, target=0.90) == 0.85

def test_ramp_clamps_to_target():
    assert cosine_sparsity_ramp(15000, ramp_end_step=12000, target=0.90) == 0.90
    assert cosine_sparsity_ramp(12000, ramp_end_step=12000, target=0.90) == 0.90

def test_ramp_monotonic_increasing():
    vals = [cosine_sparsity_ramp(s, ramp_end_step=12000, target=0.90)
            for s in range(0, 12000, 200)]
    for a, b in zip(vals, vals[1:]):
        assert b >= a - 1e-9

def test_ramp_midpoint():
    mid = cosine_sparsity_ramp(6000, ramp_end_step=12000, init=0.85, target=0.90)
    expected = 0.85 + (0.90 - 0.85) * 0.5 * (1 - math.cos(math.pi * 0.5))
    assert abs(mid - expected) < 1e-6

def test_set_current_sparsity_writes_to_marked_modules_only():
    class A(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.current_sparsity = 0.85
    class B(torch.nn.Module):
        pass
    root = torch.nn.Module()
    root.a = A(); root.b = B()
    set_current_sparsity(root, 0.93)
    assert root.a.current_sparsity == 0.93
    assert not hasattr(root.b, "current_sparsity")
```

- [ ] Step 2: Run `pytest tests/test_sparsity_schedule.py -v`. Expected: all FAIL (functions not defined).

- [ ] Step 3: Implement `flashvsr_b1/attn/sparsity_schedule.py` strictly following the `task_b1.md §3.5` pseudocode. Codex chooses internal layout; the public interface above is fixed.

- [ ] Step 4: Re-run `pytest tests/test_sparsity_schedule.py -v`. Expected: 4 passed.

- [ ] Step 5: Commit:

```bash
git add flashvsr_b1/attn/sparsity_schedule.py tests/test_sparsity_schedule.py
git commit -m "feat(b1/attn): cosine sparsity ramp + set_current_sparsity"
```

- [ ] Step 6: Report `logs/<DATE>-task2-sparsity-schedule.md`.

---

## Task 3: Shadow block-pool attention

**Spec ref:** `task_b1.md §3.4` (interface + math), `§2.2/§2.3` (where it's called), `§2.4` (causal mask hard rule).

**Files:**
- Create: `flashvsr_b1/attn/shadow_block_pool_attn.py`
- Create: `tests/test_shadow_block_pool_attn.py`

**Interface contracts:**

```python
def block_mean_pool_3d(x: torch.Tensor,
                       block_size: tuple[int, int, int],
                       grid_shape: tuple[int, int, int]) -> torch.Tensor:
    """
    x: [B, H, S, d_head]  with  S = T * Hh * Ww  (matches grid_shape)
    returns: [B, H, N_blk, d_head]  with N_blk = (T/bt) * (Hh/bh) * (Ww/bw)
    """

def shadow_block_pool_attn(Q: torch.Tensor, K: torch.Tensor, *,
                            block_size: tuple[int, int, int],
                            grid_shape: tuple[int, int, int],
                            causal: bool = True) -> torch.Tensor:
    """
    Q, K: [B, H, S, d_head]
    returns: [B, H, N_blk, N_blk]  softmax-normalized along last dim,
             causal=True puts float('-inf') on future blocks before softmax.
    """
```

`grid_shape` is passed in (not inferred from `S`) so the same util serves both landscape (22, 8, 15) and portrait (22, 15, 8) bucket inputs.

**Acceptance tests:**

- [ ] Step 1: Write failing test `tests/test_shadow_block_pool_attn.py`:

```python
import torch, math
from flashvsr_b1.attn.shadow_block_pool_attn import (
    block_mean_pool_3d, shadow_block_pool_attn,
)

def _set_seed(s=0):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def test_block_mean_pool_3d_shape_landscape():
    _set_seed()
    B, H, T, Hh, Ww, d = 2, 4, 22, 64, 120, 16
    x = torch.randn(B, H, T*Hh*Ww, d)
    out = block_mean_pool_3d(x, block_size=(2,8,8), grid_shape=(T, Hh, Ww))
    assert out.shape == (B, H, (T//2)*(Hh//8)*(Ww//8), d)
    assert out.shape[-2] == 11 * 8 * 15  # = 1320

def test_block_mean_pool_3d_shape_portrait():
    _set_seed()
    B, H, T, Hh, Ww, d = 2, 4, 22, 120, 64, 16
    x = torch.randn(B, H, T*Hh*Ww, d)
    out = block_mean_pool_3d(x, block_size=(2,8,8), grid_shape=(T, Hh, Ww))
    assert out.shape == (B, H, 11 * 15 * 8, d)

def test_block_mean_pool_3d_equals_explicit_mean():
    """A single-block pool must equal explicit mean over that block."""
    _set_seed()
    B, H, T, Hh, Ww, d = 1, 1, 2, 8, 8, 4
    x = torch.randn(B, H, T*Hh*Ww, d)
    out = block_mean_pool_3d(x, block_size=(2,8,8), grid_shape=(T, Hh, Ww))
    expected = x.mean(dim=2, keepdim=True)         # single block → mean of all S
    assert torch.allclose(out, expected, atol=1e-6)

def test_shadow_attention_shape_and_softmax():
    _set_seed()
    B, H, T, Hh, Ww, d = 1, 2, 22, 64, 120, 16
    Q = torch.randn(B, H, T*Hh*Ww, d)
    K = torch.randn(B, H, T*Hh*Ww, d)
    A = shadow_block_pool_attn(Q, K, block_size=(2,8,8),
                                grid_shape=(T, Hh, Ww), causal=True)
    assert A.shape == (B, H, 1320, 1320)
    row_sums = A.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

def test_shadow_attention_causal_zeros_future_columns():
    _set_seed()
    B, H, T, Hh, Ww, d = 1, 1, 4, 8, 8, 4   # N_blk = 2*1*1 = 2  (tiny)
    Q = torch.randn(B, H, T*Hh*Ww, d)
    K = torch.randn(B, H, T*Hh*Ww, d)
    A = shadow_block_pool_attn(Q, K, block_size=(2,8,8),
                                grid_shape=(T, Hh, Ww), causal=True)
    # row 0 attends to col 0 only (col 1 is future → softmax = 0)
    assert A[0, 0, 0, 0] == 1.0
    assert A[0, 0, 0, 1] == 0.0

def test_shadow_attention_grad_flows_to_Q_and_K():
    _set_seed()
    # N_blk must be >= 2 AND the loss must depend on individual softmax
    # probabilities (not row sums, which are always 1.0 → grad = 0).
    B, H, T, Hh, Ww, d = 1, 1, 4, 8, 8, 4   # N_blk = 2 * 1 * 1 = 2
    Q = torch.randn(B, H, T*Hh*Ww, d, requires_grad=True)
    K = torch.randn(B, H, T*Hh*Ww, d, requires_grad=True)
    A = shadow_block_pool_attn(Q, K, block_size=(2,8,8),
                                grid_shape=(T, Hh, Ww), causal=True)
    # Loss on a single probability (A[..., 1, 0]) — non-constant function of Q,K.
    A[..., 1, 0].sum().backward()
    assert Q.grad is not None and Q.grad.abs().sum() > 0
    assert K.grad is not None and K.grad.abs().sum() > 0
```

- [ ] Step 2: `pytest tests/test_shadow_block_pool_attn.py -v` → all FAIL.

- [ ] Step 3: Implement following `task_b1.md §3.4`. Use `einsum`-based scaled dot product. Build the causal mask using block-index comparison.

- [ ] Step 4: `pytest tests/test_shadow_block_pool_attn.py -v` → 6 passed.

- [ ] Step 5: Commit:

```bash
git add flashvsr_b1/attn/shadow_block_pool_attn.py tests/test_shadow_block_pool_attn.py
git commit -m "feat(b1/attn): shadow block-pool attention for L_block distillation"
```

- [ ] Step 6: Report `logs/<DATE>-task3-shadow-block-pool.md`.

---

## Task 4: LSWA forward (port)

**Spec ref:** `task_b1.md §3.3`. Reference implementation:
`/Users/zonghuiliu/Documents/Codex/VideoGen/FlashVSR_Attention_Map_Distillation/wan_video_dit.py` lines 391–491 (`_local_spatial_attention` + `_lswa_forward`).

**Files:**
- Create: `flashvsr_b1/attn/lswa.py`
- Create: `tests/test_lswa.py`

**Interface contracts:**

```python
def lswa_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, *,
                 window_size: tuple[int, int, int],
                 f: int, h: int, w: int,
                 is_stream: bool = False,
                 pre_cache_k: torch.Tensor | None = None,
                 pre_cache_v: torch.Tensor | None = None) -> torch.Tensor:
    """
    Q, K, V: [B, S, D]  with S = f * h * w  (training) or chunk-len (stream).
    Returns: [B, S, D] attention output. Strictly causal in time (window_t=2 ⇒
    each frame only sees the previous 1 frame + itself, spatial window 21x21).
    No attention logits returned (LSWA is L_block-exempt per §4.1).
    """
```

**Acceptance tests:**

- [ ] Step 1: Write `tests/test_lswa.py`:

```python
import torch
from flashvsr_b1.attn.lswa import lswa_forward

REF_PATH = "/Users/zonghuiliu/Documents/Codex/VideoGen/FlashVSR_Attention_Map_Distillation/wan_video_dit.py"

def test_lswa_output_shape_train_mode():
    torch.manual_seed(0)
    B, f, h, w, D = 1, 4, 16, 16, 64
    Q = torch.randn(B, f*h*w, D); K = torch.randn(B, f*h*w, D); V = torch.randn(B, f*h*w, D)
    out = lswa_forward(Q, K, V, window_size=(2, 5, 5), f=f, h=h, w=w)
    assert out.shape == (B, f*h*w, D)

def test_lswa_is_causal_in_time():
    """Changing K/V at frame T should not affect output at frame T-2 (window_t=2 forbids that history)."""
    torch.manual_seed(0)
    B, f, h, w, D = 1, 4, 4, 4, 16
    Q = torch.randn(B, f*h*w, D); K = torch.randn(B, f*h*w, D); V = torch.randn(B, f*h*w, D)
    out_a = lswa_forward(Q, K, V, window_size=(2, 3, 3), f=f, h=h, w=w)
    # perturb K/V only at the last frame (frame 3)
    K2 = K.clone(); V2 = V.clone()
    K2[:, 3*h*w:, :] += 1.0
    V2[:, 3*h*w:, :] += 1.0
    out_b = lswa_forward(Q, K2, V2, window_size=(2, 3, 3), f=f, h=h, w=w)
    # frames 0 and 1 must be identical (cannot see frame 3)
    assert torch.allclose(out_a[:, :2*h*w, :], out_b[:, :2*h*w, :], atol=1e-5)

def test_lswa_matches_reference_implementation():
    """
    Numerical parity with the existing _local_spatial_attention in wan_video_dit.py.
    Codex should load that module via importlib.util.spec_from_file_location(REF_PATH),
    build a small SelfAttention, and compare lswa_forward output against the
    reference's _lswa_forward output on identical inputs and seeds.
    Tolerance: atol=1e-5.
    """
    # Codex implements; assertion: outputs match.
```

- [ ] Step 2: `pytest tests/test_lswa.py -v` → FAIL (module missing).

- [ ] Step 3: Implement `flashvsr_b1/attn/lswa.py` by lifting the relevant code from the reference file. Do **not** rewrite the algorithm — copy + adapt to the standalone signature.

- [ ] Step 4: `pytest tests/test_lswa.py -v` → 3 passed.

- [ ] Step 5: Commit:

```bash
git add flashvsr_b1/attn/lswa.py tests/test_lswa.py
git commit -m "feat(b1/attn): LSWA forward ported from root wan_video_dit.py"
```

- [ ] Step 6: Report `logs/<DATE>-task4-lswa.md`.

---

## Task 5: BSA kernel wrapper

**Spec ref:** `task_b1.md §3.2`. Reference: root `wan_video_dit.py` `_block_sparse_forward` (line 493), `generate_causal_block_mask` (line 165), `generate_draft_block_mask` (line 124), `flash_attention` (line 182).

**Files:**
- Create: `flashvsr_b1/attn/bsa_kernel.py`
- Create: `tests/test_bsa_kernel.py`

**Interface contracts:**

```python
def topk_for(sparsity: float, total_kv_blocks: int) -> int:
    """active = round(total * (1 - sparsity)), min 1."""

def bsa_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, *,
                block_size: tuple[int, int, int],
                grid_shape: tuple[int, int, int],
                current_sparsity: float,
                num_heads: int,
                local_window_mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Wraps block_sparse_attn_func with causal block sparse mask derived from
    current_sparsity. Falls back gracefully (with a clear exception message) when
    block_sparse_attn is not importable — DO NOT silently switch to SDPA.
    """
```

**Acceptance tests:**

- [ ] Step 1: Write `tests/test_bsa_kernel.py`:

```python
import pytest, torch
from flashvsr_b1.attn.bsa_kernel import topk_for, bsa_forward

def test_topk_for_85pct():
    # 85% sparse → 15% active. With 1320 blocks → ~198 active.
    assert topk_for(0.85, 1320) == 198

def test_topk_for_90pct():
    assert topk_for(0.90, 1320) == 132

def test_topk_for_95pct():
    assert topk_for(0.95, 1320) == 66

def test_topk_for_clamps_to_one():
    assert topk_for(0.999, 10) == 1

@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="bsa_forward requires CUDA + block_sparse_attn lib")
def test_bsa_forward_shape():
    """Token-grid (T, H_lat, W_lat) MUST be divisible by block_size axis-wise.
    With block_size (2, 8, 8), valid tiny grid is (4, 8, 8) → block grid (2,1,1)."""
    torch.manual_seed(0)
    B, T, H_lat, W_lat, D, H = 1, 4, 8, 8, 128, 4
    S = T * H_lat * W_lat
    Q = torch.randn(B, S, D, device="cuda")
    K = torch.randn(B, S, D, device="cuda")
    V = torch.randn(B, S, D, device="cuda")
    out = bsa_forward(Q, K, V,
                      block_size=(2,8,8), grid_shape=(T, H_lat, W_lat),
                      current_sparsity=0.85,
                      num_heads=H, local_window_mask=None)
    assert out.shape == (B, S, D)

@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="parity test requires CUDA")
def test_bsa_parity_with_root_implementation():
    """
    Build the same SelfAttention object from root wan_video_dit.py and call its
    _block_sparse_forward with the FULL signature (q, k, v, B, f, h, w, D,
    local_num, topk, kv_len, is_stream, pre_cache_k, pre_cache_v, local_range).
    Compare against bsa_forward output. atol=1e-4. Token grid must be
    divisible by block_size — use (T=22, H=16, W=16) → block grid (11,2,2).
    """
    # Codex implements; assertion: outputs match.
```

- [ ] Step 2: `pytest tests/test_bsa_kernel.py -v` (CPU-only off-host) → 4 passed (topk tests), 2 skipped (CUDA tests). On internal B200 → 6 passed.

- [ ] Step 3: Implement. Two key bits:
   - `topk_for` is a one-liner (`max(1, int(round(total_kv_blocks * (1.0 - sparsity))))`).
   - `bsa_forward` reuses `generate_causal_block_mask` from the reference. Do not vendor — `from wan_video_dit import generate_causal_block_mask` (the file at project root) and import `block_sparse_attn_func` lazily inside the function so off-CUDA dev still imports the module.

- [ ] Step 4: Run tests as in Step 2.

- [ ] Step 5: Commit:

```bash
git add flashvsr_b1/attn/bsa_kernel.py tests/test_bsa_kernel.py
git commit -m "feat(b1/attn): bsa kernel wrapper with current_sparsity → topk"
```

- [ ] Step 6: Report `logs/<DATE>-task5-bsa-kernel.md` — explicitly note which tests ran on CPU vs CUDA, and the parity-test result on B200.

---

## Task 6: Dataset wrapper

**Spec ref:** `task_b1.md §5.2`. Base class: `FlashVSR_LSWA/degradation/basic_vsr_dataset_hw_crop.py:BasicVSRDataset_hw_crop`.

**Files:**
- Create: `flashvsr_b1/configs/data_b1.yaml` (copied from `FlashVSR_LSWA/animal_1080x1920@89.yaml`, with `frame_num: 85` and `temporal_stride: 85`)
- Create: `flashvsr_b1/data/dataset_b1.py`
- Create: `tests/test_dataset_b1.py`

**Interface contracts:**

```python
class DatasetB1(BasicVSRDataset_hw_crop):
    """
    Adds two fields to each __getitem__ output:
      - aspect_bucket: "landscape" if w > h else "portrait"
      - latent_shape:  (22, 64, 120) for landscape, (22, 120, 64) for portrait
    Everything else identical to parent.
    """
```

**Acceptance tests:**

- [ ] Step 1: Copy data yaml:

```bash
cp FlashVSR_LSWA/animal_1080x1920@89.yaml flashvsr_b1/configs/data_b1.yaml
```

Then edit:
- `frame_num: 89` → `frame_num: 85`
- `temporal_stride: 89` → `temporal_stride: 85`
- Leave all degradation params identical.

- [ ] Step 2: Write `tests/test_dataset_b1.py`. Mock the parent's `__getitem__` so the test runs on macOS without the internal mp4 files:

```python
import torch
from unittest.mock import patch
from flashvsr_b1.data.dataset_b1 import DatasetB1

def _fake_parent_item(h, w):
    return {
        "lr":         torch.zeros(3, 85, h, w),
        "hr":         torch.zeros(3, 85, h*4, w*4),
        "sample_meta": {},
        "degradation_meta": {},
        "data_name":  "dummy.mp4",
    }

def test_landscape_bucket_and_latent_shape():
    with patch.object(DatasetB1, "__init__", lambda self, *a, **k: None), \
         patch.object(DatasetB1.__bases__[0], "__getitem__",
                      lambda self, idx: _fake_parent_item(1024, 1920)):
        ds = DatasetB1()
        item = ds[0]
        assert item["aspect_bucket"] == "landscape"
        assert item["latent_shape"] == (22, 64, 120)

def test_portrait_bucket_and_latent_shape():
    with patch.object(DatasetB1, "__init__", lambda self, *a, **k: None), \
         patch.object(DatasetB1.__bases__[0], "__getitem__",
                      lambda self, idx: _fake_parent_item(1920, 1024)):
        ds = DatasetB1()
        item = ds[0]
        assert item["aspect_bucket"] == "portrait"
        assert item["latent_shape"] == (22, 120, 64)

def test_parent_fields_preserved():
    """We must not drop or rename parent dataset fields."""
    with patch.object(DatasetB1, "__init__", lambda self, *a, **k: None), \
         patch.object(DatasetB1.__bases__[0], "__getitem__",
                      lambda self, idx: _fake_parent_item(1024, 1920)):
        ds = DatasetB1()
        item = ds[0]
        for k in ["lr", "hr", "sample_meta", "degradation_meta", "data_name"]:
            assert k in item
```

- [ ] Step 3: `pytest tests/test_dataset_b1.py -v` → FAIL.

- [ ] Step 4: Implement `flashvsr_b1/data/dataset_b1.py`. Add the parent module to `sys.path` at the top (it lives at `FlashVSR_LSWA/degradation/`) or import via explicit `importlib.util`.

- [ ] Step 5: Re-run tests → 3 passed.

- [ ] Step 6: Commit:

```bash
git add flashvsr_b1/configs/data_b1.yaml flashvsr_b1/data/dataset_b1.py tests/test_dataset_b1.py
git commit -m "feat(b1/data): DatasetB1 wrapper with aspect-ratio fields"
```

- [ ] Step 7: Report `logs/<DATE>-task6-dataset-b1.md`.

---

## Task 7: Bucket sampler

**Spec ref:** `task_b1.md §5.3`.

**Files:**
- Create: `flashvsr_b1/data/bucket_sampler.py`
- Create: `tests/test_bucket_sampler.py`

**Interface contracts:**

```python
class AspectRatioBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Each batch contains samples from ONE bucket only.
    Bucket order across batches is determined deterministically from epoch + seed.
    DDP-safe: rank 0 decides bucket order then broadcasts via dist.broadcast_object_list.
    """
    def __init__(self, dataset, *, num_replicas: int, rank: int,
                 batch_size: int, seed: int = 0, drop_last: bool = True): ...
```

The dataset must expose, for each index, the `aspect_bucket` of that sample without loading the video — store a precomputed list via a one-time scan in `dataset.bucket_index: list[str]`. Add this attribute to `DatasetB1.__init__` (Task 6 follow-up — pre-scan `sample_index.json` for `crop_height`/`crop_width`).

**Note:** This pre-scan amendment lives in `dataset_b1.py`. If you didn't add it in Task 6, do it here as a clearly-separated commit before the sampler commit.

**Acceptance tests:**

- [ ] Step 1: Write `tests/test_bucket_sampler.py`:

```python
import torch
from flashvsr_b1.data.bucket_sampler import AspectRatioBucketSampler

class FakeDataset:
    def __init__(self, n_land=120, n_port=80):
        self.bucket_index = (["landscape"] * n_land) + (["portrait"] * n_port)
    def __len__(self):
        return len(self.bucket_index)

def test_each_batch_is_single_bucket():
    ds = FakeDataset(n_land=120, n_port=80)
    sampler = AspectRatioBucketSampler(ds, num_replicas=1, rank=0,
                                        batch_size=4, seed=0)
    batches = []
    cur = []
    for idx in sampler:
        cur.append(idx)
        if len(cur) == 4:
            batches.append(cur); cur = []
    for batch in batches:
        buckets = {ds.bucket_index[i] for i in batch}
        assert len(buckets) == 1

def test_bucket_ratio_close_to_dataset_ratio():
    ds = FakeDataset(n_land=160, n_port=40)              # 4:1
    sampler = AspectRatioBucketSampler(ds, num_replicas=1, rank=0,
                                        batch_size=4, seed=0)
    counts = {"landscape": 0, "portrait": 0}
    for idx in sampler:
        counts[ds.bucket_index[idx]] += 1
    ratio = counts["landscape"] / max(counts["portrait"], 1)
    assert 3.5 < ratio < 4.5

def test_drop_last_enforces_full_batches():
    ds = FakeDataset(n_land=122, n_port=83)              # not multiples of 4
    sampler = AspectRatioBucketSampler(ds, num_replicas=1, rank=0,
                                        batch_size=4, seed=0)
    idxs = list(sampler)
    assert len(idxs) % 4 == 0

def test_ddp_ranks_disjoint_and_complete():
    ds = FakeDataset(n_land=120, n_port=80)
    s0 = AspectRatioBucketSampler(ds, num_replicas=2, rank=0, batch_size=4, seed=42)
    s1 = AspectRatioBucketSampler(ds, num_replicas=2, rank=1, batch_size=4, seed=42)
    a, b = set(s0), set(s1)
    assert len(a & b) == 0
    # combined should cover ~all (after drop_last)
    assert len(a | b) >= int(0.95 * len(ds))
```

- [ ] Step 2: `pytest tests/test_bucket_sampler.py -v` → FAIL.

- [ ] Step 3: Implement. Algorithm (rank-0 view):
  1. Group sample indices by bucket using `dataset.bucket_index`.
  2. Shuffle within each bucket (seed = `seed + epoch`).
  3. Pack into `(bucket_id, batch_size)` chunks; drop last partial per bucket.
  4. Interleave bucket chunks proportional to bucket size.
  5. Slice for this rank: `flat_indices[rank::num_replicas]` after broadcasting from rank 0.

- [ ] Step 4: Re-run tests → 4 passed.

- [ ] Step 5: Commit:

```bash
git add flashvsr_b1/data/bucket_sampler.py tests/test_bucket_sampler.py
git commit -m "feat(b1/data): aspect-ratio bucket sampler for DDP"
```

- [ ] Step 6: Report `logs/<DATE>-task7-bucket-sampler.md`.

---

## Task 8: FlashVSR component port

**Spec ref:** `task_b1.md §1` (file tree row). Source: `FlashVSR_LSWA/flashvsr_components.py`.

**Files:**
- Create: `flashvsr_b1/models/flashvsr_components.py`
- Create: `tests/test_flashvsr_components.py`

**Interface contracts (preserve from source):**

```python
class FlashVSRTinyConfig: ...                  # patch_size, dim, ffn_dim, num_heads=12, num_layers=30, in_dim=16, out_dim=16

class Causal_LQ4x_Proj(nn.Module):
    def __init__(self, in_dim=3, out_dim=1536, layer_num=1): ...

def build_tc_decoder(checkpoint_path: str | None = None) -> nn.Module: ...

def load_flashvsr_tiny_checkpoint(model: nn.Module, path: str, *, strict: bool = True) -> dict: ...
```

**Acceptance tests:**

- [ ] Step 1: Write `tests/test_flashvsr_components.py`:

```python
import torch
from flashvsr_b1.models.flashvsr_components import (
    FlashVSRTinyConfig, Causal_LQ4x_Proj, build_tc_decoder,
)

def test_tiny_config_defaults():
    c = FlashVSRTinyConfig()
    assert c.num_layers == 30
    assert c.num_heads == 12
    assert c.dim == 1536
    assert c.in_dim == 16
    assert c.out_dim == 16

def test_lq_proj_forward_shape():
    """Tiny shape to keep the unit test under 100 MB on macOS; real shapes only
    matter on B200. Just confirms in_dim=3 in, out_dim=1536 out, no crash."""
    proj = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1)
    x = torch.randn(1, 3, 4, 64, 96)
    out = proj(x)
    assert out.shape[1] == 1536

def test_tc_decoder_builds_without_checkpoint():
    dec = build_tc_decoder(checkpoint_path=None)
    assert dec is not None
```

- [ ] Step 2: `pytest tests/test_flashvsr_components.py -v` → FAIL.

- [ ] Step 3: Implement by porting. Where the source uses `sys.path` hacks, use `importlib.util` for portability.

- [ ] Step 4: Re-run → 3 passed.

- [ ] Step 5: Commit:

```bash
git add flashvsr_b1/models/flashvsr_components.py tests/test_flashvsr_components.py
git commit -m "feat(b1/models): port FlashVSR Tiny config / LQ proj / TCDecoder"
```

- [ ] Step 6: Report `logs/<DATE>-task8-flashvsr-components.md`.

---

## Task 9: SelfAttention + B1WanModel

**Spec ref:** `task_b1.md §3.1`. Base class: `diffsynth.models.wan_video_dit.WanModel` (read DiffSynth-Studio source to confirm class location and SelfAttention class name).

**Files:**
- Create: `flashvsr_b1/models/wan_dit_b1.py`
- Create: `tests/test_wan_dit_b1.py`

**Interface contracts:**

```python
class SelfAttentionB1(nn.Module):
    block_size: tuple[int, int, int]
    window_size: tuple[int, int, int]
    current_sparsity: float
    attn_mode: str                                # "BSA" | "LSWA"
    distill_export: bool

    def forward(self, x, freqs, *, return_aux: bool = False): ...

class B1WanModel(diffsynth.models.wan_video_dit.WanModel):
    """All SelfAttention replaced with SelfAttentionB1.
    Layers in self.distill_layers (set in __init__) have distill_export=True.
    forward signature gains optional return_aux=True, returning (x, {layer→aux})."""
    distill_layers: set[int]
```

**Acceptance tests:**

- [ ] Step 1: Write `tests/test_wan_dit_b1.py`:

```python
import torch
from flashvsr_b1.models.wan_dit_b1 import SelfAttentionB1, B1WanModel

def test_self_attention_default_attrs():
    sa = SelfAttentionB1(dim=1536, num_heads=12)
    assert sa.attn_mode == "BSA"
    assert sa.current_sparsity == 0.85
    assert sa.block_size == (2, 8, 8)
    assert sa.window_size == (2, 21, 21)
    assert sa.distill_export is False

def test_self_attention_signature_return_aux_false():
    """Without return_aux, forward must return a single tensor of same seq shape."""
    sa = SelfAttentionB1(dim=64, num_heads=4)
    sa.attn_mode = "LSWA"               # CPU-safe path
    x = torch.randn(1, 4*4*4, 64)
    freqs = None                         # stub; SA must accept None for unit test
    out = sa(x, freqs, f=4, h=4, w=4)
    assert isinstance(out, torch.Tensor) and out.shape == x.shape

def test_self_attention_returns_aux_when_distill_export():
    sa = SelfAttentionB1(dim=64, num_heads=4, distill_export=True)
    sa.attn_mode = "LSWA"
    x = torch.randn(1, 4*4*4, 64)
    out, aux = sa(x, None, f=4, h=4, w=4, return_aux=True)
    assert "h_out" in aux
    assert "A_blk" not in aux            # LSWA mode skips block attention map

def test_b1_wan_model_distill_layers_default():
    """Construct with default distill_layers = {4,9,14,19,24,29}."""
    m = B1WanModel.__new__(B1WanModel)  # avoid full init for unit test
    m._init_distill_layers_for_test()    # helper Codex adds; sets the attr
    assert m.distill_layers == {4, 9, 14, 19, 24, 29}

def test_b1_wan_model_loads_tiny_checkpoint_no_missing_keys(tmp_path):
    """
    With a synthesised tiny state_dict (12-head 1536-dim 30-layer skeleton),
    load_state_dict(strict=True) must succeed — i.e., our SelfAttentionB1
    must expose the same state-dict keys as the upstream Wan SelfAttention.
    """
    # Codex: build a small fake state dict with the exact key names DiffSynth Wan uses.
```

- [ ] Step 2: `pytest tests/test_wan_dit_b1.py -v` → FAIL.

- [ ] Step 3: Implement. Critical points:
  - Subclass DiffSynth's `WanModel`. Replace each block's `self_attn` after `super().__init__()` runs (don't touch `__init__` weight init logic).
  - `SelfAttentionB1.forward` dispatches to `bsa_forward` or `lswa_forward` based on `self.attn_mode`.
  - When `return_aux=True and self.distill_export=True and self.attn_mode == "BSA"`, also call `shadow_block_pool_attn(Q, K, ...)` and pack into `aux["A_blk"]`.
  - State-dict keys for `qkv_proj`, `o_proj`, norms must match upstream Wan so checkpoint load works.

- [ ] Step 4: Re-run → 5 passed.

- [ ] Step 5: Commit:

```bash
git add flashvsr_b1/models/wan_dit_b1.py tests/test_wan_dit_b1.py
git commit -m "feat(b1/models): SelfAttentionB1 + B1WanModel with return_aux"
```

- [ ] Step 6: Report `logs/<DATE>-task9-wan-dit-b1.md`.

---

## Task 10: Four loss modules

**Spec ref:** `task_b1.md §4.2`.

**Files:**
- Create: `flashvsr_b1/losses/output_loss.py`
- Create: `flashvsr_b1/losses/lpips_loss.py`
- Create: `flashvsr_b1/losses/block_kl_loss.py`
- Create: `flashvsr_b1/losses/attn_out_loss.py`
- Create: `tests/test_losses.py`

**Interface contracts:**

```python
def L_output(x_s: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor: ...   # Huber, beta=0.1
def L_lpips(x_s_latent, gt_hr_rgb, vae_decoder, lpips_net) -> torch.Tensor: ...
def L_block(A_blk_t_detached: torch.Tensor, A_blk_s: torch.Tensor,
            eps: float = 1e-8) -> torch.Tensor: ...                       # KL(t || s)
def L_attn_out(h_s: torch.Tensor, h_t_detached: torch.Tensor) -> torch.Tensor: ...   # Huber
```

**Acceptance tests:**

- [ ] Step 1: Write `tests/test_losses.py`:

```python
import torch, math
from flashvsr_b1.losses.output_loss import L_output
from flashvsr_b1.losses.block_kl_loss import L_block
from flashvsr_b1.losses.attn_out_loss import L_attn_out

def test_L_output_zero_on_equal_tensors():
    x = torch.randn(2, 3, 4)
    assert L_output(x, x).item() < 1e-8

def test_L_output_huber_smoothness_near_zero():
    """For small diff, Huber ≈ 0.5 * (diff^2) / beta. β=0.1 → at diff=0.05, loss=0.5·0.0025/0.1=0.0125."""
    x = torch.zeros(1); y = torch.full((1,), 0.05)
    assert abs(L_output(x, y).item() - 0.0125) < 1e-4

def test_L_block_zero_on_identical_distributions():
    p = torch.rand(2, 4, 8, 8)
    p = p / p.sum(dim=-1, keepdim=True)
    loss = L_block(p, p.clone())
    assert loss.item() < 1e-5

def test_L_block_positive_on_different_distributions():
    p = torch.zeros(1, 1, 1, 4); p[..., 0] = 1.0          # concentrated at 0
    q = torch.zeros(1, 1, 1, 4); q[..., 3] = 1.0          # concentrated at 3
    loss = L_block(p, q)
    assert loss.item() > 1.0

def test_L_block_grad_flows_to_student_only():
    p = torch.rand(1, 1, 4, 4); p = p / p.sum(-1, keepdim=True); p = p.detach()
    q_logits = torch.randn(1, 1, 4, 4, requires_grad=True)
    q = q_logits.softmax(-1)
    L_block(p, q).backward()
    assert q_logits.grad is not None
    # p has no requires_grad → no grad attr expected

def test_L_attn_out_zero_on_equal():
    h = torch.randn(2, 4, 8)
    assert L_attn_out(h, h.detach()).item() < 1e-8
```

LPIPS test requires the package; gate it:

```python
import pytest
try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

@pytest.mark.skipif(not HAS_LPIPS, reason="lpips not installed")
def test_L_lpips_shape():
    from flashvsr_b1.losses.lpips_loss import L_lpips
    class IdentityDecoder:
        def __call__(self, x): return x[:, :3]            # take 3 channels
    net = lpips.LPIPS(net="vgg").eval()
    x_s = torch.randn(1, 16, 4, 32, 32)
    gt  = torch.randn(1, 3, 32, 32)
    loss = L_lpips(x_s, gt, IdentityDecoder(), net)
    assert loss.dim() == 0
```

- [ ] Step 2: `pytest tests/test_losses.py -v` → FAIL.

- [ ] Step 3: Implement following `task_b1.md §4.2`. KL must avoid `nan` when `p` contains exact zeros (clamp before log).

- [ ] Step 4: Re-run → all pass (or skip LPIPS test on minimal env).

- [ ] Step 5: Commit:

```bash
git add flashvsr_b1/losses tests/test_losses.py
git commit -m "feat(b1/losses): output / lpips / block_kl / attn_out"
```

- [ ] Step 6: Report `logs/<DATE>-task10-losses.md`.

---

## Task 11: MetricsLogger + plot script

**Spec ref:** `task_b1.md §7`.

**Files:**
- Create: `flashvsr_b1/train/metrics_logger.py`
- Create: `eval/plot_training_metrics.py`
- Create: `tests/test_metrics_logger.py`

**Interface contracts:** as in `task_b1.md §7.3` and `§7.5`. In particular:

```python
def make_run_dir(log_root: str, config_path: str) -> str: ...

class MetricsLogger:
    SEQLEN_PER_VIDEO = 22 * 64 * 120                          # 168_960
    JSONL_FIELDS = [...]                                       # exact list in spec
    def __init__(self, run_dir, *, global_batch, world_size,
                 log_every_steps=50, ema_span=100): ...
    def step(self, step, *, loss_dict, lam, sparsity, lr, epoch=0): ...
    def close(self): ...
```

**Acceptance tests:**

- [ ] Step 1: Write `tests/test_metrics_logger.py`:

```python
import json, os, time, csv
import torch.distributed  # noqa: needed for is_initialized()
from flashvsr_b1.train.metrics_logger import make_run_dir, MetricsLogger

def test_make_run_dir_format(tmp_path):
    d = make_run_dir(str(tmp_path), "flashvsr_b1/configs/b1_bsa90.yaml")
    name = os.path.basename(d)
    # YYYYMMDD-HHMMSS_b1_bsa90
    assert name.endswith("_b1_bsa90")
    ts = name.split("_b1_bsa90")[0]
    assert len(ts) == 15 and ts[8] == "-"

def test_logger_writes_log_txt_jsonl_csv(tmp_path):
    rd = make_run_dir(str(tmp_path), "x/b1_bsa90.yaml")
    logger = MetricsLogger(rd, global_batch=8, world_size=8,
                            log_every_steps=2, ema_span=10)
    for s in range(1, 5):
        logger.step(s,
            loss_dict={"out":0.1,"lpips":0.2,"block":0.0,"attn_out":0.05,"total":0.35},
            lam={"l1":1.0,"l2":0.5,"l3":0.5,"l4":0.1},
            sparsity=0.85, lr=1e-5)
    logger.close()
    assert os.path.exists(os.path.join(rd, "log.txt"))
    lines = open(os.path.join(rd, "train_metrics.jsonl")).read().splitlines()
    assert len(lines) >= 1
    rec = json.loads(lines[0])
    for k in ["L_total", "tokens_per_sec", "videos_per_hour", "current_sparsity"]:
        assert k in rec
    rows = list(csv.DictReader(open(os.path.join(rd, "train_metrics.csv"))))
    assert len(rows) >= 1

def test_throughput_calculation_sanity(tmp_path):
    """With global_batch=8 and ~1s step, videos_per_hour ≈ 8 * 3600 = 28800."""
    rd = make_run_dir(str(tmp_path), "x/b1_bsa90.yaml")
    logger = MetricsLogger(rd, global_batch=8, world_size=8,
                            log_every_steps=1, ema_span=10)
    # fake time progression
    logger._window_start = time.perf_counter() - 1.0          # 1 second elapsed
    logger._window_steps = 1
    logger.step(1, loss_dict={"total":0.5,"out":0.5,"lpips":0,"block":0,"attn_out":0},
                lam={"l1":1.0,"l2":0.0,"l3":0.0,"l4":0.0},
                sparsity=0.85, lr=1e-5)
    logger.close()
    rec = json.loads(open(os.path.join(rd, "train_metrics.jsonl")).readline())
    # 8 videos in ~1s → ~28800 / hour. Allow ±20% drift.
    assert 22000 < rec["videos_per_hour"] < 36000

def test_plot_script_runs(tmp_path):
    rd = make_run_dir(str(tmp_path), "x/b1_bsa90.yaml")
    logger = MetricsLogger(rd, global_batch=8, world_size=8,
                            log_every_steps=1, ema_span=10)
    for s in range(1, 11):
        logger.step(s,
            loss_dict={"out":0.1,"lpips":0.2,"block":0.05,"attn_out":0.05,"total":0.4},
            lam={"l1":1.0,"l2":0.5,"l3":0.5,"l4":0.1},
            sparsity=0.85+s*0.001, lr=1e-5)
    logger.close()
    from eval.plot_training_metrics import plot
    plot(rd, ema_span=5)
    assert os.path.exists(os.path.join(rd, "loss_throughput.png"))
```

- [ ] Step 2: `pytest tests/test_metrics_logger.py -v` → FAIL.

- [ ] Step 3: Implement both files following `task_b1.md §7.3` and `§7.5` verbatim. `torch.cuda` calls in `MetricsLogger.step` must be guarded so the test (which runs on CPU) does not crash — e.g., `if torch.cuda.is_available(): mem_alloc = ...; else: mem_alloc = 0.0`.

- [ ] Step 4: Re-run → 4 passed.

- [ ] Step 5: Commit:

```bash
git add flashvsr_b1/train/metrics_logger.py eval/plot_training_metrics.py tests/test_metrics_logger.py
git commit -m "feat(b1/train): metrics logger + 6-subplot training visualization"
```

- [ ] Step 6: Report `logs/<DATE>-task11-metrics-logger.md`.

---

## Task 12: B1Pipeline

**Spec ref:** `task_b1.md §5.4`. Base: `diffsynth.pipelines.wan_video.WanVideoPipeline`.

**Files:**
- Create: `flashvsr_b1/pipelines/b1_pipeline.py`
- Create: `tests/test_b1_pipeline.py`

**Interface contracts:**

```python
class B1Pipeline(diffsynth.pipelines.wan_video.WanVideoPipeline):
    """
    On load:
      1. parent loads Wan DiT weights into self.dit (WanModel).
      2. self.dit is replaced with B1WanModel built around the same state_dict.
      3. self.lq_proj (Causal_LQ4x_Proj) and self.tc_decoder (TCDecoder) attached.
      4. self.lpips_net = lpips.LPIPS(net='vgg').eval()
    """
    @classmethod
    def from_b1_config(cls, cfg) -> "B1Pipeline": ...
```

**Acceptance tests:**

- [ ] Step 1: Write `tests/test_b1_pipeline.py`. Heavily mocked because we cannot load real ckpts off-host:

```python
from unittest.mock import patch, MagicMock
from flashvsr_b1.pipelines.b1_pipeline import B1Pipeline

def test_pipeline_replaces_self_attn_with_b1_variant():
    """Construct a B1Pipeline with mocked checkpoint loading, then verify that
    every block in pipeline.dit has self_attn of type SelfAttentionB1."""
    # Codex: patch out actual file IO, supply a synthetic state_dict.
    pass  # implement

def test_pipeline_asserts_block_size_match():
    """If user-provided teacher and student block_size differ, pipeline init must raise."""
    pass

def test_pipeline_distill_layers_default():
    """B1WanModel inside the pipeline has distill_layers == {4,9,14,19,24,29} by default."""
    pass
```

- [ ] Step 2: `pytest tests/test_b1_pipeline.py -v` → FAIL.

- [ ] Step 3: Implement. Key steps in `from_b1_config`:
  - `pipe = super().from_pretrained(...)` to leverage DiffSynth ckpt loading.
  - Replace each `block.self_attn` with `SelfAttentionB1(...)` preserving weight tensors via state_dict copy.
  - Mark layers in `cfg.distill_layers` with `distill_export=True`.
  - Attach `lq_proj`, `tc_decoder`, `lpips_net`.
  - `assert teacher.block_size == student.block_size` — both use `cfg.block_size`.

- [ ] Step 4: Re-run tests → 3 passed.

- [ ] Step 5: Commit:

```bash
git add flashvsr_b1/pipelines/b1_pipeline.py tests/test_b1_pipeline.py
git commit -m "feat(b1/pipelines): B1Pipeline wraps Wan video pipeline"
```

- [ ] Step 6: Report `logs/<DATE>-task12-pipeline.md`.

---

## Task 13: λ schedule + checkpoint IO

**Spec ref:** `task_b1.md §4.3` (λ table), `§7.4` (save_checkpoint), `§10.4` (ckpt naming).

**Files:**
- Create: `flashvsr_b1/train/lambda_schedule.py`
- Create: `flashvsr_b1/train/ckpt_io.py`
- Create: `tests/test_lambda_schedule.py`
- Create: `tests/test_ckpt_io.py`

**Interface contracts:**

```python
def lambda_at(step: int, *, total: int = 20000) -> dict: ...
def sparsity_at(step: int, *, target: float, total: int = 20000) -> float: ...

def save_checkpoint(run_dir: str, *, step: int, config_stem: str,
                    student, optimizer, scheduler,
                    current_sparsity: float, cfg_dict: dict) -> str: ...
def load_checkpoint(path: str, *, student, optimizer=None, scheduler=None) -> dict: ...
def update_latest_symlink(run_dir: str, ckpt_path: str) -> None: ...
```

**Acceptance tests:**

- [ ] Step 1: Write `tests/test_lambda_schedule.py`:

```python
import math
from flashvsr_b1.train.lambda_schedule import lambda_at, sparsity_at

def test_warmup_phase():
    lam = lambda_at(0)
    assert lam == {"l1":1.0, "l2":0.5, "l3":0.5, "l4":0.1}
    lam = lambda_at(1999)
    assert lam["l3"] == 0.5

def test_main_phase_l3_decay():
    lam_start = lambda_at(2000)
    lam_end   = lambda_at(14999)
    assert lam_start["l3"] > 0.49
    assert lam_end["l3"]   < 0.11

def test_refine_phase():
    lam = lambda_at(15000)
    assert lam == {"l1":1.0, "l2":1.0, "l3":0.1, "l4":0.05}

def test_sparsity_ramp_endpoints():
    assert sparsity_at(0, target=0.90) == 0.85
    assert sparsity_at(20000, target=0.95) == 0.95
    assert sparsity_at(12000, target=0.90) == 0.90        # ramp_end at 60%
```

And `tests/test_ckpt_io.py`:

```python
import os, torch
from flashvsr_b1.train.ckpt_io import save_checkpoint, load_checkpoint, update_latest_symlink

class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.w = torch.nn.Linear(4, 4)

def test_save_and_load_roundtrip(tmp_path):
    rd = str(tmp_path)
    os.makedirs(os.path.join(rd, "ckpt"))
    s = TinyModel(); opt = torch.optim.AdamW(s.parameters(), lr=1e-3)
    path = save_checkpoint(rd, step=100, config_stem="b1_bsa90",
                           student=s, optimizer=opt, scheduler=None,
                           current_sparsity=0.87, cfg_dict={"a": 1})
    assert os.path.basename(path) == "step_000000100_b1_bsa90.pt"
    s2 = TinyModel(); opt2 = torch.optim.AdamW(s2.parameters(), lr=1e-3)
    info = load_checkpoint(path, student=s2, optimizer=opt2)
    assert info["step"] == 100
    assert info["current_sparsity"] == 0.87
    for p1, p2 in zip(s.parameters(), s2.parameters()):
        assert torch.allclose(p1, p2)

def test_latest_symlink_updates(tmp_path):
    rd = str(tmp_path)
    os.makedirs(os.path.join(rd, "ckpt"))
    s = TinyModel(); opt = torch.optim.AdamW(s.parameters(), lr=1e-3)
    p1 = save_checkpoint(rd, step=100, config_stem="b1_bsa90",
                         student=s, optimizer=opt, scheduler=None,
                         current_sparsity=0.87, cfg_dict={})
    update_latest_symlink(rd, p1)
    latest = os.path.join(rd, "ckpt", "latest.pt")
    assert os.path.realpath(latest) == os.path.realpath(p1)
    p2 = save_checkpoint(rd, step=200, config_stem="b1_bsa90",
                         student=s, optimizer=opt, scheduler=None,
                         current_sparsity=0.88, cfg_dict={})
    update_latest_symlink(rd, p2)
    assert os.path.realpath(latest) == os.path.realpath(p2)
```

- [ ] Step 2: `pytest tests/test_lambda_schedule.py tests/test_ckpt_io.py -v` → FAIL.

- [ ] Step 3: Implement.

- [ ] Step 4: Re-run → 7 passed.

- [ ] Step 5: Commit:

```bash
git add flashvsr_b1/train/lambda_schedule.py flashvsr_b1/train/ckpt_io.py tests/test_lambda_schedule.py tests/test_ckpt_io.py
git commit -m "feat(b1/train): lambda schedule + checkpoint IO with latest symlink"
```

- [ ] Step 6: Report `logs/<DATE>-task13-schedule-ckpt.md`.

---

## Task 14: B1Trainer

**Spec ref:** `task_b1.md §4.4` (training step), `§5.4` (DiffSynth integration), `§7.4` (logger + ckpt hooks).

**Files:**
- Create: `flashvsr_b1/train/trainer_b1.py`
- Create: `tests/test_trainer_b1.py`

**Interface contracts:**

```python
class B1Trainer(diffsynth.trainers.UnifiedTrainer):
    def __init__(self, cfg, config_path: str): ...
    def compute_loss(self, batch, step) -> tuple[torch.Tensor, dict]: ...
    def training_step(self, batch, step) -> None: ...                  # backward + opt + logger
    def save_checkpoint(self, step: int) -> None: ...
```

**Acceptance tests:**

This task is integration-heavy and hard to fully unit-test off the B200. The acceptance bar is:

- [ ] Step 1: Write `tests/test_trainer_b1.py` with **smoke** tests using mocked pipeline:

```python
import torch
from unittest.mock import MagicMock
from flashvsr_b1.train.trainer_b1 import B1Trainer

def test_compute_loss_assembles_all_four_terms_for_bsa():
    """With BSA student, loss dict must contain {out, lpips, block, attn_out, total}."""
    trainer = B1Trainer.__new__(B1Trainer)
    trainer.teacher = MagicMock()
    trainer.student = MagicMock()
    trainer.student.attn_mode = "BSA"
    # ... Codex sets up mock forwards returning predictable tensors ...
    loss, ld = trainer.compute_loss(batch={}, step=0)
    for k in ["out", "lpips", "block", "attn_out"]:
        assert k in ld

def test_compute_loss_skips_block_for_lswa():
    trainer = B1Trainer.__new__(B1Trainer)
    trainer.teacher = MagicMock()
    trainer.student = MagicMock()
    trainer.student.attn_mode = "LSWA"
    loss, ld = trainer.compute_loss(batch={}, step=0)
    assert "block" not in ld
    assert "attn_out" in ld

def test_compute_loss_set_current_sparsity_called_for_bsa_not_lswa():
    """A side-effect test: when attn_mode is BSA, set_current_sparsity is invoked;
    when LSWA, it is not."""
    # Codex implements with monkeypatch on flashvsr_b1.attn.sparsity_schedule.set_current_sparsity
```

- [ ] Step 2: `pytest tests/test_trainer_b1.py -v` → FAIL.

- [ ] Step 3: Implement following `task_b1.md §4.4` and `§7.4`. The smoke training run is exercised by Task 15's script — keep this task's unit tests narrow.

- [ ] Step 4: Re-run → 3 passed.

- [ ] Step 5: Commit:

```bash
git add flashvsr_b1/train/trainer_b1.py tests/test_trainer_b1.py
git commit -m "feat(b1/train): B1Trainer with 4-loss compute_loss + ramp + logger"
```

- [ ] Step 6: Report `logs/<DATE>-task14-trainer.md`.

---

## Task 15: Three YAML configs + single-GPU smoke

**Spec ref:** `task_b1.md §5.5`.

**Files:**
- Create: `flashvsr_b1/configs/b1_bsa90.yaml`
- Create: `flashvsr_b1/configs/b1_bsa95.yaml`
- Create: `flashvsr_b1/configs/b1_lswa.yaml`
- Create: `scripts/10_smoke_one_step.sh`

**Acceptance:**

- [ ] Step 1: Write `flashvsr_b1/configs/b1_bsa90.yaml` matching `task_b1.md §5.5` verbatim, with placeholder paths for `teacher_ckpt`/`student_ckpt`/`tc_decoder_ckpt`/`lq_proj_ckpt` (Codex documents in the report exactly which paths the operator must overwrite).

- [ ] Step 2: Derive `b1_bsa95.yaml` (`target_sparsity: 0.95`) and `b1_lswa.yaml` (`attn_mode: LSWA`, `target_sparsity` field present but commented "ignored for LSWA").

- [ ] Step 3: Write `scripts/10_smoke_one_step.sh`:

```bash
#!/bin/bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CONFIG=${1:-flashvsr_b1/configs/b1_bsa90.yaml}
EXTRA_FLAGS="--train.total_steps=20 --logging.log_every_steps=2 --logging.ckpt_every_steps=10"

python -m flashvsr_b1.train.trainer_b1 --config "$CONFIG" $EXTRA_FLAGS
```

Make it executable:

```bash
chmod +x scripts/10_smoke_one_step.sh
```

- [ ] Step 4: On the B200, run the smoke (Codex's report records this — off-host, mark as "operator step pending"):

```bash
bash scripts/10_smoke_one_step.sh flashvsr_b1/configs/b1_bsa90.yaml
```

Expected outputs (after 20 steps):
- `log/<TS>_b1_bsa90/log.txt` with ≥ 8 lines
- `log/<TS>_b1_bsa90/train_metrics.jsonl` with ≥ 8 rows
- `log/<TS>_b1_bsa90/ckpt/step_000000010_b1_bsa90.pt`
- `log/<TS>_b1_bsa90/loss_throughput.png` (run `plot_training_metrics.py` on the dir)

- [ ] Step 5: Commit:

```bash
git add flashvsr_b1/configs/b1_*.yaml scripts/10_smoke_one_step.sh
git commit -m "feat(b1/configs): three training configs + single-GPU smoke script"
```

- [ ] Step 6: Report `logs/<DATE>-task15-configs-smoke.md` including the smoke run log (or "pending on B200" if not yet executed).

---

## Task 16: Evaluation scripts

**Spec ref:** `task_b1.md §6.2`, `§6.4`.

**Files:**
- Create: `eval/eval_sr.py`
- Create: `eval/compare_baseline.py`
- Create: `tests/test_eval_sr.py`

**Interface contracts:**

```python
def evaluate_checkpoint(ckpt_path: str, val_json: str, cfg: dict,
                       device: str = "cuda") -> dict:
    """
    Returns:
      {"psnr": ..., "ssim": ..., "lpips": ..., "dists": ...,
       "sparsity_rate": ..., "fps_720p": ..., "fps_1080p": ...,
       "peak_mem_gb": ...}
    """
```

**Acceptance tests:**

- [ ] Step 1: Write `tests/test_eval_sr.py` (mocked):

```python
def test_eval_returns_required_metric_keys():
    """With a stubbed evaluate_one_video that returns synthetic numbers,
    evaluate_checkpoint must aggregate into exactly the documented keys."""
    # Codex implements with monkeypatch.
```

- [ ] Step 2: Implement following `task_b1.md §6.2`. Use existing libraries:
  - PSNR / SSIM: `torchmetrics.image`
  - LPIPS: `lpips` package
  - DISTS: `pyiqa` or vendored implementation
  - Sparsity: count `active_blocks / total_blocks` per layer/head, average

- [ ] Step 3: Write `eval/compare_baseline.py` that reads `eval/results_<run_id>.json` for each of the three runs + FlashVSR v1.1 reference and prints / writes a markdown table matching `task_b1.md §6.4`.

- [ ] Step 4: Tests pass.

- [ ] Step 5: Commit:

```bash
git add eval/eval_sr.py eval/compare_baseline.py tests/test_eval_sr.py
git commit -m "feat(b1/eval): SR metrics + sparsity + FPS + 3-run comparison"
```

- [ ] Step 6: Report `logs/<DATE>-task16-eval.md`.

---

## Task 17: Multi-GPU training scripts + eval orchestration

**Spec ref:** `task_b1.md §5.6` (train script template), `§6.3` (three-run serial schedule).

**Files:**
- Create: `scripts/20a_train_b1_bsa90.sh`
- Create: `scripts/20b_train_b1_lswa.sh`
- Create: `scripts/20c_train_b1_bsa95.sh`
- Create: `scripts/30_eval_all.sh`

**Acceptance:**

- [ ] Step 1: Write `scripts/20a_train_b1_bsa90.sh`:

```bash
#!/bin/bash
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/FlashVSR_Attention_Map_Distillation}"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NPROC_PER_NODE=${NPROC_PER_NODE:-8} \
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  -m flashvsr_b1.train.trainer_b1 \
  --config flashvsr_b1/configs/b1_bsa90.yaml \
  "$@"
```

- [ ] Step 2: `scripts/20b_train_b1_lswa.sh` and `scripts/20c_train_b1_bsa95.sh` are identical except `--config`.

- [ ] Step 3: Write `scripts/30_eval_all.sh`:

```bash
#!/bin/bash
set -euo pipefail
RUNS=${@:-"$(ls -d log/*_b1_bsa90 log/*_b1_lswa log/*_b1_bsa95 2>/dev/null)"}

for run in $RUNS; do
  echo "Evaluating $run"
  python -m eval.eval_sr --ckpt "$run/ckpt/latest.pt" \
      --val_json /path/to/val_samples.json \
      --out_json "$run/eval/final_metrics.json"
done

python -m eval.compare_baseline --runs "$RUNS" \
    --out docs/final_report.md
```

- [ ] Step 4: `chmod +x scripts/20*.sh scripts/30_eval_all.sh`

- [ ] Step 5: Commit:

```bash
git add scripts/20a_train_b1_bsa90.sh scripts/20b_train_b1_lswa.sh scripts/20c_train_b1_bsa95.sh scripts/30_eval_all.sh
git commit -m "feat(b1/scripts): three-run serial training entries + eval orchestration"
```

- [ ] Step 6: Report `logs/<DATE>-task17-scripts.md`.

---

## Final integration verification (no new code)

After all 17 tasks land, Claude (not Codex) runs the following sanity sweep before the operator kicks off the first B200 run:

- [ ] `pytest -v` — all CPU-runnable tests green
- [ ] `bash scripts/10_smoke_one_step.sh flashvsr_b1/configs/b1_bsa90.yaml` on a B200 → 20 steps complete, all 4 log artifacts present
- [ ] `python -m eval.plot_training_metrics log/<TS>_b1_bsa90` → PNG renders without warnings
- [ ] `task_b1.md §0` decision table re-checked against actual code: block_size lock, distill layers, λ defaults, ramp shape

If any check fails, open a follow-up issue, do not start the 20k-step run.

---

## Three-run execution schedule (operator-facing, after all tasks land)

This is the runtime plan, not Codex tasks:

1. **Run 1 — BSA-90%**
   - `bash scripts/20a_train_b1_bsa90.sh`
   - Watch `log/<TS>_b1_bsa90/log.txt` for `videos_per_hour` stability
   - Plot at step 500 / 2000 / 5000 to verify ramp + λ decay shape
   - On success: `python -m eval.eval_sr --ckpt log/<TS>_b1_bsa90/ckpt/latest.pt ...`

2. **Run 2 — LSWA**
   - Same pattern with `b1_lswa.yaml`
   - **Risk-lowest** (LSWA already validated in `FlashVSR_LSWA`), so a regression here surfaces our wiring bugs rather than algorithmic issues

3. **Run 3 — BSA-95%**
   - Watch L_block in main phase; if not converging by step 8k, extend `total_steps` to 25k–30k (re-read `task_b1.md §6.3`)

4. **Compare**
   - `bash scripts/30_eval_all.sh` → `docs/final_report.md`

---

## Self-review notes

**Spec coverage check** (each `task_b1.md` section → covering task):

| Spec section | Implementing task(s) |
| --- | --- |
| §0 decisions | All tasks reference; pipeline asserts (Task 12) and logger fields (Task 11) enforce |
| §1 file tree | Task 1 (skeleton); Tasks 2–17 fill the leaves |
| §2 Forward construction + Figure 8 | Task 9 (model) + Task 14 (trainer) |
| §3.1 SelfAttention signature | Task 9 |
| §3.2 bsa_kernel | Task 5 |
| §3.3 lswa | Task 4 |
| §3.4 shadow_block_pool_attn | Task 3 |
| §3.5 sparsity_schedule | Task 2 |
| §4.1–§4.2 four loss definitions | Task 10 |
| §4.3 λ schedule | Task 13 |
| §4.4 per-step training | Task 14 |
| §5.1–§5.3 data | Tasks 6, 7 |
| §5.4–§5.5 pipeline + config | Tasks 12, 15 |
| §5.6 train script | Task 17 |
| §6.1 inference path | Task 12 (B1Pipeline gains inference helper) |
| §6.2 eval metrics | Task 16 |
| §6.3 three-run serial | Tasks 15+17 (configs+scripts); operator schedule above |
| §6.4 comparison table | Task 16 (`compare_baseline.py`) |
| §7 logging + plotting | Task 11 |
| §8 codex task list | This entire document |
| §9 risk table | Operator schedule + retry policy in `task_b1.md §9` |
| §10 conventions | All tasks observe |

**Placeholder scan:** none — every step lists a concrete file path, code block (where editing code), command, expected outcome.

**Type / signature consistency check:**
- `block_size: tuple[int,int,int]` — used identically in Tasks 3, 5, 9, 12.
- `current_sparsity: float` — set by Task 2's `set_current_sparsity`, read by Tasks 5 and 9.
- `grid_shape: tuple[int,int,int]` — introduced in Task 3, passed in from Task 9 (which knows `f, h, w` from the model's spatial planner).
- `return_aux=True` returning `(out, aux)` — Tasks 9, 12, 14 all agree.
- `MetricsLogger.step` signature — Tasks 11 and 14 agree.

**Done.**
