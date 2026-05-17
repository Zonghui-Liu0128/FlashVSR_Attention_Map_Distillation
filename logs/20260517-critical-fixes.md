# 2026-05-17 Critical B1 Integration Fixes

## Scope

Fixed the blocking B1 sparse one-step integration gaps identified after the 17-task pass:

- Added a runnable `flashvsr_b1.train.trainer_b1` module entry point with OmegaConf loading, torchrun/DDP init, optimizer construction, dataloader construction, autocast, checkpoint cadence, optional eval cadence, final checkpoint save, and metric cleanup.
- Removed trainer teacher=student fallback; trainer now requires a separate teacher exposed by `B1Pipeline`.
- Updated block-size validation to inspect teacher/student attention modules instead of comparing config defaults.
- Added `B1WanModel.b1_forward(LR_latent, z_t, t_star, return_aux=False)` and made trainer call only that contract, including DDP-wrapped models.
- Updated `B1Pipeline.from_b1_config` to expose `pipe.student`, build a separate frozen BSA-85 teacher, freeze LPIPS, attach `prepare_batch`, and move projector/decoder/lpips modules to the DiT device.
- Added temporal-block causal masking to BSA draft masks before `block_sparse_attn_func`.
- Changed shadow block-pool attention from flat-index triangular masking to block-time causal masking.
- Fixed BSA parity test unpacking to tolerate reference implementations returning either a tensor or a tuple.
- Updated review/tests to assert the corrected `b1_forward` and temporal-causal behavior.

## Files Modified

- `flashvsr_b1/train/trainer_b1.py`
- `flashvsr_b1/pipelines/b1_pipeline.py`
- `flashvsr_b1/models/wan_dit_b1.py`
- `flashvsr_b1/attn/bsa_kernel.py`
- `flashvsr_b1/attn/shadow_block_pool_attn.py`
- `tests/test_b1_pipeline.py`
- `tests/test_bsa_kernel.py`
- `tests/test_trainer_b1.py`
- `tests/review_logic/test_review_real_logic.py`

## Verification

Command:

```bash
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python -m pytest tests/ -v
```

Result:

```text
79 passed, 3 skipped in 28.06s
```

Skipped tests were CUDA/BSA or optional LPIPS dependent on this macOS CPU environment.

## B200 Verification Required

- Run `tests/test_bsa_kernel.py::test_bsa_forward_shape` with CUDA and `block_sparse_attn` installed.
- Run `tests/test_bsa_kernel.py::test_bsa_parity_with_root_implementation` on B200. The local root `wan_video_dit.py` currently returns a tuple from `_block_sparse_forward`; the test now handles either tuple or tensor return type, but numerical parity still requires the production kernel.
- Run a short torchrun smoke with real FlashVSR teacher/student checkpoints and data config to verify second teacher loading, DDP wrapping, checkpoint save/load expectations, and memory fit.
- Validate real eval cadence on B200 once `_evaluate_one_video` and `_measure_fps` are implemented; this macOS path only verifies that the train loop can call the eval hook when a validation JSON exists.
