# Task 14 - B1Trainer

## What changed

- Added `flashvsr_b1/train/trainer_b1.py`.
- Added `tests/test_trainer_b1.py`.

## Implementation notes

- DiffSynth-Studio in this checkout does not contain `diffsynth.trainers.UnifiedTrainer`; the actual base used by the training examples is `diffsynth.diffusion.DiffusionTrainingModule`, so `B1Trainer` subclasses that.
- `compute_loss` follows `task_b1.md` section 4.4:
  - prepares batch into `LR_latent, z_t, t_star, gt_hr`;
  - applies sparsity ramp only for BSA;
  - runs teacher under `torch.no_grad()` and detaches aux tensors;
  - runs student with aux;
  - builds `out`, `lpips`, `attn_out`, and BSA-only `block`;
  - composes total loss with `lambda_at(step)`.
- `training_step` performs backward, grad clipping, optimizer step/zero, and `MetricsLogger.step`.
- `save_checkpoint` is rank-0 only and delegates to `flashvsr_b1.train.ckpt_io.save_checkpoint`, then updates `latest.pt`.

## TDD result

RED command:

```bash
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python -m pytest tests/test_trainer_b1.py -v
```

RED output:

```text
collected 3 items

tests/test_trainer_b1.py::test_compute_loss_assembles_all_four_terms_for_bsa FAILED [ 33%]
tests/test_trainer_b1.py::test_compute_loss_skips_block_for_lswa FAILED  [ 66%]
tests/test_trainer_b1.py::test_compute_loss_set_current_sparsity_called_for_bsa_only FAILED [100%]

ModuleNotFoundError: No module named 'flashvsr_b1.train.trainer_b1'
============================== 3 failed in 0.78s ===============================
```

GREEN command:

```bash
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python -m pytest tests/test_trainer_b1.py -v
```

GREEN output:

```text
collected 3 items

tests/test_trainer_b1.py::test_compute_loss_assembles_all_four_terms_for_bsa PASSED [ 33%]
tests/test_trainer_b1.py::test_compute_loss_skips_block_for_lswa PASSED  [ 66%]
tests/test_trainer_b1.py::test_compute_loss_set_current_sparsity_called_for_bsa_only PASSED [100%]

============================== 3 passed in 0.71s ===============================
```

## Concerns

- `__init__` has real pipeline construction hooks, but checkpoint-heavy instantiation is intentionally not covered here. Task 15 smoke should exercise that path.
- DiffSynth trainer naming differs from the spec placeholder; this module uses the actual local DiffSynth training base.
