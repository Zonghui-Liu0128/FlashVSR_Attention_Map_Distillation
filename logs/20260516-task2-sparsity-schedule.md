# Task 2: Sparsity Schedule

## What Changed

- Added `tests/test_sparsity_schedule.py` with the five Task 2 acceptance tests:
  - ramp starts at `init`
  - ramp clamps to `target` at and after `ramp_end_step`
  - ramp is monotonic increasing
  - midpoint follows the cosine formula
  - `set_current_sparsity` only mutates modules that already define `current_sparsity`
- Added `flashvsr_b1/attn/sparsity_schedule.py` with:
  - `cosine_sparsity_ramp(step: int, *, ramp_end_step: int, init: float = 0.85, target: float = 0.90) -> float`
  - `set_current_sparsity(model: torch.nn.Module, rate: float) -> None`

## Why

Task 2 provides the runtime sparsity ramp used by BSA student training. The function reaches target sparsity at `int(total_steps * 0.6)` according to `task_b1.md §3.5` and `§4.3`, while `set_current_sparsity` updates only attention modules marked with a `current_sparsity` attribute.

## Test Design

The tests are the five acceptance checks from `docs/superpowers/plans/2026-05-16-vsr-b1-sparse-onestep.md` Task 2. They validate the exact public API, numerical cosine behavior, clamping, monotonicity, and module traversal behavior.

## Red Run

Default base Python could not import `torch`, so I switched to the existing `flashvsr` conda environment, which has PyTorch and pytest installed.

Command:

```bash
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python -m pytest tests/test_sparsity_schedule.py -v
```

Result before implementation:

```text
collected 0 items / 1 error
E   ModuleNotFoundError: No module named 'flashvsr_b1.attn.sparsity_schedule'
```

## Green Run

Command:

```bash
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python -m pytest tests/test_sparsity_schedule.py -v
```

Result after implementation:

```text
collected 5 items

tests/test_sparsity_schedule.py::test_ramp_init PASSED
tests/test_sparsity_schedule.py::test_ramp_clamps_to_target PASSED
tests/test_sparsity_schedule.py::test_ramp_monotonic_increasing PASSED
tests/test_sparsity_schedule.py::test_ramp_midpoint PASSED
tests/test_sparsity_schedule.py::test_set_current_sparsity_writes_to_marked_modules_only PASSED

5 passed in 0.69s
```

## Git Status Before Report

```text
?? DiffSynth-Studio/
?? data/
?? flashvsr_b1/attn/sparsity_schedule.py
?? tests/test_sparsity_schedule.py
```

`DiffSynth-Studio/` and `data/` were already untracked before this task. No commit was made, per instruction.

## Claude Review Iteration History

- Pending Claude review.

## Debug Notes

- The plan text says "Expected: 4 passed" for Task 2 green, but the same Task 2 acceptance block contains five tests. I kept all five requested acceptance tests and verified all five pass.
