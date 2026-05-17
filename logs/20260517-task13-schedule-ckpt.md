# Task 13: Lambda schedule + checkpoint IO

## Status

Completed. Implemented the Task 13 foreground utilities without modifying
`flashvsr_b1/train/__init__.py` and without committing.

Note: the prompt says the final focused run should show 7 PASS, but the
verbatim requested test files contain 6 test functions. The focused pytest run
therefore reports `6 passed`.

## Files created

- `flashvsr_b1/train/lambda_schedule.py`
- `flashvsr_b1/train/ckpt_io.py`
- `tests/test_lambda_schedule.py`
- `tests/test_ckpt_io.py`

## Test design

- `tests/test_lambda_schedule.py` covers warmup, main cosine decay, refine
  constants, and sparsity ramp endpoints through the Task 2 ramp utility.
- `tests/test_ckpt_io.py` covers checkpoint save/load roundtrip and replacing
  `ckpt/latest.pt` with a relative symlink to the latest checkpoint file.

## RED run

Command:

```bash
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python -m pytest tests/test_lambda_schedule.py tests/test_ckpt_io.py -v
```

Result before implementation:

```text
collected 0 items / 2 errors
ERROR tests/test_lambda_schedule.py
E   ModuleNotFoundError: No module named 'flashvsr_b1.train.lambda_schedule'
ERROR tests/test_ckpt_io.py
E   ModuleNotFoundError: No module named 'flashvsr_b1.train.ckpt_io'
============================== 2 errors in 0.78s ===============================
```

## GREEN run

Command:

```bash
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python -m pytest tests/test_lambda_schedule.py tests/test_ckpt_io.py -v
```

Result after implementation/refactor:

```text
tests/test_lambda_schedule.py::test_warmup_phase PASSED
tests/test_lambda_schedule.py::test_main_phase_l3_decay PASSED
tests/test_lambda_schedule.py::test_refine_phase PASSED
tests/test_lambda_schedule.py::test_sparsity_ramp_endpoints PASSED
tests/test_ckpt_io.py::test_save_and_load_roundtrip PASSED
tests/test_ckpt_io.py::test_latest_symlink_updates PASSED
============================== 6 passed in 1.33s ===============================
```

## Debug notes

- `lambda_schedule.py` follows `task_b1.md §4.3` directly.
- `sparsity_at` delegates to `cosine_sparsity_ramp` with
  `ramp_end_step=int(total * 0.6)`.
- Checkpoint names use `step_<N09d>_<config_stem>.pt`.
- `latest.pt` is replaced via `os.path.lexists` and points to the checkpoint
  basename, making the symlink relative within `run_dir/ckpt`.
