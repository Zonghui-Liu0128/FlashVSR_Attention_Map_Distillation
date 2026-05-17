# Task 12 - B1Pipeline

Status: DONE_WITH_CONCERNS

## What changed

- Created `flashvsr_b1/pipelines/b1_pipeline.py`.
- Added `B1Pipeline.from_b1_config(cfg)`:
  - validates teacher/student `block_size` agreement,
  - calls the DiffSynth Wan parent loader,
  - wraps the loaded `pipe.dit` with `B1WanModel.from_wan_model(...)`,
  - attaches `lq_proj`, `tc_decoder`, and optional lazy `lpips_net`.
- Extended `flashvsr_b1/models/wan_dit_b1.py` with `B1WanModel.from_wan_model(...)` so a parent-loaded Wan model can keep its already-loaded state while replacing each block's `self_attn`.
- Created `tests/test_b1_pipeline.py` with mocked parent loading and mocked FlashVSR/LPIPS attachments.

## DiffSynth check

Confirmed the source class exists at:

- `DiffSynth-Studio/diffsynth/pipelines/wan_video.py`
- class name: `WanVideoPipeline`
- loader pattern: `@staticmethod from_pretrained(...)` returns a populated pipeline with `pipe.dit`.

Concern: in this local macOS environment, importing the real pipeline currently fails because `torchvision` is unavailable. The tests patch `WanVideoPipeline.from_pretrained`, so they verify our integration logic without importing real checkpoints or full DiffSynth runtime dependencies.

## TDD notes

Initial red run:

```text
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python -m pytest tests/test_b1_pipeline.py -v
collected 3 items
tests/test_b1_pipeline.py::test_pipeline_replaces_self_attn_with_b1_variant FAILED
tests/test_b1_pipeline.py::test_pipeline_asserts_block_size_match FAILED
tests/test_b1_pipeline.py::test_pipeline_distill_layers_default FAILED
```

The failures were `NotImplementedError` from the minimal `B1Pipeline.from_b1_config` skeleton.

Final focused verification:

```text
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python -m pytest tests/test_b1_pipeline.py -v
collected 3 items
tests/test_b1_pipeline.py::test_pipeline_replaces_self_attn_with_b1_variant PASSED
tests/test_b1_pipeline.py::test_pipeline_asserts_block_size_match PASSED
tests/test_b1_pipeline.py::test_pipeline_distill_layers_default PASSED
============================== 3 passed in 0.76s ===============================
```

Regression check after touching `B1WanModel`:

```text
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python -m pytest tests/test_wan_dit_b1.py -v
============================== 4 passed in 0.83s ===============================
```

## Files created or modified

- `flashvsr_b1/pipelines/b1_pipeline.py`
- `tests/test_b1_pipeline.py`
- `flashvsr_b1/models/wan_dit_b1.py`
- `logs/20260517-task12-pipeline.md`

`flashvsr_b1/pipelines/__init__.py` was not modified.

## Git status at report time

```text
 M docs/superpowers/plans/2026-05-16-vsr-b1-sparse-onestep.md
 M flashvsr_b1/models/wan_dit_b1.py
?? DiffSynth-Studio/
?? data/
?? flashvsr_b1/pipelines/b1_pipeline.py
?? tests/test_b1_pipeline.py
```

The modified plan file and untracked `DiffSynth-Studio/` and `data/` entries were pre-existing in the worktree for this task and were not changed here.
