# Fix C: B1Pipeline / B1WanModel `nn.Module.__init__` after `cls.__new__`

Status: DONE

## Scope

`B1Pipeline.from_b1_config` and `B1WanModel.from_wan_model` both built instances
via `cls.__new__(cls)` and copied attributes via `__dict__.update(base_pipe.__dict__)`.
On production this happened to work because the real DiffSynth `WanVideoPipeline`
returned by `from_pretrained` is itself an `nn.Module` whose `__dict__` already
contains `_modules`, `_parameters`, `_buffers`, etc. With test mocks (or any
non-`nn.Module` base), those internal dicts were absent and the next module
assignment crashed inside `nn.Module.__setattr__` with
`cannot assign module before Module.__init__() call`.

This is also a real production fragility — any future change to upstream
`WanVideoPipeline.from_pretrained` that omits `_modules` would break us.

## Files Modified

- `flashvsr_b1/pipelines/b1_pipeline.py`
  - Insert `torch.nn.Module.__init__(pipe)` between `cls.__new__(cls)` and
    `__dict__.update(...)`.
  - Local DiffSynth-unavailable fallback `WanVideoPipeline` now inherits
    `torch.nn.Module` so that `B1Pipeline` → `WanVideoPipeline` →
    `nn.Module` chain is consistent on macOS dev.
- `flashvsr_b1/models/wan_dit_b1.py`
  - Insert `torch.nn.Module.__init__(b1_model)` between `cls.__new__(cls)` and
    `__dict__.update(...)`.
- `tests/review_logic/test_b1_contract_gaps.py::test_prepare_batch_outputs_channels_accepted_by_wan_patch_embedding`
  - Insert `torch.nn.Module.__init__(pipe)` after the direct `__new__` call.

Total: +4 lines, 1 modified line across 3 files.

## Out-of-Scope Revert

The first Codex attempt also added a silent "channel grouping" (`view(...).mean(dim=2)`)
in `prepare_batch` to make `lr_latent.shape[1]` match `pipe.dit.in_dim`. This
silently averaged 96 channel groups into 1 to paper over the contract gap that
`test_prepare_batch_outputs_channels_accepted_by_wan_patch_embedding` is meant
to expose (LQ proj outputs `cfg.dim=1536` channels; Wan DiT patch_embedding
expects `cfg.in_dim=16` channels).

Reverted in this commit. The channel mismatch is a real architectural design
gap and is tracked separately as **Issue H** (`prepare_batch / b1_forward
channel pairing`) for future resolution. This Fix C log explicitly defers it.

## Verification

Targeted command (macOS):

```bash
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python -m pytest \
    tests/test_b1_pipeline.py \
    tests/review_logic/test_b1_contract_gaps.py::test_prepare_batch_outputs_channels_accepted_by_wan_patch_embedding \
    tests/review_logic/test_review_real_logic.py::test_C3_pipeline_constructs_separate_teacher \
    -v
```

Result: 4 passed (the three pipeline tests and the C3 separate-teacher test) +
1 failed (`test_prepare_batch_outputs_channels_accepted_by_wan_patch_embedding`,
deferred to Issue H).

Full suite (macOS):

```bash
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python -m pytest tests/ -v
```

Result: `79 passed, 3 skipped, 4 failed in 14.34s`. The 4 remaining failures
are all in the upcoming fix queue:

- `test_trainer_accepts_b1wanmodel_layer_aux_contract` → Fix A (Task #30)
- `test_bsa_block_size_is_compatible_with_flashvsr_patchified_grid` → Fix B (Task #29)
- `test_bucket_sampler_keeps_bucket_choice_synchronized_across_ranks` → Fix D (Task #31)
- `test_prepare_batch_outputs_channels_accepted_by_wan_patch_embedding` → Fix H (Task #34, new)

No regressions on previously-passing tests.

## B200 Re-Verification Required

On B200 (`pull` to this commit, then):

```bash
python -m pytest \
    tests/test_b1_pipeline.py \
    tests/review_logic/test_review_real_logic.py::test_C3_pipeline_constructs_separate_teacher \
    -v
```

Expected: 4 passed. (CUDA-skipped tests stay skipped; `prepare_batch` channel
test stays failing pending Issue H.)
