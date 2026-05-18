# B200 pytest fix — Issues F + G

Status: DONE_WITH_CONCERNS

Concern: local macOS verification matches the expected acceptance output, but B200 CUDA verification was not available in this environment.

## Files modified

- `flashvsr_b1/attn/bsa_kernel.py`
- `tests/test_lswa.py`
- `tests/test_bsa_kernel.py`

## Changes

- Replaced hardcoded macOS `wan_video_dit.py` paths in the LSWA and BSA parity tests with repo-relative `Path(__file__).resolve().parents[1] / "wan_video_dit.py"`.
- Added missing-reference skips for the parity tests.
- Added a temporary `utils` shim around `wan_video_dit.py` imports in `bsa_kernel._load_reference_module`.
- Added the same temporary `utils` shim to `tests/test_bsa_kernel.py::test_bsa_parity_with_root_implementation`.

## Pytest output

Command:

```bash
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python -m pytest tests/test_bsa_kernel.py tests/test_lswa.py -v
```

Result:

```text
collected 9 items

tests/test_bsa_kernel.py::test_topk_for_85pct PASSED
tests/test_bsa_kernel.py::test_topk_for_90pct PASSED
tests/test_bsa_kernel.py::test_topk_for_95pct PASSED
tests/test_bsa_kernel.py::test_topk_for_clamps_to_one PASSED
tests/test_bsa_kernel.py::test_bsa_forward_shape SKIPPED
tests/test_bsa_kernel.py::test_bsa_parity_with_root_implementation SKIPPED
tests/test_lswa.py::test_lswa_output_shape_train_mode PASSED
tests/test_lswa.py::test_lswa_is_causal_in_time PASSED
tests/test_lswa.py::test_lswa_matches_reference_implementation PASSED

7 passed, 2 skipped in 0.82s
```

Command:

```bash
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python -m pytest tests/ -v
```

Result:

```text
collected 89 items

1 failed, 85 passed, 3 skipped in 27.43s
```

Failure retained from the expected macOS baseline:

```text
FAILED tests/review_logic/test_b1_contract_gaps.py::test_prepare_batch_outputs_channels_accepted_by_wan_patch_embedding
AssertionError: prepare_batch produced 1536 channels and z_t 1536 channels, but Wan patch_embedding expects in_dim=16.
```

## B200 verification required

On B200, the three previously-failing tests must run and pass after this commit:

- `tests/test_bsa_kernel.py::test_bsa_forward_shape` (was `ModuleNotFoundError`)
- `tests/test_bsa_kernel.py::test_bsa_parity_with_root_implementation` (was `FileNotFoundError`)
- `tests/test_lswa.py::test_lswa_matches_reference_implementation` (was `FileNotFoundError`)
