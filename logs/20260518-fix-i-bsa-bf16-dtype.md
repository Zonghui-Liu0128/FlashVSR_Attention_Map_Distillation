# B200 pytest fix - Issue I: BSA bf16 dtype

Status: DONE

## Files modified

- `tests/test_bsa_kernel.py`

## Changes

- Updated `test_bsa_forward_shape` to build Q/K/V as `torch.bfloat16`.
- Added an explicit output dtype assertion for `bsa_forward`.
- Updated `test_bsa_parity_with_root_implementation` to build Q/K/V as `torch.bfloat16`.
- Cast the reference `SelfAttention` module to bf16.
- Kept the parity test to shape parity only, as requested.

## Pytest output

Command:

```bash
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python -m pytest tests/test_bsa_kernel.py -v
```

Result:

```text
tests/test_bsa_kernel.py::test_topk_for_85pct PASSED
tests/test_bsa_kernel.py::test_topk_for_90pct PASSED
tests/test_bsa_kernel.py::test_topk_for_95pct PASSED
tests/test_bsa_kernel.py::test_topk_for_clamps_to_one PASSED
tests/test_bsa_kernel.py::test_bsa_forward_shape SKIPPED
tests/test_bsa_kernel.py::test_bsa_parity_with_root_implementation SKIPPED

4 passed, 2 skipped in 0.82s
```

Command:

```bash
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python -m pytest tests/ -v
```

Result:

```text
FAILED tests/review_logic/test_b1_contract_gaps.py::test_prepare_batch_outputs_channels_accepted_by_wan_patch_embedding

1 failed, 85 passed, 3 skipped in 29.14s
```

The failing test is the known Issue H:

```text
AssertionError: prepare_batch produced 1536 channels and z_t 1536 channels,
but Wan patch_embedding expects in_dim=16.
```

## B200 verification required

On B200, the two BSA CUDA tests must now run and pass instead of crashing with:

```text
RuntimeError: BlockSparseAttention only support fp16 and bf16 data type
```

Specifically:

- `tests/test_bsa_kernel.py::test_bsa_forward_shape`
- `tests/test_bsa_kernel.py::test_bsa_parity_with_root_implementation`
