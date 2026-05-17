# Task 5: BSA kernel wrapper

## Implemented

- Added `tests/test_bsa_kernel.py` verbatim from the task prompt.
- Added `flashvsr_b1/attn/bsa_kernel.py` with:
  - `topk_for(sparsity, total_kv_blocks)` using `max(1, int(round(total_kv_blocks * (1.0 - sparsity))))`.
  - `bsa_forward(...)` as a thin wrapper around `block_sparse_attn_func`.
  - Lazy `block_sparse_attn` import inside `bsa_forward`; missing library raises:
    `block_sparse_attn library required for BSA mode — install from FlashVSR repo`
  - Reference-module loading via `importlib.util.spec_from_file_location`.
  - Top-k mask construction through `generate_draft_block_mask`.
  - Q/K/V partition, kernel reorder, and output reorder back to `(B, S, D)`.
- Updated `flashvsr_b1/attn/__init__.py` to expose `bsa_forward` and `topk_for`.

## Reference copied/adapted

- `wan_video_dit.py:124-161`: `generate_draft_block_mask` call contract.
- `wan_video_dit.py:182-216`: low-level `block_sparse_attn_func` argument setup.
- `wan_video_dit.py:493-540`: BSA path partition/reorder/reverse pattern.
- `wan_video_dit.py:74-98`: default local-window mask builder used when `local_window_mask=None`, because the current reference `generate_draft_block_mask` asserts that `local_attn_mask` is not `None`.

## Why lazy import

`block_sparse_attn` is CUDA-specific and is not available on the macOS development environment. Importing it at module import time would break CPU-only tests and any code path that merely imports the package. `bsa_forward` imports it only when the BSA kernel is actually executed, and it raises a clear RuntimeError instead of silently falling back to SDPA.

## Pytest RED

Command:

```bash
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python -m pytest tests/test_bsa_kernel.py -v
```

Result:

```text
collected 6 items

tests/test_bsa_kernel.py::test_topk_for_85pct FAILED
tests/test_bsa_kernel.py::test_topk_for_90pct FAILED
tests/test_bsa_kernel.py::test_topk_for_95pct FAILED
tests/test_bsa_kernel.py::test_topk_for_clamps_to_one FAILED
tests/test_bsa_kernel.py::test_bsa_forward_shape SKIPPED
tests/test_bsa_kernel.py::test_bsa_parity_with_root_implementation SKIPPED

4 failed, 2 skipped in 0.67s
```

The failures were the expected `NotImplementedError` from the initial stubs.

## Pytest GREEN

Command:

```bash
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python -m pytest tests/test_bsa_kernel.py -v
```

Result:

```text
collected 6 items

tests/test_bsa_kernel.py::test_topk_for_85pct PASSED
tests/test_bsa_kernel.py::test_topk_for_90pct PASSED
tests/test_bsa_kernel.py::test_topk_for_95pct PASSED
tests/test_bsa_kernel.py::test_topk_for_clamps_to_one PASSED
tests/test_bsa_kernel.py::test_bsa_forward_shape SKIPPED
tests/test_bsa_kernel.py::test_bsa_parity_with_root_implementation SKIPPED

4 passed, 2 skipped in 0.76s
```

## Concerns

- The parity and shape tests are skipped on this macOS environment because CUDA is unavailable. Real kernel parity still needs to be exercised on the B200 environment with `block_sparse_attn` installed.
- The current reference `generate_draft_block_mask` requires a non-`None` local window mask, while this task's public interface allows `local_window_mask=None`. The wrapper builds the same default shifted local mask used by `_block_sparse_forward` when no mask is supplied.
- Staging could not be completed in this sandbox. `git add tests/test_bsa_kernel.py flashvsr_b1/attn/bsa_kernel.py flashvsr_b1/attn/__init__.py logs/20260517-task5-bsa-kernel.md` failed because Git could not create `.git/index.lock` (`Operation not permitted`). A direct `touch .git/index.lock` failed with the same error, so this appears to be an environment permission limitation rather than a stale lock.
