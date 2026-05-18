# Fix D: Bucket Sampler DDP Synchronization

Status: DONE

Files modified:
- `flashvsr_b1/data/bucket_sampler.py`
- `tests/test_bucket_sampler.py`

## Summary

`AspectRatioBucketSampler` now shards by single-bucket super-chunks of size
`batch_size * num_replicas`. Every rank follows the same super-chunk order, and
rank `r` consumes its per-rank slice from each super-chunk. This keeps bucket
choice synchronized across DDP ranks while still assigning disjoint item indices.

Added `test_super_chunk_same_bucket_and_disjoint_across_ranks` to cover the
3-rank DDP invariant.

## Regression Check

Before the sampler fix, the new regression test failed as intended:

```text
tests/test_bucket_sampler.py::test_super_chunk_same_bucket_and_disjoint_across_ranks FAILED
AssertionError: step 0 ranks disagree on bucket: [{'portrait'}, {'landscape'}, {'portrait'}]
```

## Targeted Pytest

Command:

```bash
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python -m pytest \
    tests/test_bucket_sampler.py \
    tests/review_logic/test_b1_contract_gaps.py::test_bucket_sampler_keeps_bucket_choice_synchronized_across_ranks \
    tests/review_logic/test_review_real_logic.py::test_C12_bucket_sampler_strict_same_bucket_per_batch \
    -v
```

Output:

```text
collected 7 items

tests/test_bucket_sampler.py::test_each_batch_is_single_bucket PASSED
tests/test_bucket_sampler.py::test_bucket_ratio_close_to_dataset_ratio PASSED
tests/test_bucket_sampler.py::test_drop_last_enforces_full_batches PASSED
tests/test_bucket_sampler.py::test_ddp_ranks_disjoint_and_complete PASSED
tests/test_bucket_sampler.py::test_super_chunk_same_bucket_and_disjoint_across_ranks PASSED
tests/review_logic/test_b1_contract_gaps.py::test_bucket_sampler_keeps_bucket_choice_synchronized_across_ranks PASSED
tests/review_logic/test_review_real_logic.py::test_C12_bucket_sampler_strict_same_bucket_per_batch PASSED

7 passed in 0.72s
```

Note: the prompt expected 6 passed, but the command collected 7 tests because
`tests/test_bucket_sampler.py` now contains 5 tests, plus the 2 explicitly named
review tests.

## Full Suite

Command:

```bash
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python -m pytest tests/ -v
```

Output:

```text
collected 87 items

tests/review_logic/test_b1_contract_gaps.py::test_trainer_accepts_b1wanmodel_layer_aux_contract PASSED
tests/review_logic/test_b1_contract_gaps.py::test_bsa_block_size_is_compatible_with_flashvsr_patchified_grid PASSED
tests/review_logic/test_b1_contract_gaps.py::test_prepare_batch_outputs_channels_accepted_by_wan_patch_embedding FAILED
tests/review_logic/test_b1_contract_gaps.py::test_bucket_sampler_keeps_bucket_choice_synchronized_across_ranks PASSED
...
tests/test_bucket_sampler.py::test_each_batch_is_single_bucket PASSED
tests/test_bucket_sampler.py::test_bucket_ratio_close_to_dataset_ratio PASSED
tests/test_bucket_sampler.py::test_drop_last_enforces_full_batches PASSED
tests/test_bucket_sampler.py::test_ddp_ranks_disjoint_and_complete PASSED
tests/test_bucket_sampler.py::test_super_chunk_same_bucket_and_disjoint_across_ranks PASSED
...
tests/test_wan_dit_b1.py::test_b1_wan_model_distill_layers_default PASSED

FAILED tests/review_logic/test_b1_contract_gaps.py::test_prepare_batch_outputs_channels_accepted_by_wan_patch_embedding
1 failed, 83 passed, 3 skipped in 27.61s
```

Remaining failure:

```text
AssertionError: prepare_batch produced 1536 channels and z_t 1536 channels,
but Wan patch_embedding expects in_dim=16.
```
