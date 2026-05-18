# B200 Pytest Fix B - BSA Grid Contract Test

Status: DONE

## Files Modified

- `tests/review_logic/test_b1_contract_gaps.py`
- `tests/review_logic/test_review_real_logic.py`

## Targeted Pytest

Command:

```bash
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python -m pytest \
    tests/review_logic/test_b1_contract_gaps.py::test_bsa_block_size_is_compatible_with_flashvsr_patchified_grid \
    tests/review_logic/test_review_real_logic.py::test_C5b_bsa_partition_succeeds_on_latent_grid \
    tests/review_logic/test_review_real_logic.py::test_C13_metrics_logger_seqlen_constant_matches_spec_latent_grid \
    -v
```

Output:

```text
============================= test session starts ==============================
platform darwin -- Python 3.11.13, pytest-9.0.3, pluggy-1.6.0 -- /Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python
cachedir: .pytest_cache
rootdir: /Users/zonghuiliu/Documents/Codex/VideoGen/FlashVSR_Attention_Map_Distillation
configfile: pytest.ini
collecting ... collected 3 items

tests/review_logic/test_b1_contract_gaps.py::test_bsa_block_size_is_compatible_with_flashvsr_patchified_grid PASSED [ 33%]
tests/review_logic/test_review_real_logic.py::test_C5b_bsa_partition_succeeds_on_latent_grid PASSED [ 66%]
tests/review_logic/test_review_real_logic.py::test_C13_metrics_logger_seqlen_constant_matches_spec_latent_grid PASSED [100%]

============================== 3 passed in 0.81s ===============================
```

## Full Suite Pytest

Command:

```bash
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python -m pytest tests/ -v
```

Output summary:

```text
============================= test session starts ==============================
platform darwin -- Python 3.11.13, pytest-9.0.3, pluggy-1.6.0 -- /Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python
cachedir: .pytest_cache
rootdir: /Users/zonghuiliu/Documents/Codex/VideoGen/FlashVSR_Attention_Map_Distillation
configfile: pytest.ini
collecting ... collected 86 items

tests/review_logic/test_b1_contract_gaps.py::test_trainer_accepts_b1wanmodel_layer_aux_contract FAILED [  1%]
tests/review_logic/test_b1_contract_gaps.py::test_bsa_block_size_is_compatible_with_flashvsr_patchified_grid PASSED [  2%]
tests/review_logic/test_b1_contract_gaps.py::test_prepare_batch_outputs_channels_accepted_by_wan_patch_embedding FAILED [  3%]
tests/review_logic/test_b1_contract_gaps.py::test_bucket_sampler_keeps_bucket_choice_synchronized_across_ranks FAILED [  4%]
...
tests/test_wan_dit_b1.py::test_b1_wan_model_distill_layers_default PASSED [100%]

=================================== FAILURES ===================================
FAILED tests/review_logic/test_b1_contract_gaps.py::test_trainer_accepts_b1wanmodel_layer_aux_contract
FAILED tests/review_logic/test_b1_contract_gaps.py::test_prepare_batch_outputs_channels_accepted_by_wan_patch_embedding
FAILED tests/review_logic/test_b1_contract_gaps.py::test_bucket_sampler_keeps_bucket_choice_synchronized_across_ranks
=================== 3 failed, 80 passed, 3 skipped in 27.71s ===================
```

Known remaining failures correspond to Fix A, Fix D, and Fix H.
