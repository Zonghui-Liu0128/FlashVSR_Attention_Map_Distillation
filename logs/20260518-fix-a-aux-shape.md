# Fix A Aux Shape Report

Status: DONE

Files modified:
- `flashvsr_b1/models/wan_dit_b1.py`
- `tests/review_logic/test_b1_contract_gaps.py`

Notes:
- `B1WanModel.forward` now aggregates aux as `aux["h_out"][layer_idx]` and `aux["A_blk"][layer_idx]`.
- `test_trainer_accepts_b1wanmodel_layer_aux_contract` now uses the same spec-shaped aux contract.
- `task_b1.md`, `flashvsr_b1/train/trainer_b1.py`, `tests/test_trainer_b1.py`, and `tests/review_logic/test_review_real_logic.py` were not modified.

## Targeted acceptance

Command:

```bash
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python -m pytest \
    tests/review_logic/test_b1_contract_gaps.py::test_trainer_accepts_b1wanmodel_layer_aux_contract \
    tests/review_logic/test_review_real_logic.py::test_C10_compute_loss_end_to_end_smoke \
    tests/test_trainer_b1.py \
    -v
```

Output:

```text
============================= test session starts ==============================
platform darwin -- Python 3.11.13, pytest-9.0.3, pluggy-1.6.0 -- /Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python
cachedir: .pytest_cache
rootdir: /Users/zonghuiliu/Documents/Codex/VideoGen/FlashVSR_Attention_Map_Distillation
configfile: pytest.ini
collecting ... collected 5 items

tests/review_logic/test_b1_contract_gaps.py::test_trainer_accepts_b1wanmodel_layer_aux_contract PASSED [ 20%]
tests/review_logic/test_review_real_logic.py::test_C10_compute_loss_end_to_end_smoke PASSED [ 40%]
tests/test_trainer_b1.py::test_compute_loss_assembles_all_four_terms_for_bsa PASSED [ 60%]
tests/test_trainer_b1.py::test_compute_loss_skips_block_for_lswa PASSED  [ 80%]
tests/test_trainer_b1.py::test_compute_loss_set_current_sparsity_called_for_bsa_only PASSED [100%]

============================== 5 passed in 0.71s ===============================
```

## Full suite

Command:

```bash
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python -m pytest tests/ -v
```

Output:

```text
============================= test session starts ==============================
platform darwin -- Python 3.11.13, pytest-9.0.3, pluggy-1.6.0 -- /Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python
cachedir: .pytest_cache
rootdir: /Users/zonghuiliu/Documents/Codex/VideoGen/FlashVSR_Attention_Map_Distillation
configfile: pytest.ini
collecting ... collected 86 items

tests/review_logic/test_b1_contract_gaps.py::test_trainer_accepts_b1wanmodel_layer_aux_contract PASSED [  1%]
tests/review_logic/test_b1_contract_gaps.py::test_bsa_block_size_is_compatible_with_flashvsr_patchified_grid PASSED [  2%]
tests/review_logic/test_b1_contract_gaps.py::test_prepare_batch_outputs_channels_accepted_by_wan_patch_embedding FAILED [  3%]
tests/review_logic/test_b1_contract_gaps.py::test_bucket_sampler_keeps_bucket_choice_synchronized_across_ranks FAILED [  4%]
...
tests/test_wan_dit_b1.py::test_b1_wan_model_distill_layers_default PASSED [100%]

=================================== FAILURES ===================================
_____ test_prepare_batch_outputs_channels_accepted_by_wan_patch_embedding ______
E       AssertionError: prepare_batch produced 1536 channels and z_t 1536 channels, but Wan patch_embedding expects in_dim=16.
E       assert 1536 == 16

______ test_bucket_sampler_keeps_bucket_choice_synchronized_across_ranks _______
E           AssertionError: DDP step 0 uses different buckets across ranks: rank0=portrait, rank1=landscape.
E           assert 'portrait' == 'landscape'

=========================== short test summary info ============================
FAILED tests/review_logic/test_b1_contract_gaps.py::test_prepare_batch_outputs_channels_accepted_by_wan_patch_embedding
FAILED tests/review_logic/test_b1_contract_gaps.py::test_bucket_sampler_keeps_bucket_choice_synchronized_across_ranks
=================== 2 failed, 81 passed, 3 skipped in 27.51s ===================
```
