# Fix E: LPIPS 5D Tensor Handling

Status: DONE

## Files modified

- `flashvsr_b1/losses/lpips_loss.py`
- `tests/test_losses.py`

## Summary

- Added `_flatten_video_to_bchw` so LPIPS receives 4D BCHW tensors.
- `L_lpips` now supports 4D BCHW and 5D BCTHW inputs, flattening 5D video frames into the batch axis.
- Added symmetric batch broadcasting for mixed 5D/4D cases.
- Added mock-based tests that verify shape handling without requiring the `lpips` package.
- Added unsupported-rank rejection coverage.

## B200 verification required

macOS can only verify the mock-based shape logic because `lpips` is not installed locally. The real `tests/test_losses.py::test_L_lpips_shape` remains skipped on macOS and must be verified on B200 with `lpips` installed.

## Pytest output: `tests/test_losses.py`

Command:

```bash
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python -m pytest tests/test_losses.py -v
```

Output:

```text
============================= test session starts ==============================
platform darwin -- Python 3.11.13, pytest-9.0.3, pluggy-1.6.0 -- /Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python
cachedir: .pytest_cache
rootdir: /Users/zonghuiliu/Documents/Codex/VideoGen/FlashVSR_Attention_Map_Distillation
configfile: pytest.ini
collecting ... collected 9 items

tests/test_losses.py::test_L_output_zero_on_equal_tensors PASSED         [ 11%]
tests/test_losses.py::test_L_output_huber_smoothness_near_zero PASSED    [ 22%]
tests/test_losses.py::test_L_block_zero_on_identical_distributions PASSED [ 33%]
tests/test_losses.py::test_L_block_positive_on_different_distributions PASSED [ 44%]
tests/test_losses.py::test_L_block_grad_flows_to_student_only PASSED     [ 55%]
tests/test_losses.py::test_L_attn_out_zero_on_equal PASSED               [ 66%]
tests/test_losses.py::test_L_lpips_shape SKIPPED (lpips not installed)   [ 77%]
tests/test_losses.py::test_L_lpips_shape_handling_with_mock_lpips_net PASSED [ 88%]
tests/test_losses.py::test_L_lpips_rejects_unsupported_ndim PASSED       [100%]

========================= 8 passed, 1 skipped in 0.83s =========================
```

## Pytest output: `tests/`

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
collecting ... collected 89 items

tests/review_logic/test_b1_contract_gaps.py::test_trainer_accepts_b1wanmodel_layer_aux_contract PASSED [  1%]
tests/review_logic/test_b1_contract_gaps.py::test_bsa_block_size_is_compatible_with_flashvsr_patchified_grid PASSED [  2%]
tests/review_logic/test_b1_contract_gaps.py::test_prepare_batch_outputs_channels_accepted_by_wan_patch_embedding FAILED [  3%]
tests/review_logic/test_b1_contract_gaps.py::test_bucket_sampler_keeps_bucket_choice_synchronized_across_ranks PASSED [  4%]
...
tests/test_losses.py::test_L_lpips_shape SKIPPED (lpips not installed)   [ 68%]
tests/test_losses.py::test_L_lpips_shape_handling_with_mock_lpips_net PASSED [ 69%]
tests/test_losses.py::test_L_lpips_rejects_unsupported_ndim PASSED       [ 70%]
...
tests/test_wan_dit_b1.py::test_b1_wan_model_distill_layers_default PASSED [100%]

=================================== FAILURES ===================================
_____ test_prepare_batch_outputs_channels_accepted_by_wan_patch_embedding ______

E       AssertionError: prepare_batch produced 1536 channels and z_t 1536 channels, but Wan patch_embedding expects in_dim=16.
E       assert 1536 == 16
E        +  where 16 = FakeDit().in_dim
E        +    where FakeDit() = B1Pipeline(
E          (dit): FakeDit()
E          (lq_proj): FakeLQProj()
E        ).dit

tests/review_logic/test_b1_contract_gaps.py:122: AssertionError
=========================== short test summary info ============================
FAILED tests/review_logic/test_b1_contract_gaps.py::test_prepare_batch_outputs_channels_accepted_by_wan_patch_embedding
=================== 1 failed, 85 passed, 3 skipped in 27.46s ===================
```
