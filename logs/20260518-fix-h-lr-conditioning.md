# Fix H - LR Conditioning Contract

Status: DONE

## Files Modified

- `flashvsr_b1/pipelines/b1_pipeline.py`
- `flashvsr_b1/models/wan_dit_b1.py`
- `tests/review_logic/test_b1_contract_gaps.py`
- `tests/review_logic/test_review_real_logic.py`
- `tests/test_trainer_b1.py`
- `tests/test_wan_dit_b1.py`

## Summary

- `B1Pipeline.prepare_batch` now returns `LR_latents` as upstream-style `list[Tensor]` token tensors and `z_t` as a 16-channel pre-patch VAE latent.
- `B1WanModel.b1_forward` now rejects tensor LR conditioning, forwards raw `z_t`, and threads `LQ_latents` into `forward`.
- `B1WanModel.forward` now adds `LQ_latents[layer_idx]` inside the DiT block loop before `_forward_block_b1`, matching vendored `wan_video_dit.py:862-864`.
- Tests were updated to pin the new list-of-token conditioning contract.

## Pytest Output

Targeted acceptance command:

```text
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python -m pytest \
    tests/review_logic/test_b1_contract_gaps.py \
    tests/review_logic/test_review_real_logic.py \
    tests/test_b1_pipeline.py \
    tests/test_trainer_b1.py \
    tests/test_wan_dit_b1.py \
    -v

collected 39 items
39 passed in 1.78s
```

Full suite command:

```text
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python -m pytest tests/ -v

collected 91 items
88 passed, 3 skipped in 28.21s
```

## Fixture Adjustments Beyond Prompt

- Updated `tests/review_logic/test_review_real_logic.py::test_C4_trainer_model_call_matches_B1WanModel_forward_signature` to expect `B1WanModel.b1_forward` parameter name `LR_latents` instead of `LR_latent`, matching the required new method signature.

No test fixture depended on the old additive `LR_latent + z_t` contract in a way that required stopping for adjudication.
