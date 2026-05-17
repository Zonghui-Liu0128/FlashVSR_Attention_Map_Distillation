# Task 9 - Wan DiT B1 Attention Wrapper

## What Changed

- Added `flashvsr_b1/models/wan_dit_b1.py` with:
  - `SelfAttentionB1`
  - `B1WanModel`
  - BSA/LSWA dispatch
  - distillation aux export for layers `{4, 9, 14, 19, 24, 29}`
- Updated `flashvsr_b1/attn/lswa.py` to require explicit `num_heads`.
- Updated `tests/test_lswa.py` call sites to pass `num_heads`.
- Added `tests/test_wan_dit_b1.py` from the task prompt.

## TDD Red Run

Command:

```bash
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python -m pytest tests/test_wan_dit_b1.py -v
```

Result before implementation:

```text
ModuleNotFoundError: No module named 'flashvsr_b1.models.wan_dit_b1'
collected 0 items / 1 error
```

The prompt expected four failing tests, but pytest stops at collection while the module is absent. This is the expected red failure for a missing production module.

## DiffSynth SelfAttention Key Layout

`DiffSynth-Studio/diffsynth/models/wan_video_dit.py` defines `SelfAttention` with separate upstream projection keys:

- `q.weight`, `q.bias`
- `k.weight`, `k.bias`
- `v.weight`, `v.bias`
- `o.weight`, `o.bias`
- `norm_q.weight`
- `norm_k.weight`

There is no upstream `qkv_proj`/`o_proj` fused layout in this DiffSynth copy.

## State-Dict Load Story

`SelfAttentionB1` uses the task-requested fused modules:

- `qkv_proj: Linear(dim, 3 * dim)`
- `o_proj: Linear(dim, dim)`

To preserve strict checkpoint loading from upstream Wan checkpoints, `SelfAttentionB1` registers a load-state-dict pre-hook that remaps:

- upstream `q/k/v` weights and biases into concatenated `qkv_proj`
- upstream `o` into `o_proj`
- upstream `norm_q`/`norm_k` load directly because B1 keeps those modules

Probe command:

```bash
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python - <<'PY'
from flashvsr_b1.models.wan_dit_b1 import B1WanModel, wan_video_dit

kwargs = dict(
    dim=8, in_dim=4, ffn_dim=16, out_dim=4, text_dim=6, freq_dim=8,
    eps=1e-6, patch_size=(1, 1, 1), num_heads=2, num_layers=2,
    has_image_input=False,
)
src = wan_video_dit.WanModel(**kwargs)
dst = B1WanModel(**kwargs)
print(dst.load_state_dict(src.state_dict(), strict=True))
PY
```

Result:

```text
<All keys matched successfully>
```

Strict loading works through the remap hook. Native key parity does not exist because the task explicitly asked for fused B1 projection names while DiffSynth upstream uses split `q/k/v/o` names.

## num_heads Propagation Path

- `B1WanModel` reads each upstream block's `old_attn.num_heads` while replacing `block.self_attn`.
- `SelfAttentionB1.forward(...)` passes `self.num_heads` into `lswa_forward(...)`.
- `lswa_forward(...)` passes `num_heads` into `_local_spatial_attention(...)`.
- `_local_spatial_attention(...)` asserts `D % num_heads == 0` and no longer has a heuristic fallback.

## Verification

Command:

```bash
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python -m pytest tests/test_wan_dit_b1.py tests/test_lswa.py -v
```

Result:

```text
collected 7 items

tests/test_wan_dit_b1.py::test_self_attention_default_attrs PASSED
tests/test_wan_dit_b1.py::test_self_attention_forward_lswa_mode_no_aux PASSED
tests/test_wan_dit_b1.py::test_self_attention_returns_aux_for_distill_layer_lswa PASSED
tests/test_wan_dit_b1.py::test_b1_wan_model_distill_layers_default PASSED
tests/test_lswa.py::test_lswa_output_shape_train_mode PASSED
tests/test_lswa.py::test_lswa_is_causal_in_time PASSED
tests/test_lswa.py::test_lswa_matches_reference_implementation PASSED

7 passed in 0.79s
```

## Deferred / B200 Notes

- BSA mode still requires CUDA and the `block_sparse_attn` package at runtime, as expected.
- CPU tests only exercise LSWA mode and checkpoint key remapping.
- No state-dict gap is deferred to B200; the strict-load remap probe passes for a tiny Wan model.
