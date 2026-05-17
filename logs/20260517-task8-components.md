# Task 8 - FlashVSR Components

## Ported APIs

- `FlashVSRTinyConfig`: adapted from LSWA dataclass. Kept the requested public fields and defaults: `patch_size=(1, 2, 2)`, `dim=1536`, `ffn_dim=8960`, `num_heads=12`, `num_layers=30`, `in_dim=16`, `out_dim=16`.
- `Causal_LQ4x_Proj`: ported LSWA helper modules (`RMS_norm`, `CausalConv3d`, `PixelShuffle3d`) and projection stack. Adapted the constructor defaults to `in_dim=3`, `out_dim=1536`, `layer_num=1`. Adapted `forward` to return a tensor with channel dimension at index 1 for the B1 public API test contract.
- `build_tc_decoder`: adapted from LSWA `build_tcdecoder`. The LSWA version constructs `TAEHV` on CUDA with `bfloat16`; this B1 wrapper returns a minimal `nn.Module` stub when `checkpoint_path is None` so CPU-only macOS tests do not require CUDA or a checkpoint.
- `load_flashvsr_tiny_checkpoint`: adapted from LSWA checkpoint helpers. Supports nested checkpoint dicts, `.safetensors`, strict loading, and the LSWA non-strict prefix/shape normalization path.

## TCDecoder Stub Strategy

`build_tc_decoder(checkpoint_path=None)` returns an identity-style `_TCDecoderStub(nn.Module)`. This keeps the no-checkpoint path cheap and portable for unit tests. When a checkpoint path is supplied, the port constructs the LSWA-style `TAEHV` decoder on CPU, loads weights with non-strict normalization, sets train mode, and clears temporal memory.

## LSWA Dependency Concerns

- `flashvsr_b1/models/flashvsr_components.py` uses the same `sys.path` bootstrap style as `flashvsr_b1/data/dataset_b1.py` to make `FlashVSR_LSWA` importable if later LSWA-local dependencies are needed.
- The upstream TC decoder builder defaults to CUDA and `torch.bfloat16`, which is not safe for the CPU-only macOS test path.
- The shell default `python` at `/Users/zonghuiliu/anaconda3/bin/python` does not have `torch`; the project env at `/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python` has `torch 2.6.0`.

## Pytest Output

Initial RED run after adding only `tests/test_flashvsr_components.py`:

```text
============================= test session starts ==============================
platform darwin -- Python 3.10.11, pytest-9.0.3, pluggy-1.6.0 -- /Users/zonghuiliu/anaconda3/bin/python
cachedir: .pytest_cache
rootdir: /Users/zonghuiliu/Documents/Codex/VideoGen/FlashVSR_Attention_Map_Distillation
configfile: pytest.ini
plugins: anyio-4.0.0
collecting ... collected 0 items / 1 error

==================================== ERRORS ====================================
______________ ERROR collecting tests/test_flashvsr_components.py ______________
ImportError while importing test module '/Users/zonghuiliu/Documents/Codex/VideoGen/FlashVSR_Attention_Map_Distillation/tests/test_flashvsr_components.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../../../anaconda3/lib/python3.10/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/test_flashvsr_components.py:1: in <module>
    import torch
E   ModuleNotFoundError: No module named 'torch'
=========================== short test summary info ============================
ERROR tests/test_flashvsr_components.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
=============================== 1 error in 0.07s ===============================
```

Final requested command:

```text
============================= test session starts ==============================
platform darwin -- Python 3.10.11, pytest-9.0.3, pluggy-1.6.0 -- /Users/zonghuiliu/anaconda3/bin/python
cachedir: .pytest_cache
rootdir: /Users/zonghuiliu/Documents/Codex/VideoGen/FlashVSR_Attention_Map_Distillation
configfile: pytest.ini
plugins: anyio-4.0.0
collecting ... collected 0 items / 1 error

==================================== ERRORS ====================================
______________ ERROR collecting tests/test_flashvsr_components.py ______________
ImportError while importing test module '/Users/zonghuiliu/Documents/Codex/VideoGen/FlashVSR_Attention_Map_Distillation/tests/test_flashvsr_components.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../../../anaconda3/lib/python3.10/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/test_flashvsr_components.py:1: in <module>
    import torch
E   ModuleNotFoundError: No module named 'torch'
=========================== short test summary info ============================
ERROR tests/test_flashvsr_components.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
=============================== 1 error in 0.07s ===============================
```

Final torch-enabled project env command:

```text
============================= test session starts ==============================
platform darwin -- Python 3.11.13, pytest-9.0.3, pluggy-1.6.0 -- /Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python
cachedir: .pytest_cache
rootdir: /Users/zonghuiliu/Documents/Codex/VideoGen/FlashVSR_Attention_Map_Distillation
configfile: pytest.ini
collecting ... collected 3 items

tests/test_flashvsr_components.py::test_tiny_config_defaults PASSED      [ 33%]
tests/test_flashvsr_components.py::test_lq_proj_forward_shape PASSED     [ 66%]
tests/test_flashvsr_components.py::test_tc_decoder_builds_without_checkpoint PASSED [100%]

============================== 3 passed in 2.09s ===============================
```
