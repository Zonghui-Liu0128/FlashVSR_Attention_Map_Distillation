# Fix J - Vendor LSWA degradation

Status: BLOCKED

## Summary

Vendored LSWA `degradation/` and `sample_index.py` into `flashvsr_b1/data/`, removed the sibling `FlashVSR_LSWA` `sys.path` bootstrap from the B1 dataset wrapper, and removed the unused bootstrap from `flashvsr_b1/models/flashvsr_components.py`.

Import verification is blocked in the requested conda env because `cv2` is not installed:

```text
ModuleNotFoundError: No module named 'cv2'
```

Per the Fix J instruction, full pytest was not run after this new vendored import error.

## Files Added

- `flashvsr_b1/data/sample_index.py` - 16,360 bytes
- `flashvsr_b1/data/degradation/__init__.py` - 60 bytes
- `flashvsr_b1/data/degradation/basic_vsr_dataset_hw_crop.py` - 34,472 bytes
- `flashvsr_b1/data/degradation/dataset_common_utils.py` - 36,206 bytes
- `flashvsr_b1/data/degradation/degradations.py` - 31,743 bytes
- `flashvsr_b1/data/degradation/operators.py` - 7,735 bytes

## Files Modified

- `flashvsr_b1/data/degradation/basic_vsr_dataset_hw_crop.py`
  - Only authorized vendored edit: `from sample_index import build_sample_records_from_metadata` -> `from ..sample_index import build_sample_records_from_metadata`
- `flashvsr_b1/data/dataset_b1.py`
  - Removed sibling LSWA `sys.path` bootstrap and changed import to `.degradation.basic_vsr_dataset_hw_crop`
- `flashvsr_b1/models/flashvsr_components.py`
  - Removed unused sibling LSWA `sys.path` bootstrap

## Verbatim Import-Smoke Output

Command:

```bash
/Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python -c "
from flashvsr_b1.data.dataset_b1 import DatasetB1
from flashvsr_b1.data.degradation.basic_vsr_dataset_hw_crop import BasicVSRDataset_hw_crop
from flashvsr_b1.data.sample_index import build_sample_records_from_metadata
print('vendored degradation modules import OK')
"
```

stdout:

```text
```

stderr:

```text
Traceback (most recent call last):
  File "<string>", line 2, in <module>
  File "/Users/zonghuiliu/Documents/Codex/VideoGen/FlashVSR_Attention_Map_Distillation/flashvsr_b1/data/dataset_b1.py", line 5, in <module>
    from .degradation.basic_vsr_dataset_hw_crop import BasicVSRDataset_hw_crop
  File "/Users/zonghuiliu/Documents/Codex/VideoGen/FlashVSR_Attention_Map_Distillation/flashvsr_b1/data/degradation/basic_vsr_dataset_hw_crop.py", line 26, in <module>
    import cv2
ModuleNotFoundError: No module named 'cv2'
```

## Verbatim Pytest Output

Not run. The import smoke failed with a new vendored import error, and the instruction said to stop and report if that occurred.

## Dependency Callout

The vendored files import these third-party packages that must be present in the B200 conda env:

- `cv2` / OpenCV
- `numpy`
- `torch`
- `torchvision`
- `scipy`
- `pandas`
- `imageio`
- `PIL` / Pillow
- `einops` is still required elsewhere in the project import graph, including `flashvsr_b1/models/flashvsr_components.py`

## Scope Checks

- `flashvsr_b1/data/__init__.py` was not modified.
- `flashvsr_b1/configs/data_b1.yaml` was not modified.
- Docs, task plans, and tests were not modified.
- `flashvsr_b1/models/flashvsr_components.py` has no remaining `_LSWA_ROOT`, `os.`, or `sys.` references.
- Vendored LSWA files are byte-identical to upstream except for the one authorized relative import in `basic_vsr_dataset_hw_crop.py`.
