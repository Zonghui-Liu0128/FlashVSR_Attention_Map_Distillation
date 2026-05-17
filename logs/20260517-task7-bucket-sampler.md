# Task 7: Aspect-Ratio Bucket Sampler

## AspectRatioBucketSampler chunk interleaving

`AspectRatioBucketSampler` first groups dataset indices by `dataset.bucket_index`, then shuffles each bucket with `torch.Generator().manual_seed(seed + epoch)`. Each bucket is packed into `batch_size` chunks, with partial tail chunks dropped when `drop_last=True`.

The sampler then merges bucket chunks with a deterministic weighted interleave. At each output position it chooses the bucket with the largest proportional lag relative to its total chunk count, so buckets with more chunks are emitted more often while every emitted chunk still contains indices from only one bucket. After the global chunk order is built, each DDP rank takes whole chunks at positions `rank, rank + num_replicas, rank + 2 * num_replicas, ...`, which keeps rank-local batches single-bucket and makes ranks deterministic without broadcast.

## DatasetB1 sample-list hook

I read `../FlashVSR_LSWA/degradation/basic_vsr_dataset_hw_crop.py` before changing `DatasetB1`. The parent `BasicVSRDataset_hw_crop.__init__` loads the sample index JSON through `sample_json_path`, extracts `sample_index["samples"]`, applies `max_samples`, optional `random.shuffle(samples)`, repeats by `data_repeat`, and stores the resulting list on `self.imgs`.

`DatasetB1.__init__` now calls `super().__init__(opt)` and builds `self.bucket_index` from `self.imgs`. Each sample record uses the confirmed parent field names `crop_height` and `crop_width`; the bucket is `"landscape"` when `crop_width > crop_height`, otherwise `"portrait"`. This avoids loading video and keeps `len(self.bucket_index) == len(self)`.

## Pytest output

RED before implementation:

```text
============================= test session starts ==============================
platform darwin -- Python 3.11.13, pytest-9.0.3, pluggy-1.6.0 -- /Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python
cachedir: .pytest_cache
rootdir: /Users/zonghuiliu/Documents/Codex/VideoGen/FlashVSR_Attention_Map_Distillation
configfile: pytest.ini
collecting ... collected 0 items / 1 error

==================================== ERRORS ====================================
________________ ERROR collecting tests/test_bucket_sampler.py _________________
ImportError while importing test module '/Users/zonghuiliu/Documents/Codex/VideoGen/FlashVSR_Attention_Map_Distillation/tests/test_bucket_sampler.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../../../anaconda3/envs/flashvsr/lib/python3.11/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/test_bucket_sampler.py:2: in <module>
    from flashvsr_b1.data.bucket_sampler import AspectRatioBucketSampler
E   ModuleNotFoundError: No module named 'flashvsr_b1.data.bucket_sampler'
=========================== short test summary info ============================
ERROR tests/test_bucket_sampler.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
=============================== 1 error in 0.97s ===============================
```

GREEN after implementation:

```text
============================= test session starts ==============================
platform darwin -- Python 3.11.13, pytest-9.0.3, pluggy-1.6.0 -- /Users/zonghuiliu/anaconda3/envs/flashvsr/bin/python
cachedir: .pytest_cache
rootdir: /Users/zonghuiliu/Documents/Codex/VideoGen/FlashVSR_Attention_Map_Distillation
configfile: pytest.ini
collecting ... collected 4 items

tests/test_bucket_sampler.py::test_each_batch_is_single_bucket PASSED    [ 25%]
tests/test_bucket_sampler.py::test_bucket_ratio_close_to_dataset_ratio PASSED [ 50%]
tests/test_bucket_sampler.py::test_drop_last_enforces_full_batches PASSED [ 75%]
tests/test_bucket_sampler.py::test_ddp_ranks_disjoint_and_complete PASSED [100%]

============================== 4 passed in 0.71s ===============================
```

## Concerns about parent class interface

`self.imgs` is the usable parent hook and is already the list consumed by `__len__` / `__getitem__`, but it is not documented as a formal public API. If the parent later renames that attribute, `DatasetB1.__init__` will fail fast with a clear error instead of silently building a stale or mismatched bucket index.

DatasetB1 bucket_index population is production-only code path, validated on B200
