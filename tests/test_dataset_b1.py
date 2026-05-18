import sys
import types
import torch
from unittest.mock import patch


class _FakeBasicVSRDataset_hw_crop:
    def __getitem__(self, idx):
        raise NotImplementedError


# Post-Fix-J: degradation/ is vendored under flashvsr_b1.data, so dataset_b1
# now does `from .degradation.basic_vsr_dataset_hw_crop import ...` (relative
# package import). Stub the modules under the new namespace so cv2 / imageio
# / pandas don't have to be installed in the test env. Stub the parent
# package too so the `from .degradation...` resolves without executing
# `flashvsr_b1/data/degradation/__init__.py`.
_fake_pkg_root = types.ModuleType("flashvsr_b1.data.degradation")
_fake_parent = types.ModuleType(
    "flashvsr_b1.data.degradation.basic_vsr_dataset_hw_crop"
)
_fake_parent.BasicVSRDataset_hw_crop = _FakeBasicVSRDataset_hw_crop
sys.modules.setdefault("flashvsr_b1.data.degradation", _fake_pkg_root)
sys.modules["flashvsr_b1.data.degradation.basic_vsr_dataset_hw_crop"] = _fake_parent

from flashvsr_b1.data.dataset_b1 import DatasetB1


def _fake_parent_item(h, w):
    return {
        "lr": torch.zeros(3, 85, h, w),
        "hr": torch.zeros(3, 85, h * 4, w * 4),
        "sample_meta": {},
        "degradation_meta": {},
        "data_name": "dummy.mp4",
    }


def test_landscape_bucket_and_latent_shape():
    with patch.object(DatasetB1, "__init__", lambda self, *a, **k: None), patch.object(
        DatasetB1.__bases__[0],
        "__getitem__",
        lambda self, idx: _fake_parent_item(1024, 1920),
    ):
        ds = DatasetB1()
        item = ds[0]
        assert item["aspect_bucket"] == "landscape"
        assert item["latent_shape"] == (22, 64, 120)


def test_portrait_bucket_and_latent_shape():
    with patch.object(DatasetB1, "__init__", lambda self, *a, **k: None), patch.object(
        DatasetB1.__bases__[0],
        "__getitem__",
        lambda self, idx: _fake_parent_item(1920, 1024),
    ):
        ds = DatasetB1()
        item = ds[0]
        assert item["aspect_bucket"] == "portrait"
        assert item["latent_shape"] == (22, 120, 64)


def test_parent_fields_preserved():
    with patch.object(DatasetB1, "__init__", lambda self, *a, **k: None), patch.object(
        DatasetB1.__bases__[0],
        "__getitem__",
        lambda self, idx: _fake_parent_item(1024, 1920),
    ):
        ds = DatasetB1()
        item = ds[0]
        for k in ["lr", "hr", "sample_meta", "degradation_meta", "data_name"]:
            assert k in item
