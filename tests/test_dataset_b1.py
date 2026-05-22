import sys
import types
import torch
from unittest.mock import patch


class _FakeBasicVSRDataset_hw_crop:
    def __getitem__(self, idx):
        raise NotImplementedError


# Post-Fix-J: degradation/ is vendored under flashvsr_b1.data, so dataset_b1
# now does `from .degradation.basic_vsr_dataset_hw_crop import ...` (relative
# package import). Stub only the heavy BasicVSR module so cv2 / imageio / pandas
# don't have to be installed in the test env, while leaving the real package
# importable for sibling degradation modules.
import flashvsr_b1.data.degradation as _real_degradation_pkg

_fake_parent = types.ModuleType(
    "flashvsr_b1.data.degradation.basic_vsr_dataset_hw_crop"
)
_fake_parent.BasicVSRDataset_hw_crop = _FakeBasicVSRDataset_hw_crop
sys.modules.setdefault("flashvsr_b1.data.degradation", _real_degradation_pkg)
sys.modules["flashvsr_b1.data.degradation.basic_vsr_dataset_hw_crop"] = _fake_parent

from flashvsr_b1.data.dataset_b1 import DatasetB1


def _fake_parent_item(h, w, frames=85):
    return {
        "lr": torch.zeros(3, frames, h, w),
        "hr": torch.zeros(3, frames, h, w),
        "sample_meta": {},
        "degradation_meta": {},
        "data_name": "dummy.mp4",
    }


def test_landscape_bucket_and_latent_shape():
    with patch.object(DatasetB1, "__init__", lambda self, *a, **k: None), patch.object(
        DatasetB1.__bases__[0],
        "__getitem__",
        lambda self, idx: _fake_parent_item(32, 64),
    ):
        ds = DatasetB1()
        item = ds[0]
        assert item["aspect_bucket"] == "landscape"
        assert item["latent_shape"] == (22, 2, 4)


def test_portrait_bucket_and_latent_shape():
    with patch.object(DatasetB1, "__init__", lambda self, *a, **k: None), patch.object(
        DatasetB1.__bases__[0],
        "__getitem__",
        lambda self, idx: _fake_parent_item(64, 32),
    ):
        ds = DatasetB1()
        item = ds[0]
        assert item["aspect_bucket"] == "portrait"
        assert item["latent_shape"] == (22, 4, 2)


def test_parent_fields_preserved():
    with patch.object(DatasetB1, "__init__", lambda self, *a, **k: None), patch.object(
        DatasetB1.__bases__[0],
        "__getitem__",
        lambda self, idx: _fake_parent_item(32, 64),
    ):
        ds = DatasetB1()
        item = ds[0]
        for k in ["lr", "hr", "sample_meta", "degradation_meta", "data_name"]:
            assert k in item


def test_latent_shape_time_follows_truncated_45_frame_clip():
    with patch.object(DatasetB1, "__init__", lambda self, *a, **k: None), patch.object(
        DatasetB1.__bases__[0],
        "__getitem__",
        lambda self, idx: _fake_parent_item(32, 64, frames=45),
    ):
        ds = DatasetB1()
        item = ds[0]
        assert item["lr"].shape[1] == 45
        assert item["latent_shape"] == (12, 2, 4)


def test_latent_shape_supports_960x720_93_frame_csv_target():
    with patch.object(DatasetB1, "__init__", lambda self, *a, **k: None), patch.object(
        DatasetB1.__bases__[0],
        "__getitem__",
        lambda self, idx: _fake_parent_item(720, 960, frames=93),
    ):
        ds = DatasetB1()
        item = ds[0]

    assert item["aspect_bucket"] == "landscape"
    assert item["latent_shape"] == (24, 45, 60)


def test_real_parent_tchw_zero_one_video_is_normalized_to_flashvsr_contract():
    aigc_input = torch.tensor(
        [
            [[[0.0, 0.5], [1.0, 0.25]], [[0.5, 1.0], [0.0, 0.75]], [[1.0, 0.0], [0.5, 0.25]]],
            [[[0.25, 0.75], [0.5, 1.0]], [[1.0, 0.5], [0.25, 0.0]], [[0.0, 0.25], [0.75, 1.0]]],
        ]
    ).repeat(1, 1, 8, 8)
    parent_item = {
        "aigc_input": aigc_input,
        "read_input": torch.ones(2, 3, 16, 16),
        "sample_meta": {},
        "degradation_meta": {},
        "data_name": "dummy.mp4",
    }
    with patch.object(DatasetB1, "__init__", lambda self, *a, **k: None), patch.object(
        DatasetB1.__bases__[0],
        "__getitem__",
        lambda self, idx: parent_item,
    ):
        ds = DatasetB1()
        item = ds[0]

    assert item["lr"].shape == (3, 2, 16, 16)
    assert item["hr"].shape == (3, 2, 16, 16)
    assert torch.allclose(item["lr"], parent_item["aigc_input"].permute(1, 0, 2, 3).mul(2).sub(1))
    assert item["lr"].amin().item() == -1.0
    assert item["lr"].amax().item() == 1.0
