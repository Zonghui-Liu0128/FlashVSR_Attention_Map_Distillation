from __future__ import annotations

import os, sys
from typing import Any

_LSWA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "FlashVSR_LSWA"))
if _LSWA_ROOT not in sys.path:
    sys.path.insert(0, _LSWA_ROOT)

from degradation.basic_vsr_dataset_hw_crop import BasicVSRDataset_hw_crop


class DatasetB1(BasicVSRDataset_hw_crop):
    """Wrap BasicVSRDataset_hw_crop with aspect_bucket and latent_shape fields."""

    def __init__(self, opt: dict[str, Any]):
        super().__init__(opt)
        # BasicVSRDataset_hw_crop stores its loaded, shuffled, repeated sample
        # records in self.imgs; each record carries crop_height/crop_width.
        samples = getattr(self, "imgs", None)
        if samples is None:
            raise RuntimeError("BasicVSRDataset_hw_crop did not expose self.imgs for bucket indexing")
        self.bucket_index: list[str] = [self._bucket_for_sample(sample) for sample in samples]
        if len(self.bucket_index) != len(self):
            raise RuntimeError(f"bucket_index length {len(self.bucket_index)} does not match dataset length {len(self)}")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = super().__getitem__(idx)
        h, w = item["lr"].shape[-2:]
        landscape = w > h
        item["aspect_bucket"] = "landscape" if landscape else "portrait"
        item["latent_shape"] = (22, 64, 120) if landscape else (22, 120, 64)
        return item

    @staticmethod
    def _bucket_for_sample(sample: dict[str, Any]) -> str:
        crop_height = int(sample["crop_height"])
        crop_width = int(sample["crop_width"])
        return "landscape" if crop_width > crop_height else "portrait"
