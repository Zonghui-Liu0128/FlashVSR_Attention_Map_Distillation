from __future__ import annotations

import os, sys
from typing import Any

_LSWA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "FlashVSR_LSWA"))
if _LSWA_ROOT not in sys.path:
    sys.path.insert(0, _LSWA_ROOT)

from degradation.basic_vsr_dataset_hw_crop import BasicVSRDataset_hw_crop


class DatasetB1(BasicVSRDataset_hw_crop):
    """Wrap BasicVSRDataset_hw_crop with aspect_bucket and latent_shape fields."""

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = super().__getitem__(idx)
        h, w = item["lr"].shape[-2:]
        landscape = w > h
        item["aspect_bucket"] = "landscape" if landscape else "portrait"
        item["latent_shape"] = (22, 64, 120) if landscape else (22, 120, 64)
        return item
