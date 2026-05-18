from __future__ import annotations

from typing import Any

from .degradation.basic_vsr_dataset_hw_crop import BasicVSRDataset_hw_crop


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
        # LSWA parent (`degradation/basic_vsr_dataset_hw_crop.py:266-273`) returns
        #   `aigc_input` — degraded LR tensor upsampled to GT size (3, T, H, W)
        #   `read_input` — HR/GT tensor (3, T, H, W)
        # Both are at HR pixel resolution. Re-key to `lr` / `hr` for the
        # `B1Pipeline.prepare_batch` contract. The `not in` guards keep this
        # compatible with mock fixtures that provide `lr` / `hr` directly.
        if "lr" not in item and "aigc_input" in item:
            item["lr"] = item["aigc_input"]
        if "hr" not in item and "read_input" in item:
            item["hr"] = item["read_input"]
        if "lr" not in item:
            raise KeyError(
                f"DatasetB1 expected 'lr' (or 'aigc_input') in parent item; "
                f"got keys: {sorted(item.keys())}"
            )
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
