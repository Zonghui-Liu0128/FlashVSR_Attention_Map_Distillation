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
        #   `aigc_input` — degraded LR tensor upsampled to GT size (T, 3, H, W)
        #   `read_input` — HR/GT tensor (T, 3, H, W)
        # Both are at HR pixel resolution in [0, 1]. Re-key to `lr` / `hr` and
        # normalize to FlashVSR's CTHW, [-1, 1] input contract.
        if "lr" not in item and "aigc_input" in item:
            item["lr"] = item["aigc_input"]
        if "hr" not in item and "read_input" in item:
            item["hr"] = item["read_input"]
        if "lr" not in item:
            raise KeyError(
                f"DatasetB1 expected 'lr' (or 'aigc_input') in parent item; "
                f"got keys: {sorted(item.keys())}"
            )
        if "hr" not in item:
            raise KeyError(
                f"DatasetB1 expected 'hr' (or 'read_input') in parent item; "
                f"got keys: {sorted(item.keys())}"
            )
        item["lr"] = self._to_flashvsr_video_tensor(item["lr"], "lr")
        item["hr"] = self._to_flashvsr_video_tensor(item["hr"], "hr")
        if item["lr"].shape[1] != item["hr"].shape[1]:
            raise ValueError(
                f"DatasetB1 expected lr/hr to have the same frame count; "
                f"got lr={tuple(item['lr'].shape)}, hr={tuple(item['hr'].shape)}"
            )
        h, w = item["lr"].shape[-2:]
        landscape = w > h
        item["aspect_bucket"] = "landscape" if landscape else "portrait"
        item["latent_shape"] = self._latent_shape_from_video(
            frame_count=int(item["lr"].shape[1]),
            height=int(h),
            width=int(w),
        )
        return item

    @staticmethod
    def _to_flashvsr_video_tensor(video, name: str):
        if video.ndim != 4:
            raise ValueError(
                f"DatasetB1 expected {name} as 4D CTHW or TCHW RGB video; "
                f"got shape={tuple(video.shape)}"
            )
        if video.shape[0] == 3:
            out = video.contiguous()
        elif video.shape[1] == 3:
            out = video.permute(1, 0, 2, 3).contiguous()
        else:
            raise ValueError(
                f"DatasetB1 expected {name} as CTHW or TCHW RGB video; "
                f"got shape={tuple(video.shape)}"
            )

        if out.numel() and float(out.amin()) >= 0.0 and float(out.amax()) <= 1.0:
            out = out.mul(2).sub(1)
        return out

    @staticmethod
    def _latent_time_from_frame_count(frame_count: int) -> int:
        if frame_count <= 0:
            raise ValueError(f"DatasetB1 expected positive frame count; got {frame_count}")
        return (frame_count + 3) // 4

    @classmethod
    def _latent_shape_from_video(cls, *, frame_count: int, height: int, width: int) -> tuple[int, int, int]:
        if height <= 0 or width <= 0:
            raise ValueError(f"DatasetB1 expected positive video shape; got height={height}, width={width}")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"DatasetB1 expected H/W divisible by 16 for Causal_LQ4x_Proj; "
                f"got height={height}, width={width}"
            )
        return (cls._latent_time_from_frame_count(frame_count), height // 16, width // 16)

    @staticmethod
    def _bucket_for_sample(sample: dict[str, Any]) -> str:
        crop_height = int(sample["crop_height"])
        crop_width = int(sample["crop_width"])
        return "landscape" if crop_width > crop_height else "portrait"
