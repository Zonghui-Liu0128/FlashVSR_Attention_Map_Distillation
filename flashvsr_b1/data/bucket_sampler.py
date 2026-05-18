from __future__ import annotations

from collections import defaultdict
from typing import Iterator

import torch
from torch.utils.data.distributed import DistributedSampler


class AspectRatioBucketSampler(DistributedSampler):
    def __init__(
        self,
        dataset,
        *,
        num_replicas: int,
        rank: int,
        batch_size: int,
        seed: int = 0,
        drop_last: bool = True,
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False,
            seed=seed,
            drop_last=drop_last,
        )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        bucket_index = getattr(dataset, "bucket_index", None)
        if bucket_index is None:
            raise ValueError("dataset must expose bucket_index: list[str]")
        if len(bucket_index) != len(dataset):
            raise ValueError(
                f"dataset.bucket_index length {len(bucket_index)} does not match len(dataset) {len(dataset)}"
            )
        invalid = set(bucket_index) - {"landscape", "portrait"}
        if invalid:
            raise ValueError(f"dataset.bucket_index contains unsupported buckets: {sorted(invalid)}")

        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        super_chunks = self._build_interleaved_chunks()
        start = self.rank * self.batch_size
        end = start + self.batch_size
        indices = [idx for chunk in super_chunks for idx in chunk[start:end]]
        return iter(indices)

    def __len__(self) -> int:
        return len(self._build_interleaved_chunks()) * self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_interleaved_chunks(self) -> list[list[int]]:
        chunks_by_bucket: dict[str, list[list[int]]] = {}
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        for bucket, indices in self._group_indices().items():
            perm = torch.randperm(len(indices), generator=generator).tolist()
            shuffled = [indices[i] for i in perm]
            chunks_by_bucket[bucket] = self._chunk_indices(shuffled)
        return self._interleave_chunks(chunks_by_bucket)

    def _group_indices(self) -> dict[str, list[int]]:
        grouped: dict[str, list[int]] = defaultdict(list)
        for idx, bucket in enumerate(self.dataset.bucket_index):
            grouped[bucket].append(idx)
        return dict(grouped)

    def _chunk_indices(self, indices: list[int]) -> list[list[int]]:
        super_chunk_size = self.batch_size * self.num_replicas
        chunks = []
        for start in range(0, len(indices), super_chunk_size):
            chunk = indices[start : start + super_chunk_size]
            if len(chunk) == super_chunk_size or not self.drop_last:
                chunks.append(chunk)
        return chunks

    @staticmethod
    def _interleave_chunks(chunks_by_bucket: dict[str, list[list[int]]]) -> list[list[int]]:
        ordered_buckets = sorted(chunks_by_bucket)
        total_by_bucket = {bucket: len(chunks_by_bucket[bucket]) for bucket in ordered_buckets}
        remaining = sum(total_by_bucket.values())
        if remaining == 0:
            return []

        cursor = {bucket: 0 for bucket in ordered_buckets}
        emitted = {bucket: 0 for bucket in ordered_buckets}
        interleaved: list[list[int]] = []
        while len(interleaved) < remaining:
            emitted_total = len(interleaved)
            candidates = [bucket for bucket in ordered_buckets if cursor[bucket] < total_by_bucket[bucket]]
            bucket = max(
                candidates,
                key=lambda name: (
                    (emitted_total + 1) * total_by_bucket[name] - emitted[name] * remaining,
                    total_by_bucket[name],
                    name,
                ),
            )
            interleaved.append(chunks_by_bucket[bucket][cursor[bucket]])
            cursor[bucket] += 1
            emitted[bucket] += 1
        return interleaved
