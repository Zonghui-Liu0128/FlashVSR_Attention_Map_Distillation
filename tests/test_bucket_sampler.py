import torch
from flashvsr_b1.data.bucket_sampler import AspectRatioBucketSampler

class FakeDataset:
    def __init__(self, n_land=120, n_port=80):
        self.bucket_index = (["landscape"] * n_land) + (["portrait"] * n_port)
    def __len__(self):
        return len(self.bucket_index)

def test_each_batch_is_single_bucket():
    ds = FakeDataset(n_land=120, n_port=80)
    sampler = AspectRatioBucketSampler(ds, num_replicas=1, rank=0,
                                        batch_size=4, seed=0)
    batches = []
    cur = []
    for idx in sampler:
        cur.append(idx)
        if len(cur) == 4:
            batches.append(cur); cur = []
    for batch in batches:
        buckets = {ds.bucket_index[i] for i in batch}
        assert len(buckets) == 1

def test_bucket_ratio_close_to_dataset_ratio():
    ds = FakeDataset(n_land=160, n_port=40)
    sampler = AspectRatioBucketSampler(ds, num_replicas=1, rank=0,
                                        batch_size=4, seed=0)
    counts = {"landscape": 0, "portrait": 0}
    for idx in sampler:
        counts[ds.bucket_index[idx]] += 1
    ratio = counts["landscape"] / max(counts["portrait"], 1)
    assert 3.5 < ratio < 4.5

def test_drop_last_enforces_full_batches():
    ds = FakeDataset(n_land=122, n_port=83)
    sampler = AspectRatioBucketSampler(ds, num_replicas=1, rank=0,
                                        batch_size=4, seed=0)
    idxs = list(sampler)
    assert len(idxs) % 4 == 0

def test_ddp_ranks_disjoint_and_complete():
    ds = FakeDataset(n_land=120, n_port=80)
    s0 = AspectRatioBucketSampler(ds, num_replicas=2, rank=0, batch_size=4, seed=42)
    s1 = AspectRatioBucketSampler(ds, num_replicas=2, rank=1, batch_size=4, seed=42)
    a, b = set(s0), set(s1)
    assert len(a & b) == 0
    assert len(a | b) >= int(0.95 * len(ds))
