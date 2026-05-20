import json

import torch

from flashvsr_b1.train.memory_trace import MemoryTrace, summarize_tensors


def test_memory_trace_writes_range_events_with_tensor_shapes(tmp_path):
    trace = MemoryTrace(tmp_path, enabled=True, nvtx=False, rank=0)

    with trace.range("unit.phase", tensor=torch.zeros(2, 3, dtype=torch.float16)):
        trace.record("unit.inner", tensors={"nested": [torch.ones(1, 4)]})
    trace.close()

    path = tmp_path / "memory_trace_rank0.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "unit.phase.start",
        "unit.inner",
        "unit.phase.end",
    ]
    assert records[0]["tensors"]["tensor"]["shape"] == [2, 3]
    assert records[0]["tensors"]["tensor"]["dtype"] == "torch.float16"
    assert records[1]["tensors"]["nested"][0]["shape"] == [1, 4]


def test_memory_trace_disabled_does_not_create_file(tmp_path):
    trace = MemoryTrace(tmp_path, enabled=False, nvtx=False, rank=0)

    with trace.range("disabled.phase"):
        trace.record("disabled.inner", tensors={"x": torch.zeros(1)})
    trace.dump_memory_summary("disabled_oom")
    trace.close()

    assert not (tmp_path / "memory_trace_rank0.jsonl").exists()
    assert not list(tmp_path.glob("memory_summary_*.txt"))


def test_memory_trace_dump_summary_writes_cpu_safe_file(tmp_path):
    trace = MemoryTrace(tmp_path, enabled=True, nvtx=False, rank=0)

    path = trace.dump_memory_summary("oom_step_0")
    trace.close()

    assert path is not None
    text = path.read_text()
    assert "cuda_available" in text


def test_summarize_tensors_limits_nested_payloads():
    payload = {
        "a": torch.zeros(1, 2),
        "b": [torch.zeros(3), torch.zeros(4), torch.zeros(5)],
    }

    summary = summarize_tensors(payload, max_items=2)

    assert summary["a"]["shape"] == [1, 2]
    assert len(summary["b"]) == 2
    assert summary["b"][1]["shape"] == [4]
