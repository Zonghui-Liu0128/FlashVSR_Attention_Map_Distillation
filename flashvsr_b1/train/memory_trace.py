from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _safe_label(label: str) -> str:
    out = []
    for ch in str(label):
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "event"


def summarize_tensors(value: Any, *, max_items: int = 8) -> Any:
    if torch.is_tensor(value):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "numel": int(value.numel()),
            "requires_grad": bool(value.requires_grad),
            "estimated_mb": float(value.numel() * value.element_size() / 1024**2),
        }
    if isinstance(value, dict):
        return {
            str(k): summarize_tensors(v, max_items=max_items)
            for k, v in list(value.items())[:max_items]
        }
    if isinstance(value, (list, tuple)):
        return [
            summarize_tensors(v, max_items=max_items)
            for v in list(value)[:max_items]
        ]
    return None


def _json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        return summarize_tensors(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class MemoryTrace:
    def __init__(
        self,
        run_dir: str | os.PathLike[str],
        *,
        enabled: bool,
        nvtx: bool = True,
        rank: int = 0,
        max_items: int = 8,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.enabled = bool(enabled)
        self.nvtx = bool(nvtx)
        self.rank = int(rank)
        self.max_items = int(max_items)
        self._fp = None
        if self.enabled:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._fp = open(self.run_dir / f"memory_trace_rank{self.rank}.jsonl", "a", buffering=1)

    @classmethod
    def from_env(cls, run_dir: str | os.PathLike[str], *, rank: int = 0) -> "MemoryTrace":
        enabled = _env_bool("FLASHVSR_MEM_TRACE", False)
        nvtx = _env_bool("FLASHVSR_NVTX", enabled)
        max_items = int(os.environ.get("FLASHVSR_MEM_TRACE_MAX_ITEMS", "8"))
        return cls(run_dir, enabled=enabled, nvtx=nvtx, rank=rank, max_items=max_items)

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def _cuda_stats(self) -> dict[str, Any]:
        if not torch.cuda.is_available():
            return {"cuda_available": False}
        stats: dict[str, Any] = {"cuda_available": True}
        try:
            device = torch.cuda.current_device()
            stats["device"] = int(device)
            stats["allocated_gb"] = torch.cuda.memory_allocated(device) / 1024**3
            stats["reserved_gb"] = torch.cuda.memory_reserved(device) / 1024**3
            stats["max_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3
            stats["max_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 1024**3
            free, total = torch.cuda.mem_get_info(device)
            stats["free_gb"] = free / 1024**3
            stats["total_gb"] = total / 1024**3
        except Exception as exc:
            stats["cuda_stats_error"] = repr(exc)
        return stats

    def record(self, event: str, *, tensors: Any = None, **fields: Any) -> None:
        if not self.enabled or self._fp is None:
            return
        record = {
            "event": str(event),
            "time": time.time(),
            "rank": self.rank,
            "pid": os.getpid(),
            "cuda": self._cuda_stats(),
        }
        if tensors is not None:
            record["tensors"] = summarize_tensors(tensors, max_items=self.max_items)
        tensor_fields = {k: v for k, v in fields.items() if torch.is_tensor(v)}
        plain_fields = {k: v for k, v in fields.items() if k not in tensor_fields}
        if tensor_fields:
            existing = record.get("tensors", {})
            record["tensors"] = {
                **(existing if isinstance(existing, dict) else {"value": existing}),
                **summarize_tensors(tensor_fields, max_items=self.max_items),
            }
        if plain_fields:
            record["fields"] = _json_safe(plain_fields)
        self._fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    @contextmanager
    def range(self, name: str, *, tensors: Any = None, **fields: Any) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        nvtx_pushed = False
        if self.nvtx and torch.cuda.is_available():
            try:
                torch.cuda.nvtx.range_push(str(name))
                nvtx_pushed = True
            except Exception:
                nvtx_pushed = False
        start = time.perf_counter()
        self.record(f"{name}.start", tensors=tensors, **fields)
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.record(f"{name}.end", elapsed_ms=elapsed_ms)
            if nvtx_pushed:
                try:
                    torch.cuda.nvtx.range_pop()
                except Exception:
                    pass

    def dump_memory_summary(self, label: str) -> Path | None:
        if not self.enabled:
            return None
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / f"memory_summary_{_safe_label(label)}_rank{self.rank}.txt"
        with open(path, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(self._cuda_stats(), indent=2, sort_keys=True) + "\n")
            if torch.cuda.is_available():
                try:
                    fp.write("\n")
                    fp.write(torch.cuda.memory_summary())
                    fp.write("\n")
                except Exception as exc:
                    fp.write(f"memory_summary_error={exc!r}\n")
        return path


_GLOBAL_TRACE: MemoryTrace | None = None


def set_memory_trace(trace: MemoryTrace | None) -> None:
    global _GLOBAL_TRACE
    _GLOBAL_TRACE = trace


def get_memory_trace() -> MemoryTrace | None:
    return _GLOBAL_TRACE
