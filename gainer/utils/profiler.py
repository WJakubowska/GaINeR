"""Lightweight step profiler to time sections with optional CUDA sync.

Enable by setting environment variable GAINER_PROFILE=1.
By default, reports every 200 steps and resets accumulators.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional

import torch


class SectionTimer:
    def __init__(self, name: str, profiler: "StepProfiler") -> None:
        self.name = name
        self.profiler = profiler
        self.t0: Optional[float] = None

    def __enter__(self):
        if self.profiler.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.profiler.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        assert self.t0 is not None
        dt_ms = (time.perf_counter() - self.t0) * 1e3
        self.profiler.record(self.name, dt_ms)


class StepProfiler:
    def __init__(self, enabled: Optional[bool] = None, every_n: int = 200, sync_cuda: bool = True) -> None:
        self.enabled = enabled if enabled is not None else os.getenv("GAINER_PROFILE", "0") == "1"
        self.every_n = every_n
        self.sync_cuda = sync_cuda
        self._acc: Dict[str, float] = {}
        self._cnt: Dict[str, int] = {}

    def section(self, name: str) -> SectionTimer:
        return SectionTimer(name, profiler=self)

    def record(self, name: str, dt_ms: float) -> None:
        if not self.enabled:
            return
        self._acc[name] = self._acc.get(name, 0.0) + dt_ms
        self._cnt[name] = self._cnt.get(name, 0) + 1

    def maybe_report(self, prefix: str = "PROF") -> None:
        if not self.enabled:
            return
        if not self._acc:
            return
        parts = []
        for k in sorted(self._acc.keys()):
            tot = self._acc[k]
            cnt = max(1, self._cnt.get(k, 1))
            parts.append(f"{k}={tot/cnt:.2f}ms")
        print(f"[{prefix}] " + " ".join(parts))
        self._acc.clear()
        self._cnt.clear()
