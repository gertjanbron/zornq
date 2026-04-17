from __future__ import annotations

import time
import tracemalloc


class PeakMemoryTracker:
    def __init__(self) -> None:
        self.start_time = 0.0
        self.elapsed_sec = 0.0
        self.peak_mb = 0.0

    def __enter__(self) -> "PeakMemoryTracker":
        self.start_time = time.perf_counter()
        tracemalloc.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.elapsed_sec = time.perf_counter() - self.start_time
        self.peak_mb = peak / (1024.0 * 1024.0)
