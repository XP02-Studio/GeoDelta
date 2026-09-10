"""
High-Precision Latency Profiler and SLA Budget Tracker
Provides millisecond-accurate timing for each pipeline stage and SLA compliance verification.
"""

import time
from contextlib import contextmanager
from typing import Dict, List, Optional
import numpy as np

class PipelineProfiler:
    """Tracks latency metrics across pipeline execution stages."""
    def __init__(self, sla_budget_ms: float = 420.0):
        self.sla_budget_ms = sla_budget_ms
        self.timings: Dict[str, float] = {}

    @contextmanager
    def measure(self, stage_name: str):
        """Context manager to measure execution time of a specific block in milliseconds."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.timings[stage_name] = elapsed_ms

    def get_total_latency(self) -> float:
        """Returns total measured pipeline latency in ms."""
        return sum(self.timings.values())

    def is_within_sla(self) -> bool:
        """Checks if total latency is within the SLA budget (<420ms)."""
        return self.get_total_latency() <= self.sla_budget_ms

    def summary(self) -> Dict[str, float]:
        """Returns clean dictionary of stage timings and total."""
        res = {k: round(v, 2) for k, v in self.timings.items()}
        res["total_ms"] = round(self.get_total_latency(), 2)
        res["budget_ms"] = self.sla_budget_ms
        res["sla_pass"] = self.is_within_sla()
        return res


class BenchmarkStats:
    """Aggregates multi-run benchmarking statistics."""
    def __init__(self, latencies: List[float]):
        self.latencies = np.array(latencies, dtype=np.float64)

    def stats(self) -> Dict[str, float]:
        if len(self.latencies) == 0:
            return {}
        return {
            "iterations": len(self.latencies),
            "mean_ms": round(float(np.mean(self.latencies)), 2),
            "std_ms": round(float(np.std(self.latencies)), 2),
            "min_ms": round(float(np.min(self.latencies)), 2),
            "max_ms": round(float(np.max(self.latencies)), 2),
            "p50_ms": round(float(np.percentile(self.latencies, 50)), 2),
            "p95_ms": round(float(np.percentile(self.latencies, 95)), 2),
            "p99_ms": round(float(np.percentile(self.latencies, 99)), 2)
        }
