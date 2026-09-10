"""
Latency SLA Verification Test (<420ms Target)
"""

import numpy as np
import pytest
import time
from src.config import LATENCY_BUDGET_MS
from src.pipeline import DeepLearningCore

def test_pipeline_latency_budget():
    core = DeepLearningCore()

    # Generate synthetic satellite image pair
    H, W = 256, 256
    t1 = np.random.randint(50, 200, (H, W, 3), dtype=np.uint8)
    t2 = t1.copy()
    t2[50:100, 50:120] = np.random.randint(210, 250, (50, 70, 3), dtype=np.uint8)

    # Warmup
    for _ in range(3):
        _ = core.analyze(t1, t2, query="new runway")

    # Measure
    latencies = []
    for _ in range(5):
        t_start = time.perf_counter()
        result = core.analyze(t1, t2, query="new runway")
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        latencies.append(elapsed_ms)

        assert result["status"] == "SUCCESS"
        assert "instances" in result
        assert "latency" in result

    mean_latency = float(np.mean(latencies))
    print(f"\n[Test Result] Average Pipeline Latency: {mean_latency:.2f} ms (Budget: {LATENCY_BUDGET_MS} ms)")
    
    assert mean_latency <= LATENCY_BUDGET_MS, (
        f"Pipeline latency {mean_latency:.2f}ms exceeded SLA budget of {LATENCY_BUDGET_MS}ms!"
    )
