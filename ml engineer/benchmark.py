"""
Latency Benchmarking & SLA Compliance Suite
Tests end-to-end inference latency against the strict ~420ms SLA budget.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from src.config import LATENCY_BUDGET_MS
from src.pipeline import DeepLearningCore
from src.optimization.profiler import BenchmarkStats

def run_benchmark(iterations: int = 30, warmup: int = 5, use_fp16: bool = True):
    print("=" * 65)
    print("       DEEP LEARNING CORE LATENCY BENCHMARK (<420ms SLA)")
    print("=" * 65)
    print(f"[*] Iterations: {iterations} | Warmup: {warmup} | FP16: {use_fp16}")
    print(f"[*] Target SLA Latency Budget: <{LATENCY_BUDGET_MS} ms")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Hardware Device: {device.upper()}")
    if device == "cuda":
        print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")

    # Initialize Core
    core = DeepLearningCore(device=device, use_fp16=use_fp16)

    # Generate synthetic satellite tile pair
    H, W = 256, 256
    t1 = np.random.randint(60, 180, (H, W, 3), dtype=np.uint8)
    t2 = t1.copy()
    # Add simulated airstrip/bunker (Red)
    t2[40:120, 100:130] = np.random.randint(200, 240, (80, 30, 3), dtype=np.uint8)
    # Add simulated dirt road (Yellow)
    t2[160:175, 20:230] = np.random.randint(180, 210, (15, 210, 3), dtype=np.uint8)
    # Add simulated cloud shadow (Green)
    t2[190:230, 40:80] = np.clip(t2[190:230, 40:80] - 60, 0, 255)

    query = "new runway"

    # Warmup
    print("\n[*] Warming up inference pipeline...")
    for _ in range(warmup):
        _ = core.analyze(t1, t2, query=query)
    print("[OK] Warmup complete.")

    # Benchmark runs
    print(f"\n[*] Executing {iterations} timed benchmark runs...")
    latencies = []
    stage_breakdowns = {
        "1_image_preprocessing": [],
        "2_changeformer_forward_pass": [],
        "3_instance_polygon_clustering": [],
        "4_threat_classification": [],
        "5_rs_clip_semantic_matching": [],
        "6_payload_assembly": []
    }

    for i in range(iterations):
        t0 = time.perf_counter()
        result = core.analyze(t1, t2, query=query)
        t_elapsed = (time.perf_counter() - t0) * 1000.0
        latencies.append(t_elapsed)

        breakdown = result.get("latency", {}).get("breakdown_ms", {})
        for stage, val in breakdown.items():
            if stage in stage_breakdowns:
                stage_breakdowns[stage].append(val)

        if (i + 1) % 10 == 0 or (i + 1) == iterations:
            print(f"    - Iteration {i+1:2d}/{iterations}: {t_elapsed:6.2f} ms")

    # Compute statistics
    stats = BenchmarkStats(latencies).stats()

    print("\n" + "=" * 65)
    print("STAGE-BY-STAGE LATENCY BREAKDOWN (AVERAGE):")
    print("-" * 65)
    for stage, vals in stage_breakdowns.items():
        if vals:
            avg_stage = float(np.mean(vals))
            print(f"  {stage:<35} : {avg_stage:6.2f} ms ({avg_stage / stats['mean_ms'] * 100:4.1f}%)")
    print("-" * 65)

    print("END-TO-END LATENCY SUMMARY:")
    print("-" * 65)
    print(f"  Mean Latency       : {stats['mean_ms']:6.2f} ms")
    print(f"  P50 (Median)       : {stats['p50_ms']:6.2f} ms")
    print(f"  P95 Latency        : {stats['p95_ms']:6.2f} ms")
    print(f"  P99 Latency        : {stats['p99_ms']:6.2f} ms")
    print(f"  Min / Max          : {stats['min_ms']:6.2f} ms / {stats['max_ms']:6.2f} ms")
    print(f"  SLA Budget Target  : {LATENCY_BUDGET_MS:6.2f} ms")
    print("-" * 65)

    sla_passed = stats["mean_ms"] <= LATENCY_BUDGET_MS
    status_str = "PASSED [SLA COMPLIANT]" if sla_passed else "FAILED [EXCEEDED BUDGET]"
    print(f"  STATUS             : {status_str}")
    print("=" * 65)

    return sla_passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Learning Core Latency Benchmark")
    parser.add_argument("--iterations", type=int, default=25, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 acceleration")
    args = parser.parse_args()

    passed = run_benchmark(iterations=args.iterations, warmup=args.warmup, use_fp16=not args.no_fp16)
    sys.exit(0 if passed else 1)
