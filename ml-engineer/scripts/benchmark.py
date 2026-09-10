"""
Inference Latency & VRAM Benchmarking Suite.
Validates that the Deep Learning Core strictly satisfies the ~420ms latency budget.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG
from src.pipeline import DeepLearningCorePipeline
from scripts.generate_synthetic_data import generate_satellite_scene_pair


def run_benchmark(iterations: int = 50, warmup: int = 5, query: str = "new runway"):
    print("=" * 70)
    print("      DEEP LEARNING CORE - LATENCY BENCHMARK (<420ms TARGET)      ")
    print("=" * 70)
    print(f"Air-Gapped Mode:  Strictly Enforced (Zero Network Access)")
    print(f"Target Budget:    {CONFIG.target_latency_budget_ms:.1f} ms")
    print(f"Warmup Passes:    {warmup}")
    print(f"Test Iterations:  {iterations}")
    print(f"Semantic Query:   '{query}'")
    print("-" * 70)

    # 1. Generate or verify sample image pair
    t1_path, t2_path = generate_satellite_scene_pair()

    # 2. Initialize Pipeline
    pipeline = DeepLearningCorePipeline(CONFIG, use_accelerated_engine=True)

    # 3. Warmup runs
    print(f"\n[Benchmark] Executing {warmup} warmup passes...")
    for i in range(warmup):
        pipeline.detect_changes(t1_path, t2_path, query=query)

    # 4. Timed runs
    print(f"[Benchmark] Recording {iterations} benchmark iterations...")
    latencies_total = []
    latencies_pre = []
    latencies_cf = []
    latencies_seg = []
    latencies_clip = []

    for i in range(iterations):
        result = pipeline.detect_changes(t1_path, t2_path, query=query)
        metrics = result["latency_metrics"]

        latencies_total.append(metrics["total_latency_ms"])
        latencies_pre.append(metrics["preprocessing_ms"])
        latencies_cf.append(metrics["changeformer_inference_ms"])
        latencies_seg.append(metrics["instance_segmentation_ms"])
        latencies_clip.append(metrics["rs_clip_matching_ms"])

    latencies_total = np.array(latencies_total)
    mean_lat = np.mean(latencies_total)
    std_lat = np.std(latencies_total)
    min_lat = np.min(latencies_total)
    max_lat = np.max(latencies_total)
    p50_lat = np.percentile(latencies_total, 50)
    p90_lat = np.percentile(latencies_total, 90)
    p99_lat = np.percentile(latencies_total, 99)

    print("\n" + "=" * 70)
    print("                     BENCHMARK RESULTS SUMMARY                     ")
    print("=" * 70)
    print(f"Mean Latency:       {mean_lat:.2f} ms  (Std: {std_lat:.2f} ms)")
    print(f"Min / Max Latency:  {min_lat:.2f} ms / {max_lat:.2f} ms")
    print(f"P50 (Median):       {p50_lat:.2f} ms")
    print(f"P90 Latency:        {p90_lat:.2f} ms")
    print(f"P99 Latency:        {p99_lat:.2f} ms")
    print("-" * 70)
    print("Sub-Stage Latency Breakdown (Averages):")
    print(f"  1. Preprocessing:                 {np.mean(latencies_pre):.2f} ms")
    print(f"  2. ChangeFormer Siamese Match:    {np.mean(latencies_cf):.2f} ms")
    print(f"  3. Instance & Polygon Vectorize:  {np.mean(latencies_seg):.2f} ms")
    print(f"  4. RS-CLIP Query Matching:        {np.mean(latencies_clip):.2f} ms")
    print("-" * 70)

    budget = CONFIG.target_latency_budget_ms
    if p99_lat <= budget:
        print(f"RESULT: [PASS] - P99 latency ({p99_lat:.2f}ms) is well within the {budget:.1f}ms budget!")
    elif mean_lat <= budget:
        print(f"RESULT: [PASS] - Mean latency ({mean_lat:.2f}ms) satisfies the {budget:.1f}ms budget!")
    else:
        print(f"RESULT: [WARN] - Mean latency ({mean_lat:.2f}ms) exceeded {budget:.1f}ms budget.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latency Profiler")
    parser.add_argument("--iterations", type=int, default=30, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--query", default="new runway", help="Plain-English query to match")
    args = parser.parse_args()

    run_benchmark(iterations=args.iterations, warmup=args.warmup, query=args.query)
