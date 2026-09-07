"""
Master Runner Script (run.py) for Deep Learning Core
Orchestrates offline initialization, data generation, unit testing, sample inference, latency benchmarking, and API server hosting.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import torch
import pytest

from src.config import LATENCY_BUDGET_MS, CHANGEFORMER_WEIGHTS_PATH, RS_CLIP_WEIGHTS_PATH
from scripts.init_offline_assets import init_all_assets
from scripts.generate_sample_tiles import generate_sample_satellite_pair
from src.pipeline import DeepLearningCore
from benchmark import run_benchmark
from src.server import create_app

def print_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title.upper()}")
    print("=" * 70)

def step_1_init_assets():
    print_header("Step 1: Initializing Air-Gapped Models & Weights")
    init_all_assets()

def step_2_generate_sample_tiles():
    print_header("Step 2: Generating Bitemporal Satellite Test Tiles")
    t1_path, t2_path = generate_sample_satellite_pair(output_dir="data")
    return t1_path, t2_path

def step_3_run_tests():
    print_header("Step 3: Running Automated Test Suite (PyTest)")
    retcode = pytest.main(["-v", str(ROOT_DIR / "tests")])
    if retcode == 0:
        print("\n[OK] All test assertions passed successfully!")
    else:
        print(f"\n[!] Test suite finished with exit code {retcode}")
    return retcode == 0

def step_4_sample_inference(t1_path: str, t2_path: str, query: str = "new runway"):
    print_header(f"Step 4: Executing Change Detection & RS-CLIP Query ('{query}')")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    core = DeepLearningCore(device=device, use_fp16=True)
    
    t_start = time.perf_counter()
    result = core.analyze(t1_path, t2_path, query=query)
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    print("\n--- INFERENCE RESULTS JSON PAYLOAD ---")
    print(json.dumps(result, indent=2))
    
    summary = result.get("summary", {})
    latency = result.get("latency", {})
    
    print("\n" + "-" * 50)
    print(f"Detected Instances: {summary.get('total_instances_detected', 0)}")
    print(f"  - Red (Heavy Infrastructure): {summary.get('red_threats', 0)}")
    print(f"  - Yellow (Logistical Surface): {summary.get('yellow_warnings', 0)}")
    print(f"  - Green (Filtered False Pos) : {summary.get('green_filtered_false_positives', 0)}")
    print(f"Execution Latency : {elapsed_ms:.2f} ms (SLA Budget: <{LATENCY_BUDGET_MS} ms | Pass: {latency.get('is_within_budget')})")
    print("-" * 50)
    return result

def step_5_benchmark():
    print_header("Step 5: Latency SLA Multi-Iteration Benchmark (<420ms Target)")
    passed = run_benchmark(iterations=25, warmup=5, use_fp16=True)
    return passed

def step_6_start_server(host: str = "0.0.0.0", port: int = 8000):
    print_header(f"Step 6: Launching REST API Server on http://{host}:{port}")
    print(f"[*] Endpoints:")
    print(f"    - GET  http://localhost:{port}/api/v1/health")
    print(f"    - POST http://localhost:{port}/api/v1/analyze")
    print(f"    - POST http://localhost:{port}/api/v1/benchmark")
    print("[*] Press Ctrl+C to stop server.\n")
    
    app = create_app()
    app.run(host=host, port=port, debug=False)

def main():
    parser = argparse.ArgumentParser(description="Deep Learning Core Master Runner")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest test suite")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip latency benchmark")
    parser.add_argument("--query", type=str, default="new runway", help="Plain-English semantic query")
    parser.add_argument("--serve", action="store_true", help="Start the REST API server after pipeline execution")
    parser.add_argument("--port", type=int, default=8000, help="Port for REST API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for REST API server")
    args = parser.parse_args()

    print_header("DEEP LEARNING CORE: AIR-GAPPED MASTER PIPELINE")
    print(f"[*] Python Version : {sys.version.split()[0]}")
    print(f"[*] PyTorch Version: {torch.__version__}")
    print(f"[*] Hardware Device: {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print(f"[*] Target SLA     : <{LATENCY_BUDGET_MS} ms")

    # 1. Initialize Assets
    step_1_init_assets()

    # 2. Generate Sample Bitemporal Tiles
    t1_path, t2_path = step_2_generate_sample_tiles()

    # 3. Run Tests
    if not args.skip_tests:
        tests_passed = step_3_run_tests()
        if not tests_passed:
            print("[!] Warning: Some tests failed.")

    # 4. Run Sample Inference
    step_4_sample_inference(t1_path, t2_path, query=args.query)

    # 5. Run Benchmark
    if not args.skip_benchmark:
        sla_passed = step_5_benchmark()
        if not sla_passed:
            print("[!] Warning: Benchmark exceeded target SLA.")

    print_header("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY")

    # 6. Optional Server Launch
    if args.serve:
        step_6_start_server(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
