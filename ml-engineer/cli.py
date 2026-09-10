"""
Command-Line Interface (CLI) for Deep Learning Core
Execute offline ChangeFormer and RS-CLIP inference directly from terminal.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import DeepLearningCore

def main():
    parser = argparse.ArgumentParser(description="Deep Learning Core Satellite Change Analyzer")
    parser.add_argument("--t1", type=str, required=True, help="Path to T1 satellite image")
    parser.add_argument("--t2", type=str, required=True, help="Path to T2 satellite image")
    parser.add_argument("--query", type=str, default=None, help="Plain-English semantic query (e.g. 'new runway')")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path (prints to stdout if omitted)")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 mixed precision")
    args = parser.parse_args()

    t1_path = Path(args.t1)
    t2_path = Path(args.t2)

    if not t1_path.exists():
        print(f"[!] Error: T1 image file not found: {t1_path}", file=sys.stderr)
        sys.exit(1)
    if not t2_path.exists():
        print(f"[!] Error: T2 image file not found: {t2_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Processing satellite pair: {t1_path.name} vs {t2_path.name}")
    if args.query:
        print(f"[*] Semantic Query: '{args.query}'")

    core = DeepLearningCore(use_fp16=not args.no_fp16)
    result = core.analyze(str(t1_path), str(t2_path), query=args.query)

    result_json = json.dumps(result, indent=2)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result_json)
        print(f"[OK] Analysis saved to: {out_path}")
    else:
        print("\n" + "="*50)
        print("ANALYSIS PAYLOAD OUTPUT:")
        print("="*50)
        print(result_json)

    # Print summary & SLA validation
    summary = result.get("summary", {})
    latency = result.get("latency", {})
    print("\n" + "-"*50)
    print(f"Detected Instances : {summary.get('total_instances_detected', 0)}")
    print(f"  - Red (Heavy Infrastructure): {summary.get('red_threats', 0)}")
    print(f"  - Yellow (Logistical Surface): {summary.get('yellow_warnings', 0)}")
    print(f"  - Green (Filtered False Pos) : {summary.get('green_filtered_false_positives', 0)}")
    print(f"Total Pipeline Latency: {latency.get('total_pipeline_ms')} ms (Target: <420ms | Pass: {latency.get('is_within_budget')})")
    print("-"*50)

if __name__ == "__main__":
    main()
