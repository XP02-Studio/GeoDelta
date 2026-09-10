"""
Interactive Demonstration of Deep Learning Core.
Runs full end-to-end Change Detection, Traffic Light Tagging, and RS-CLIP Semantic Matching.
"""

import os
import sys
import json
from pathlib import Path
import cv2

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG
from src.pipeline import DeepLearningCorePipeline
from scripts.generate_synthetic_data import generate_satellite_scene_pair


def main():
    print("=" * 75)
    print("      DEEP LEARNING CORE: AIR-GAPPED CHANGE DETECTION & RS-CLIP      ")
    print("=" * 75)

    # 1. Generate multi-temporal satellite images
    t1_path, t2_path = generate_satellite_scene_pair()

    # 2. Initialize pipeline
    pipeline = DeepLearningCorePipeline(CONFIG, use_accelerated_engine=True)

    # 3. Test queries
    queries = [
        "new runway",
        "concrete bunker fortification",
        "dirt road extension",
        "cloud shadow"
    ]

    out_dir = Path("data/output")
    out_dir.mkdir(parents=True, exist_ok=True)

    for query in queries:
        print(f"\n[DEMO] Processing satellite passes with Query: '{query}'...")
        result = pipeline.detect_changes(t1_path, t2_path, query=query)

        # Print latency summary
        lat = result["latency_metrics"]
        print(f"  -> Total Latency: {lat['total_latency_ms']} ms (Budget: {result['target_latency_budget_ms']} ms | Met: {result['budget_satisfied']})")
        print(f"     Breakdown: Preproc: {lat['preprocessing_ms']}ms | ChangeFormer: {lat['changeformer_inference_ms']}ms | Segment: {lat['instance_segmentation_ms']}ms | RS-CLIP: {lat['rs_clip_matching_ms']}ms")

        # Threat summary
        ts = result["threat_summary"]
        print(f"  -> Detected {ts['total_instances_detected']} instances: "
              f"Red ({CONFIG.traffic_light.red.hex}): {ts['red_heavy_infrastructure']} | "
              f"Yellow ({CONFIG.traffic_light.yellow.hex}): {ts['yellow_logistical_surface']} | "
              f"Green ({CONFIG.traffic_light.green.hex}): {ts['green_filtered_false_positives']}")

        # Top matched instances
        instances = result["instances"]
        for inst in instances[:3]:
            print(f"     * [{inst['id']}] {inst['threat_level'].upper()} ({inst['hex_color']}) - {inst['category']} | "
                  f"Area: {inst['area_pixels']}px | RS-CLIP Score: {inst['semantic_match_score']} | Centroid: {inst['centroid']}")

        # Save visualization overlay
        safe_query_name = query.replace(" ", "_")
        overlay_path = out_dir / f"overlay_{safe_query_name}.png"
        blended_path = out_dir / f"blended_{safe_query_name}.png"

        cv2.imwrite(str(overlay_path), cv2.cvtColor(result["artifacts"]["color_overlay"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(blended_path), cv2.cvtColor(result["artifacts"]["blended_overlay"], cv2.COLOR_RGB2BGR))
        print(f"  -> Saved overlay visualization: {blended_path}")

    print("\n" + "=" * 75)
    print("                    DEMO COMPLETED SUCCESSFULLY                     ")
    print("=" * 75)


if __name__ == "__main__":
    main()
