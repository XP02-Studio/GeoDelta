"""
Synthetic Satellite Image Pair Generator for Air-Gapped Testing.
Generates multi-temporal T1 and T2 satellite scenes featuring:
  - Heavy Infrastructure (bunkers, airstrip runway additions) -> RED
  - Logistical Surface Changes (dirt road extension, forest clearing) -> YELLOW
  - False Positives (shadow displacement, cloud glare) -> GREEN
"""

import os
from pathlib import Path
from typing import Tuple
import numpy as np
import cv2


def generate_satellite_scene_pair(
    size: Tuple[int, int] = (512, 512),
    output_dir: str = "data/sample_scenes"
) -> Tuple[Path, Path]:
    """Generates T1 and T2 multi-temporal satellite images with realistic structural changes."""
    h, w = size
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Base Terrain Background (Terrain / Vegetation texture)
    np.random.seed(42)
    # Greenish terrain with noise
    terrain_base = np.zeros((h, w, 3), dtype=np.uint8)
    terrain_base[:, :, 0] = np.random.randint(45, 75, (h, w), dtype=np.uint8)   # R
    terrain_base[:, :, 1] = np.random.randint(90, 130, (h, w), dtype=np.uint8)  # G (Vegetation)
    terrain_base[:, :, 2] = np.random.randint(40, 65, (h, w), dtype=np.uint8)   # B

    # Add Perlin-like low-frequency elevation gradients
    blur_kernel = 61
    terrain_base = cv2.GaussianBlur(terrain_base, (blur_kernel, blur_kernel), 0)

    # Existing pre-pass features in T1
    t1 = terrain_base.copy()
    # Existing river or natural feature
    cv2.polylines(
        t1,
        [np.array([[0, 120], [150, 180], [320, 240], [512, 300]], dtype=np.int32)],
        isClosed=False,
        color=(30, 60, 90),
        thickness=18
    )

    t2 = t1.copy()

    # --- Scenario Additions in T2 ---

    # 1. Heavy Infrastructure: New Reinforced Bunker & Concrete Pad (RED #EF4444)
    bunker_x, bunker_y, bunker_w, bunker_h = 340, 80, 80, 70
    cv2.rectangle(t2, (bunker_x, bunker_y), (bunker_x + bunker_w, bunker_y + bunker_h), (180, 185, 190), -1)
    # Bunker roof ventilation & shadows
    cv2.rectangle(t2, (bunker_x + 15, bunker_y + 15), (bunker_x + 65, bunker_y + 55), (120, 125, 130), -1)
    cv2.circle(t2, (bunker_x + 40, bunker_y + 35), 8, (70, 75, 80), -1)

    # 2. Heavy Infrastructure: New Airstrip / Runway Addition (RED #EF4444)
    runway_pts = np.array([[60, 380], [280, 480], [290, 460], [70, 360]], dtype=np.int32)
    cv2.fillPoly(t2, [runway_pts], (160, 165, 170))
    # Runway center stripe markings
    cv2.line(t2, (65, 370), (285, 470), (230, 230, 230), 2)

    # 3. Logistical Surface Change: New Dirt Access Road Extension (YELLOW #EAB308)
    road_pts = np.array([[220, 0], [250, 150], [300, 280], [350, 360]], dtype=np.int32)
    cv2.polylines(t2, [road_pts], isClosed=False, color=(160, 125, 80), thickness=8)

    # 4. Logistical Surface Change: Forest Clearing / Trench Staging Area (YELLOW #EAB308)
    clearing_x, clearing_y, clearing_w, clearing_h = 60, 60, 90, 50
    cv2.rectangle(t2, (clearing_x, clearing_y), (clearing_x + clearing_w, clearing_y + clearing_h), (140, 110, 70), -1)

    # 5. False Positive: Cloud Cover / Transient Bright Reflection (GREEN #22C55E)
    cloud_center = (420, 420)
    cv2.circle(t2, cloud_center, 40, (245, 248, 252), -1)
    # Blur the cloud patch to mimic atmospheric diffusion
    cloud_roi = t2[360:480, 360:480]
    t2[360:480, 360:480] = cv2.GaussianBlur(cloud_roi, (25, 25), 0)

    # 6. False Positive: Shadow Displacement (GREEN #22C55E)
    # T1 had a dark tree shadow at (180, 200), in T2 it has shifted to (195, 210)
    cv2.ellipse(t1, (180, 200), (35, 15), 45, 0, 360, (25, 30, 20), -1)
    cv2.ellipse(t2, (195, 210), (35, 15), 45, 0, 360, (25, 30, 20), -1)

    # Save generated images
    t1_path = out_path / "satellite_pass_t1.png"
    t2_path = out_path / "satellite_pass_t2.png"

    # Convert RGB to BGR for cv2.imwrite
    cv2.imwrite(str(t1_path), cv2.cvtColor(t1, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(t2_path), cv2.cvtColor(t2, cv2.COLOR_RGB2BGR))

    print(f"[Synthetic Generator] Generated multi-temporal satellite image pair:")
    print(f"  - Pass T1: {t1_path}")
    print(f"  - Pass T2: {t2_path}")
    return t1_path, t2_path


if __name__ == "__main__":
    generate_satellite_scene_pair()
