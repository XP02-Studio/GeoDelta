"""
Sample Bitemporal Satellite Tile Generator
Creates synthetic bitemporal satellite image pairs (T1 and T2) showcasing:
1. Heavy Infrastructure Addition (Red): Runway & Bunker
2. Logistical Surface Change (Yellow): Dirt Road & Forest Clearing
3. Filtered False Positive (Green): Cloud & Ground Shadow
"""

import cv2
import numpy as np
from pathlib import Path

def generate_sample_satellite_pair(output_dir: str = "data"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    H, W = 256, 256
    np.random.seed(42)

    # 1. Base T1 Terrain (Forest, Fields, Natural Earth)
    # Background texture
    t1 = np.zeros((H, W, 3), dtype=np.uint8)
    t1[:, :, 0] = np.random.randint(40, 70, (H, W))   # R
    t1[:, :, 1] = np.random.randint(90, 130, (H, W))  # G (Vegetation)
    t1[:, :, 2] = np.random.randint(30, 60, (H, W))   # B
    t1 = cv2.GaussianBlur(t1, (5, 5), 0)

    # 2. Construct T2 with realistic military/intelligence changes
    t2 = t1.copy()

    # Change 1: Runway Expansion / Bunker (Red #EF4444)
    # High-reflectance concrete strip
    cv2.rectangle(t2, (120, 30), (160, 140), (195, 200, 205), -1) # Concrete runway
    cv2.rectangle(t2, (130, 150), (170, 180), (130, 135, 140), -1) # Bunker building
    cv2.rectangle(t2, (125, 35), (155, 135), (220, 225, 230), 1)

    # Change 2: Logistical Dirt Road & Clearing (Yellow #EAB308)
    # Linear dirt track
    pts = np.array([[20, 200], [80, 210], [150, 225], [230, 230]], np.int32)
    cv2.polylines(t2, [pts], False, (140, 110, 80), 6) # Brown dirt road
    cv2.circle(t2, (60, 190), 15, (135, 105, 75), -1) # Staging ground clearing

    # Change 3: Cloud Shadow / Transient false positive (Green #22C55E)
    # Diffuse dark region in top left
    shadow_mask = np.zeros((H, W), dtype=np.float32)
    cv2.ellipse(shadow_mask, (50, 60), (30, 20), 30, 0, 360, 1.0, -1)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (15, 15), 0)
    for c in range(3):
        t2[:, :, c] = np.clip(t2[:, :, c] * (1.0 - shadow_mask * 0.6), 0, 255).astype(np.uint8)

    # Save images as RGB
    t1_bgr = cv2.cvtColor(t1, cv2.COLOR_RGB2BGR)
    t2_bgr = cv2.cvtColor(t2, cv2.COLOR_RGB2BGR)

    t1_path = out_dir / "sample_t1.png"
    t2_path = out_dir / "sample_t2.png"

    cv2.imwrite(str(t1_path), t1_bgr)
    cv2.imwrite(str(t2_path), t2_bgr)

    print(f"[OK] Generated synthetic satellite tile pair:")
    print(f"    - T1 (Pre-event) : {t1_path}")
    print(f"    - T2 (Post-event): {t2_path}")

    return str(t1_path), str(t2_path)

if __name__ == "__main__":
    generate_sample_satellite_pair()
