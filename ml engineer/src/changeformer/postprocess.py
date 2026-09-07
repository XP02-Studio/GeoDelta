"""
Post-Processing & Traffic Light Instance Segmentation Engine.
Converts raw probability masks into instance-segmented polygons with unique IDs
and classifies them into the Traffic Light threat taxonomy:
  - Red (#EF4444): Heavy Infrastructure (Bunkers, Airstrips, Buildings)
  - Yellow (#EAB308): Logistical Surface (Dirt roads, Clearings, Bridges)
  - Green (#22C55E): Filtered False Positives (Shadows, Clouds, Seasonal shifts)
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict

from src.config import CONFIG, TrafficLightConfig


@dataclass
class PolygonInstance:
    instance_id: str
    threat_level: str          # "red", "yellow", "green"
    hex_color: str             # "#EF4444", "#EAB308", "#22C55E"
    category: str              # "Heavy Infrastructure", "Logistical Surface", "Filtered False Positive"
    description: str
    confidence: float          # Change detection probability
    area_pixels: int
    bbox: List[int]            # [x_min, y_min, width, height]
    centroid: List[float]      # [cx, cy]
    polygon: List[List[int]]   # List of [x, y] coordinates
    semantic_label: Optional[str] = None
    semantic_match_score: Optional[float] = None


class TrafficLightPostProcessor:
    """
    Instance segmentation, polygon vectorization, and Traffic Light threat categorizer.
    """
    def __init__(self, config: Optional[TrafficLightConfig] = None):
        self.config = config or CONFIG.traffic_light
        self.color_map = {
            "red": self.config.red.hex,       # #EF4444
            "yellow": self.config.yellow.hex, # #EAB308
            "green": self.config.green.hex,   # #22C55E
        }

    def _classify_threat_level(
        self,
        prob_crop: np.ndarray,
        t1_crop: np.ndarray,
        t2_crop: np.ndarray,
        contour: np.ndarray,
        area: int
    ) -> Tuple[str, str, str]:
        """
        Applies spectral and geometric heuristics to assign initial Traffic Light category.
        Can be further refined by RS-CLIP plain-English semantic matching.
        """
        # Calculate spectral statistics for false positive filtering
        t1_gray = cv2.cvtColor(t1_crop, cv2.COLOR_RGB2GRAY).astype(np.float32) if len(t1_crop.shape) == 3 else t1_crop
        t2_gray = cv2.cvtColor(t2_crop, cv2.COLOR_RGB2GRAY).astype(np.float32) if len(t2_crop.shape) == 3 else t2_crop

        mean_diff = np.mean(np.abs(t2_gray - t1_gray))
        t1_mean = np.mean(t1_gray)
        t2_mean = np.mean(t2_gray)

        # Heuristic 1: Shadow detection (Very low intensity in T2 compared to T1 or vice-versa)
        is_shadow = (t2_mean < 45 and t1_mean > 70) or (t1_mean < 45 and t2_mean > 70)

        # Heuristic 2: Cloud / High reflectance glare (Extremely bright saturated patch in T2)
        is_cloud = t2_mean > 225 and mean_diff > 80

        # Heuristic 3: Low-contrast diffuse change (Seasonal illumination shift)
        is_diffuse_illumination = mean_diff < 15

        if is_shadow or is_cloud or is_diffuse_illumination:
            # Tagged as Filtered False Positive -> GREEN (#22C55E)
            return (
                "green",
                self.config.green.hex,
                self.config.green.category
            )

        # Geometric properties: Perimeter-to-Area ratio & Compactness
        perimeter = cv2.arcLength(contour, True)
        compactness = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)

        # Heuristic 4: Large rectangular or high-compactness structures -> RED (#EF4444)
        # Heavy infrastructure (bunkers, large buildings, missile pads)
        if area >= 300 or (area >= 120 and compactness > 0.45):
            return (
                "red",
                self.config.red.hex,
                self.config.red.category
            )

        # Heuristic 5: Elongated, narrow or moderate irregular surface changes -> YELLOW (#EAB308)
        # Dirt roads, clearings, bridge extensions, vehicle tracks
        return (
            "yellow",
            self.config.yellow.hex,
            self.config.yellow.category
        )

    def process(
        self,
        prob_map: np.ndarray,
        t1_img: np.ndarray,
        t2_img: np.ndarray,
        prob_threshold: Optional[float] = None
    ) -> Tuple[List[PolygonInstance], np.ndarray, np.ndarray]:
        """
        Extracts instance segmented polygons and generates Traffic Light output.

        Args:
            prob_map: [H, W] float array in range [0, 1]
            t1_img: [H, W, 3] uint8 RGB image
            t2_img: [H, W, 3] uint8 RGB image
            prob_threshold: Probability cutoff for change binarization

        Returns:
            instances: List of PolygonInstance objects
            instance_mask: [H, W] uint16 labeled mask
            color_overlay: [H, W, 3] RGB overlay with Traffic Light colors
        """
        threshold = prob_threshold or self.config.change_probability_threshold
        binary_mask = (prob_map >= threshold).astype(np.uint8) * 255

        # Morphological noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

        # Find individual connected contours
        contours, _ = cv2.findContours(
            cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        h, w = prob_map.shape[:2]
        instance_mask = np.zeros((h, w), dtype=np.uint16)
        color_overlay = np.zeros((h, w, 3), dtype=np.uint8)

        instances: List[PolygonInstance] = []
        inst_counter = 1

        for cnt in contours:
            area = int(cv2.contourArea(cnt))
            if area < self.config.min_instance_area_pixels:
                continue

            # Bounding box & centroid
            x, y, bw, bh = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
            else:
                cx, cy = float(x + bw / 2), float(y + bh / 2)

            # Extract simplified polygon points
            epsilon = 0.015 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            polygon_pts = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]

            # Extract image crops for analysis
            y_max, x_max = min(h, y + bh), min(w, x + bw)
            t1_crop = t1_img[y:y_max, x:x_max]
            t2_crop = t2_img[y:y_max, x:x_max]
            prob_crop = prob_map[y:y_max, x:x_max]

            mean_confidence = float(np.mean(prob_crop))

            # Classify into Traffic Light Taxonomy
            threat_level, hex_color, category = self._classify_threat_level(
                prob_crop, t1_crop, t2_crop, cnt, area
            )

            desc = getattr(self.config, threat_level).description

            instance_id = f"inst_{inst_counter:03d}"
            inst_counter += 1

            instance = PolygonInstance(
                instance_id=instance_id,
                threat_level=threat_level,
                hex_color=hex_color,
                category=category,
                description=desc,
                confidence=round(mean_confidence, 4),
                area_pixels=area,
                bbox=[int(x), int(y), int(bw), int(bh)],
                centroid=[round(cx, 2), round(cy, 2)],
                polygon=polygon_pts
            )
            instances.append(instance)

            # Paint instance ID on instance mask
            cv2.drawContours(instance_mask, [cnt], -1, int(inst_counter - 1), thickness=-1)

            # Parse RGB color for overlay
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            cv2.drawContours(color_overlay, [cnt], -1, (r, g, b), thickness=-1)

        return instances, instance_mask, color_overlay

    def create_alpha_blended_overlay(
        self,
        base_image: np.ndarray,
        color_overlay: np.ndarray,
        alpha: float = 0.45
    ) -> np.ndarray:
        """Blends the Traffic Light color mask smoothly on top of satellite image T2."""
        mask_active = (color_overlay > 0).any(axis=-1)
        blended = base_image.copy()
        blended[mask_active] = cv2.addWeighted(
            base_image[mask_active], 1.0 - alpha,
            color_overlay[mask_active], alpha, 0
        )
        return blended
