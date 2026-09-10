"""
Threat Classifier & Traffic Light Categorization Engine
Maps detected polygon instances into Red, Yellow, and Green traffic light categories.
"""

import cv2
import numpy as np
from typing import Any, Dict, List, Tuple, Union
from src.config import COLOR_RED, COLOR_YELLOW, COLOR_GREEN
from src.postprocessing.instance_extractor import ChangeInstance

class ThreatClassifier:
    """
    Classifies satellite change instances into operational threat categories:
    - RED (#EF4444): Heavy infrastructure additions (bunkers, airstrips, large buildings).
    - YELLOW (#EAB308): Logistical surface changes (dirt roads, clearings, bridge extensions).
    - GREEN (#22C55E): Filtered false positives (shadows, transient clouds, seasonal foliage).
    """

    def __init__(self, shadow_diff_threshold: float = 40.0, edge_density_threshold: float = 0.08):
        self.shadow_diff_threshold = shadow_diff_threshold
        self.edge_density_threshold = edge_density_threshold

    def is_false_positive_shadow_or_cloud(self, instance: ChangeInstance) -> Tuple[bool, str]:
        """
        Determines if a detected change is an ephemeral shadow, cloud, or lighting artifact.
        """
        t1_gray = cv2.cvtColor(instance.cropped_patch_t1, cv2.COLOR_RGB2GRAY)
        t2_gray = cv2.cvtColor(instance.cropped_patch_t2, cv2.COLOR_RGB2GRAY)

        mean_t1 = float(np.mean(t1_gray))
        mean_t2 = float(np.mean(t2_gray))
        brightness_diff = abs(mean_t1 - mean_t2)

        # Canny edge detection in T2 patch
        edges_t2 = cv2.Canny(t2_gray, 50, 150)
        edge_density = float(np.sum(edges_t2 > 0)) / (edges_t2.shape[0] * edges_t2.shape[1])

        # 1. Cloud Detection: very bright saturated patch in T2 with low texture complexity
        if mean_t2 > 215 and edge_density < self.edge_density_threshold:
            return True, "Cloud Cover / Ephemeral Vapor"

        # 2. Shadow Detection: significant darkening without high structural edge sharpness
        if mean_t2 < 45 and (mean_t1 - mean_t2) > self.shadow_diff_threshold and edge_density < self.edge_density_threshold:
            return True, "Cloud Shadow / Ground Shadow"

        # 3. Very low solidity with diffuse boundary and low mean change score
        if instance.solidity < 0.35 and instance.mean_change_score < 0.55 and edge_density < 0.04:
            return True, "Seasonal Vegetation / Foliage Variation"

        return False, ""

    def classify_instance(self, instance: ChangeInstance) -> Dict[str, Union[str, float]]:
        """
        Categorizes an individual instance and assigns its Traffic Light color code and category.
        """
        is_fp, fp_reason = self.is_false_positive_shadow_or_cloud(instance)

        # Case 1: Green (#22C55E) - Suspected changes filtered out as false positives
        if is_fp:
            return {
                "threat_level": "GREEN",
                "color_hex": COLOR_GREEN,
                "category": "Filtered False Positive",
                "sub_type": fp_reason,
                "severity": 0,
                "is_alert": False
            }

        # Extract features for Red vs Yellow
        t2_gray = cv2.cvtColor(instance.cropped_patch_t2, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(t2_gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / (edges.shape[0] * edges.shape[1])

        # Case 2: Yellow (#EAB308) - Logistical surface changes (roads, clearings, linear features)
        # Elongated aspect ratio (>2.8 or <0.35) or low-to-moderate solidity with medium area
        is_linear = (instance.aspect_ratio > 2.5 or instance.aspect_ratio < 0.4)
        is_surface_clearing = (instance.solidity < 0.70 and edge_density < 0.15)

        if is_linear or is_surface_clearing:
            sub_type = "Dirt Road / Supply Route" if is_linear else "Forest Clearing / Staging Ground"
            return {
                "threat_level": "YELLOW",
                "color_hex": COLOR_YELLOW,
                "category": "Logistical Surface Change",
                "sub_type": sub_type,
                "severity": 1,
                "is_alert": True
            }

        # Case 3: Red (#EF4444) - Heavy infrastructure additions (bunkers, airstrips, buildings)
        # High structural solidity, high edge density, compact building/structure footprint
        sub_type = "Heavy Infrastructure Addition"
        if instance.area_pixels > 2000 and (instance.aspect_ratio > 2.0 or instance.aspect_ratio < 0.5):
            sub_type = "Airstrip / Runway Expansion"
        elif instance.solidity > 0.75 and edge_density > 0.12:
            sub_type = "Fortified Bunker / Concrete Building"
        elif instance.area_pixels > 500:
            sub_type = "Permanent Facility / Warehouse"

        return {
            "threat_level": "RED",
            "color_hex": COLOR_RED,
            "category": "Heavy Infrastructure Addition",
            "sub_type": sub_type,
            "severity": 2,
            "is_alert": True
        }
