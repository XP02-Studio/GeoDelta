"""
Payload Formatter for Backend & Frontend Consumption
Assembles polygon instances, threat classifications, RS-CLIP match confidences, and latency metrics into standardized JSON.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
from src.postprocessing.instance_extractor import ChangeInstance

class PayloadFormatter:
    """Formats change detection and semantic matching results into clean structured JSON."""

    @staticmethod
    def format_response(
        instances: List[ChangeInstance],
        classifications: List[Dict[str, Any]],
        confidences: Optional[List[float]] = None,
        query: Optional[str] = None,
        latency_metrics: Optional[Dict[str, float]] = None,
        image_shape: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """
        Creates the standardized payload.
        """
        H, W = image_shape if image_shape else (256, 256)
        formatted_instances = []

        total_red = 0
        total_yellow = 0
        total_green = 0

        for idx, (inst, cls_info) in enumerate(zip(instances, classifications)):
            conf = float(confidences[idx]) if confidences and idx < len(confidences) else 1.0

            threat_lvl = cls_info["threat_level"]
            if threat_lvl == "RED":
                total_red += 1
            elif threat_lvl == "YELLOW":
                total_yellow += 1
            elif threat_lvl == "GREEN":
                total_green += 1

            inst_payload = {
                "instance_id": inst.instance_id,
                "threat_level": cls_info["threat_level"],
                "color_hex": cls_info["color_hex"],
                "category": cls_info["category"],
                "sub_type": cls_info["sub_type"],
                "is_alert": cls_info["is_alert"],
                "match_confidence": round(conf, 4),
                "geometry": {
                    "bbox_pixels": inst.bbox,                  # [x, y, w, h]
                    "bbox_normalized": inst.bbox_normalized,  # [x, y, w, h] in [0, 1]
                    "polygon_pixels": inst.polygon,           # [[x, y], ...]
                    "polygon_normalized": inst.polygon_normalized,
                    "centroid_pixels": inst.centroid,
                    "area_pixels": inst.area_pixels
                },
                "metrics": {
                    "aspect_ratio": inst.aspect_ratio,
                    "solidity": inst.solidity,
                    "change_score": inst.mean_change_score
                }
            }
            formatted_instances.append(inst_payload)

        total_latency = sum(latency_metrics.values()) if latency_metrics else 0.0

        return {
            "status": "SUCCESS",
            "timestamp": time.time(),
            "query": query,
            "summary": {
                "total_instances_detected": len(instances),
                "red_threats": total_red,
                "yellow_warnings": total_yellow,
                "green_filtered_false_positives": total_green,
                "tile_dimensions": {"height": H, "width": W}
            },
            "instances": formatted_instances,
            "latency": {
                "total_pipeline_ms": round(total_latency, 2),
                "budget_ms": 420.0,
                "is_within_budget": total_latency <= 420.0,
                "breakdown_ms": {k: round(v, 2) for k, v in (latency_metrics or {}).items()}
            }
        }
