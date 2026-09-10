"""
Unit Tests for Traffic Light Threat Categorization & Polygon Extraction
"""

import cv2
import numpy as np
import pytest
from src.config import COLOR_RED, COLOR_YELLOW, COLOR_GREEN
from src.postprocessing.instance_extractor import InstanceExtractor
from src.postprocessing.threat_classifier import ThreatClassifier
from src.postprocessing.payload_formatter import PayloadFormatter

def test_instance_polygon_extraction():
    extractor = InstanceExtractor(min_area=10, threshold=0.4)

    # Synthetic heatmap with 2 distinct change clusters
    heatmap = np.zeros((256, 256), dtype=np.float32)
    heatmap[30:70, 30:70] = 0.85   # Cluster 1 (Square)
    heatmap[150:170, 50:220] = 0.75 # Cluster 2 (Elongated rectangle)

    t1 = np.zeros((256, 256, 3), dtype=np.uint8)
    t2 = np.ones((256, 256, 3), dtype=np.uint8) * 200

    instances = extractor.extract_instances(heatmap, t1, t2)

    assert len(instances) == 2, f"Expected 2 instances, found {len(instances)}"
    assert instances[0].instance_id == "inst_001"
    assert instances[1].instance_id == "inst_002"

    for inst in instances:
        assert len(inst.polygon) >= 3, "Polygon must contain at least 3 vertices"
        assert len(inst.bbox) == 4, "Bounding box must be [x, y, w, h]"
        assert inst.cropped_patch_t2.shape == (64, 64, 3), "Cropped patch must be standard 64x64x3"

def test_traffic_light_threat_classification():
    extractor = InstanceExtractor(min_area=10)
    classifier = ThreatClassifier()

    # 1. Red instance (Large compact structure)
    heatmap_red = np.zeros((256, 256), dtype=np.float32)
    heatmap_red[40:120, 40:120] = 0.9
    t1 = np.ones((256, 256, 3), dtype=np.uint8) * 50
    t2 = np.ones((256, 256, 3), dtype=np.uint8) * 220
    # add structural texture
    t2[40:120:5, 40:120] = 30

    inst_red = extractor.extract_instances(heatmap_red, t1, t2)[0]
    res_red = classifier.classify_instance(inst_red)
    assert res_red["threat_level"] == "RED"
    assert res_red["color_hex"] == COLOR_RED
    assert res_red["is_alert"] is True

    # 2. Yellow instance (Elongated dirt road)
    heatmap_yel = np.zeros((256, 256), dtype=np.float32)
    heatmap_yel[100:112, 20:230] = 0.8
    inst_yel = extractor.extract_instances(heatmap_yel, t1, t2)[0]
    res_yel = classifier.classify_instance(inst_yel)
    assert res_yel["threat_level"] == "YELLOW"
    assert res_yel["color_hex"] == COLOR_YELLOW

    # 3. Green instance (Diffuse cloud shadow false positive)
    heatmap_grn = np.zeros((256, 256), dtype=np.float32)
    heatmap_grn[30:80, 30:80] = 0.6
    t1_shadow = np.ones((256, 256, 3), dtype=np.uint8) * 160
    t2_shadow = np.ones((256, 256, 3), dtype=np.uint8) * 30 # Dark shadow
    inst_grn = extractor.extract_instances(heatmap_grn, t1_shadow, t2_shadow)[0]
    res_grn = classifier.classify_instance(inst_grn)
    assert res_grn["threat_level"] == "GREEN"
    assert res_grn["color_hex"] == COLOR_GREEN
    assert res_grn["is_alert"] is False

def test_payload_formatter():
    extractor = InstanceExtractor()
    classifier = ThreatClassifier()

    heatmap = np.zeros((256, 256), dtype=np.float32)
    heatmap[50:100, 50:100] = 0.8
    t1 = np.zeros((256, 256, 3), dtype=np.uint8)
    t2 = np.ones((256, 256, 3), dtype=np.uint8) * 200

    instances = extractor.extract_instances(heatmap, t1, t2)
    classifications = [classifier.classify_instance(i) for i in instances]
    confidences = [0.94]

    payload = PayloadFormatter.format_response(
        instances=instances,
        classifications=classifications,
        confidences=confidences,
        query="new runway",
        latency_metrics={"1_forward": 45.2, "2_post": 8.1}
    )

    assert payload["status"] == "SUCCESS"
    assert payload["query"] == "new runway"
    assert payload["instances"][0]["match_confidence"] == 0.94
    assert "geometry" in payload["instances"][0]
    assert "latency" in payload
    assert payload["latency"]["budget_ms"] == 420.0
