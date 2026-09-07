"""
Integration Tests for End-to-End Deep Learning Pipeline.
"""

import pytest
import numpy as np
from src.pipeline import DeepLearningCore
from scripts.generate_sample_tiles import generate_sample_satellite_pair

def test_full_pipeline_execution(tmp_path):
    t1_path, t2_path = generate_sample_satellite_pair(output_dir=str(tmp_path))

    pipeline = DeepLearningCore()
    result = pipeline.analyze(
        image_t1=t1_path,
        image_t2=t2_path,
        query="new runway"
    )

    assert result["status"] == "SUCCESS"
    assert "summary" in result
    assert "latency" in result
    assert result["latency"]["total_pipeline_ms"] > 0
    assert "instances" in result
    assert len(result["instances"]) > 0

    first_inst = result["instances"][0]
    assert "threat_level" in first_inst
    assert "color_hex" in first_inst
    assert "geometry" in first_inst
    assert "match_confidence" in first_inst
    assert first_inst["threat_level"] in ["RED", "YELLOW", "GREEN"]
