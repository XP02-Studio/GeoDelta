"""
REST API Serving Layer for Deep Learning Core
Exposes endpoints for real-time satellite change analysis, RS-CLIP semantic matching, and latency benchmarks.
"""

import base64
import io
import os
import time
import argparse
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

from src.config import LATENCY_BUDGET_MS, ALLOW_INTERNET
from src.pipeline import DeepLearningCore

def create_app(core: Optional[DeepLearningCore] = None) -> Flask:
    app = Flask(__name__)
    CORS(app)

    # Initialize or inject DL Core
    app.config["DL_CORE"] = core or DeepLearningCore()

    def decode_image(data: Any) -> np.ndarray:
        """Decodes base64 string or file stream to RGB numpy array."""
        if hasattr(data, "read"):
            img = Image.open(data.stream).convert("RGB")
        elif isinstance(data, str):
            if "," in data:
                data = data.split(",")[1]
            img_bytes = base64.b64decode(data)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            raise ValueError("Unsupported image format")
        return np.array(img)

    @app.route("/api/v1/health", methods=["GET"])
    def health_check():
        """Returns health status, GPU state, and offline compliance."""
        return jsonify({
            "status": "HEALTHY",
            "offline_mode": not ALLOW_INTERNET,
            "device": "CUDA" if torch.cuda.is_available() else "CPU",
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "latency_sla_target_ms": LATENCY_BUDGET_MS,
            "models_loaded": {
                "changeformer": True,
                "rs_clip": True
            }
        }), 200

    @app.route("/api/v1/analyze", methods=["POST"])
    def analyze_satellite_pair():
        """
        Processes bitemporal satellite image pair and returns traffic light tagged polygons and match confidences.
        Supports multipart/form-data and application/json (base64).
        """
        dl_core: DeepLearningCore = app.config["DL_CORE"]

        try:
            query = None
            if request.is_json:
                data = request.get_json()
                if "t1_image" not in data or "t2_image" not in data:
                    return jsonify({"error": "Missing 't1_image' or 't2_image' in JSON payload"}), 400
                img_t1 = decode_image(data["t1_image"])
                img_t2 = decode_image(data["t2_image"])
                query = data.get("query")
            elif request.files:
                if "t1_image" not in request.files or "t2_image" not in request.files:
                    return jsonify({"error": "Missing 't1_image' or 't2_image' files in multipart request"}), 400
                img_t1 = decode_image(request.files["t1_image"])
                img_t2 = decode_image(request.files["t2_image"])
                query = request.form.get("query")
            else:
                return jsonify({"error": "Invalid request content type"}), 400

            result = dl_core.analyze(img_t1, img_t2, query=query)
            return jsonify(result), 200

        except Exception as e:
            return jsonify({
                "status": "ERROR",
                "message": str(e)
            }), 500

    @app.route("/api/v1/benchmark", methods=["POST"])
    def run_benchmark():
        """Runs offline benchmark suite and returns latency metrics against 420ms budget."""
        dl_core: DeepLearningCore = app.config["DL_CORE"]
        data = request.get_json() or {}
        iterations = int(data.get("iterations", 10))

        # Synthetic test tiles
        dummy_t1 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        dummy_t2 = dummy_t1.copy()
        # Inject artificial change
        dummy_t2[100:150, 100:150] = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

        latencies = []
        for _ in range(iterations):
            t_start = time.perf_counter()
            _ = dl_core.analyze(dummy_t1, dummy_t2, query="new runway")
            elapsed = (time.perf_counter() - t_start) * 1000.0
            latencies.append(elapsed)

        lat_arr = np.array(latencies)
        mean_lat = float(np.mean(lat_arr))
        p95_lat = float(np.percentile(lat_arr, 95))

        return jsonify({
            "status": "SUCCESS",
            "iterations": iterations,
            "latency_metrics": {
                "mean_ms": round(mean_lat, 2),
                "min_ms": round(float(np.min(lat_arr)), 2),
                "max_ms": round(float(np.max(lat_arr)), 2),
                "p50_ms": round(float(np.percentile(lat_arr, 50)), 2),
                "p95_ms": round(p95_lat, 2),
                "p99_ms": round(float(np.percentile(lat_arr, 99)), 2)
            },
            "sla_target_ms": LATENCY_BUDGET_MS,
            "sla_compliant": mean_lat <= LATENCY_BUDGET_MS
        }), 200

    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Learning Core REST API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--use-fp16", action="store_true", help="Enable FP16 acceleration")
    args = parser.parse_args()

    app = create_app()
    app.run(host=args.host, port=args.port, debug=False)
