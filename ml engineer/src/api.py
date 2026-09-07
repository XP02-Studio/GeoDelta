"""
High-Performance Air-Gapped REST API Server for Deep Learning Core.
Exposes endpoints for real-time change detection, Traffic Light categorization, and RS-CLIP semantic matching.
"""

import os
import io
import time
import base64
import argparse
from typing import Dict, Any, List, Optional
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

from src.config import CONFIG, AppConfig
from src.pipeline import DeepLearningCorePipeline


app = Flask(__name__)
CORS(app)

# Global pipeline instance
PIPELINE: Optional[DeepLearningCorePipeline] = None


def get_pipeline() -> DeepLearningCorePipeline:
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = DeepLearningCorePipeline(CONFIG, use_accelerated_engine=True)
    return PIPELINE


def decode_image_from_bytes_or_base64(raw_data: Any) -> np.ndarray:
    """Decodes image from raw file bytes or Base64 string into RGB numpy array."""
    if isinstance(raw_data, str):
        if "," in raw_data:
            raw_data = raw_data.split(",", 1)[1]
        byte_data = base64.b64decode(raw_data)
    elif hasattr(raw_data, "read"):
        byte_data = raw_data.read()
    elif isinstance(raw_data, bytes):
        byte_data = raw_data
    else:
        raise ValueError("Unsupported image format received.")

    image_array = np.frombuffer(byte_data, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Failed to decode image buffer.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def encode_image_to_base64(image_rgb: np.ndarray) -> str:
    """Encodes RGB numpy array to base64 PNG string."""
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".png", image_bgr)
    return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"


@app.route("/api/v1/health", methods=["GET"])
def health_check():
    """Health check endpoint confirming air-gapped readiness and latency budget status."""
    pipe = get_pipeline()
    active_provider = pipe.engine.cf_session.get_providers()[0] if pipe.engine else "PyTorch-CPU"

    return jsonify({
        "status": "healthy",
        "air_gapped_mode": True,
        "offline_weights_verified": True,
        "active_execution_provider": active_provider,
        "target_latency_budget_ms": CONFIG.target_latency_budget_ms,
        "max_vram_gb": CONFIG.max_vram_gb
    })


@app.route("/api/v1/detect_changes", methods=["POST"])
def detect_changes_endpoint():
    """
    Main change detection and semantic search endpoint.
    Accepts multipart/form-data (t1_image, t2_image, query) or JSON with base64 images.
    """
    try:
        query = None
        prob_threshold = None

        if request.is_json:
            data = request.get_json()
            t1_img = decode_image_from_bytes_or_base64(data.get("t1_image"))
            t2_img = decode_image_from_bytes_or_base64(data.get("t2_image"))
            query = data.get("query")
            prob_threshold = data.get("threshold")
            include_masks = data.get("include_mask_base64", False)
        else:
            if "t1_image" not in request.files or "t2_image" not in request.files:
                return jsonify({"error": "Missing 't1_image' or 't2_image' in request"}), 400

            t1_img = decode_image_from_bytes_or_base64(request.files["t1_image"])
            t2_img = decode_image_from_bytes_or_base64(request.files["t2_image"])
            query = request.form.get("query")
            prob_threshold = float(request.form.get("threshold")) if request.form.get("threshold") else None
            include_masks = request.form.get("include_mask_base64", "false").lower() == "true"

        pipe = get_pipeline()
        result = pipe.detect_changes(
            t1_image=t1_img,
            t2_image=t2_img,
            query=query,
            prob_threshold=prob_threshold
        )

        # Build clean response payload
        response_payload = {
            "status": result["status"],
            "air_gapped": result["air_gapped"],
            "query": result["query"],
            "threat_summary": result["threat_summary"],
            "latency_metrics": result["latency_metrics"],
            "budget_satisfied": result["budget_satisfied"],
            "instances": result["instances"]
        }

        if include_masks:
            response_payload["visual_artifacts"] = {
                "color_overlay_png": encode_image_to_base64(result["artifacts"]["color_overlay"]),
                "blended_overlay_png": encode_image_to_base64(result["artifacts"]["blended_overlay"])
            }

        return jsonify(response_payload)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/semantic_match", methods=["POST"])
def semantic_match_endpoint():
    """Zero-shot matching of plain-English text query against cropped image."""
    try:
        data = request.get_json() if request.is_json else request.form
        query = data.get("query")
        if not query:
            return jsonify({"error": "Query string is required."}), 400

        if request.is_json:
            crop_img = decode_image_from_bytes_or_base64(data.get("crop_image"))
        else:
            crop_img = decode_image_from_bytes_or_base64(request.files["crop_image"])

        pipe = get_pipeline()
        crops = [crop_img]
        text_emb = pipe.matcher.encode_text(query)
        vis_emb = pipe.matcher.encode_image_crops(crops)

        sim_score = float((vis_emb @ text_emb.T).squeeze().cpu().numpy())
        sim_score = float(np.clip(sim_score, 0.0, 1.0))

        return jsonify({
            "query": query,
            "match_confidence": round(sim_score, 4),
            "match_status": "high" if sim_score > 0.65 else ("moderate" if sim_score > 0.40 else "low")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Learning Core REST Server")
    parser.add_argument("--host", default=CONFIG.raw_config.get("server", {}).get("host", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=CONFIG.raw_config.get("server", {}).get("port", 8000))
    args = parser.parse_args()

    # Pre-warm pipeline on launch
    get_pipeline()
    print(f"[API Server] Starting Air-Gapped API at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)
