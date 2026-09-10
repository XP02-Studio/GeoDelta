"""
Global Configuration & Air-Gapped Path Manager
Strictly enforces offline local file loading and latency budget constraints.
"""

import os
from pathlib import Path

# Workspace Root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Air-Gapped & Offline Security Flags
ALLOW_INTERNET = False
LOCAL_FILES_ONLY = True
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TORCH_HOME"] = str(ROOT_DIR / "models" / "torch_cache")

# Persistent Local Weight Directories
MODELS_DIR = ROOT_DIR / "models"
CHANGEFORMER_DIR = MODELS_DIR / "changeformer"
RS_CLIP_DIR = MODELS_DIR / "rs_clip"
OPTIMIZED_DIR = MODELS_DIR / "optimized"

# Ensure directories exist
for directory in [MODELS_DIR, CHANGEFORMER_DIR, RS_CLIP_DIR, OPTIMIZED_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Weight File Paths
CHANGEFORMER_WEIGHTS_PATH = CHANGEFORMER_DIR / "changeformer_levir.pth"
CHANGEFORMER_ONNX_PATH = OPTIMIZED_DIR / "changeformer_fp16.onnx"
CHANGEFORMER_TRT_ENGINE_PATH = OPTIMIZED_DIR / "changeformer_fp16.engine"

RS_CLIP_WEIGHTS_PATH = RS_CLIP_DIR / "pytorch_model.bin"
RS_CLIP_CONFIG_PATH = RS_CLIP_DIR / "config.json"
RS_CLIP_VOCAB_PATH = RS_CLIP_DIR / "vocab.json"
RS_CLIP_VISUAL_ONNX_PATH = OPTIMIZED_DIR / "rs_clip_visual_fp16.onnx"
RS_CLIP_TEXT_ONNX_PATH = OPTIMIZED_DIR / "rs_clip_text_fp16.onnx"

# Performance & SLA Target
LATENCY_BUDGET_MS = 420.0  # Strict SLA in milliseconds
DEFAULT_INPUT_SIZE = (256, 256) # Standard tile dimensions (H, W)
DEFAULT_EMBED_DIM = 512

# Threat Level Color Definitions (Hex & RGB)
COLOR_RED = "#EF4444"      # Heavy Infrastructure Additions (Bunkers, Airstrips, Buildings)
COLOR_YELLOW = "#EAB308"   # Logistical Surface Changes (Roads, Clearings, Bridge Extensions)
COLOR_GREEN = "#22C55E"    # Filtered False Positives (Shadows, Cloud Cover, Seasonal)

THREAT_COLORS = {
    "RED": COLOR_RED,
    "YELLOW": COLOR_YELLOW,
    "GREEN": COLOR_GREEN
}

# Post-Processing Parameters
MIN_CONTOUR_AREA_PIXELS = 16
MAX_CONTOUR_AREA_PIXELS = 256 * 256
CHANGE_THRESHOLD = 0.45
FALSE_POSITIVE_SHADOW_THRESHOLD = 0.25
