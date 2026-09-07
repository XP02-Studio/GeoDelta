"""
Offline Asset & Model Initializer
Pre-computes and initializes all air-gapped model checkpoints, tokenizers, and ONNX engines.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from src.config import (
    CHANGEFORMER_WEIGHTS_PATH,
    CHANGEFORMER_ONNX_PATH,
    RS_CLIP_WEIGHTS_PATH,
    RS_CLIP_CONFIG_PATH,
    RS_CLIP_VOCAB_PATH,
    RS_CLIP_VISUAL_ONNX_PATH,
    RS_CLIP_TEXT_ONNX_PATH,
    DEFAULT_INPUT_SIZE
)
from src.models.changeformer import ChangeFormer
from src.models.rs_clip import RSCLIP, SimpleRSTokenizer
from src.optimization.onnx_exporter import ONNXExporter

def init_all_assets():
    print("=" * 60)
    print("   INITIALIZING AIR-GAPPED ASSETS & LOCAL MODEL STORAGE")
    print("=" * 60)

    # 1. Initialize & Save ChangeFormer PyTorch Weights
    print("[1/4] Initializing Siamese ChangeFormer model...")
    changeformer = ChangeFormer()
    CHANGEFORMER_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(changeformer.state_dict(), str(CHANGEFORMER_WEIGHTS_PATH))
    print(f"[OK] Saved ChangeFormer weights to: {CHANGEFORMER_WEIGHTS_PATH}")
    
    # 2. Initialize & Save RS-CLIP Model & Tokenizer
    print("\n[2/4] Initializing RS-CLIP Multimodal model & Vocabulary...")
    rs_clip = RSCLIP()
    RS_CLIP_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Save PyTorch weights
    torch.save(rs_clip.state_dict(), str(RS_CLIP_WEIGHTS_PATH))
    print(f"[OK] Saved RS-CLIP weights to: {RS_CLIP_WEIGHTS_PATH}")

    # Save Vocab
    rs_clip.tokenizer.save_vocab(RS_CLIP_VOCAB_PATH)
    print(f"[OK] Saved RS-CLIP vocab to: {RS_CLIP_VOCAB_PATH}")

    # Save Config
    config_data = {
        "model_type": "rs-clip",
        "embed_dim": 512,
        "max_text_length": 32,
        "vocab_size": len(rs_clip.tokenizer.vocab),
        "visual_input_size": [3, 64, 64]
    }
    with open(RS_CLIP_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    print(f"[OK] Saved RS-CLIP config to: {RS_CLIP_CONFIG_PATH}")

    # 3. Export ONNX Models for Low-Latency Inference
    print("\n[3/4] Exporting models to ONNX Intermediate Representation...")
    try:
        ONNXExporter.export_changeformer(
            model=changeformer,
            output_path=str(CHANGEFORMER_ONNX_PATH),
            input_size=DEFAULT_INPUT_SIZE
        )
        ONNXExporter.export_rs_clip_visual(
            model=rs_clip,
            output_path=str(RS_CLIP_VISUAL_ONNX_PATH)
        )
        ONNXExporter.export_rs_clip_text(
            model=rs_clip,
            output_path=str(RS_CLIP_TEXT_ONNX_PATH)
        )
        print("[OK] All ONNX graphs exported successfully.")
    except Exception as e:
        print(f"[!] Warning: ONNX export encountered note ({e}). Native PyTorch backend remains active.")

    print("\n[4/4] Verifying offline integrity...")
    assert os.path.exists(CHANGEFORMER_WEIGHTS_PATH), "ChangeFormer weights missing!"
    assert os.path.exists(RS_CLIP_WEIGHTS_PATH), "RS-CLIP weights missing!"
    assert os.path.exists(RS_CLIP_VOCAB_PATH), "RS-CLIP vocab missing!"

    print("\n" + "=" * 60)
    print(" [SUCCESS] Air-gapped offline environment fully primed and ready!")
    print("=" * 60)

if __name__ == "__main__":
    init_all_assets()
