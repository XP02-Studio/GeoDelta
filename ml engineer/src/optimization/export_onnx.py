"""
ONNX Model Exporter for ChangeFormer and RS-CLIP.
Converts PyTorch computation graphs into optimized ONNX graphs with FP16 support and dynamic axes.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.onnx

from src.config import CONFIG, AppConfig
from src.changeformer.model import build_changeformer, ChangeFormer
from src.rs_clip.text_encoder import RSCLIPTextEncoder
from src.rs_clip.visual_encoder import RSCLIPVisualEncoder
from src.rs_clip.matcher import RSCLIPMatcher


class WrapperChangeFormer(torch.nn.Module):
    """Wrapper to ensure single clean output tensor for ONNX export."""
    def __init__(self, model: ChangeFormer):
        super().__init__()
        self.model = model

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        _, probs = self.model(t1, t2)
        return probs


def export_changeformer_onnx(
    output_path: Optional[Path] = None,
    input_size: Tuple[int, int] = (256, 256),
    opset_version: int = 17,
    fp16: bool = False
) -> Path:
    """Exports ChangeFormer Siamese model to ONNX."""
    output_path = output_path or CONFIG.changeformer.onnx_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ONNX Export] Initializing ChangeFormer for ONNX export (Resolution: {input_size})...")
    model = build_changeformer(device="cpu")
    wrapper = WrapperChangeFormer(model).eval()

    h, w = input_size
    dummy_t1 = torch.randn(1, 3, h, w, dtype=torch.float32)
    dummy_t2 = torch.randn(1, 3, h, w, dtype=torch.float32)

    print(f"[ONNX Export] Exporting to: {output_path}")
    torch.onnx.export(
        wrapper,
        (dummy_t1, dummy_t2),
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_t1", "input_t2"],
        output_names=["change_probability"],
        dynamic_axes={
            "input_t1": {0: "batch_size"},
            "input_t2": {0: "batch_size"},
            "change_probability": {0: "batch_size"}
        }
    )

    print(f"[ONNX Export] ChangeFormer exported successfully to {output_path} (Size: {output_path.stat().st_size / 1e6:.2f} MB)")
    return output_path


def export_rs_clip_onnx(
    visual_output_path: Optional[Path] = None,
    text_output_path: Optional[Path] = None,
    opset_version: int = 17
) -> Tuple[Path, Path]:
    """Exports RS-CLIP visual and text encoders to ONNX."""
    visual_output_path = visual_output_path or CONFIG.rs_clip.onnx_visual_path
    text_output_path = text_output_path or CONFIG.rs_clip.onnx_text_path

    visual_output_path.parent.mkdir(parents=True, exist_ok=True)
    text_output_path.parent.mkdir(parents=True, exist_ok=True)

    print("[ONNX Export] Initializing RS-CLIP encoders for ONNX export...")
    matcher = RSCLIPMatcher(device="cpu")
    vis_enc = matcher.visual_encoder.eval()
    txt_enc = matcher.text_encoder.eval()

    # Visual branch
    dummy_img = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    print(f"[ONNX Export] Exporting RS-CLIP Visual Encoder to: {visual_output_path}")
    torch.onnx.export(
        vis_enc,
        dummy_img,
        str(visual_output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["image_crops"],
        output_names=["visual_embeddings"],
        dynamic_axes={
            "image_crops": {0: "batch_size"},
            "visual_embeddings": {0: "batch_size"}
        }
    )

    # Text branch
    dummy_text = torch.ones(1, 77, dtype=torch.long)
    print(f"[ONNX Export] Exporting RS-CLIP Text Encoder to: {text_output_path}")
    torch.onnx.export(
        txt_enc,
        dummy_text,
        str(text_output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["text_tokens"],
        output_names=["text_embeddings"],
        dynamic_axes={
            "text_tokens": {0: "batch_size"},
            "text_embeddings": {0: "batch_size"}
        }
    )

    print("[ONNX Export] RS-CLIP encoders exported successfully.")
    return visual_output_path, text_output_path


def export_all():
    """Exports both ChangeFormer and RS-CLIP models."""
    cf_path = export_changeformer_onnx()
    vis_path, txt_path = export_rs_clip_onnx()
    return cf_path, vis_path, txt_path


if __name__ == "__main__":
    export_all()
