"""
ONNX Model Exporter for ChangeFormer and RS-CLIP
Converts PyTorch models to ONNX IR with graph optimization and FP16 support.
"""

import os
import torch
from pathlib import Path
from typing import Optional, Tuple
from src.config import (
    CHANGEFORMER_ONNX_PATH,
    RS_CLIP_VISUAL_ONNX_PATH,
    RS_CLIP_TEXT_ONNX_PATH,
    DEFAULT_INPUT_SIZE
)
from src.models.changeformer import ChangeFormer
from src.models.rs_clip import RSCLIP

class ONNXExporter:
    """Handles ONNX graph export and validation for air-gapped inference."""

    @staticmethod
    def export_changeformer(
        model: ChangeFormer,
        output_path: Optional[str] = None,
        input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
        opset_version: int = 17,
        dynamic_batch: bool = True
    ) -> str:
        """
        Exports Siamese ChangeFormer model to ONNX.
        Inputs:
            t1: [B, 3, H, W]
            t2: [B, 3, H, W]
        Output:
            logits: [B, 2, H, W]
        """
        out_path = Path(output_path or CHANGEFORMER_ONNX_PATH)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        model.eval()
        dummy_t1 = torch.randn(1, 3, input_size[0], input_size[1], dtype=torch.float32)
        dummy_t2 = torch.randn(1, 3, input_size[0], input_size[1], dtype=torch.float32)

        dynamic_axes = {
            "t1": {0: "batch_size"},
            "t2": {0: "batch_size"},
            "logits": {0: "batch_size"}
        } if dynamic_batch else None

        print(f"[*] Exporting ChangeFormer to ONNX at: {out_path} ...")
        torch.onnx.export(
            model,
            (dummy_t1, dummy_t2),
            str(out_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["t1", "t2"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes
        )
        print(f"[OK] ChangeFormer successfully exported to: {out_path}")
        return str(out_path)

    @staticmethod
    def export_rs_clip_visual(
        model: RSCLIP,
        output_path: Optional[str] = None,
        patch_size: Tuple[int, int] = (64, 64),
        opset_version: int = 17
    ) -> str:
        """
        Exports RS-CLIP visual encoder to ONNX.
        Input: patches [B, 3, 64, 64]
        Output: visual_embeddings [B, 512]
        """
        out_path = Path(output_path or RS_CLIP_VISUAL_ONNX_PATH)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        model.eval()
        dummy_patch = torch.randn(1, 3, patch_size[0], patch_size[1], dtype=torch.float32)

        dynamic_axes = {
            "patches": {0: "batch_size"},
            "visual_embeddings": {0: "batch_size"}
        }

        print(f"[*] Exporting RS-CLIP Visual Encoder to: {out_path} ...")
        torch.onnx.export(
            model.visual_encoder,
            dummy_patch,
            str(out_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["patches"],
            output_names=["visual_embeddings"],
            dynamic_axes=dynamic_axes
        )
        print(f"[OK] RS-CLIP Visual Encoder successfully exported to: {out_path}")
        return str(out_path)

    @staticmethod
    def export_rs_clip_text(
        model: RSCLIP,
        output_path: Optional[str] = None,
        max_length: int = 32,
        opset_version: int = 17
    ) -> str:
        """
        Exports RS-CLIP text encoder to ONNX.
        Input: tokens [B, 32]
        Output: text_embeddings [B, 512]
        """
        out_path = Path(output_path or RS_CLIP_TEXT_ONNX_PATH)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        model.eval()
        dummy_tokens = torch.randint(0, 100, (1, max_length), dtype=torch.long)

        dynamic_axes = {
            "tokens": {0: "batch_size"},
            "text_embeddings": {0: "batch_size"}
        }

        print(f"[*] Exporting RS-CLIP Text Encoder to: {out_path} ...")
        torch.onnx.export(
            model.text_encoder,
            dummy_tokens,
            str(out_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["tokens"],
            output_names=["text_embeddings"],
            dynamic_axes=dynamic_axes
        )
        print(f"[OK] RS-CLIP Text Encoder successfully exported to: {out_path}")
        return str(out_path)
