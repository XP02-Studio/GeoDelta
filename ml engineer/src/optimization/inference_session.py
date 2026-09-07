"""
Unified Accelerated Inference Session Manager.
Coordinates ChangeFormer and RS-CLIP accelerated execution pipelines.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch

from src.config import CONFIG, AppConfig
from src.optimization.export_onnx import export_all
from src.optimization.trt_engine import RuntimeSessionManager
from src.optimization.io_binding import FastIOBindingRunner


class AcceleratedInferenceEngine:
    """
    High-performance engine providing optimized execution for ChangeFormer and RS-CLIP.
    Utilizes ONNXRuntime, TensorRT layer fusion, FP16, and I/O Binding.
    """
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or CONFIG
        self.cf_session = None
        self.cf_io_runner = None
        self.clip_vis_session = None
        self.clip_vis_io_runner = None
        self.clip_txt_session = None
        self.clip_txt_io_runner = None

        self._initialize_engines()

    def _initialize_engines(self):
        """Ensures ONNX models are compiled and sessions are initialized."""
        cf_onnx = self.config.changeformer.onnx_path
        vis_onnx = self.config.rs_clip.onnx_visual_path
        txt_onnx = self.config.rs_clip.onnx_text_path

        # If ONNX models don't exist yet, export them automatically
        if not (cf_onnx.exists() and vis_onnx.exists() and txt_onnx.exists()):
            print("[Inference Engine] Compiling and exporting ONNX models...")
            export_all()

        # Initialize ONNXRuntime sessions with hardware execution providers
        use_fp16 = self.config.optimization.fp16_enabled
        self.cf_session = RuntimeSessionManager.create_session(cf_onnx, use_fp16=use_fp16)
        self.cf_io_runner = FastIOBindingRunner(self.cf_session)

        self.clip_vis_session = RuntimeSessionManager.create_session(vis_onnx, use_fp16=use_fp16)
        self.clip_vis_io_runner = FastIOBindingRunner(self.clip_vis_session)

        self.clip_txt_session = RuntimeSessionManager.create_session(txt_onnx, use_fp16=use_fp16)
        self.clip_txt_io_runner = FastIOBindingRunner(self.clip_txt_session)

        print("[Inference Engine] All accelerated inference sessions successfully loaded.")

    def run_changeformer(self, t1_tensor: np.ndarray, t2_tensor: np.ndarray) -> np.ndarray:
        """
        Executes ChangeFormer inference with I/O Binding.
        Args:
            t1_tensor: [1, 3, H, W] float32 numpy array
            t2_tensor: [1, 3, H, W] float32 numpy array
        Returns:
            [H, W] float32 change probability map
        """
        feed = {
            "input_t1": t1_tensor,
            "input_t2": t2_tensor
        }

        if self.config.optimization.io_binding_enabled:
            outputs = self.cf_io_runner.run_with_binding(feed)
        else:
            outputs = self.cf_session.run(None, feed)

        # Output shape: [1, 1, H, W] -> [H, W]
        prob_map = outputs[0][0, 0]
        return prob_map

    def run_clip_visual(self, crops_tensor: np.ndarray) -> np.ndarray:
        """
        Executes RS-CLIP visual encoder on image crops with I/O Binding.
        Args:
            crops_tensor: [N, 3, 224, 224] float32 numpy array
        Returns:
            [N, 512] float32 visual embeddings
        """
        if len(crops_tensor) == 0:
            return np.empty((0, 512), dtype=np.float32)

        feed = {"image_crops": crops_tensor}

        if self.config.optimization.io_binding_enabled:
            outputs = self.clip_vis_io_runner.run_with_binding(feed)
        else:
            outputs = self.clip_vis_session.run(None, feed)

        return outputs[0]

    def run_clip_text(self, text_tokens: np.ndarray) -> np.ndarray:
        """
        Executes RS-CLIP text encoder on token IDs with I/O Binding.
        Args:
            text_tokens: [B, 77] int64 numpy array
        Returns:
            [B, 512] float32 text embeddings
        """
        feed = {"text_tokens": text_tokens}

        if self.config.optimization.io_binding_enabled:
            outputs = self.clip_txt_io_runner.run_with_binding(feed)
        else:
            outputs = self.clip_txt_session.run(None, feed)

        return outputs[0]
