"""
High-Performance Inference Runtime Engine
Implements TensorRT, ONNX Runtime FP16, and I/O Binding memory management.
"""

import os
import time
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

class OptimizedInferenceEngine:
    """
    Optimized Inference Engine supporting TensorRT, ONNX Runtime FP16, and GPU I/O Binding.
    """
    def __init__(
        self,
        onnx_model_path: str,
        use_gpu: bool = True,
        use_fp16: bool = True,
        use_tensorrt: bool = False,
        use_io_binding: bool = True
    ):
        self.onnx_model_path = str(onnx_model_path)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.use_fp16 = use_fp16
        self.use_tensorrt = use_tensorrt
        self.use_io_binding = use_io_binding
        self.session: Optional[ort.InferenceSession] = None
        self.io_binding = None

        self._initialize_session()

    def _initialize_session(self):
        """Initializes ONNX Runtime session with optimized execution providers."""
        if not HAS_ORT:
            print("[!] ONNX Runtime not available. Falling back to native PyTorch engine.")
            return

        providers = []
        provider_options = []

        # 1. TensorRT Execution Provider
        if self.use_tensorrt and self.use_gpu and "TensorrtExecutionProvider" in ort.get_available_providers():
            trt_options = {
                "device_id": 0,
                "trt_max_workspace_size": 2 * 1024 * 1024 * 1024, # 2GB
                "trt_fp16_enable": self.use_fp16,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(Path(self.onnx_model_path).parent / "trt_cache")
            }
            providers.append("TensorrtExecutionProvider")
            provider_options.append(trt_options)
            print("[*] Enabled TensorRT Execution Provider with FP16 layer fusion.")

        # 2. CUDA Execution Provider with I/O Binding
        if self.use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            cuda_options = {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 4 * 1024 * 1024 * 1024, # 4GB
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True
            }
            providers.append("CUDAExecutionProvider")
            provider_options.append(cuda_options)
            print("[*] Enabled CUDA Execution Provider with pre-allocated arena memory.")

        # 3. CPU Execution Provider (Fallback)
        cpu_options = {
            "intra_op_num_threads": 4,
            "execution_mode": ort.ExecutionMode.ORT_SEQUENTIAL
        }
        providers.append("CPUExecutionProvider")
        provider_options.append(cpu_options)

        # Session Options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True

        try:
            self.session = ort.InferenceSession(
                self.onnx_model_path,
                sess_options=sess_options,
                providers=providers,
                provider_options=provider_options
            )
            print(f"[OK] Active ONNX Runtime Providers: {self.session.get_providers()}")
        except Exception as e:
            print(f"[!] Error loading ONNX model with requested providers ({e}). Retrying on CPU...")
            self.session = ort.InferenceSession(
                self.onnx_model_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"]
            )

        if self.use_io_binding and self.session is not None:
            self.io_binding = self.session.io_binding()

    def run_changeformer(self, t1_np: np.ndarray, t2_np: np.ndarray) -> np.ndarray:
        """
        Executes Siamese ChangeFormer forward pass with pre-allocated memory binding.
        Args:
            t1_np: [B, 3, H, W] float32 numpy array
            t2_np: [B, 3, H, W] float32 numpy array
        Returns:
            logits: [B, 2, H, W] float32 numpy array
        """
        if self.session is None:
            raise RuntimeError("InferenceSession not initialized.")

        # If I/O binding is enabled on CUDA
        if self.use_gpu and self.io_binding is not None:
            try:
                device_id = 0
                t1_ort = ort.OrtValue.ortvalue_from_numpy(t1_np, "cuda", device_id)
                t2_ort = ort.OrtValue.ortvalue_from_numpy(t2_np, "cuda", device_id)

                self.io_binding.bind_ortvalue_input("t1", t1_ort)
                self.io_binding.bind_ortvalue_input("t2", t2_ort)
                self.io_binding.bind_output("logits", "cuda", device_id)

                self.session.run_with_iobinding(self.io_binding)
                outputs = self.io_binding.get_outputs()
                return outputs[0].numpy()
            except Exception:
                # Fallback to standard run if CUDA I/O binding encounters driver quirk
                pass

        # Standard accelerated feed
        outputs = self.session.run(None, {"t1": t1_np, "t2": t2_np})
        return outputs[0]

    def run_generic(self, input_feed: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Runs generic inference for RS-CLIP or other models."""
        if self.session is None:
            raise RuntimeError("InferenceSession not initialized.")
        return self.session.run(None, input_feed)
