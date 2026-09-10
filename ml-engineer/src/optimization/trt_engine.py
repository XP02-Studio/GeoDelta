"""
TensorRT Engine Builder & High-Performance Execution Manager.
Handles layer fusion, FP16 precision compilation, and execution provider orchestration.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np

import onnxruntime as ort
from src.config import CONFIG, AppConfig


class TensorRTBuilder:
    """
    Compiles ONNX graphs into standalone TensorRT serialized engines (.engine).
    Applies layer fusion, FP16 precision, and kernel auto-tuning.
    """
    @staticmethod
    def is_tensorrt_available() -> bool:
        try:
            import tensorrt as trt
            return True
        except ImportError:
            return False

    @classmethod
    def build_engine_from_onnx(
        cls,
        onnx_file_path: Path,
        engine_file_path: Path,
        fp16_enabled: bool = True,
        max_workspace_bytes: int = 2147483648
    ) -> bool:
        """Compiles ONNX model into TensorRT engine if NVIDIA TensorRT is available."""
        if not cls.is_tensorrt_available():
            print("[TensorRT Builder] TensorRT Python library not found. Skipping native engine compilation.")
            return False

        import tensorrt as trt

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, TRT_LOGGER)

        print(f"[TensorRT Builder] Parsing ONNX file: {onnx_file_path}")
        with open(onnx_file_path, "rb") as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(f"[TensorRT Error] {parser.get_error(error)}")
                return False

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_bytes)

        if fp16_enabled and builder.platform_has_tf32:
            print("[TensorRT Builder] Enabling FP16 mixed precision and TF32 Tensor Cores...")
            config.set_flag(trt.BuilderFlag.FP16)

        print(f"[TensorRT Builder] Building serialized engine: {engine_file_path}...")
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("[TensorRT Builder] Engine build failed.")
            return False

        engine_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(engine_file_path, "wb") as f:
            f.write(serialized_engine)

        print(f"[TensorRT Builder] TensorRT engine compiled successfully to {engine_file_path}")
        return True


class RuntimeSessionManager:
    """
    Manages ONNXRuntime inference sessions with prioritized hardware acceleration:
    1. TensorrtExecutionProvider (FP16 Layer Fusion)
    2. CUDAExecutionProvider (GPU Tensor Cores)
    3. CPUExecutionProvider (Optimized multi-threaded CPU fallback)
    """
    @staticmethod
    def create_session(onnx_path: Path, use_fp16: bool = True) -> ort.InferenceSession:
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model file not found at: {onnx_path}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = min(8, os.cpu_count() or 4)

        available_providers = ort.get_available_providers()
        print(f"[Runtime Manager] Available Execution Providers: {available_providers}")

        providers = []
        provider_options = []

        if "TensorrtExecutionProvider" in available_providers:
            providers.append("TensorrtExecutionProvider")
            provider_options.append({
                "device_id": 0,
                "trt_max_workspace_size": 2147483648,
                "trt_fp16_enable": use_fp16,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(onnx_path.parent / "trt_cache")
            })

        if "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
            provider_options.append({
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            })

        providers.append("CPUExecutionProvider")
        provider_options.append({})

        session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=providers,
            provider_options=provider_options
        )

        active_provider = session.get_providers()[0]
        print(f"[Runtime Manager] Initialized {onnx_path.name} with provider: {active_provider}")
        return session
