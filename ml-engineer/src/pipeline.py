"""
Deep Learning Core: End-to-End Change Detection & Semantic Intelligence Pipeline
Orchestrates ChangeFormer Siamese inference, Traffic Light polygon extraction, RS-CLIP semantic matching, and latency tracking.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.config import (
    CHANGEFORMER_WEIGHTS_PATH,
    CHANGEFORMER_ONNX_PATH,
    RS_CLIP_WEIGHTS_PATH,
    RS_CLIP_VOCAB_PATH,
    LATENCY_BUDGET_MS,
    DEFAULT_INPUT_SIZE
)
from src.models.changeformer import ChangeFormer
from src.models.rs_clip import RSCLIP
from src.postprocessing.instance_extractor import InstanceExtractor, ChangeInstance
from src.postprocessing.threat_classifier import ThreatClassifier
from src.postprocessing.payload_formatter import PayloadFormatter
from src.matching.semantic_engine import SemanticMatchingEngine
from src.optimization.profiler import PipelineProfiler
from src.optimization.runtime_engine import OptimizedInferenceEngine

class DeepLearningCore:
    """
    Unified Deep Learning Core Pipeline for Air-Gapped Satellite Analysis.
    """
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_onnx: bool = True,
        use_fp16: bool = True,
        use_tensorrt: bool = False,
        use_io_binding: bool = True
    ):
        self.device = device
        self.use_onnx = use_onnx
        self.use_fp16 = use_fp16 and (device == "cuda")
        self.profiler = PipelineProfiler(sla_budget_ms=LATENCY_BUDGET_MS)

        print(f"[*] Initializing Deep Learning Core (Device: {self.device}, FP16: {self.use_fp16}, ONNX: {self.use_onnx})...")

        # 1. Initialize ChangeFormer Model
        self.changeformer_pt = ChangeFormer.load_local(str(CHANGEFORMER_WEIGHTS_PATH), device=self.device)
        self.optimized_engine: Optional[OptimizedInferenceEngine] = None

        if self.use_onnx and os.path.exists(CHANGEFORMER_ONNX_PATH):
            try:
                self.optimized_engine = OptimizedInferenceEngine(
                    onnx_model_path=str(CHANGEFORMER_ONNX_PATH),
                    use_gpu=(self.device == "cuda"),
                    use_fp16=self.use_fp16,
                    use_tensorrt=use_tensorrt,
                    use_io_binding=use_io_binding
                )
            except Exception as e:
                print(f"[!] Could not start ONNX engine ({e}). Using PyTorch backend.")

        # 2. Initialize RS-CLIP Multimodal Model
        self.rs_clip = RSCLIP.load_local(
            weights_path=str(RS_CLIP_WEIGHTS_PATH),
            vocab_path=str(RS_CLIP_VOCAB_PATH),
            device=self.device
        )
        self.semantic_engine = SemanticMatchingEngine(self.rs_clip, device=self.device)

        # 3. Initialize Post-Processing Engines
        self.instance_extractor = InstanceExtractor()
        self.threat_classifier = ThreatClassifier()

        print("[OK] Deep Learning Core initialized successfully and ready for air-gapped inference.")

    def _preprocess_images(self, img_t1: np.ndarray, img_t2: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """Preprocesses RGB images into normalized float32 tensors."""
        H_orig, W_orig = img_t1.shape[:2]

        # Resize to standard size if needed
        if (H_orig, W_orig) != DEFAULT_INPUT_SIZE:
            t1_resized = cv2.resize(img_t1, DEFAULT_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
            t2_resized = cv2.resize(img_t2, DEFAULT_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        else:
            t1_resized = img_t1
            t2_resized = img_t2

        t1_norm = (t1_resized.astype(np.float32) / 255.0).transpose(2, 0, 1) # [3, H, W]
        t2_norm = (t2_resized.astype(np.float32) / 255.0).transpose(2, 0, 1) # [3, H, W]

        # Satellite standard normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

        t1_norm = (t1_norm - mean) / std
        t2_norm = (t2_norm - mean) / std

        t1_tensor = torch.from_numpy(t1_norm).unsqueeze(0).to(self.device) # [1, 3, H, W]
        t2_tensor = torch.from_numpy(t2_norm).unsqueeze(0).to(self.device)

        return t1_tensor, t2_tensor, (H_orig, W_orig)

    def analyze(
        self,
        image_t1: Union[np.ndarray, str],
        image_t2: Union[np.ndarray, str],
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end inference on bitemporal satellite image pair.
        Args:
            image_t1: Time 1 image (RGB array or file path)
            image_t2: Time 2 image (RGB array or file path)
            query: Optional plain-English query (e.g. "new runway")
        Returns:
            Formatted JSON payload with traffic light instances, match confidence, and latency breakdown.
        """
        profiler = PipelineProfiler(sla_budget_ms=LATENCY_BUDGET_MS)

        # 1. Image Loading & Preprocessing
        with profiler.measure("1_image_preprocessing"):
            if isinstance(image_t1, (str, Path)):
                img1 = cv2.imread(str(image_t1))
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            else:
                img1 = image_t1

            if isinstance(image_t2, (str, Path)):
                img2 = cv2.imread(str(image_t2))
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            else:
                img2 = image_t2

            t1_tensor, t2_tensor, orig_shape = self._preprocess_images(img1, img2)

        # 2. ChangeFormer Siamese Inference
        with profiler.measure("2_changeformer_forward_pass"):
            if self.optimized_engine is not None:
                # Optimized ONNX/TensorRT execution
                t1_np = t1_tensor.cpu().numpy()
                t2_np = t2_tensor.cpu().numpy()
                logits_np = self.optimized_engine.run_changeformer(t1_np, t2_np)
                prob_map = 1.0 / (1.0 + np.exp(-logits_np[0, 1])) # Sigmoid on change class
            else:
                # PyTorch JIT / standard execution
                with torch.no_grad():
                    if self.use_fp16:
                        with torch.cuda.amp.autocast():
                            logits = self.changeformer_pt(t1_tensor, t2_tensor)
                    else:
                        logits = self.changeformer_pt(t1_tensor, t2_tensor)
                    probs = F.softmax(logits, dim=1)
                    prob_map = probs[0, 1].cpu().numpy()

            # Resize probability map back to original dimensions if different
            if prob_map.shape != orig_shape:
                prob_map = cv2.resize(prob_map, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR)

        # 3. Instance Segmentation & Polygon Extraction
        with profiler.measure("3_instance_polygon_clustering"):
            instances = self.instance_extractor.extract_instances(
                change_heatmap=prob_map,
                image_t1=img1,
                image_t2=img2
            )

        # 4. Traffic Light Threat Classification
        with profiler.measure("4_threat_classification"):
            classifications = [self.threat_classifier.classify_instance(inst) for inst in instances]

        # 5. RS-CLIP Semantic Matching (if query provided)
        confidences = None
        if query and instances:
            with profiler.measure("5_rs_clip_semantic_matching"):
                confidences = self.semantic_engine.match_instances_with_query(instances, query)

        # 6. JSON Payload Formatting
        with profiler.measure("6_payload_assembly"):
            payload = PayloadFormatter.format_response(
                instances=instances,
                classifications=classifications,
                confidences=confidences,
                query=query,
                latency_metrics=profiler.timings,
                image_shape=orig_shape
            )

        return payload
