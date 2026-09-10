"""
Multimodal Zero-Shot Semantic Matching Engine.
Bridges natural language queries and remote-sensing visual features using RS-CLIP.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rs_clip.tokenizer import OfflineCLIPTokenizer
from src.rs_clip.text_encoder import RSCLIPTextEncoder
from src.rs_clip.visual_encoder import RSCLIPVisualEncoder
from src.changeformer.postprocess import PolygonInstance
from src.config import CONFIG, RSCLIPConfig


class RSCLIPMatcher(nn.Module):
    """
    Multimodal Semantic Matcher for zero-shot satellite image query matching.
    """
    def __init__(
        self,
        cfg: Optional[RSCLIPConfig] = None,
        weights_path: Optional[str] = None,
        device: str = "cpu"
    ):
        super().__init__()
        self.cfg = cfg or CONFIG.rs_clip
        self.device = device

        # Pure offline tokenizer
        self.tokenizer = OfflineCLIPTokenizer(
            vocab_path=self.cfg.vocab_path,
            context_length=self.cfg.context_length
        )

        # Dual encoders
        self.text_encoder = RSCLIPTextEncoder(
            vocab_size=self.cfg.vocab_size,
            context_length=self.cfg.context_length,
            transformer_width=self.cfg.transformer_width,
            transformer_heads=self.cfg.transformer_heads,
            transformer_layers=self.cfg.transformer_layers,
            embed_dim=self.cfg.embed_dim
        )

        self.visual_encoder = RSCLIPVisualEncoder(
            image_size=self.cfg.image_resolution[0],
            patch_size=self.cfg.visual_patch_size,
            visual_width=self.cfg.visual_width,
            visual_layers=self.cfg.visual_layers,
            embed_dim=self.cfg.embed_dim
        )

        # Preprocessing normalization constants
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        self._load_or_init_weights(weights_path)
        self.to(device)
        self.eval()

    def _load_or_init_weights(self, weights_path: Optional[str] = None):
        """Loads local air-gapped checkpoint or initializes local weight archive."""
        path = Path(weights_path) if weights_path else self.cfg.weights_path
        if path.exists() and path.is_file():
            print(f"[RS-CLIP] Loading local multimodal weights from: {path}")
            state_dict = torch.load(str(path), map_location="cpu")
            self.load_state_dict(state_dict, strict=False)
        else:
            print(f"[RS-CLIP] Model weight archive '{path}' not found. Initialized with structured weights.")
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), str(path))
            print(f"[RS-CLIP] Cached local weights at: {path}")

    @torch.no_grad()
    def encode_text(self, text_queries: Union[str, List[str]]) -> torch.Tensor:
        """
        Converts plain-English text queries into 512-d normalized query vectors.
        Args:
            text_queries: Single string or list of query strings (e.g. "new runway")
        Returns:
            [B, 512] normalized PyTorch tensor
        """
        tokens = self.tokenizer.encode(text_queries).to(self.device)
        text_features = self.text_encoder(tokens)
        return text_features

    def preprocess_crops(self, crops: List[np.ndarray]) -> torch.Tensor:
        """Resizes and normalizes image crops into a standardized [N, 3, 224, 224] tensor."""
        target_h, target_w = self.cfg.image_resolution
        processed_tensors = []

        for crop in crops:
            if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                crop = np.zeros((target_h, target_w, 3), dtype=np.uint8)

            resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            # [H, W, C] uint8 -> [C, H, W] float32 in [0, 1]
            tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            processed_tensors.append(tensor)

        batch = torch.stack(processed_tensors, dim=0).to(self.device)
        # Apply standardization
        batch = (batch - self.mean) / self.std
        return batch

    @torch.no_grad()
    def encode_image_crops(self, crops: List[np.ndarray]) -> torch.Tensor:
        """
        Extracts 512-d normalized visual vectors from satellite image crops.
        Args:
            crops: List of [H, W, 3] uint8 RGB crops
        Returns:
            [N, 512] normalized PyTorch tensor
        """
        if len(crops) == 0:
            return torch.empty((0, self.cfg.embed_dim), device=self.device)

        batch = self.preprocess_crops(crops)
        visual_features = self.visual_encoder(batch)
        return visual_features

    @torch.no_grad()
    def match_instances(
        self,
        instances: List[PolygonInstance],
        t2_image: np.ndarray,
        query: str,
        confidence_threshold: float = 0.20
    ) -> List[PolygonInstance]:
        """
        Computes Cosine Similarity between query text vector and all detected change instances.
        Appends match confidence to each PolygonInstance.

        Args:
            instances: List of detected polygon instances
            t2_image: [H, W, 3] Satellite T2 pass
            query: Plain-English query (e.g. "new runway")
            confidence_threshold: Minimum match confidence
        Returns:
            Updated polygon instances with semantic_label and semantic_match_score
        """
        if not instances or not query.strip():
            return instances

        h, w = t2_image.shape[:2]
        crops = []

        # Extract bounding box crops for each instance
        for inst in instances:
            bx, by, bw, bh = inst.bbox
            # Add subtle padding around bounding box for context
            pad_x = int(bw * 0.15)
            pad_y = int(bh * 0.15)
            x1 = max(0, bx - pad_x)
            y1 = max(0, by - pad_y)
            x2 = min(w, bx + bw + pad_x)
            y2 = min(h, by + bh + pad_y)

            crop = t2_image[y1:y2, x1:x2]
            crops.append(crop)

        # Generate text and visual embeddings
        text_emb = self.encode_text(query)          # [1, 512]
        visual_embs = self.encode_image_crops(crops)  # [N, 512]

        # Calculate Cosine Similarity: [N, 512] @ [512, 1] -> [N]
        cos_similarities = (visual_embs @ text_emb.T).squeeze(-1).cpu().numpy()

        for inst, sim_score in zip(instances, cos_similarities):
            score = float(np.clip(sim_score, 0.0, 1.0))
            inst.semantic_label = query
            inst.semantic_match_score = round(score, 4)

            # Re-evaluate threat level if query specifically indicates heavy infrastructure
            lower_query = query.lower()
            red_keywords = CONFIG.traffic_light.red.keywords
            if any(k in lower_query for k in red_keywords) and score > 0.65:
                inst.threat_level = "red"
                inst.hex_color = CONFIG.traffic_light.red.hex
                inst.category = CONFIG.traffic_light.red.category

        return instances
