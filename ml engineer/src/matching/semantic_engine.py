"""
Semantic Matching Engine using RS-CLIP
Bridges plain-English natural language queries with ChangeFormer detected instance crops.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from src.models.rs_clip import RSCLIP
from src.postprocessing.instance_extractor import ChangeInstance

class SemanticMatchingEngine:
    """
    Multimodal zero-shot matching engine for remote sensing change intelligence.
    """
    def __init__(self, rs_clip_model: RSCLIP, device: str = "cpu"):
        self.model = rs_clip_model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

        # Remote Sensing normalization constants
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def preprocess_patches(self, patches: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocesses a list of RGB image patches [H, W, 3] into a normalized batch tensor [B, 3, H, W].
        """
        if not patches:
            return torch.empty((0, 3, 64, 64), device=self.device)

        batch_arr = np.stack(patches, axis=0) # [B, H, W, 3]
        tensor = torch.from_numpy(batch_arr).permute(0, 3, 1, 2).float() / 255.0 # [B, 3, H, W]
        tensor = tensor.to(self.device)
        tensor = (tensor - self.mean) / self.std
        return tensor

    def match_instances_with_query(
        self,
        instances: List[ChangeInstance],
        query: str
    ) -> List[float]:
        """
        Calculates match confidence scores between a plain-English query and extracted instances.
        Args:
            instances: List of ChangeInstance objects containing cropped image patches
            query: Plain-English query string (e.g., "new runway")
        Returns:
            List of confidence scores [0.0, 1.0] corresponding to each instance
        """
        if not instances or not query.strip():
            return [1.0 for _ in instances]

        # Extract T2 patches from instances
        patches = [inst.cropped_patch_t2 for inst in instances]
        batch_tensor = self.preprocess_patches(patches)

        with torch.no_grad():
            confidences = self.model.compute_similarity(batch_tensor, query)
            scores = confidences.cpu().numpy().tolist()

        return [round(float(s), 4) for s in scores]

    def zero_shot_classify_instance(
        self,
        instance: ChangeInstance,
        candidate_labels: List[str]
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Performs multi-class zero-shot classification across candidate remote sensing categories.
        """
        patch_tensor = self.preprocess_patches([instance.cropped_patch_t2])
        
        with torch.no_grad():
            img_feat = self.model.encode_image(patch_tensor) # [1, 512]
            text_feats = self.model.encode_text(candidate_labels, device=self.device) # [N, 512]

            logits = (img_feat @ text_feats.T).squeeze(0) # [N]
            probs = F.softmax(logits * 5.0, dim=0).cpu().numpy()

        scores_dict = {label: round(float(prob), 4) for label, prob in zip(candidate_labels, probs)}
        best_idx = int(np.argmax(probs))
        best_label = candidate_labels[best_idx]
        best_score = float(probs[best_idx])

        return best_label, best_score, scores_dict
