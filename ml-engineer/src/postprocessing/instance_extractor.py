"""
Instance Segmentation & Polygon Extraction Module
Converts ChangeFormer heatmaps into discrete polygon instances with bounding boxes and crops.
"""

import cv2
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from src.config import MIN_CONTOUR_AREA_PIXELS, MAX_CONTOUR_AREA_PIXELS, CHANGE_THRESHOLD

@dataclass
class ChangeInstance:
    instance_id: str
    bbox: List[int]                  # [x, y, width, height]
    bbox_normalized: List[float]     # [x_norm, y_norm, w_norm, h_norm] in [0, 1]
    polygon: List[List[int]]         # [[x1, y1], [x2, y2], ...]
    polygon_normalized: List[List[float]]
    area_pixels: float
    centroid: List[float]            # [cx, cy]
    aspect_ratio: float
    solidity: float
    mean_change_score: float
    cropped_patch_t2: np.ndarray     # RGB patch [H, W, 3]
    cropped_patch_t1: np.ndarray     # RGB patch [H, W, 3]


class InstanceExtractor:
    """
    Extracts instance-segmented polygons and morphological features from change detection masks.
    """
    def __init__(self, min_area: int = MIN_CONTOUR_AREA_PIXELS,
                 max_area: int = MAX_CONTOUR_AREA_PIXELS,
                 threshold: float = CHANGE_THRESHOLD):
        self.min_area = min_area
        self.max_area = max_area
        self.threshold = threshold

    def extract_instances(self,
                          change_heatmap: np.ndarray,
                          image_t1: np.ndarray,
                          image_t2: np.ndarray) -> List[ChangeInstance]:
        """
        Processes change probability map and extracts vector polygon clusters.
        Args:
            change_heatmap: [H, W] float array in range [0, 1]
            image_t1: [H, W, 3] uint8 RGB image
            image_t2: [H, W, 3] uint8 RGB image
        Returns:
            List of ChangeInstance objects
        """
        H, W = change_heatmap.shape

        # 1. Binary Thresholding
        binary_mask = (change_heatmap >= self.threshold).astype(np.uint8) * 255

        # 2. Morphological noise filtering (opening removes speckles, closing bridges gaps)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 3. Contour Detection
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        instances: List[ChangeInstance] = []

        for idx, cnt in enumerate(contours, start=1):
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            # Bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            if w <= 0 or h <= 0:
                continue

            # Polygon approximation
            epsilon = 0.015 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            poly_points = approx.reshape(-1, 2).tolist()
            if len(poly_points) < 3:
                # Fallback to bounding box polygon if approximation is degenerate
                poly_points = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

            # Normalized coordinates
            poly_norm = [[round(p[0] / W, 4), round(p[1] / H, 4)] for p in poly_points]
            bbox_norm = [round(x / W, 4), round(y / H, 4), round(w / W, 4), round(h / H, 4)]

            # Centroid
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
            else:
                cx = float(x + w / 2)
                cy = float(y + h / 2)

            # Aspect ratio & Solidity
            aspect_ratio = float(w) / float(h) if h > 0 else 1.0
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / float(hull_area) if hull_area > 0 else 1.0

            # Mean change score in contour
            instance_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(instance_mask, [cnt], -1, 255, -1)
            mean_score = float(np.mean(change_heatmap[instance_mask > 0]))

            # Crop patches from T1 and T2 (with padding)
            pad_x = max(2, int(w * 0.1))
            pad_y = max(2, int(h * 0.1))
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(W, x + w + pad_x)
            y1 = min(H, y + h + pad_y)

            crop_t1 = image_t1[y0:y1, x0:x1]
            crop_t2 = image_t2[y0:y1, x0:x1]

            # Resize crops to standard patch size for neural visual encoder
            crop_t1_std = cv2.resize(crop_t1, (64, 64), interpolation=cv2.INTER_LINEAR)
            crop_t2_std = cv2.resize(crop_t2, (64, 64), interpolation=cv2.INTER_LINEAR)

            instance = ChangeInstance(
                instance_id=f"inst_{idx:03d}",
                bbox=[x, y, w, h],
                bbox_normalized=bbox_norm,
                polygon=poly_points,
                polygon_normalized=poly_norm,
                area_pixels=float(area),
                centroid=[round(cx, 2), round(cy, 2)],
                aspect_ratio=round(aspect_ratio, 3),
                solidity=round(solidity, 3),
                mean_change_score=round(mean_score, 4),
                cropped_patch_t1=crop_t1_std,
                cropped_patch_t2=crop_t2_std
            )
            instances.append(instance)

        return instances
