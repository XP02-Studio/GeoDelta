"""
Multi-Scale Difference MLP Decoder for ChangeFormer.
Fuses multi-scale spatial difference features into high-resolution change probability maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class MLPProjection(nn.Module):
    """Linear projection and normalization layer for multi-scale feature alignment."""
    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MultiScaleDifferenceDecoder(nn.Module):
    """
    Decodes multi-scale difference features from Siamese transformer stages
    into dense pixel-wise change probability masks.
    """
    def __init__(
        self,
        embed_dims: List[int] = [64, 128, 256, 512],
        decoder_dim: int = 256,
        num_classes: int = 2,
        drop_rate: float = 0.1
    ):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.num_classes = num_classes

        # Difference projections for each scale
        self.proj1 = MLPProjection(embed_dims[0], decoder_dim)
        self.proj2 = MLPProjection(embed_dims[1], decoder_dim)
        self.proj3 = MLPProjection(embed_dims[2], decoder_dim)
        self.proj4 = MLPProjection(embed_dims[3], decoder_dim)

        # Multi-scale feature fusion block
        self.fusion = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_rate),
            nn.Conv2d(decoder_dim, decoder_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim // 2),
            nn.ReLU(inplace=True),
        )

        # Final classification head predicting [B, num_classes, H, W]
        self.classifier = nn.Conv2d(decoder_dim // 2, num_classes, kernel_size=1)

    def forward(self, diff_features: List[torch.Tensor], target_shape: torch.Size) -> torch.Tensor:
        """
        Args:
            diff_features: List of 4 difference tensors from Stage 1 to Stage 4.
            target_shape: (H, W) target spatial dimensions of the original input image.
        Returns:
            Logits tensor of shape [B, num_classes, H, W]
        """
        d1, d2, d3, d4 = diff_features

        # Project all stages to decoder_dim
        p1 = self.proj1(d1)  # [B, decoder_dim, H/4, W/4]
        p2 = self.proj2(d2)  # [B, decoder_dim, H/8, W/8]
        p3 = self.proj3(d3)  # [B, decoder_dim, H/16, W/16]
        p4 = self.proj4(d4)  # [B, decoder_dim, H/32, W/32]

        # Upsample all projections to stage 1 spatial resolution (H/4, W/4)
        target_stage1_size = (p1.shape[2], p1.shape[3])
        p2_up = F.interpolate(p2, size=target_stage1_size, mode="bilinear", align_corners=False)
        p3_up = F.interpolate(p3, size=target_stage1_size, mode="bilinear", align_corners=False)
        p4_up = F.interpolate(p4, size=target_stage1_size, mode="bilinear", align_corners=False)

        # Concatenate multi-scale representations
        fused = torch.cat([p1, p2_up, p3_up, p4_up], dim=1)  # [B, decoder_dim * 4, H/4, W/4]
        fused = self.fusion(fused)

        # Predict change logits at 1/4 resolution
        logits_low = self.classifier(fused)  # [B, num_classes, H/4, W/4]

        # Upsample directly to target resolution (H, W)
        logits = F.interpolate(logits_low, size=target_shape, mode="bilinear", align_corners=False)
        return logits
