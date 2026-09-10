"""
RS-CLIP Visual Encoder for Remote Sensing Imagery.
Converts cropped bounding box regions from ChangeFormer into 512-dimensional normalized embeddings.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VisualPatchEmbedding(nn.Module):
    def __init__(self, image_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, embed_dim, H/P, W/P] -> [B, N, embed_dim]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisualTransformerLayer(nn.Module):
    def __init__(self, d_model: int = 768, nhead: int = 12, dim_feedforward: int = 3072, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.ln_1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x


class RSCLIPVisualEncoder(nn.Module):
    """
    Multimodal Visual Transformer Encoder branch of RS-CLIP.
    Extracts 512-d normalized visual embeddings from satellite image crops.
    """
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        visual_width: int = 768,
        visual_layers: int = 6,
        visual_heads: int = 12,
        embed_dim: int = 512
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_embed = VisualPatchEmbedding(image_size, patch_size, in_channels, visual_width)
        num_patches = self.patch_embed.num_patches

        self.class_embedding = nn.Parameter(torch.empty(1, 1, visual_width))
        self.positional_embedding = nn.Parameter(torch.empty(1, num_patches + 1, visual_width))
        nn.init.normal_(self.class_embedding, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.02)

        self.ln_pre = nn.LayerNorm(visual_width)
        self.layers = nn.ModuleList([
            VisualTransformerLayer(
                d_model=visual_width,
                nhead=visual_heads,
                dim_feedforward=visual_width * 4
            )
            for _ in range(visual_layers)
        ])
        self.ln_post = nn.LayerNorm(visual_width)
        self.visual_projection = nn.Parameter(torch.empty(visual_width, embed_dim))
        nn.init.normal_(self.visual_projection, std=visual_width ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, 3, 224, 224] RGB satellite image crop tensors
        Returns:
            [N, embed_dim] L2-normalized visual feature embeddings
        """
        b = x.shape[0]
        patches = self.patch_embed(x)  # [B, num_patches, visual_width]

        # Prepend [CLS] token
        cls_tokens = self.class_embedding.expand(b, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)  # [B, num_patches + 1, visual_width]
        x = x + self.positional_embedding[:, :x.shape[1], :]
        x = self.ln_pre(x)

        for layer in self.layers:
            x = layer(x)

        # Extract CLS token feature
        cls_feature = self.ln_post(x[:, 0, :])

        # Project to 512-d space
        embedding = cls_feature @ self.visual_projection

        # Unit L2 Normalization
        embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding
