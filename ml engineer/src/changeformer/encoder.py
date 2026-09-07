"""
Hierarchical Transformer Encoder for ChangeFormer Siamese Network.
Extracts multi-scale spatial features across multiple resolution stages.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class OverlapPatchEmbed(nn.Module):
    """
    Overlapping Patch Embedding module to preserve local spatial continuity.
    """
    def __init__(self, patch_size: int = 7, stride: int = 4, in_channels: int = 3, embed_dim: int = 64):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, h, w = x.shape
        # [B, C, H, W] -> [B, H*W, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w


class EfficientAttention(nn.Module):
    """
    Efficient Multi-Head Self-Attention with spatial reduction for low computational complexity.
    """
    def __init__(self, dim: int, num_heads: int = 8, sr_ratio: int = 1, drop_rate: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop_rate)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr is not None:
            # Spatial reduction: [B, N, C] -> [B, C, H, W] -> SR -> [B, N_sr, C]
            x_reshaped = x.permute(0, 2, 1).reshape(b, c, h, w)
            x_sr = self.sr(x_reshaped).reshape(b, c, -1).permute(0, 2, 1)
            x_sr = self.norm(x_sr)
            kv = self.kv(x_sr).reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        out = self.drop(out)
        return out


class MixFFN(nn.Module):
    """
    Mix-FeedForward Network with depth-wise convolution for positional encoding.
    """
    def __init__(self, in_features: int, hidden_features: int = None, drop_rate: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        x = self.fc1(x)
        # Depthwise 3x3 conv
        x = x.transpose(1, 2).view(b, -1, h, w)
        x = self.dwconv(x)
        x = self.act(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4, sr_ratio: int = 1, drop_rate: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads=num_heads, sr_ratio=sr_ratio, drop_rate=drop_rate)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, hidden_features=dim * mlp_ratio, drop_rate=drop_rate)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.mlp(self.norm2(x), h, w)
        return x


class HierarchicalTransformerEncoder(nn.Module):
    """
    4-Stage Hierarchical Transformer Encoder for Multi-Scale Feature Extraction.
    Outputs feature maps at 1/4, 1/8, 1/16, and 1/32 of the input spatial resolution.
    """
    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: List[int] = [64, 128, 256, 512],
        num_heads: List[int] = [1, 2, 4, 8],
        mlp_ratios: List[int] = [4, 4, 4, 4],
        depths: List[int] = [2, 2, 2, 2],
        sr_ratios: List[int] = [8, 4, 2, 1],
        drop_rate: float = 0.1
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.depths = depths

        # Patch embedding stages
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_channels=in_channels, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_channels=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_channels=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_channels=embed_dims[2], embed_dim=embed_dims[3])

        # Transformer blocks for each stage
        self.block1 = nn.ModuleList([
            TransformerBlock(embed_dims[0], num_heads[0], mlp_ratios[0], sr_ratios[0], drop_rate)
            for _ in range(depths[0])
        ])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        self.block2 = nn.ModuleList([
            TransformerBlock(embed_dims[1], num_heads[1], mlp_ratios[1], sr_ratios[1], drop_rate)
            for _ in range(depths[1])
        ])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        self.block3 = nn.ModuleList([
            TransformerBlock(embed_dims[2], num_heads[2], mlp_ratios[2], sr_ratios[2], drop_rate)
            for _ in range(depths[2])
        ])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        self.block4 = nn.ModuleList([
            TransformerBlock(embed_dims[3], num_heads[3], mlp_ratios[3], sr_ratios[3], drop_rate)
            for _ in range(depths[3])
        ])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        b = x.shape[0]

        # Stage 1 (1/4 resolution)
        x1, h1, w1 = self.patch_embed1(x)
        for blk in self.block1:
            x1 = blk(x1, h1, w1)
        x1 = self.norm1(x1)
        f1 = x1.reshape(b, h1, w1, self.embed_dims[0]).permute(0, 3, 1, 2).contiguous()
        features.append(f1)

        # Stage 2 (1/8 resolution)
        x2, h2, w2 = self.patch_embed2(f1)
        for blk in self.block2:
            x2 = blk(x2, h2, w2)
        x2 = self.norm2(x2)
        f2 = x2.reshape(b, h2, w2, self.embed_dims[1]).permute(0, 3, 1, 2).contiguous()
        features.append(f2)

        # Stage 3 (1/16 resolution)
        x3, h3, w3 = self.patch_embed3(f2)
        for blk in self.block3:
            x3 = blk(x3, h3, w3)
        x3 = self.norm3(x3)
        f3 = x3.reshape(b, h3, w3, self.embed_dims[2]).permute(0, 3, 1, 2).contiguous()
        features.append(f3)

        # Stage 4 (1/32 resolution)
        x4, h4, w4 = self.patch_embed4(f3)
        for blk in self.block4:
            x4 = blk(x4, h4, w4)
        x4 = self.norm4(x4)
        f4 = x4.reshape(b, h4, w4, self.embed_dims[3]).permute(0, 3, 1, 2).contiguous()
        features.append(f4)

        return features
