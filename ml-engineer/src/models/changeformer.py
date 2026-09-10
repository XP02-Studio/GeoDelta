"""
ChangeFormer: Transformer-Based Siamese Network for Remote Sensing Change Detection
Implements hierarchical multi-scale Transformer encoders with spatial reduction attention (SRA)
and an MLP multi-scale difference fusion decoder.
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class SpatialReductionAttention(nn.Module):
    """Efficient Multi-Head Attention with Spatial Reduction (SRA) to maintain <420ms latency."""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, sr_ratio: int = 1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.sr_ratio = sr_ratio

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr is not None:
            x_spatial = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_reduced = self.sr(x_spatial).reshape(B, C, -1).permute(0, 2, 1)
            x_reduced = self.norm(x_reduced)
            kv = self.kv(x_reduced).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out


class MLP(nn.Module):
    """Feedforward Multi-Layer Perceptron with GELU activation."""
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Hierarchical Transformer Stage Block."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True, sr_ratio: int = 1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialReductionAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding for spatial reduction."""
    def __init__(self, img_size: int = 256, patch_size: int = 7, stride: int = 4, in_chans: int = 3, embed_dim: int = 64):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class HierarchicalEncoder(nn.Module):
    """4-stage Hierarchical Vision Transformer Encoder for ChangeFormer."""
    def __init__(self, in_chans: int = 3, embed_dims: List[int] = [64, 128, 256, 512],
                 num_heads: List[int] = [1, 2, 4, 8], mlp_ratios: List[int] = [4, 4, 4, 4],
                 sr_ratios: List[int] = [8, 4, 2, 1], depths: List[int] = [2, 2, 2, 2]):
        super().__init__()
        self.embed_dims = embed_dims

        # Stage 1 (1/4 scale)
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.block1 = nn.ModuleList([
            TransformerBlock(embed_dims[0], num_heads[0], mlp_ratios[0], sr_ratio=sr_ratios[0])
            for _ in range(depths[0])
        ])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        # Stage 2 (1/8 scale)
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.block2 = nn.ModuleList([
            TransformerBlock(embed_dims[1], num_heads[1], mlp_ratios[1], sr_ratio=sr_ratios[1])
            for _ in range(depths[1])
        ])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        # Stage 3 (1/16 scale)
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.block3 = nn.ModuleList([
            TransformerBlock(embed_dims[2], num_heads[2], mlp_ratios[2], sr_ratio=sr_ratios[2])
            for _ in range(depths[2])
        ])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        # Stage 4 (1/32 scale)
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        self.block4 = nn.ModuleList([
            TransformerBlock(embed_dims[3], num_heads[3], mlp_ratios[3], sr_ratio=sr_ratios[3])
            for _ in range(depths[3])
        ])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        B = x.shape[0]
        outs = []

        # Stage 1
        x, H1, W1 = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H1, W1)
        x1 = self.norm1(x).reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        # Stage 2
        x, H2, W2 = self.patch_embed2(x1)
        for blk in self.block2:
            x = blk(x, H2, W2)
        x2 = self.norm2(x).reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x2)

        # Stage 3
        x, H3, W3 = self.patch_embed3(x2)
        for blk in self.block3:
            x = blk(x, H3, W3)
        x3 = self.norm3(x).reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x3)

        # Stage 4
        x, H4, W4 = self.patch_embed4(x3)
        for blk in self.block4:
            x = blk(x, H4, W4)
        x4 = self.norm4(x).reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x4)

        return outs


class MLPDecoder(nn.Module):
    """Multi-Scale MLP Feature Fusion Decoder for Change Detection."""
    def __init__(self, embed_dims: List[int] = [64, 128, 256, 512], decoder_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.linear_c4 = nn.Sequential(nn.Conv2d(embed_dims[3], decoder_dim, 1), nn.BatchNorm2d(decoder_dim), nn.ReLU(inplace=True))
        self.linear_c3 = nn.Sequential(nn.Conv2d(embed_dims[2], decoder_dim, 1), nn.BatchNorm2d(decoder_dim), nn.ReLU(inplace=True))
        self.linear_c2 = nn.Sequential(nn.Conv2d(embed_dims[1], decoder_dim, 1), nn.BatchNorm2d(decoder_dim), nn.ReLU(inplace=True))
        self.linear_c1 = nn.Sequential(nn.Conv2d(embed_dims[0], decoder_dim, 1), nn.BatchNorm2d(decoder_dim), nn.ReLU(inplace=True))

        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_dim, decoder_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim // 2),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Conv2d(decoder_dim // 2, num_classes, kernel_size=1)

    def forward(self, diff_features: List[torch.Tensor], orig_size: Tuple[int, int]) -> torch.Tensor:
        d1, d2, d3, d4 = diff_features
        _c4 = self.linear_c4(d4)
        _c4 = F.interpolate(_c4, size=d1.shape[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(d3)
        _c3 = F.interpolate(_c3, size=d1.shape[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(d2)
        _c2 = F.interpolate(_c2, size=d1.shape[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(d1)

        fused = torch.cat([_c4, _c3, _c2, _c1], dim=1)
        feat = self.fuse(fused)
        out = self.classifier(feat)
        out = F.interpolate(out, size=orig_size, mode='bilinear', align_corners=False)
        return out


class ChangeFormer(nn.Module):
    """
    Siamese ChangeFormer model for remote sensing bitemporal change detection.
    Extracts multi-scale features for T1 and T2 through twin Siamese encoders,
    computes absolute multi-scale difference maps, and decodes into change logits.
    """
    def __init__(self, in_chans: int = 3, embed_dims: List[int] = [64, 128, 256, 512],
                 decoder_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.encoder = HierarchicalEncoder(in_chans=in_chans, embed_dims=embed_dims)
        self.decoder = MLPDecoder(embed_dims=embed_dims, decoder_dim=decoder_dim, num_classes=num_classes)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for bitemporal satellite tile pair.
        Args:
            t1: [B, 3, H, W] - Time 1 imagery
            t2: [B, 3, H, W] - Time 2 imagery
        Returns:
            logits: [B, 2, H, W] - Semantic change logits (channel 0: no-change, channel 1: change)
        """
        orig_size = (t1.shape[2], t1.shape[3])

        # Siamese feature extraction
        feats_t1 = self.encoder(t1)
        feats_t2 = self.encoder(t2)

        # Multi-scale absolute difference computation
        diff_features = [torch.abs(f1 - f2) for f1, f2 in zip(feats_t1, feats_t2)]

        # Multi-scale MLP decoding
        logits = self.decoder(diff_features, orig_size)
        return logits

    @classmethod
    def load_local(cls, weights_path: Optional[str] = None, device: str = "cpu") -> "ChangeFormer":
        """
        Loads ChangeFormer model strictly from local disk for 100% offline air-gapped readiness.
        """
        model = cls()
        if weights_path and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print(f"[OK] ChangeFormer loaded weights from local path: {weights_path}")
        else:
            print(f"[*] ChangeFormer initialized with default architecture (ready for local training/eval).")
        model.to(device)
        model.eval()
        return model
