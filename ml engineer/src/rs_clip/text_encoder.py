"""
RS-CLIP Text Transformer Encoder.
Converts tokenized natural language queries into 512-dimensional normalized embeddings.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TextTransformerLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int = 8, dim_feedforward: int = 2048, dropout: float = 0.0):
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

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN Transformer block
        norm_x = self.ln_1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, attn_mask=attn_mask)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x


class RSCLIPTextEncoder(nn.Module):
    """
    Multimodal Text Encoder branch of RS-CLIP.
    Maps plain-English queries (e.g. 'new runway') to unit-normalized 512-d feature space.
    """
    def __init__(
        self,
        vocab_size: int = 49408,
        context_length: int = 77,
        transformer_width: int = 512,
        transformer_heads: int = 8,
        transformer_layers: int = 6,
        embed_dim: int = 512
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        nn.init.normal_(self.positional_embedding, std=0.01)

        self.layers = nn.ModuleList([
            TextTransformerLayer(
                d_model=transformer_width,
                nhead=transformer_heads,
                dim_feedforward=transformer_width * 4
            )
            for _ in range(transformer_layers)
        ])
        self.ln_final = nn.LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        nn.init.normal_(self.text_projection, std=transformer_width ** -0.5)

    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_tokens: [B, context_length] token indices
        Returns:
            [B, embed_dim] L2-normalized text embedding vector
        """
        b, seq_len = text_tokens.shape
        x = self.token_embedding(text_tokens) + self.positional_embedding[:seq_len]

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)

        # Pool feature from the EOT (highest token index) or argmax
        eot_indices = text_tokens.argmax(dim=-1)
        pooled = x[torch.arange(b), eot_indices]

        # Project to 512-d multimodal space
        embedding = pooled @ self.text_projection

        # Unit L2 Normalization
        embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding
