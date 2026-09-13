"""
Self-contained RS-CLIP text encoder adapter for the backend.
Loads the RS-CLIP text model weights and encodes plain-English queries
into 512-D normalized vectors for Qdrant semantic search.
"""

import json
import os
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

VECTOR_SIZE = 512


class _SimpleTokenizer:
    """Minimal air-gapped tokenizer matching ml-engineer/src/models/rs_clip.py."""

    DEFAULT_VOCAB = [
        "<pad>", "<start>", "<end>", "<unk>",
        "new", "runway", "airstrip", "military", "bunker", "hangar", "aircraft", "jet",
        "building", "structure", "fortified", "concrete", "barracks", "storage", "depot",
        "dirt", "road", "trench", "forest", "clearing", "bridge", "extension", "pathway",
        "vehicle", "convoy", "truck", "tank", "artillery", "crane", "construction", "excavation",
        "shadow", "cloud", "cover", "seasonal", "foliage", "vegetation", "water", "river",
        "heavy", "infrastructure", "logistical", "surface", "threat", "perimeter", "fence",
    ]

    def __init__(self, vocab_file=None, max_length: int = 32):
        self.max_length = max_length
        self.vocab = {}
        if vocab_file and os.path.exists(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)
        else:
            self.vocab = {word: idx for idx, word in enumerate(self.DEFAULT_VOCAB)}
        self.pad_id = self.vocab.get("<pad>", 0)
        self.start_id = self.vocab.get("<start>", 1)
        self.end_id = self.vocab.get("<end>", 2)
        self.unk_id = self.vocab.get("<unk>", 3)

    def encode(self, text: str) -> torch.Tensor:
        tokens = re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)
        ids = [self.start_id]
        for tok in tokens:
            ids.append(self.vocab.get(tok, self.unk_id))
        ids.append(self.end_id)
        if len(ids) > self.max_length:
            ids = ids[: self.max_length - 1] + [self.end_id]
        else:
            ids += [self.pad_id] * (self.max_length - len(ids))
        return torch.tensor(ids, dtype=torch.long)


class TextTransformerEncoder(nn.Module):
    """
    Exact replica of ml-engineer/src/models/rs_clip.py TextTransformerEncoder.
    Parameter names MUST match for load_state_dict to work.
    """
    def __init__(self, vocab_size: int = 1000, max_length: int = 32, embed_dim: int = 512,
                 num_heads: int = 8, num_layers: int = 4):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.randn(max_length, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_final = nn.LayerNorm(embed_dim)
        self.text_projection = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, seq_len = input_ids.shape
        x = self.token_embedding(input_ids) + self.positional_embedding[:seq_len]
        x = self.transformer(x)
        x = self.ln_final(x)
        pooled = x.mean(dim=1)
        feat = self.text_projection(pooled)
        feat = F.normalize(feat, p=2, dim=-1)
        return feat


_model: TextTransformerEncoder | None = None
_tokenizer: _SimpleTokenizer | None = None


def _get_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    _tokenizer = _SimpleTokenizer()

    weights_dir = Path(os.getenv("RSCLIP_WEIGHTS_DIR", "ml-engineer/models/rs_clip"))
    weights_path = weights_dir / "pytorch_model.bin"
    vocab_path = weights_dir / "vocab.json"

    if vocab_path.exists():
        _tokenizer = _SimpleTokenizer(vocab_file=str(vocab_path))
        print(f"[clip_adapter] Loaded vocab from {vocab_path}")

    _model = TextTransformerEncoder(vocab_size=len(_tokenizer.vocab))

    if weights_path.exists():
        state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        _model.load_state_dict(state_dict, strict=False)
        print(f"[clip_adapter] Loaded RS-CLIP text weights from {weights_path}")
    else:
        print(f"[clip_adapter] No weights at {weights_path}; using init weights")

    _model.eval()
    return _model, _tokenizer


def encode(text: str) -> list[float]:
    """Encode a plain-English query to a 512-D normalized vector."""
    model, tokenizer = _get_model()
    tokens = tokenizer.encode(text).unsqueeze(0)
    with torch.no_grad():
        vec = model(tokens)
    result = vec.squeeze(0).tolist()
    if len(result) != VECTOR_SIZE:
        raise ValueError(f"Expected {VECTOR_SIZE}-D vector, got {len(result)}-D")
    return result
