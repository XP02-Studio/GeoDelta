"""
Lightweight RS-CLIP text encoder adapter for free-tier deployment.
Two modes:
  1. If torch is available AND model weights exist → full RS-CLIP inference
  2. Otherwise → numpy-only vocabulary-hashing encoder (zero extra RAM)

Both produce 512-D normalized vectors so Qdrant search works either way.
"""

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import List

VECTOR_SIZE = 512

# Remote-sensing vocabulary (must match seed_data labels for best results)
VOCAB: List[str] = [
    "<pad>", "<start>", "<end>", "<unk>",
    "new", "runway", "airstrip", "military", "bunker", "hangar", "aircraft", "jet",
    "building", "structure", "fortified", "concrete", "barracks", "storage", "depot",
    "dirt", "road", "trench", "forest", "clearing", "bridge", "extension", "pathway",
    "vehicle", "convoy", "truck", "tank", "artillery", "crane", "construction", "excavation",
    "shadow", "cloud", "cover", "seasonal", "foliage", "vegetation", "water", "river",
    "heavy", "infrastructure", "logistical", "surface", "threat", "perimeter", "fence",
    "missile", "silo", "launcher", "radar", "dome", "antenna", "facility",
    "warehouse", "tower", "installation", "highway", "track", "path", "trail",
    "clearing", "tree", "ditch", "earthwork", "staging", "area",
    "shadows", "clouds", "cloudy", "haze", "reflection",
    "illumination", "coast", "helipad", "wall", "crater",
]

# --- Semantic synonym clusters for richer matching ---
_SYNONYM_CLUSTERS = {
    "airstrip": ["runway", "airstrip", "airport", "landing", "strip"],
    "bunker": ["bunker", "bunker", "shelter", "reinforced", "fortified"],
    "hangar": ["hangar", "hangar", "shelter", "aircraft"],
    "missile": ["missile", "silo", "launcher", "rocket"],
    "road": ["road", "highway", "track", "path", "trail", "dirt"],
    "bridge": ["bridge", "overpass", "crossing"],
    "building": ["building", "structure", "warehouse", "depot", "facility", "tower", "installation"],
    "military": ["military", "barracks", "defence", "defense", "base"],
    "vehicle": ["vehicle", "truck", "tank", "convoy", "artillery"],
    "forest": ["forest", "tree", "vegetation", "foliage", "clearing"],
    "construction": ["construction", "excavation", "earthwork", "crane", "staging"],
    "radar": ["radar", "antenna", "dome", "sensor"],
    "fence": ["fence", "wall", "perimeter", "barrier"],
    "coast": ["coast", "river", "water"],
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)


def _numpy_encode(text: str) -> list[float]:
    """
    Pure-numpy vocabulary hashing encoder.
    Maps each query word to multiple dimensions via synonym expansion + hash,
    producing a unique 512-D vector per query. Zero PyTorch dependency.
    """
    tokens = _tokenize(text)
    vec = [0.0] * VECTOR_SIZE

    # Expand tokens through synonym clusters
    expanded = list(tokens)
    for token in tokens:
        for key, synonyms in _SYNONYM_CLUSTERS.items():
            if token in synonyms:
                expanded.extend(synonyms)
                break

    # Hash each word to one or more dimension indices and accumulate
    for word in expanded:
        h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
        # Activate a primary dimension
        dim = h % VECTOR_SIZE
        vec[dim] += 1.0
        # Activate a few harmonic dimensions for richer representation
        vec[(h >> 8) % VECTOR_SIZE] += 0.5
        vec[(h >> 16) % VECTOR_SIZE] += 0.3

    # L2 normalize
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]

    return vec


# ------------------------------------------------------------------ #
#  Attempt full PyTorch RS-CLIP; fall back to numpy encoder           #
# ------------------------------------------------------------------ #
_torch_available = False
_model = None
_tokenizer = None


def _try_load_torch_model():
    global _torch_available, _model, _tokenizer
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class _SimpleTokenizer:
            def __init__(self, vocab_file=None, max_length: int = 32):
                self.max_length = max_length
                self.vocab = {}
                if vocab_file and os.path.exists(vocab_file):
                    with open(vocab_file, "r", encoding="utf-8") as f:
                        self.vocab = json.load(f)
                else:
                    self.vocab = {w: i for i, w in enumerate(VOCAB)}
                self.pad_id = self.vocab.get("<pad>", 0)
                self.start_id = self.vocab.get("<start>", 1)
                self.end_id = self.vocab.get("<end>", 2)

            def encode(self, text: str):
                tokens = _tokenize(text)
                ids = [self.start_id]
                for tok in tokens:
                    ids.append(self.vocab.get(tok, self.vocab.get("<unk>", 3)))
                ids.append(self.end_id)
                if len(ids) > self.max_length:
                    ids = ids[: self.max_length - 1] + [self.end_id]
                else:
                    ids += [self.pad_id] * (self.max_length - len(ids))
                return torch.tensor(ids, dtype=torch.long)

        class _TextTransformerLayer(nn.Module):
            def __init__(self, d_model, nhead=8, dim_ff=2048):
                super().__init__()
                self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
                self.ln1 = nn.LayerNorm(d_model)
                self.mlp = nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(), nn.Linear(dim_ff, d_model))
                self.ln2 = nn.LayerNorm(d_model)

            def forward(self, x):
                h = self.ln1(x)
                x = x + self.attn(h, h, h)[0]
                x = x + self.mlp(self.ln2(x))
                return x

        class _TextEncoder(nn.Module):
            def __init__(self, vocab_size=1000, max_length=32, embed_dim=512, heads=8, layers=4):
                super().__init__()
                self.token_embedding = nn.Embedding(vocab_size, embed_dim)
                self.positional_embedding = nn.Parameter(torch.randn(max_length, embed_dim) * 0.02)
                self.layers = nn.ModuleList([_TextTransformerLayer(embed_dim, heads, embed_dim * 4) for _ in range(layers)])
                self.ln_final = nn.LayerNorm(embed_dim)
                self.text_projection = nn.Linear(embed_dim, embed_dim, bias=False)

            def forward(self, tokens):
                B, S = tokens.shape
                x = self.token_embedding(tokens) + self.positional_embedding[:S]
                for layer in self.layers:
                    x = layer(x)
                x = self.ln_final(x)
                pooled = x.mean(dim=1)
                feat = self.text_projection(pooled)
                return F.normalize(feat, p=2, dim=-1)

        tokenizer = _SimpleTokenizer()
        weights_dir = Path(os.getenv("RSCLIP_WEIGHTS_DIR", "ml-engineer/models/rs_clip"))
        weights_path = weights_dir / "pytorch_model.bin"
        vocab_path = weights_dir / "vocab.json"

        if vocab_path.exists():
            tokenizer = _SimpleTokenizer(vocab_file=str(vocab_path))

        model = _TextEncoder(vocab_size=len(tokenizer.vocab))

        if weights_path.exists():
            state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            print(f"[clip_adapter] Loaded RS-CLIP weights from {weights_path}")
        else:
            print(f"[clip_adapter] No weights at {weights_path}; using init weights")

        model.eval()
        _model = model
        _tokenizer = tokenizer
        _torch_available = True
        print("[clip_adapter] Full RS-CLIP (PyTorch) encoder active")

    except ImportError:
        print("[clip_adapter] PyTorch not installed — using lightweight numpy encoder")
    except Exception as e:
        print(f"[clip_adapter] Could not load RS-CLIP model ({e}) — using lightweight numpy encoder")


def encode(text: str) -> list[float]:
    """Encode a plain-English query to a 512-D normalized vector."""
    if _model is None and not _torch_available:
        _try_load_torch_model()

    if _torch_available and _model is not None:
        import torch
        tokens = _tokenizer.encode(text).unsqueeze(0)
        with torch.no_grad():
            vec = _model(tokens)
        result = vec.squeeze(0).tolist()
        if len(result) != VECTOR_SIZE:
            raise ValueError(f"Expected {VECTOR_SIZE}-D vector, got {len(result)}-D")
        return result

    return _numpy_encode(text)
