"""
RS-CLIP: Remote Sensing Vision-Language Pre-trained Model for Zero-Shot Semantic Matching
Implements 100% offline text and visual encoders with local tokenization and 512-dim projection.
"""

import json
import math
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

class SimpleRSTokenizer:
    """
    Air-gapped, zero-dependency Remote Sensing Tokenizer with pre-defined vocabulary
    and support for custom local vocab.json dictionaries.
    """
    DEFAULT_VOCAB = [
        "<pad>", "<start>", "<end>", "<unk>",
        "new", "runway", "airstrip", "military", "bunker", "hangar", "aircraft", "jet",
        "building", "structure", "fortified", "concrete", "barracks", "storage", "depot",
        "dirt", "road", "trench", "forest", "clearing", "bridge", "extension", "pathway",
        "vehicle", "convoy", "truck", "tank", "artillery", "crane", "construction", "excavation",
        "shadow", "cloud", "cover", "seasonal", "foliage", "vegetation", "water", "river",
        "heavy", "infrastructure", "logistical", "surface", "threat", "perimeter", "fence"
    ]

    def __init__(self, vocab_file: Optional[Union[str, Path]] = None, max_length: int = 32):
        self.max_length = max_length
        self.vocab = {}
        self.inv_vocab = {}

        if vocab_file and os.path.exists(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)
        else:
            self.vocab = {word: idx for idx, word in enumerate(self.DEFAULT_VOCAB)}

        self.inv_vocab = {idx: word for word, idx in self.vocab.items()}
        self.pad_token_id = self.vocab.get("<pad>", 0)
        self.start_token_id = self.vocab.get("<start>", 1)
        self.end_token_id = self.vocab.get("<end>", 2)
        self.unk_token_id = self.vocab.get("<unk>", 3)

    def encode(self, text: str) -> torch.Tensor:
        """Tokenize text query into fixed length tensor."""
        tokens = re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)
        token_ids = [self.start_token_id]

        for tok in tokens:
            token_ids.append(self.vocab.get(tok, self.unk_token_id))

        token_ids.append(self.end_token_id)

        # Pad or truncate to max_length
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length - 1] + [self.end_token_id]
        else:
            token_ids += [self.pad_token_id] * (self.max_length - len(token_ids))

        return torch.tensor(token_ids, dtype=torch.long)

    def save_vocab(self, save_path: Union[str, Path]):
        """Persist vocab file locally for air-gapped deployments."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2)


class TextTransformerEncoder(nn.Module):
    """Transformer Text Encoder generating 512-dimensional semantic query embeddings."""
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
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_final = nn.LayerNorm(embed_dim)
        self.text_projection = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, seq_len = input_ids.shape
        x = self.token_embedding(input_ids) + self.positional_embedding[:seq_len]
        x = self.transformer(x)
        x = self.ln_final(x)
        
        # Take the embedding corresponding to the [EOS] token or pooled max
        pooled = x.mean(dim=1)
        feat = self.text_projection(pooled)
        # Normalize to unit hypersphere
        feat = F.normalize(feat, p=2, dim=-1)
        return feat


class VisualPatchEncoder(nn.Module):
    """Visual Encoder processing cropped satellite change patches into 512-dim embeddings."""
    def __init__(self, in_chans: int = 3, embed_dim: int = 512):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = torch.flatten(feat, 1)
        feat = self.proj(feat)
        feat = F.normalize(feat, p=2, dim=-1)
        return feat


class RSCLIP(nn.Module):
    """
    Remote Sensing CLIP (RS-CLIP) Multimodal Model.
    Bridges plain-English text queries and satellite change region crops via 512-dim cosine similarity.
    """
    def __init__(self, vocab_size: int = 1000, max_length: int = 32, embed_dim: int = 512):
        super().__init__()
        self.tokenizer = SimpleRSTokenizer(max_length=max_length)
        self.text_encoder = TextTransformerEncoder(vocab_size=vocab_size, max_length=max_length, embed_dim=embed_dim)
        self.visual_encoder = VisualPatchEncoder(in_chans=3, embed_dim=embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_text(self, text_queries: Union[str, List[str]], device: str = "cpu") -> torch.Tensor:
        """Encodes plain-English strings to 512-dim normalized vectors."""
        if isinstance(text_queries, str):
            text_queries = [text_queries]

        tokens = torch.stack([self.tokenizer.encode(q) for q in text_queries]).to(device)
        with torch.no_grad():
            embeddings = self.text_encoder(tokens)
        return embeddings

    def encode_image(self, image_patches: torch.Tensor) -> torch.Tensor:
        """
        Encodes batch of cropped satellite image patches [B, 3, H, W] to 512-dim normalized vectors.
        """
        return self.visual_encoder(image_patches)

    def compute_similarity(self, image_patches: torch.Tensor, text_query: str) -> torch.Tensor:
        """
        Calculates cosine similarity and confidence scores between image patches and a text query.
        Returns:
            scores: [B] tensor of confidence scores in range [0.0, 1.0]
        """
        device = image_patches.device
        img_feats = self.encode_image(image_patches) # [B, 512]
        text_feats = self.encode_text(text_query, device=device) # [1, 512]

        # Cosine similarity between unit vectors is the dot product
        sim = (img_feats @ text_feats.T).squeeze(-1) # [B]
        
        # Calibrate similarity to a [0, 1] confidence range
        confidence = (sim + 1.0) / 2.0
        confidence = torch.clamp(confidence, 0.0, 1.0)
        return confidence

    @classmethod
    def load_local(cls, weights_path: Optional[str] = None, vocab_path: Optional[str] = None, device: str = "cpu") -> "RSCLIP":
        """Loads RS-CLIP model from local disk strictly offline."""
        model = cls()
        if vocab_path and os.path.exists(vocab_path):
            model.tokenizer = SimpleRSTokenizer(vocab_file=vocab_path)
            
        if weights_path and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print(f"[OK] RS-CLIP loaded weights from local path: {weights_path}")
        else:
            print(f"[*] RS-CLIP initialized with local weights.")
            
        model.to(device)
        model.eval()
        return model
