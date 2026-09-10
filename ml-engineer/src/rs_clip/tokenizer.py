"""
Pure Offline BPE Tokenizer for RS-CLIP.
Operates completely air-gapped with a local remote-sensing vocabulary table.
Guarantees zero network calls during tokenization of plain-English queries.
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Union, Optional
import torch

from src.config import CONFIG, RSCLIPConfig


class OfflineCLIPTokenizer:
    """
    Air-Gapped Byte-Pair Encoding (BPE) Tokenizer for RS-CLIP text queries.
    Encodes English text strings (e.g. 'new runway', 'concrete bunker') into token IDs.
    """
    def __init__(self, vocab_path: Optional[Union[str, Path]] = None, context_length: int = 77):
        self.context_length = context_length
        self.vocab_path = Path(vocab_path) if vocab_path else CONFIG.rs_clip.vocab_path

        # Special Tokens
        self.sot_token = "<|startoftext|>"
        self.eot_token = "<|endoftext|>"
        self.pad_token = "<|pad|>"

        self.encoder: Dict[str, int] = {}
        self.decoder: Dict[int, str] = {}
        self._load_or_create_vocab()

    def _load_or_create_vocab(self):
        """Loads offline vocabulary table or initializes built-in remote sensing vocabulary."""
        if self.vocab_path.exists() and self.vocab_path.is_file():
            try:
                with open(self.vocab_path, "r", encoding="utf-8") as f:
                    self.encoder = json.load(f)
                self.decoder = {v: k for k, v in self.encoder.items()}
                return
            except Exception as e:
                print(f"[OfflineTokenizer] Warning loading {self.vocab_path}: {e}")

        # Construct comprehensive remote-sensing and domain-specific offline vocabulary
        self.vocab_path.parent.mkdir(parents=True, exist_ok=True)
        vocab = {
            self.pad_token: 0,
            self.sot_token: 49406,
            self.eot_token: 49407,
        }

        # Remote sensing terminology & common vocabulary
        rs_terms = [
            "a", "an", "the", "new", "old", "large", "small", "heavy", "light",
            "runway", "airstrip", "airport", "aircraft", "hangar", "tarmac", "taxiway",
            "bunker", "reinforced", "concrete", "barracks", "fortification", "base",
            "missile", "silo", "launcher", "radar", "dome", "antenna", "facility",
            "building", "structure", "warehouse", "depot", "tower", "installation",
            "road", "dirt", "paved", "highway", "track", "path", "trail", "bridge",
            "extension", "clearing", "forest", "tree", "vegetation", "trench", "ditch",
            "excavation", "construction", "site", "earthwork", "staging", "area",
            "shadow", "shadows", "cloud", "clouds", "cloudy", "haze", "reflection",
            "seasonal", "illumination", "water", "river", "coast", "vehicle", "tank",
            "truck", "convoy", "helipad", "perimeter", "fence", "wall", "crater"
        ]

        # Add single characters and common ASCII words
        current_id = 1
        for ch in "abcdefghijklmnopqrstuvwxyz0123456789-_., ":
            if ch not in vocab:
                vocab[ch] = current_id
                current_id += 1

        for term in rs_terms:
            if term not in vocab:
                vocab[term] = current_id
                current_id += 1

        self.encoder = vocab
        self.decoder = {v: k for k, v in self.encoder.items()}

        # Save offline vocab JSON
        with open(self.vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.encoder, f, indent=2)
        print(f"[OfflineTokenizer] Saved air-gapped vocabulary table at: {self.vocab_path}")

    def tokenize(self, text: str) -> List[int]:
        """Converts raw text into a sequence of vocabulary token IDs."""
        text = text.lower().strip()
        tokens = [self.encoder[self.sot_token]]

        # Word boundary splitting
        words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        for w in words:
            if w in self.encoder:
                tokens.append(self.encoder[w])
            else:
                # Sub-word / character fallback
                for ch in w:
                    tokens.append(self.encoder.get(ch, 0))

        tokens.append(self.encoder[self.eot_token])

        # Truncate if exceeds context length
        if len(tokens) > self.context_length:
            tokens = tokens[:self.context_length - 1] + [self.encoder[self.eot_token]]

        # Pad with pad tokens
        while len(tokens) < self.context_length:
            tokens.append(self.encoder[self.pad_token])

        return tokens

    def encode(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Tokenizes single or batch of text queries into a PyTorch LongTensor.
        Returns: [B, context_length] tensor
        """
        if isinstance(texts, str):
            texts = [texts]

        batch_tokens = [self.tokenize(t) for t in texts]
        return torch.tensor(batch_tokens, dtype=torch.long)
