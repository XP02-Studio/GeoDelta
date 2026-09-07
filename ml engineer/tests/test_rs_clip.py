"""
Unit Tests for RS-CLIP Multimodal Semantic Engine
"""

import pytest
import torch
from src.models.rs_clip import RSCLIP, SimpleRSTokenizer

def test_tokenizer_encoding():
    tokenizer = SimpleRSTokenizer(max_length=16)
    query = "new runway military bunker"
    tokens = tokenizer.encode(query)

    assert tokens.shape == (16,), "Tokenizer output length mismatch"
    assert tokens[0].item() == tokenizer.start_token_id
    assert tokenizer.vocab["runway"] in tokens.tolist()
    assert tokenizer.vocab["bunker"] in tokens.tolist()

def test_text_encoder_embeddings():
    model = RSCLIP(embed_dim=512)
    model.eval()

    embeddings = model.encode_text(["new runway", "dirt road"])
    assert embeddings.shape == (2, 512), f"Text embedding shape mismatch: {embeddings.shape}"

    # Verify unit hypersphere normalization (L2 norm = 1.0)
    norms = torch.norm(embeddings, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3), "Embeddings must be L2 normalized"

def test_visual_encoder_embeddings():
    model = RSCLIP(embed_dim=512)
    model.eval()

    patches = torch.randn(4, 3, 64, 64)
    with torch.no_grad():
        visual_feats = model.encode_image(patches)

    assert visual_feats.shape == (4, 512), f"Visual embedding shape mismatch: {visual_feats.shape}"
    norms = torch.norm(visual_feats, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3), "Visual embeddings must be L2 normalized"

def test_cosine_similarity_confidence():
    model = RSCLIP(embed_dim=512)
    model.eval()

    patches = torch.randn(3, 3, 64, 64)
    confidences = model.compute_similarity(patches, "military bunker")

    assert confidences.shape == (3,), f"Confidence output shape mismatch: {confidences.shape}"
    assert (confidences >= 0.0).all() and (confidences <= 1.0).all(), "Confidences must be in range [0, 1]"
