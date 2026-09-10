"""
Unit Tests for Siamese ChangeFormer Model
"""

import os
import pytest
import torch
from src.models.changeformer import ChangeFormer, HierarchicalEncoder, MLPDecoder

def test_hierarchical_encoder_shapes():
    encoder = HierarchicalEncoder(in_chans=3, embed_dims=[64, 128, 256, 512])
    dummy_img = torch.randn(2, 3, 256, 256)
    feats = encoder(dummy_img)

    assert len(feats) == 4, "Encoder must produce 4 hierarchical multi-scale feature maps"
    assert feats[0].shape == (2, 64, 64, 64), f"Stage 1 shape mismatch: {feats[0].shape}"
    assert feats[1].shape == (2, 128, 32, 32), f"Stage 2 shape mismatch: {feats[1].shape}"
    assert feats[2].shape == (2, 256, 16, 16), f"Stage 3 shape mismatch: {feats[2].shape}"
    assert feats[3].shape == (2, 512, 8, 8), f"Stage 4 shape mismatch: {feats[3].shape}"

def test_mlp_decoder_fusion():
    decoder = MLPDecoder(embed_dims=[64, 128, 256, 512], decoder_dim=128, num_classes=2)
    diff_feats = [
        torch.randn(2, 64, 64, 64),
        torch.randn(2, 128, 32, 32),
        torch.randn(2, 256, 16, 16),
        torch.randn(2, 512, 8, 8)
    ]
    out = decoder(diff_feats, orig_size=(256, 256))
    assert out.shape == (2, 2, 256, 256), f"Decoder output shape mismatch: {out.shape}"

def test_changeformer_forward_pass():
    model = ChangeFormer()
    model.eval()

    t1 = torch.randn(1, 3, 256, 256)
    t2 = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        logits = model(t1, t2)

    assert logits.shape == (1, 2, 256, 256), f"ChangeFormer forward shape mismatch: {logits.shape}"
    assert not torch.isnan(logits).any(), "Forward pass produced NaN values"

def test_local_offline_load(tmp_path):
    model = ChangeFormer()
    weights_path = tmp_path / "test_weights.pth"
    torch.save(model.state_dict(), str(weights_path))

    loaded_model = ChangeFormer.load_local(str(weights_path))
    assert loaded_model is not None
