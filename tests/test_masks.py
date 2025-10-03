import torch
import pytest

from masks.utils import apply_masks
from models.vision_transformer import VisionTransformer


def test_apply_masks_mismatched_shapes_handled():
    # Simulate x_pos_embed of shape [B, 64, D] and masks selecting K=9 indices
    B, N, D = 2, 64, 8
    x = torch.randn(B, N, D)

    # Create a mask with 9 positions; provide as [K] (1D)
    idx_1d = torch.tensor([0, 1, 2, 3, 8, 15, 16, 31, 63], dtype=torch.int32)

    # Apply with list containing the same indices for all batch elements
    out = apply_masks(x, [idx_1d])

    # Now full-length is preserved; shape should be [B, N, D]
    assert out.shape == (B, N, D)

    # Ensure gather index dtype normalization is applied (no RuntimeError)
    # If the function fails, an exception would be raised before this point


def test_apply_masks_per_batch_indices():
    # Different indices per batch element provided as [B, K]
    B, N, D = 2, 16, 4
    x = torch.randn(B, N, D)
    masks = [
        torch.tensor([[0, 3, 5], [1, 4, 6]], dtype=torch.int64)
    ]

    out = apply_masks(x, masks)
    # Full-length should be preserved
    assert out.shape == (B, N, D)


def test_vit_positional_embedding_broadcast_batch_gt_one():
    # Ensure positional embeddings broadcast/expand to B>1 without shape mismatch
    B, C, H, W = 3, 3, 32, 32
    model = VisionTransformer(img_size=32, patch_size=16, embed_dim=64, depth=1, num_heads=2)
    x = torch.randn(B, C, H, W)

    # Forward without masks should not assert on positional embedding shapes
    out = model(x)
    assert out.shape[0] == B


