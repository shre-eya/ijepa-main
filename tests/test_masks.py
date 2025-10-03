import torch
import pytest

from masks.utils import apply_masks


def test_apply_masks_mismatched_shapes_handled():
    # Simulate x_pos_embed of shape [B, 64, D] and masks selecting K=9 indices
    B, N, D = 2, 64, 8
    x = torch.randn(B, N, D)

    # Create a mask with 9 positions; provide as [K] (1D)
    idx_1d = torch.tensor([0, 1, 2, 3, 8, 15, 16, 31, 63], dtype=torch.int32)

    # Apply with list containing the same indices for all batch elements
    out = apply_masks(x, [idx_1d])

    # Expect output to concatenate over mask groups along batch: here 1 group -> shape [B, K, D]
    assert out.shape == (B, idx_1d.numel(), D)

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
    assert out.shape == (B, 3, D)


