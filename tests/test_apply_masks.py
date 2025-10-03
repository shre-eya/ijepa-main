import torch
import pytest

from masks.utils import apply_masks


def test_mask_keep_dtype_and_gather_no_dtype_error():
    # Create a small input tensor: B=2, N=4 patches, D=3 features
    x = torch.randn(2, 4, 3)

    # Create masks with int32 dtype to simulate upstream indices not being int64
    # Two masks for two groups to be concatenated by apply_masks
    m1 = torch.tensor([[0, 2]], dtype=torch.int32)  # shape [1, 2]
    m2 = torch.tensor([[1, 3]], dtype=torch.int32)  # shape [1, 2]
    masks = [m1, m2]

    # Verify the internal index tensor casting logic we rely on
    # (mirror the operation from apply_masks)
    mask_keep_example = m1.to(torch.int64).unsqueeze(-1).repeat(1, 1, x.size(-1))
    assert mask_keep_example.dtype == torch.int64

    # Ensure calling apply_masks does not raise dtype errors from torch.gather
    try:
        out = apply_masks(x, masks)
    except RuntimeError as e:
        pytest.fail(f"apply_masks raised RuntimeError unexpectedly: {e}")

    # Basic sanity check on output shape: concatenates along batch dim
    # Each mask keeps 2 patches, so result should have B=1+1=2 groups, N=2, D=3
    assert out.shape[1:] == (2, 3)

