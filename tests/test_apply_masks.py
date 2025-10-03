import torch
import pytest

from masks.utils import apply_masks
from train import train_step


def test_train_step_cpu_no_cuda(monkeypatch):
    # Skip if CUDA is available; this test is for CPU-only path
    if torch.cuda.is_available():
        pytest.skip("CUDA available; CPU-only scaler guard not exercised")

    # Minimal stubs to satisfy train_step's global dependencies
    class DummyEncoder(torch.nn.Module):
        def forward(self, x, masks=None):
            B = x.size(0)
            return torch.zeros(B, 4, 8)
    class DummyPredictor(torch.nn.Module):
        def forward(self, z, context_masks, target_masks):
            B = z.size(0)
            return torch.zeros(B, 4, 8)

    # Inject globals expected by train_step
    import src.train as t
    t.encoder = DummyEncoder()
    t.target_encoder = DummyEncoder()
    t.predictor = DummyPredictor()
    t.device = torch.device('cpu')
    t.optimizer = torch.optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=0.1)
    t.momentum_scheduler = iter([0.99])
    t.use_bfloat16 = False
    t.scaler = None  # ensure CPU-only path

    images = torch.randn(2, 3, 16, 16)
    context_masks = torch.zeros(2, 4, dtype=torch.bool)
    target_masks = torch.zeros(2, 4, dtype=torch.bool)

    # Should not raise NameError or require scaler on CPU
    loss_val = t.train_step(images, context_masks, target_masks)
    assert isinstance(loss_val, float)


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

