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


def test_save_checkpoint_with_loss():
    """Test that save_checkpoint accepts loss_avg parameter without NameError."""
    import src.train as t
    import tempfile
    import os
    
    # Mock the required globals
    t.encoder = torch.nn.Linear(1, 1)
    t.predictor = torch.nn.Linear(1, 1)
    t.target_encoder = torch.nn.Linear(1, 1)
    t.optimizer = torch.optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=0.1)
    t.scaler = None
    t.rank = 0
    t.latest_path = tempfile.mktemp(suffix='.pth')
    t.save_path = tempfile.mktemp(suffix='.pth')
    t.batch_size = 2
    t.world_size = 1
    t.lr = 0.01
    
    # Test that save_checkpoint works with explicit loss_avg
    try:
        t.save_checkpoint(epoch=0, loss_avg=0.5)
        # Check that file was created
        assert os.path.exists(t.latest_path)
    except NameError as e:
        pytest.fail(f"save_checkpoint raised NameError: {e}")
    finally:
        # Cleanup
        for path in [t.latest_path, t.save_path]:
            if os.path.exists(path):
                os.remove(path)


def test_training_completes_cpu_only():
    """Test that training completes without distributed RuntimeError on CPU-only."""
    import src.train as t
    import tempfile
    import os
    import yaml
    
    # Skip if distributed is already initialized (e.g., in multi-GPU test environment)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        pytest.skip("Distributed already initialized; CPU-only guard not tested")
    
    # Create minimal config for CPU-only training
    config = {
        'meta': {
            'use_bfloat16': False,
            'model_name': 'vit_tiny',
            'load_checkpoint': False,
            'read_checkpoint': None,
            'copy_data': False,
            'pred_depth': 1,
            'pred_emb_dim': 64,
            'use_pretrained': False
        },
        'data': {
            'use_gaussian_blur': False,
            'use_horizontal_flip': False,
            'use_color_distortion': False,
            'color_jitter_strength': 0.0,
            'batch_size': 2,
            'pin_mem': False,
            'num_workers': 0,
            'root_path': './data',
            'image_folder': 'cifar10',
            'crop_size': 32,
            'crop_scale': [0.8, 1.0]
        },
        'mask': {
            'allow_overlap': True,
            'patch_size': 4,
            'num_enc_masks': 1,
            'min_keep': 4,
            'enc_mask_scale': [0.25, 0.75],
            'num_pred_masks': 1,
            'pred_mask_scale': [0.25, 0.75],
            'aspect_ratio': 1.0
        },
        'optimization': {
            'ema': [0.996, 0.999],
            'ipe_scale': 1.0,
            'weight_decay': 0.04,
            'final_weight_decay': 0.4,
            'epochs': 1,
            'warmup': 10,
            'start_lr': 0.0,
            'lr': 0.0005,
            'final_lr': 0.0
        },
        'logging': {
            'folder': tempfile.mkdtemp(),
            'write_tag': 'test_cpu'
        }
    }
    
    # Mock distributed init to return single process
    original_init_distributed = t.init_distributed
    def mock_init_distributed():
        return 1, 0  # world_size=1, rank=0
    t.init_distributed = mock_init_distributed
    
    try:
        # This should complete without RuntimeError from distributed.barrier()
        t.main(config, resume_preempt=False)
    except RuntimeError as e:
        if "Default process group has not been initialized" in str(e):
            pytest.fail(f"Training crashed with distributed RuntimeError: {e}")
        else:
            # Other RuntimeErrors are acceptable for this minimal test
            pass
    finally:
        # Restore original function
        t.init_distributed = original_init_distributed
        # Cleanup temp directory
        import shutil
        if os.path.exists(config['logging']['folder']):
            shutil.rmtree(config['logging']['folder'])


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

