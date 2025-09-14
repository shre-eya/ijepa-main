import torch
from src.models.vision_transformer import vit_small

def test_uncertainty_masking():
    print("Setting up test environment...")
    
    # Create a small ViT model
    model = vit_small(img_size=224, patch_size=16)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    print(f"\nInput shape: {x.shape}")
    
    print("\nRunning tests...")
    print("-" * 50)
    
    # Test 1: Basic forward pass without uncertainty
    print("\nTest 1: Basic forward pass")
    try:
        out = model(x)
        print(f"✓ Success - Output shape: {out.shape}")
        num_patches = out.shape[1]  # Number of patches (should be 196 for 224/16)
        print(f"✓ Number of patches: {num_patches}")
    except Exception as e:
        print(f"✗ Failed - Error: {str(e)}")
    
    print("-" * 50)
    
    # Test 2: Forward pass with uncertainty estimation
    print("\nTest 2: Uncertainty estimation")
    try:
        mean_out, patch_vars = model(x, return_patch_vars=True, num_mc_samples=3)
        print(f"✓ Success:")
        print(f"  - Mean output shape: {mean_out.shape}")
        print(f"  - Patch variances shape: {patch_vars.shape}")
        print(f"  - Average patch variance: {patch_vars.mean().item():.4f}")
    except Exception as e:
        print(f"✗ Failed - Error: {str(e)}")
    
    print("-" * 50)
    
    # Test 3: Forward pass with custom mask
    print("\nTest 3: Custom masking")
    try:
        num_patches = (224 // 16) ** 2  # 14x14 = 196 patches
        custom_mask = torch.ones(batch_size, num_patches, dtype=torch.bool)
        custom_mask[:, :num_patches//2] = False  # Mask first half of patches
        
        mean_out, patch_vars = model(x, masks=custom_mask, return_patch_vars=True, num_mc_samples=3)
        print(f"✓ Success:")
        print(f"  - Mean output shape: {mean_out.shape}")
        print(f"  - Patch variances shape: {patch_vars.shape}")
        print("\nVariance analysis:")
        print(f"  - Masked patches (first half): {patch_vars[:, :num_patches//2].mean().item():.4f}")
        print(f"  - Unmasked patches (second half): {patch_vars[:, num_patches//2:].mean().item():.4f}")
    except Exception as e:
        print(f"✗ Failed - Error: {str(e)}")

if __name__ == "__main__":
    test_uncertainty_masking() 