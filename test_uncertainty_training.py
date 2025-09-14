import sys
import os

# Ensure 'src' is in the Python path dynamically
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))



import torch
import torch.nn.functional as F
from src.models.vision_transformer import vit_small
from src.masks.uncertainty_collator import UncertaintyGuidedCollator
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image

def visualize_masks(image_tensor, context_mask, target_mask, save_path):
    """
    Visualize the image with four panels:
    1. Original image
    2. Masked Context (image with gray masks showing masked regions selected by uncertainty)
    3. Reconstruction (simulated reconstruction of masked regions)
    4. Mask Overlay (original with green grid showing reconstructed regions)
    """
    # Convert image tensor to numpy
    img = image_tensor.permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min())
    
    # Create mask overlays
    H = W = int(np.sqrt(context_mask.shape[0]))
    context_mask = context_mask.reshape(H, W).float().numpy()
    target_mask = target_mask.reshape(H, W).float().numpy()
    
    # Create masked context image (gray mask for masked regions)
    masked_context = img.copy()
    mask_color = np.array([0.5, 0.5, 0.5])  # Gray color
    for i in range(H):
        for j in range(W):
            if context_mask[i, j] == 0:  # If masked (not in context)
                patch_h = img.shape[0] // H
                patch_w = img.shape[1] // W
                masked_context[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w] = mask_color

    # Create reconstruction (for demo, we'll use the original image)
    # In practice, this would be the model's reconstruction of the masked regions
    reconstruction = img.copy()
    
    # Create mask overlay (green grid showing where reconstruction happened)
    mask_overlay = img.copy()
    grid_color = np.array([0, 1, 0])  # Green color
    grid_alpha = 0.3
    for i in range(H):
        for j in range(W):
            if context_mask[i, j] == 0:  # If masked (same as gray regions)
                patch_h = img.shape[0] // H
                patch_w = img.shape[1] // W
                # Draw grid lines
                thickness = max(1, min(patch_h, patch_w) // 8)
                mask_overlay[i*patch_h:i*patch_h+thickness, j*patch_w:(j+1)*patch_w] = grid_color
                mask_overlay[(i+1)*patch_h-thickness:(i+1)*patch_h, j*patch_w:(j+1)*patch_w] = grid_color
                mask_overlay[i*patch_h:(i+1)*patch_h, j*patch_w:j*patch_w+thickness] = grid_color
                mask_overlay[i*patch_h:(i+1)*patch_h, (j+1)*patch_w-thickness:(j+1)*patch_w] = grid_color
    
    # Create figure with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original
    ax1.imshow(img)
    ax1.set_title('Original')
    ax1.axis('off')
    
    # Masked Context (showing masked regions)
    ax2.imshow(masked_context)
    ax2.set_title('Masked Context\n(Gray = Masked Regions)')
    ax2.axis('off')
    
    # Reconstruction
    ax3.imshow(reconstruction)
    ax3.set_title('Reconstruction\n(Reconstructed Masked Regions)')
    ax3.axis('off')
    
    # Mask Overlay (showing reconstructed regions)
    ax4.imshow(mask_overlay)
    ax4.set_title('Mask Overlay\n(Green = Reconstructed Regions)')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def test_uncertainty_detection():
    """Test the uncertainty detection in isolation."""
    print("\n=== Testing Uncertainty Detection ===")
    
    # Create a small ViT model
    model = vit_small(img_size=224, patch_size=16)
    model.eval()
    
    # Create dummy input
    x = torch.randn(1, 3, 224, 224)
    
    # Test uncertainty estimation
    with torch.no_grad():
        # Get all_outputs for individual variances
        all_outputs = []
        for _ in range(3):
            out, _ = model._forward_single(x, return_attn_maps=True)
            all_outputs.append(out)
        all_outputs = torch.stack(all_outputs)  # [num_mc_samples, B, N, D]
        # Compute variance per patch per sample (across D)
        indiv_patch_vars = all_outputs.var(dim=0)  # [B, N, D]
        # Save to disk
        np.save('output/individual_patch_variances.npy', indiv_patch_vars.detach().cpu().numpy())
        print('Saved individual patch variances to output/individual_patch_variances.npy')

        # Extract attention map and compute attention scores
        attn_map = model.get_last_layer_attention_map(x)  # [B, N, N]
        attn_scores = attn_map.mean(dim=1)  # [B, N] (mean over source patches)
        np.save('output/attention_scores.npy', attn_scores.detach().cpu().numpy())
        print('Saved attention scores to output/attention_scores.npy')

        # Compute mean variance per patch (across D)
        patch_vars = indiv_patch_vars.mean(dim=-1).squeeze(0)  # [N]
        attn_scores_1d = attn_scores.squeeze(0)  # [N]

        # Identify top-k high variance and top-k high attention patches
        k = 10
        top_var_indices = patch_vars.argsort(descending=True)[:k]
        top_attn_indices = attn_scores_1d.argsort(descending=True)[:k]
        # Find intersection (patches with both high variance and high attention)
        hard_patches = np.intersect1d(top_var_indices.detach().cpu().numpy(), top_attn_indices.detach().cpu().numpy())
        np.save('output/hard_patches_indices.npy', hard_patches)
        print(f'Saved indices of hard (informative) patches to output/hard_patches_indices.npy')

    # Existing code for mean/summary variance
    mean_out, patch_vars, _ = model(x, return_patch_vars=True, num_mc_samples=3)
    print("Model returned 3 values as expected")
    print("Mean output type:", type(mean_out))
    print(f"Mean output shape: {mean_out.shape}")
    print(f"Patch variances shape: {patch_vars.shape}")
    print(f"Average variance: {patch_vars.mean().item():.4f}")
    print(f"Max variance: {patch_vars.max().item():.4f}")
    np.save('output/patch_variances.npy', patch_vars.detach().cpu().numpy())
    print('Saved patch variances to output/patch_variances.npy')
    topk = 10
    flat_patch_vars = patch_vars.flatten()
    top_indices = flat_patch_vars.argsort(descending=True)[:topk].detach().cpu().numpy()
    np.save('output/most_variant_patch_indices.npy', top_indices)
    print(f'Saved indices of top {topk} most variant patches to output/most_variant_patch_indices.npy')

    return model

def test_collator(model):
    """Test the UncertaintyGuidedCollator."""
    print("\n=== Testing UncertaintyGuidedCollator ===")
    
    # Create collator with proper scale parameters
    collator = UncertaintyGuidedCollator(
        student_model_instance=model,
        input_size=224,
        patch_size=16,
        n_targets=4,  # Number of target blocks
        n_contexts=2,  # Number of context blocks
        context_mask_scale=(0.15, 0.25),  # Size range for context blocks
        target_mask_scale=(0.15, 0.25),   # Size range for target blocks
        num_mc_samples=3,
        select_uncertain_ratio=0.75,
        aspect_ratio=(0.75, 1.5),  # Allow rectangular blocks
        target_overlaps_context=False
    )
    
    # Load and prepare CIFAR-10 images
    import torchvision
    import torchvision.transforms as transforms
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model input size
        transforms.ToTensor(),
    ])
    
    # Load CIFAR-10
    dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True,
        transform=transform
    )
    
    # Select a few interesting images (e.g., first image from different classes)
    selected_indices = [0, 1000, 2000, 3000]  # These will be from different classes
    batch = [dataset[i][0] for i in selected_indices]
    
    # Process batch
    images, context_masks, target_masks = collator(batch)
    
    # Verify shapes
    print(f"Batch images shape: {images.shape}")
    print(f"Context masks shape: {context_masks.shape}")
    print(f"Target masks shape: {target_masks.shape}")
    
    # Verify mask statistics
    print(f"Average context mask coverage: {context_masks.float().mean().item():.3f}")
    print(f"Average target mask coverage: {target_masks.float().mean().item():.3f}")
    
    # Generate visualizations for each image in the batch
    for i in range(len(images)):
        visualize_masks(
            images[i], 
            context_masks[i], 
            target_masks[i],
            f'mask_visualization_{i+1}.png'
        )
    
    return collator

def test_with_real_image(model, collator):
    """Test the pipeline with a real image."""
    print("\n=== Testing with Real Image ===")
    
    # Create transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    try:
        # Try to load a test image from the CIFAR-10 dataset
        from torchvision.datasets import CIFAR10
        dataset = CIFAR10(root='./data', train=False, download=True)
        img, _ = dataset[0]
        img = transforms.Resize((224, 224))(img)
        
    except:
        # If CIFAR-10 is not available, create a synthetic image
        img = torch.randn(3, 224, 224)
        img = transforms.ToPILImage()(img)
    
    # Transform image
    img_tensor = transform(img)
    
    # Process through collator
    images, context_masks, target_masks = collator([img_tensor])
    
    # Visualize results
    visualize_masks(
        images[0], 
        context_masks[0], 
        target_masks[0],
        'mask_visualization_real.png'
    )

def test_training_step():
    """Test a single training step."""
    print("\n=== Testing Training Step ===")
    
    # Create models and optimizer
    student = vit_small(img_size=224, patch_size=16)
    teacher = vit_small(img_size=224, patch_size=16)
    predictor = student  # Simplified for testing
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    
    # Create dummy batch
    batch_size = 2
    num_patches = 196  # 14x14 patches for 224x224 image with patch_size=16
    images = torch.randn(batch_size, 3, 224, 224)
    
    # Create masks with proper shapes
    context_mask = torch.zeros(batch_size, num_patches, dtype=torch.bool)
    target_mask = torch.zeros(batch_size, num_patches, dtype=torch.bool)
    
    # Set some patches as context and target (non-overlapping)
    for i in range(batch_size):
        # Random context patches (30%)
        num_context = int(0.3 * num_patches)
        context_indices = torch.randperm(num_patches)[:num_context]
        context_mask[i, context_indices] = True
        
        # Random target patches (20%, non-overlapping with context)
        remaining_indices = torch.tensor([j for j in range(num_patches) if j not in context_indices])
        num_target = int(0.2 * num_patches)
        target_indices = remaining_indices[torch.randperm(len(remaining_indices))[:num_target]]
        target_mask[i, target_indices] = True
    
    print(f"Context mask shape: {context_mask.shape}, sum: {context_mask.sum().item()}")
    print(f"Target mask shape: {target_mask.shape}, sum: {target_mask.sum().item()}")
    
    # Training step
    optimizer.zero_grad()
    
    # Forward pass through student with context mask
    z = student(images, masks=context_mask)
    print(f"Student output shape (z): {z.shape}")
    
    # Forward pass through teacher (no masking for teacher)
    with torch.no_grad():
        h = teacher(images)
    print(f"Teacher output shape (h): {h.shape}")
    
    # Extract features for target patches
    z_masked = []
    h_masked = []
    
    for i in range(batch_size):
        # Get indices of target patches for this batch item
        target_indices = torch.where(target_mask[i])[0]
        context_indices = torch.where(context_mask[i])[0]
        
        # Map target indices to context indices positions
        target_to_context = {}
        current_pos = 0
        for j in range(num_patches):
            if context_mask[i, j]:
                target_to_context[j] = current_pos
                current_pos += 1
        
        # Extract features only for target patches that are also in context
        valid_targets = []
        for idx in target_indices:
            if idx.item() in target_to_context:
                valid_targets.append(target_to_context[idx.item()])
        
        if valid_targets:
            valid_targets = torch.tensor(valid_targets)
            z_masked.append(z[i, valid_targets])
            h_masked.append(h[i, target_indices[valid_targets]])
    
    if z_masked:
        # Stack the masked features
        z_target = torch.cat(z_masked)
        h_target = torch.cat(h_masked)
        
        print(f"Target features shape (z_target): {z_target.shape}")
        print(f"Target features shape (h_target): {h_target.shape}")
        
        # Compute loss
        loss = F.smooth_l1_loss(z_target, h_target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"Training loss: {loss.item():.4f}")
        print(f"Loss computed on target patches successfully")
    else:
        print("No valid target patches found in context")

def main():
    print("Starting uncertainty-based masking tests...")
    
    # Test uncertainty detection
    model = test_uncertainty_detection()
    
    # Test collator
    collator = test_collator(model)
    
    # Test with real image
    test_with_real_image(model, collator)
    
    # Test training step
    test_training_step()
    
    print("\nAll tests completed! Check the generated visualization files.")

if __name__ == "__main__":
    main() 