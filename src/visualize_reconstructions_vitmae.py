import os
import random

import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
    # Device selection with bonus message
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print('⚡ Using GPU for ViT-MAE (dgx-v100-01 detected)')
    else:
        print('Using CPU for ViT-MAE')

    # Output directory
    save_dir = './output/cifar10_uncertainty/reconstructions_vitmae'
    os.makedirs(save_dir, exist_ok=True)

    # Load Hugging Face ViT-MAE tiny
    from transformers import AutoImageProcessor, ViTMAEForPreTraining
    model_name = 'facebook/vit-mae-base'
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = ViTMAEForPreTraining.from_pretrained(model_name)
    model.eval().to(device)

    # CIFAR-10 to ImageNet-style normalization and 224x224 resize
    # Use the processor's image resizing/normalization defaults for correctness
    transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Pick 5 random samples
    indices = random.sample(range(len(test_set)), 5)

    with torch.no_grad():
        for i, idx in enumerate(tqdm(indices, desc='ViT-MAE reconstructing', unit='img')):
            img_tensor, _ = test_set[idx]  # [3, 224, 224], normalized
            pixel_values = img_tensor.unsqueeze(0).to(device)

            # Prepare inputs for ViT-MAE (it handles masking internally when not providing labels)
            # Mask ratio default is 0.75; can be controlled via model.config.mask_ratio
            outputs = model(pixel_values)
            # Reconstructed pixels are in outputs.logits: shape [B, num_patches, patch_size**2 * 3]

            # The processor can post-process the reconstructions back to image space
            # However, ViT-MAE returns per-patch predictions; we use helper to get image
            # Hugging Face provides a utility on the processor to postprocess reconstructed images
            # Compose the reconstructed image and the mask overlay
            
            # Decode reconstruction with processor (build PIL image from reconstructed pixel values)
            # Note: ViT-MAE returns normalized predictions in [0, 1] range per processor expectation
            recon = processor.post_process_semantic_segmentation(outputs, target_sizes=[(224, 224)]) if hasattr(processor, 'post_process_semantic_segmentation') else None
            
            # Since post-process util may not exist for MAE, manually unpatchify using model's helper if available
            if hasattr(model, 'unpatchify'):
                pred_pixel_values = model.unpatchify(outputs.logits)  # [B, 3, 224, 224]
            else:
                # Fallback: approximate via rearrangement
                B, N, Cpp = outputs.logits.shape
                p = model.config.patch_size
                H = W = int((N) ** 0.5)
                pred = outputs.logits.reshape(B, H, W, p, p, 3).permute(0, 5, 1, 3, 2, 4).reshape(B, 3, H*p, W*p)
                pred_pixel_values = pred

            # Build mask map from model (True where masked)
            # model creates a boolean mask ids_restore and mask in outputs
            if hasattr(outputs, 'mask') and outputs.mask is not None:
                # outputs.mask: [B, N], True for masked
                mask_tokens = outputs.mask[0]
            else:
                # If not exposed, approximate with model.config.mask_ratio
                N = pred_pixel_values.shape[-1] // model.config.patch_size
                N = (224 // model.config.patch_size) ** 2
                num_mask = int(N * model.config.mask_ratio)
                mask_tokens = torch.zeros(N, dtype=torch.bool, device=device)
                mask_tokens[:num_mask] = True

            # Derive pixel-space mask_hw
            p = model.config.patch_size
            H_p = W_p = 224 // p
            mask_hw = torch.zeros((224, 224), dtype=torch.bool, device=device)
            k = 0
            for gy in range(H_p):
                for gx in range(W_p):
                    y0, x0 = gy * p, gx * p
                    if mask_tokens[k]:
                        mask_hw[y0:y0+p, x0:x0+p] = True
                    k += 1

            # Prepare visualizations: original, masked context, predicted, final
            def denorm(x):
                mean = torch.tensor(processor.image_mean, device=x.device).view(1, -1, 1, 1)
                std = torch.tensor(processor.image_std, device=x.device).view(1, -1, 1, 1)
                return torch.clamp(x * std + mean, 0, 1)

            original_disp = denorm(pixel_values).cpu()
            masked_context = original_disp.clone()
            masked_context[:, :, mask_hw.cpu()] = 0.5

            pred_disp = torch.clamp(pred_pixel_values, 0, 1).detach().cpu()
            recon_only = torch.full_like(original_disp, 0.5)
            recon_only[:, :, mask_hw.cpu()] = pred_disp[:, :, mask_hw.cpu()]

            final_recon = original_disp.clone()
            final_recon[:, :, mask_hw.cpu()] = pred_disp[:, :, mask_hw.cpu()]

            # Plot 2x2
            fig, axes = plt.subplots(2, 2, figsize=(8, 8))
            axes = axes.flatten()

            axes[0].imshow(original_disp[0].permute(1, 2, 0).numpy())
            axes[0].set_title('Original')
            axes[0].axis('off')

            axes[1].imshow(masked_context[0].permute(1, 2, 0).numpy())
            axes[1].set_title('Masked Context (gray = hidden)')
            axes[1].axis('off')

            axes[2].imshow(recon_only[0].permute(1, 2, 0).numpy())
            axes[2].set_title('Predicted Reconstruction (MAE)')
            axes[2].axis('off')

            axes[3].imshow(final_recon[0].permute(1, 2, 0).numpy())
            axes[3].set_title('Final Reconstruction (Predicted + Original)')
            axes[3].axis('off')

            plt.tight_layout()
            out_file = os.path.join(save_dir, f'vitmae_reconstruction_{i+1}.png')
            plt.savefig(out_file)
            plt.close(fig)

    print(f"Saved ViT-MAE reconstructions to {save_dir}")


if __name__ == '__main__':
    main()


