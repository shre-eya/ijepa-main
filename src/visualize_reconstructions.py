import os
import random

import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

from .helper import init_model, load_checkpoint


def denorm(t, mean, std):
    mean = torch.tensor(mean, device=t.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=t.device).view(1, -1, 1, 1)
    return (t * std) + mean


def to_numpy_image(t):
    t = t.detach().cpu()
    t = torch.clamp(t, 0, 1)
    return t


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths and constants
    ckpt_path = './output/cifar10_uncertainty/jepa_uncertainty-latest.pth.tar'
    save_dir = './output/cifar10_uncertainty/reconstructions_predicted'
    os.makedirs(save_dir, exist_ok=True)

    # CIFAR-10 normalization (common for 32x32)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    transform = T.Compose([
        T.Resize(32),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Initialize model (vit_small) consistent with training
    encoder, predictor = init_model(
        device=device,
        patch_size=4,
        crop_size=32,
        model_name='vit_small',
        pred_depth=6,
        pred_emb_dim=384,
        use_pretrained=False,
    )
    target_encoder = None

    # Load checkpoint weights if available
    if os.path.exists(ckpt_path):
        try:
            _ = load_checkpoint(
                device=device,
                r_path=ckpt_path,
                encoder=encoder,
                predictor=predictor,
                target_encoder=target_encoder,
                opt=torch.optim.SGD([torch.nn.Parameter(torch.randn(1, device=device))], lr=0.1),
                scaler=None,
            )
        except Exception:
            pass

    encoder.eval()
    predictor.eval()

    # Utility: unnormalize for visualization
    def unnormalize(img):
        return denorm(img, mean, std)

    # Helper: overlay mask on an image
    def overlay_mask(img_chw, mask_bool_hw, color=(0, 1, 0), alpha=0.35):
        img = img_chw.clone()
        img = torch.clamp(img, 0, 1)
        overlay = img.clone()
        overlay[0].masked_fill_(mask_bool_hw, color[0])
        overlay[1].masked_fill_(mask_bool_hw, color[1])
        overlay[2].masked_fill_(mask_bool_hw, color[2])
        return img * (1 - alpha) + overlay * alpha

    # Build per-channel normalized gray value such that after unnormalize it becomes 0.5
    gray_norm = torch.tensor([
        (0.5 - mean[0]) / std[0],
        (0.5 - mean[1]) / std[1],
        (0.5 - mean[2]) / std[2],
    ], device=device).view(1, 3, 1, 1)

    # Select 5 random indices
    indices = random.sample(range(len(test_set)), 5)

    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            img_tensor, _ = test_set[sample_idx]
            img_bchw = img_tensor.unsqueeze(0).to(device)

            # Patching setup for 32x32 with patch_size=4
            ph = pw = 4
            grid_h = grid_w = 8
            B = 1
            N = grid_h * grid_w

            # Create a random 50% masking over patches
            # Build a boolean mask where True indicates MASKED (to be predicted)
            perm = torch.randperm(N, device=device)
            num_mask = N // 2
            masked_indices = perm[:num_mask]
            mask_bool = torch.zeros(B, N, dtype=torch.bool, device=device)
            mask_bool[0, masked_indices] = True
            context_mask_bool = ~mask_bool

            # Encode context patches only
            z = encoder(img_bchw, masks=context_mask_bool)

            # Predict masked patch embeddings
            preds = predictor(z, masks_x=context_mask_bool, masks=mask_bool)  # [B, K, D]

            # Approximate unpatchify: map embeddings -> pixels via transposed patch embedding weights
            # Use Conv2d weights from the encoder's patch embed
            W = encoder.patch_embed.proj.weight  # [D, C, ph, pw]
            b = encoder.patch_embed.proj.bias    # [D]
            D, C, _, _ = W.shape

            # Prepare empty reconstructed image in normalized space, fill non-masked with gray
            recon_norm = gray_norm.expand(B, 3, 32, 32).clone()

            # Also prepare masked-context image (original but gray on masked regions)
            img_vis = img_bchw.clone()

            mask_hw = torch.zeros((32, 32), dtype=torch.bool, device=device)

            # Iterate patches and place predictions for masked ones
            k = 0
            pred_ptr = 0
            for gy in range(grid_h):
                for gx in range(grid_w):
                    y0, x0 = gy * ph, gx * pw
                    if mask_bool[0, k]:
                        # Predicted embedding for this masked patch
                        v = preds[0, pred_ptr]  # [D]
                        pred_ptr += 1
                        # Linear back-projection: (v - b) * W^T -> [C, ph, pw]
                        patch = torch.tensordot(v - b, W, dims=([0], [0]))  # [C, ph, pw]
                        recon_norm[:, :, y0:y0+ph, x0:x0+pw] = patch.unsqueeze(0)
                        mask_hw[y0:y0+ph, x0:x0+pw] = True
                    else:
                        # Context remains original for masked-context view; gray in recon view already set
                        pass
                    k += 1

            # Build masked context visualization in pixel space
            masked_context = unnormalize(img_vis.clone())
            masked_context = torch.clamp(masked_context, 0, 1)
            masked_context[:, :, mask_hw] = 0.5

            # Unnormalize reconstructed image for display
            recon_disp = torch.clamp(unnormalize(recon_norm), 0, 1)

            # Original for display
            original_disp = torch.clamp(unnormalize(img_bchw.clone()), 0, 1)

            # Overlay visualization on original
            overlay = overlay_mask(original_disp[0], mask_hw, color=(0, 1, 0), alpha=0.35).unsqueeze(0)

            # Compose 2x2 figure
            fig, axes = plt.subplots(2, 2, figsize=(6, 6))
            axes = axes.flatten()

            axes[0].imshow(original_disp[0].permute(1, 2, 0).cpu().numpy())
            axes[0].set_title('Original')
            axes[0].axis('off')

            axes[1].imshow(masked_context[0].permute(1, 2, 0).cpu().numpy())
            axes[1].set_title('Masked Context (gray = missing)')
            axes[1].axis('off')

            axes[2].imshow(recon_disp[0].permute(1, 2, 0).cpu().numpy())
            axes[2].set_title('Reconstructed (Predicted Masked Pixels)')
            axes[2].axis('off')

            axes[3].imshow(overlay[0].permute(1, 2, 0).cpu().numpy())
            axes[3].set_title('Mask Overlay (Green = Reconstructed Regions)')
            axes[3].axis('off')

            plt.tight_layout()
            out_file = os.path.join(save_dir, f'reconstruction_predicted_{idx+1}.png')
            plt.savefig(out_file)
            plt.close(fig)

    print(f"Saved {len(indices)} predicted reconstructions to {save_dir}")


if __name__ == '__main__':
    main()


