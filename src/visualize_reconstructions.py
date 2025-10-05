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
    save_dir = './output/cifar10_uncertainty/reconstructions_decoded'
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

    # Build lightweight decoder initialized from patch embedding for cleaner pixels
    embed_dim = encoder.embed_dim  # 384 for vit_small
    ph = pw = 4
    decoder = torch.nn.ConvTranspose2d(
        in_channels=embed_dim,
        out_channels=3,
        kernel_size=(ph, pw),
        stride=ph
    ).to(device)
    # Initialize decoder as transpose of patch embedding conv
    with torch.no_grad():
        W = encoder.patch_embed.proj.weight  # [out_channels(=embed_dim), in_channels(=3), ph, pw]
        decoder.weight.copy_(W)  # [out_channels(=embed_dim), in_channels(=3), ph, pw] mapped correctly
        if decoder.bias is not None:
            decoder.bias.zero_()

    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            img_tensor, _ = test_set[sample_idx]
            img_bchw = img_tensor.unsqueeze(0).to(device)

            # Patching setup for 32x32 with patch_size=4
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

            # Place predicted embeddings into a full 8x8 grid feature map
            feat_grid = torch.zeros(B, embed_dim, grid_h, grid_w, device=device)
            mask_hw = torch.zeros((32, 32), dtype=torch.bool, device=device)

            k = 0
            pred_ptr = 0
            for gy in range(grid_h):
                for gx in range(grid_w):
                    if mask_bool[0, k]:
                        feat_grid[0, :, gy, gx] = preds[0, pred_ptr]
                        y0, x0 = gy * ph, gx * pw
                        mask_hw[y0:y0+ph, x0:x0+pw] = True
                        pred_ptr += 1
                    k += 1

            # Decode to pixel space
            decoded_img = torch.clamp(decoder(feat_grid), 0, 1)  # [B, 3, 32, 32]

            # Build masked context: original but gray on masked regions
            original_disp = torch.clamp(unnormalize(img_bchw.clone()), 0, 1)
            masked_context = original_disp.clone()
            masked_context[:, :, mask_hw] = 0.5

            # Reconstructed (decoded predicted masked pixels only)
            recon_only = torch.full_like(original_disp, 0.5)
            recon_only[:, :, mask_hw] = decoded_img[:, :, mask_hw]

            # Final reconstruction = original unmasked + decoded masked
            final_recon = original_disp.clone()
            final_recon[:, :, mask_hw] = decoded_img[:, :, mask_hw]

            # Compose 2x2 figure
            fig, axes = plt.subplots(2, 2, figsize=(6, 6))
            axes = axes.flatten()

            axes[0].imshow(original_disp[0].permute(1, 2, 0).cpu().numpy())
            axes[0].set_title('Original')
            axes[0].axis('off')

            axes[1].imshow(masked_context[0].permute(1, 2, 0).cpu().numpy())
            axes[1].set_title('Masked Context (gray = hidden)')
            axes[1].axis('off')

            axes[2].imshow(recon_only[0].permute(1, 2, 0).cpu().numpy())
            axes[2].set_title('Decoded Reconstruction (Predicted Masked Pixels)')
            axes[2].axis('off')

            axes[3].imshow(final_recon[0].permute(1, 2, 0).cpu().numpy())
            axes[3].set_title('Final Reconstruction (Predicted + Original)')
            axes[3].axis('off')

            plt.tight_layout()
            out_file = os.path.join(save_dir, f'decoded_reconstruction_{idx+1}.png')
            plt.savefig(out_file)
            plt.close(fig)

    print(f"Saved {len(indices)} decoded reconstructions to {save_dir}")


if __name__ == '__main__':
    main()


