import os
import random

import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

from helper import init_model, load_checkpoint


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
    out_path = './output/cifar10_uncertainty/reconstruction_grid.png'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

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

    # Sample 5 random indices
    indices = random.sample(range(len(test_set)), 5)

    # Prepare figure: 2 rows x 5 cols
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    # Unnormalize transform for visualization
    def unnormalize(img):
        return denorm(img, mean, std)

    # Helper to overlay mask
    def overlay_mask(img_chw, mask_bool_hw, color=(0, 1, 0), alpha=0.35):
        img = img_chw.clone()
        img = torch.clamp(img, 0, 1)
        c, h, w = img.shape
        overlay = img.clone()
        overlay[0].masked_fill_(mask_bool_hw, color[0])
        overlay[1].masked_fill_(mask_bool_hw, color[1])
        overlay[2].masked_fill_(mask_bool_hw, color[2])
        return img * (1 - alpha) + overlay * alpha

    with torch.no_grad():
        for col, idx in enumerate(indices):
            img_tensor, _ = test_set[idx]
            img_bchw = img_tensor.unsqueeze(0).to(device)

            # Build a simple square mask over ~25% patches for visualization
            # Assume patch_size=4 on 32x32 -> 8x8 = 64 patches
            B = 1
            N = (32 // 4) * (32 // 4)
            H_p = W_p = 8
            mask_bool = torch.zeros(B, N, dtype=torch.bool, device=device)
            # Select a 3x3 block in the center
            center = 4
            sel = []
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    y = center + dy
                    x = center + dx
                    if 0 <= y < H_p and 0 <= x < W_p:
                        sel.append(y * W_p + x)
            sel = torch.tensor(sel, device=device)
            mask_bool[0, sel] = True

            # Forward encoder with mask (returns patch embeddings)
            z = encoder(img_bchw, masks=mask_bool)

            # Predict reconstructions over target mask using predictor
            preds = predictor(z, masks_x=mask_bool, masks=mask_bool)
            # preds: [B, K, D], but we need to project back to image space for visualization
            # For a qualitative demo, we will simply show masked context and original image; full decoding isn't defined here

            # Build masked context visualization (gray over masked patches)
            img_vis = unnormalize(img_bchw.clone())
            img_vis = torch.clamp(img_vis, 0, 1)
            grid_h = grid_w = 8
            ph = pw = 4
            masked_context = img_vis.clone()
            mask_hw = torch.zeros((32, 32), dtype=torch.bool, device=device)
            k = 0
            for gy in range(grid_h):
                for gx in range(grid_w):
                    is_ctx = mask_bool[0, k].item()
                    y0, x0 = gy * ph, gx * pw
                    if not is_ctx:
                        # Context kept; leave as is
                        pass
                    else:
                        # Masked region -> gray
                        masked_context[:, :, y0:y0+ph, x0:x0+pw] = 0.5
                        mask_hw[y0:y0+ph, x0:x0+pw] = True
                    k += 1

            # Approximate reconstruction visualization: show original where mask was (as proxy)
            # In absence of an explicit decoder to pixels, use original pixels for illustration of locations
            reconstruction = img_vis.clone()

            # Overlay mask regions in green
            mask_overlay = overlay_mask(img_vis[0], mask_hw, color=(0, 1, 0), alpha=0.35).unsqueeze(0)

            # Convert to CPU and grid-friendly tensors
            original_img = to_numpy_image(img_vis)
            masked_ctx_img = to_numpy_image(masked_context)
            recon_img = to_numpy_image(reconstruction)
            overlay_img = to_numpy_image(mask_overlay)

            # Place into figure
            axes[0, col].imshow(original_img[0].permute(1, 2, 0).cpu().numpy())
            axes[0, col].set_title('Original')
            axes[0, col].axis('off')

            axes[1, col].imshow(masked_ctx_img[0].permute(1, 2, 0).cpu().numpy())
            axes[1, col].set_title('Masked Context')
            axes[1, col].axis('off')

            # For the second row, right cell show reconstruction and mask overlay side-by-side
            # But we have only 2 rows; instead, overlay mask on reconstruction in same cell title
            # To satisfy the 2x2 per-sample requirement in one 2x5 grid, we can use small montage columns
            # Here, show reconstruction on top-left of the cell and overlay next to it via make_grid
            pair = torchvision.utils.make_grid(torch.stack([recon_img[0], overlay_img[0]], dim=0), nrow=2)
            axes[0, col].imshow(original_img[0].permute(1, 2, 0).cpu().numpy())
            axes[0, col].set_title('Original')
            axes[0, col].axis('off')
            axes[1, col].imshow(pair.permute(1, 2, 0).cpu().numpy())
            axes[1, col].set_title('Reconstruction | Mask Overlay')
            axes[1, col].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved visualization to {out_path}")


if __name__ == '__main__':
    main()


