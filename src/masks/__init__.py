from .default import DefaultCollator
from .uncertainty_collator import UncertaintyGuidedCollator

def get_mask_collator(cfg, model=None):
    mask_cfg = cfg["mask"]  # ✅ extract mask-specific config

    # ✅ Ensure crop_size and patch_size are tuples like (32, 32)
    input_size = cfg["data"]["crop_size"]
    patch_size = mask_cfg["patch_size"]
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)

    if mask_cfg["mask_type"] == "default":
        return DefaultCollator(
            input_size=input_size,
            patch_size=patch_size,
            n_targets=mask_cfg["n_targets"],
            n_contexts=mask_cfg["n_contexts"],
            context_mask_scale=mask_cfg["context_mask_scale"],
            target_mask_scale=mask_cfg["target_mask_scale"],
        )

    elif mask_cfg["mask_type"] == "uncertainty_guided":
        return UncertaintyGuidedCollator(
            student_model_instance=model,
            input_size=input_size,
            patch_size=patch_size,
            n_targets=mask_cfg["n_targets"],
            n_contexts=mask_cfg["n_contexts"],
            context_mask_scale=mask_cfg["context_mask_scale"],
            target_mask_scale=mask_cfg["target_mask_scale"],
        )

    else:
        raise ValueError(f"Unknown mask type: {mask_cfg['mask_type']}")
