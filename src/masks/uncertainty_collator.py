# src/masks/uncertainty_collator.py

import torch
import torch.nn.functional as F
import random
import math
import os
import numpy as np

from .multiblock import MaskCollator as MultiMaskBlockCollator

class UncertaintyGuidedCollator(MultiMaskBlockCollator):
    def __init__(self,
             student_model_instance,
             input_size,
             patch_size,
             n_targets,
             n_contexts,
             context_mask_scale,
             target_mask_scale,
             num_mc_samples=3,
             candidate_target_scale_multiplier=2.0,
             select_uncertain_ratio=0.75,
             context_aspect_ratio_scale=(0.9, 1.1),
             target_aspect_ratio_scale=(0.9, 1.1),
             context_rel_pos_scale=(0.33, 0.67),
             target_rel_pos_scale=(0.33, 0.67),
             aspect_ratio=(0.9, 1.1),  
             target_rel_pos_range=(0.4, 0.6),
             target_overlaps_context=False,
             **kwargs):

        # Convert single scale values to tuples if needed
        if isinstance(context_mask_scale, (int, float)):
            context_mask_scale = (context_mask_scale * 0.8, context_mask_scale * 1.2)
        if isinstance(target_mask_scale, (int, float)):
            target_mask_scale = (target_mask_scale * 0.8, target_mask_scale * 1.2)

        super().__init__(
            input_size=input_size,
            patch_size=patch_size,
            enc_mask_scale=context_mask_scale,
            pred_mask_scale=target_mask_scale,
            aspect_ratio=aspect_ratio,
            nenc=n_contexts,
            npred=n_targets,
            min_keep=1,
            allow_overlap=target_overlaps_context,
            **kwargs
        )

        self.input_size = input_size if isinstance(input_size, tuple) else (input_size, input_size)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.n_targets = n_targets
        self.n_contexts = n_contexts
        self.student_model = student_model_instance
        self.num_mc_samples = num_mc_samples
        self.device = next(student_model_instance.parameters()).device

        self.num_candidate_targets = max(math.ceil(n_targets * candidate_target_scale_multiplier), n_targets)
        self.num_uncertain_to_select = math.ceil(n_targets * select_uncertain_ratio)
        self.num_random_to_select = n_targets - self.num_uncertain_to_select

        # Load hard patch indices if available
        self.hard_patch_indices = None
        hard_patch_path = 'output/hard_patches_indices.npy'
        if os.path.exists(hard_patch_path):
            self.hard_patch_indices = np.load(hard_patch_path)
            print(f"Loaded {len(self.hard_patch_indices)} hard patch indices from {hard_patch_path}")

        print(f"--- UncertaintyGuidedCollator Initialized ---")
        print(f"Input size: {self.input_size}, Patch size: {self.patch_size}")
        print(f"Context mask scale: {context_mask_scale}")
        print(f"Target mask scale: {target_mask_scale}")
        print(f"MC Samples: {self.num_mc_samples}, Num final targets: {self.n_targets}")
        print(f"Num candidate targets: {self.num_candidate_targets}")
        print(f"Num uncertain to select: {self.num_uncertain_to_select}, Num random to select: {self.num_random_to_select}")
        print(f"---------------------------------------------")

    def get_uncertain_patches(self, image_tensor, context_indices, candidate_target_indices):
        """
        For a given image, compute uncertainty scores for candidate patches using MC forward passes.
        
        Args:
            image_tensor: (C, H, W) tensor of the input image
            context_indices: tensor of patch indices used as context
            candidate_target_indices: list of tensors, each containing patch indices for a candidate block
            
        Returns:
            List of selected target patch indices based on uncertainty
        """
        device = self.device
        image_tensor = image_tensor.to(device)
        
        # Create a batch with just this image
        x = image_tensor.unsqueeze(0)  # [1, C, H, W]
        
        # Convert context indices to boolean mask
        patch_h = self.input_size[0] // self.patch_size[0]
        patch_w = self.input_size[1] // self.patch_size[1]
        num_patches = patch_h * patch_w
        base_mask = torch.zeros(1, num_patches, dtype=torch.bool, device=device)
        base_mask[0, context_indices] = True
        
        # For each candidate target block
        all_variances = []
        for target_indices in candidate_target_indices:
            # Create mask that includes context and this target block
            mask = base_mask.clone()
            mask[0, target_indices] = True
            
            # Get uncertainty score for this block
            with torch.no_grad():
                try:
                    _, patch_vars, _ = self.student_model(
                        x, 
                        masks=mask,
                        return_patch_vars=True,
                        num_mc_samples=self.num_mc_samples
                    )
                    # Average variance across patches in this block
                    # FIX: Ensure target_indices are within bounds and handle tensor safely
                    if isinstance(target_indices, torch.Tensor) and len(target_indices) > 0:
                        valid_indices = target_indices[target_indices < patch_vars.size(1)]
                        if len(valid_indices) > 0:
                            block_var = patch_vars[0, valid_indices].mean()
                        else:
                            block_var = torch.tensor(0.0, device=device)
                    else:
                        block_var = torch.tensor(0.0, device=device)
                except Exception as e:
                    print(f"Warning: Error in uncertainty computation: {str(e)}")
                    block_var = torch.tensor(0.0, device=device)
                
                all_variances.append(block_var)
        
        # Convert to tensor
        all_variances = torch.stack(all_variances)
        
        # Select blocks with highest variance
        num_to_select = min(self.num_uncertain_to_select, len(all_variances))
        if num_to_select > 0:
            topk_indices = torch.topk(all_variances, num_to_select).indices
            selected_targets = [candidate_target_indices[i] for i in topk_indices.tolist()]
        else:
            selected_targets = []
        
        # Add random blocks if needed
        if self.num_random_to_select > 0:
            remaining_indices = list(set(range(len(candidate_target_indices))) - set(topk_indices.tolist()))
            if remaining_indices:
                random_indices = random.sample(remaining_indices, min(self.num_random_to_select, len(remaining_indices)))
                selected_targets.extend([candidate_target_indices[i] for i in random_indices])
        
        return selected_targets

    def _sample_block_masks(self, patch_indices, num_context_blocks, num_target_blocks, image_tensor=None):
        """Sample context and target blocks with uncertainty guidance."""
        # Sample context blocks first
        context_indices_list = []
        for _ in range(num_context_blocks):
            context_block_indices, _ = self._sample_block_mask(self._sample_block_size(
                generator=torch.Generator().manual_seed(torch.randint(0, 99999, (1,)).item()),
                scale=self.enc_mask_scale,
                aspect_ratio_scale=(1.0, 1.0)
            ))
            context_indices_list.append(context_block_indices)
        
        # Combine all context indices
        context_indices = torch.cat(context_indices_list) if context_indices_list else torch.tensor([], dtype=torch.long)
        
        # Sample candidate target blocks
        candidate_target_indices = []
        for _ in range(self.num_candidate_targets):
            target_block_indices, _ = self._sample_block_mask(self._sample_block_size(
                generator=torch.Generator().manual_seed(torch.randint(0, 99999, (1,)).item()),
                scale=self.pred_mask_scale,
                aspect_ratio_scale=self.aspect_ratio
            ))
            candidate_target_indices.append(target_block_indices)
        
        # If we have an image tensor, use uncertainty to select target blocks
        if image_tensor is not None and self.student_model is not None:
            selected_targets = self.get_uncertain_patches(image_tensor, context_indices, candidate_target_indices)
            # Preferentially include hard patches if available
            if self.hard_patch_indices is not None:
                # Flatten selected_targets to a set of patch indices
                selected_indices = set()
                for block in selected_targets:
                    selected_indices.update(block.cpu().numpy().tolist())
                # Add hard patches if not already included
                for idx in self.hard_patch_indices:
                    if idx not in selected_indices:
                        # Add as a single-patch block
                        selected_targets.append(torch.tensor([idx], dtype=torch.long, device=self.device))
        else:
            # Fallback to random selection
            selected_targets = random.sample(candidate_target_indices, min(self.n_targets, len(candidate_target_indices)))
        
        return context_indices, selected_targets

    def __call__(self, batch):
        """
        Process a batch of images to create context and target masks based on uncertainty.
        
        Args:
            batch: List of (image, label) tuples from dataset
            
        Returns:
            Tuple of (collated_batch, context_masks, target_masks)
        """
        # FIX: Handle empty batches
        if not batch or len(batch) == 0:
            raise ValueError("Empty batch provided to collator")
            
        # FIX: Handle (image, label) tuples properly - extract only images
        # The dataset returns (image, label) tuples, but we only need images for masking
        if isinstance(batch[0], (tuple, list)) and len(batch[0]) == 2:
            # Extract images from (image, label) tuples
            images = [item[0] for item in batch]
        else:
            # Fallback: assume batch contains only images
            images = batch
            
        images = torch.stack(images, dim=0)
        B = images.size(0)
        device = images.device
        
        # Calculate number of patches
        patch_h = self.input_size[0] // self.patch_size[0]
        patch_w = self.input_size[1] // self.patch_size[1]
        num_patches = patch_h * patch_w
        
        all_context_indices = []
        all_target_indices = []
        
        for i in range(B):
            # Sample context and candidate target blocks
            context_indices, target_indices = self._sample_block_masks(
                patch_indices=None,  # Not needed for our implementation
                num_context_blocks=self.n_contexts,
                num_target_blocks=self.num_candidate_targets,
                image_tensor=images[i]
            )
            
            all_context_indices.append(context_indices)
            all_target_indices.append(target_indices)
        
        # Convert indices to masks
        context_masks = [torch.zeros(num_patches, dtype=torch.bool, device=device) for _ in range(B)]
        target_masks = [torch.zeros(num_patches, dtype=torch.bool, device=device) for _ in range(B)]
        
        for i in range(B):
            # FIX: Safely handle context indices with bounds checking
            if len(all_context_indices[i]) > 0:
                # Ensure indices are within bounds
                valid_context_indices = all_context_indices[i][all_context_indices[i] < num_patches]
                if len(valid_context_indices) > 0:
                    context_masks[i][valid_context_indices] = True
            
            # FIX: Safely handle target indices - each target_idx can be a tensor of indices
            for target_block in all_target_indices[i]:
                if isinstance(target_block, torch.Tensor):
                    # Handle tensor of indices (patch indices within a block)
                    # Ensure all indices are within bounds
                    valid_target_indices = target_block[target_block < num_patches]
                    if len(valid_target_indices) > 0:
                        target_masks[i][valid_target_indices] = True
                else:
                    # Handle single index (convert to tensor if needed)
                    if isinstance(target_block, (int, float)):
                        target_idx = int(target_block)
                        if 0 <= target_idx < num_patches:
                            target_masks[i][target_idx] = True
                    else:
                        # Handle other iterable types
                        try:
                            for idx in target_block:
                                if isinstance(idx, torch.Tensor):
                                    idx = idx.item() if idx.numel() == 1 else idx
                                if isinstance(idx, (int, float)):
                                    idx = int(idx)
                                    if 0 <= idx < num_patches:
                                        target_masks[i][idx] = True
                        except Exception as e:
                            print(f"Warning: Could not process target index {target_block}: {e}")
        
        context_masks = torch.stack(context_masks)
        target_masks = torch.stack(target_masks)
        
        # FIX: Follow the same pattern as other collators - return collated batch
        # Create a collated batch using the original batch structure
        collated_batch = torch.utils.data.default_collate(batch)
        
        return collated_batch, context_masks, target_masks
