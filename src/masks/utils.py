# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch


def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    
    This function preserves the sequence length N. For each mask in `masks`, we
    create a per-batch selection map of shape [B, N] and zero out the unselected
    positions in `x`. The outputs for each mask are concatenated along the batch
    dimension, resulting in shape [len(masks) * B, N, D].
    """
    all_x = []
    B, N, D = x.shape
    for m in masks:
        # Normalize mask index tensor shape and dtype before use
        # - Ensure indices are int64 (long): required by indexing/gather APIs
        # - Ensure indices have per-batch shape [B, K] so we can build a [B, N] selection map
        if m.dtype != torch.int64:
            m = m.to(torch.int64)
        if m.ndim == 1:
            # Provided as [K]; repeat for each batch element
            m = m.unsqueeze(0).repeat(B, 1)
        elif m.ndim == 2 and m.size(0) == 1 and B > 1:
            # Provided as [1, K] but input has batch B; repeat across batch
            m = m.repeat(B, 1)

        # Build selection map [B, N] with 1s at kept indices and 0s elsewhere
        select = x.new_zeros((B, N), dtype=x.dtype)
        # Use scatter to mark kept indices; values don't matter beyond non-zero
        ones = torch.ones_like(m, dtype=x.dtype)
        select.scatter_(dim=1, index=m, src=ones)
        # Expand to [B, N, D] and zero out unselected positions
        select = select.unsqueeze(-1)  # [B, N, 1]
        masked = x * select
        all_x.append(masked)

    out = torch.cat(all_x, dim=0)
    # Safety: preserve sequence length
    assert out.size(1) == N, f"apply_masks: seq length changed from {N} to {out.size(1)}"
    return out
