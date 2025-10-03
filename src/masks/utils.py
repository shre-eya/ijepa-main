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
    """
    all_x = []
    B, N, D = x.shape
    for m in masks:
        # Normalize mask index tensor shape and dtype before gather
        # - Ensure indices are int64: torch.gather requires long dtype
        # - Ensure indices have per-batch shape [B, K] so that gather on dim=1 picks K patches for each sample
        if m.dtype != torch.int64:
            m = m.to(torch.int64)
        if m.ndim == 1:
            # Provided as [K]; repeat for each batch element
            m = m.unsqueeze(0).repeat(B, 1)
        elif m.ndim == 2 and m.size(0) == 1 and B > 1:
            # Provided as [1, K] but input has batch B; repeat across batch
            m = m.repeat(B, 1)

        # Expand indices to [B, K, D] for gather along dim=1
        mask_keep = m.unsqueeze(-1).repeat(1, 1, D)

        # Gather selected patches; result is [B, K, D]
        all_x.append(torch.gather(x, dim=1, index=mask_keep))
    return torch.cat(all_x, dim=0)
