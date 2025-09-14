# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import sys

import torch

import src.models.vision_transformer as vit
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule)
from src.utils.tensors import trunc_normal_

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def load_checkpoint(
    device,
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        # -- loading encoder
        pretrained_dict = checkpoint['encoder']
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading predictor
        pretrained_dict = checkpoint['predictor']
        msg = predictor.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading target_encoder
        if target_encoder is not None:
            print(list(checkpoint.keys()))
            pretrained_dict = checkpoint['target_encoder']
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading optimizer
        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch}')
        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return encoder, predictor, target_encoder, opt, scaler, epoch


def load_pretrained_vit_model(model_name, pretrained_source='timm', **kwargs):
    """
    Load a pretrained ViT model from timm or torchvision.
    
    Args:
        model_name: Name of the model (e.g., 'vit_base_patch16_224', 'vit_large_patch16_224')
        pretrained_source: Source of pretrained weights ('timm' or 'torchvision')
        **kwargs: Additional arguments for model creation
    
    Returns:
        model: Pretrained ViT model
    """
    try:
        if pretrained_source == 'timm':
            import timm
            logger.info(f'Loading pretrained model {model_name} from timm')
            model = timm.create_model(model_name, pretrained=True, **kwargs)
            return model
        elif pretrained_source == 'torchvision':
            import torchvision.models as models
            logger.info(f'Loading pretrained model {model_name} from torchvision')
            if hasattr(models, model_name):
                model = getattr(models, model_name)(pretrained=True, **kwargs)
                return model
            else:
                raise ValueError(f'Model {model_name} not found in torchvision')
        else:
            raise ValueError(f'Unsupported pretrained source: {pretrained_source}')
    except ImportError as e:
        logger.warning(f'Could not import {pretrained_source}: {e}')
        logger.warning('Falling back to random initialization')
        return None
    except Exception as e:
        logger.warning(f'Error loading pretrained model: {e}')
        logger.warning('Falling back to random initialization')
        return None


def adapt_pretrained_model_to_ijepa(pretrained_model, target_encoder):
    """
    Adapt a pretrained ViT model to match IJEPA's encoder architecture.
    
    Args:
        pretrained_model: Pretrained ViT model from timm/torchvision
        target_encoder: IJEPA encoder to copy weights to
    
    Returns:
        target_encoder: Encoder with pretrained weights
    """
    if pretrained_model is None:
        return target_encoder
    
    logger.info('Adapting pretrained model weights to IJEPA architecture')
    
    # Get state dicts
    pretrained_state = pretrained_model.state_dict()
    target_state = target_encoder.state_dict()
    
    # Mapping from pretrained model keys to IJEPA keys
    key_mapping = {
        'patch_embed.proj.weight': 'patch_embed.proj.weight',
        'patch_embed.proj.bias': 'patch_embed.proj.bias',
        'pos_embed': 'pos_embed',
        'blocks': 'blocks',
        'norm.weight': 'norm.weight',
        'norm.bias': 'norm.bias'
    }
    
    # Copy compatible weights
    copied_keys = []
    for pretrained_key, target_key in key_mapping.items():
        if pretrained_key in pretrained_state and target_key in target_state:
            if pretrained_state[pretrained_key].shape == target_state[target_key].shape:
                target_state[target_key] = pretrained_state[pretrained_key].clone()
                copied_keys.append(target_key)
                logger.info(f'Copied {pretrained_key} -> {target_key}')
            else:
                logger.warning(f'Shape mismatch for {pretrained_key}: {pretrained_state[pretrained_key].shape} vs {target_state[target_key].shape}')
    
    # Handle block-level copying
    for i in range(min(len(pretrained_model.blocks), len(target_encoder.blocks))):
        for j, (pretrained_block, target_block) in enumerate(zip(pretrained_model.blocks[i].modules(), target_encoder.blocks[i].modules())):
            if hasattr(pretrained_block, 'weight') and hasattr(target_block, 'weight'):
                if pretrained_block.weight.shape == target_block.weight.shape:
                    target_block.weight.data = pretrained_block.weight.data.clone()
                    if hasattr(pretrained_block, 'bias') and hasattr(target_block, 'bias'):
                        target_block.bias.data = pretrained_block.bias.data.clone()
                    copied_keys.append(f'blocks.{i}.{j}')
    
    target_encoder.load_state_dict(target_state)
    logger.info(f'Successfully copied {len(copied_keys)} layers from pretrained model')
    
    return target_encoder


def init_model(
    device,
    patch_size=16,
    model_name='vit_base',
    crop_size=224,
    pred_depth=6,
    pred_emb_dim=384,
    use_pretrained=False,
    pretrained_source='timm',
    pretrained_model_name=None
):
    """
    Initialize IJEPA model with optional pretrained ViT encoder.
    
    Args:
        device: Device to place model on
        patch_size: Patch size for the model
        model_name: IJEPA model name (e.g., 'vit_base', 'vit_huge')
        crop_size: Input image size
        pred_depth: Depth of the predictor
        pred_emb_dim: Embedding dimension of the predictor
        use_pretrained: Whether to use pretrained weights
        pretrained_source: Source of pretrained weights ('timm' or 'torchvision')
        pretrained_model_name: Name of pretrained model to load
    
    Returns:
        encoder, predictor: Initialized encoder and predictor
    """
    # Create IJEPA encoder
    encoder = vit.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size)
    
    # Create predictor
    predictor = vit.__dict__['vit_predictor'](
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads)

    # Initialize weights
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    # Initialize predictor with random weights
    for m in predictor.modules():
        init_weights(m)
    
    # Handle encoder initialization
    if use_pretrained and pretrained_model_name:
        logger.info(f'Loading pretrained model: {pretrained_model_name}')
        pretrained_model = load_pretrained_vit_model(
            pretrained_model_name, 
            pretrained_source,
            img_size=crop_size,
            patch_size=patch_size
        )
        
        if pretrained_model is not None:
            encoder = adapt_pretrained_model_to_ijepa(pretrained_model, encoder)
            logger.info('Successfully loaded pretrained weights for encoder')
        else:
            logger.warning('Failed to load pretrained model, using random initialization')
            for m in encoder.modules():
                init_weights(m)
    else:
        logger.info('Using random initialization for encoder')
        for m in encoder.modules():
            init_weights(m)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    return encoder, predictor


def init_opt(
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.25
):
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler
