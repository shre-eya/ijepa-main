# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import time

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import logging
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from masks import get_mask_collator
from masks.uncertainty_collator import UncertaintyGuidedCollator

from masks.multiblock import MaskCollator as MBMaskCollator
from masks.utils import apply_masks
from utils.distributed import (
    init_distributed,
    AllReduce
)
from utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from utils.tensors import repeat_interleave_batch
from datasets.imagenet1k import make_imagenet1k

from helper import (
    load_checkpoint,
    init_model,
    init_opt)
from transforms import make_transforms

import matplotlib.pyplot as plt

# --
log_timings = True
log_freq = 10
checkpoint_freq = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

# Add this constant near the top of the file
attention_log_freq = 100  # Save attention map every 100 iterations
attention_output_dir = 'output/attention_maps'  # Directory to save attention maps
variance_log_freq = 50  # Save variance statistics every 50 iterations
variance_output_dir = 'output/train_variances'  # Directory to save variance statistics

def save_attention_map(attn_map, epoch, itr, output_dir, prefix='train'):
    """Save the attention map as an image for the first image in the batch."""
    if attn_map.ndim == 3:
        attn_map = attn_map[0]  # Take first image in batch
    attn_scores = attn_map.mean(0)  # [N]
    grid_size = int(attn_scores.shape[0] ** 0.5)
    attn_img = attn_scores.reshape(grid_size, grid_size).cpu().numpy()
    plt.imshow(attn_img, cmap='viridis')
    plt.colorbar()
    plt.title(f'Attention Map {prefix} Epoch {epoch} Iter {itr}')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{prefix}_attn_epoch{epoch}_iter{itr}.png')
    plt.savefig(save_path)
    plt.close()

def save_variance_statistics(encoder, images, context_masks, epoch, itr, output_dir, prefix='train'):
    """
    Save patch-wise variance statistics from MC forward passes.
    
    Args:
        

encoder: The encoder model
        images: Batch of images [B, C, H, W]
        context_masks: Boolean masks for context regions [B, N]
        epoch: Current epoch number
        itr: Current iteration number
        output_dir: Directory to save variance statistics
        prefix: Prefix for the saved files
    """
    encoder.eval()
    with torch.no_grad():
        try:
            # Get variance statistics from MC forward passes
            _, patch_vars, _ = encoder(
                images, 
                masks=context_masks,
                return_patch_vars=True, 
                num_mc_samples=3
            )
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save variance statistics
            variance_file = os.path.join(output_dir, f'{prefix}_variances_epoch{epoch}_iter{itr}.npy')
            np.save(variance_file, patch_vars.detach().cpu().numpy())
            
            # Save summary statistics
            mean_variance = patch_vars.mean().item()
            max_variance = patch_vars.max().item()
            min_variance = patch_vars.min().item()
            
            summary_file = os.path.join(output_dir, f'{prefix}_variance_summary_epoch{epoch}_iter{itr}.txt')
            with open(summary_file, 'w') as f:
                f.write(f"Epoch: {epoch}, Iteration: {itr}\n")
                f.write(f"Mean variance: {mean_variance:.6f}\n")
                f.write(f"Max variance: {max_variance:.6f}\n")
                f.write(f"Min variance: {min_variance:.6f}\n")
                f.write(f"Variance shape: {patch_vars.shape}\n")
            
            logger.info(f'Saved variance statistics to {variance_file} (mean: {mean_variance:.4f}, max: {max_variance:.4f})')
            
        except Exception as e:
            logger.warning(f"Failed to save variance statistics: {str(e)}")
    
    encoder.train()

def save_test_variance_statistics(encoder, test_loader, epoch, output_dir, prefix='test'):
    """
    Save variance statistics for the entire test dataset.
    
    Args:
        encoder: The encoder model
        test_loader: DataLoader for test dataset
        epoch: Current epoch number
        output_dir: Directory to save variance statistics
        prefix: Prefix for the saved files
    """
    encoder.eval()
    all_variances = []
    all_mean_variances = []
    
    with torch.no_grad():
        for itr, (images, context_masks, target_masks) in enumerate(test_loader):
            try:
                # Move to device
                images = images.to(device)
                context_masks = context_masks.to(device)
                
                # Get variance statistics from MC forward passes
                _, patch_vars, _ = encoder(
                    images, 
                    masks=context_masks,
                    return_patch_vars=True, 
                    num_mc_samples=3
                )
                
                # Store variances
                all_variances.append(patch_vars.detach().cpu().numpy())
                all_mean_variances.append(patch_vars.mean().item())
                
                # Save individual batch variances
                batch_variance_file = os.path.join(output_dir, f'{prefix}_batch_variances_epoch{epoch}_batch{itr}.npy')
                np.save(batch_variance_file, patch_vars.detach().cpu().numpy())
                
            except Exception as e:
                logger.warning(f"Failed to compute variance for test batch {itr}: {str(e)}")
    
    # Save aggregated statistics
    if all_variances:
        all_variances = np.concatenate(all_variances, axis=0)
        aggregated_file = os.path.join(output_dir, f'{prefix}_aggregated_variances_epoch{epoch}.npy')
        np.save(aggregated_file, all_variances)
        
        # Save summary statistics
        mean_variance = np.mean(all_mean_variances)
        max_variance = np.max(all_mean_variances)
        min_variance = np.min(all_mean_variances)
        
        summary_file = os.path.join(output_dir, f'{prefix}_variance_summary_epoch{epoch}.txt')
        with open(summary_file, 'w') as f:
            f.write(f"Test Epoch: {epoch}\n")
            f.write(f"Number of batches: {len(all_mean_variances)}\n")
            f.write(f"Mean variance across batches: {mean_variance:.6f}\n")
            f.write(f"Max variance across batches: {max_variance:.6f}\n")
            f.write(f"Min variance across batches: {min_variance:.6f}\n")
            f.write(f"Variance shape: {all_variances.shape}\n")
        
        logger.info(f'Saved test variance statistics to {aggregated_file} (mean: {mean_variance:.4f}, max: {max_variance:.4f})')
    
    encoder.train()

def train_step(images, context_masks, target_masks):
    """
    Training step with uncertainty-based masking.
    
    Args:
        images: Batch of images [B, C, H, W]
        context_masks: Boolean masks for context regions [B, N]
        target_masks: Boolean masks for target regions [B, N]
    """
    # Move everything to device
    images = images.to(device)
    context_masks = context_masks.to(device)
    target_masks = target_masks.to(device)
    
    # Reset gradients
    optimizer.zero_grad(set_to_none=True)
    
    # Update target encoder momentum
    m = next(momentum_scheduler)
    
    # Forward pass through student encoder
    with torch.cuda.amp.autocast(dtype=torch.bfloat16 if use_bfloat16 else torch.float32):
        # Get representations for masked images (student)
        z = encoder(images, masks=context_masks)
        
        # Get target representations (no gradients needed)
        with torch.no_grad():
            h = target_encoder(images)
            h = h.detach()
        
        # Get predictions from student
        p = predictor(z, context_masks, target_masks)
        
        # Compute loss
        loss = F.smooth_l1_loss(p, h)
    
    # Backward and optimizer step with AMP scaler guard
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    
    # Update target encoder
    with torch.no_grad():
        for param_q, param_k in zip(encoder.parameters(),
                                  target_encoder.parameters()):
            param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
    
    return loss.item()

def train_epoch(epoch):
    """Training epoch with uncertainty-based masking."""
    global encoder, predictor, target_encoder, unsupervised_loader, device, optimizer, scheduler, wd_scheduler
    global attention_log_freq, attention_output_dir, variance_log_freq, variance_output_dir, log_freq
    
    encoder.train()
    predictor.train()
    target_encoder.eval()
    
    loss_meter = AverageMeter()
    time_meter = AverageMeter()
    
    for itr, (images, context_masks, target_masks) in enumerate(unsupervised_loader):
        start_time = time.time()
        
        # Update learning rate and weight decay
        # FIX: Use .step() method instead of indexing - scheduler objects are not subscriptable
        current_lr = scheduler.step()
        current_wd = wd_scheduler.step()
        
        # Note: scheduler.step() and wd_scheduler.step() already update the optimizer param groups internally
        # but the above calls also return the current values for logging if needed
        
        # Train step
        loss = train_step(images, context_masks, target_masks)
        
        # Periodically extract and save attention map
        if itr % attention_log_freq == 0:
            encoder.eval()
            with torch.no_grad():
                attn_map = encoder.get_last_layer_attention_map(images.to(device), masks=context_masks.to(device))
                save_attention_map(attn_map, epoch, itr, attention_output_dir, prefix='train')
            encoder.train()
        
        # Periodically save variance statistics
        if itr % variance_log_freq == 0:
            save_variance_statistics(encoder, images, context_masks, epoch, itr, variance_output_dir, prefix='train')
        
        # Update meters
        loss_meter.update(loss)
        time_meter.update(time.time() - start_time)
        
        # Logging
        if (itr % log_freq == 0) or (itr == len(unsupervised_loader)-1):
            logger.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                'Time: {time.val:.2f} ({time.avg:.2f})'.format(
                    epoch, itr, len(unsupervised_loader),
                    loss=loss_meter, time=time_meter))
    
    return loss_meter.avg

def main(args, resume_preempt=False):
    # Make variables global for train_epoch function

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    # -- New pretrained model parameters
    use_pretrained = args['meta'].get('use_pretrained', False)
    pretrained_source = args['meta'].get('pretrained_source', 'timm')
    pretrained_model_name = args['meta'].get('pretrained_model_name', None)
    device = torch.device('cpu')


    # -- DATA
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    # --
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    # --

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
    patch_size = args['mask']['patch_size']  # patch-size for model training
    num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
    min_keep = args['mask']['min_keep']  # min number of patches in context block
    enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
    num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
    pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks
    aspect_ratio = args['mask']['aspect_ratio']  # aspect ratio of target blocks
    # --

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']

    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))

    # -- init model
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
        use_pretrained=use_pretrained,
        pretrained_source=pretrained_source,
        pretrained_model_name=pretrained_model_name)
    target_encoder = copy.deepcopy(encoder)
    

    # === expose key objects to module globals so helper functions can use them ===
    # train_epoch and a few other functions use encoder, predictor, target_encoder, device etc.
    # These were created in the local scope; make them available at module level.
    globals().update({
        'encoder': encoder,
        'predictor': predictor,
        'target_encoder': target_encoder,
        'device': device,
    })
    # If you want to expose more objects later (optimizer, scheduler...), add them here similarly.

    # Initialize uncertainty-guided collator
    collator = UncertaintyGuidedCollator(
        student_model_instance=encoder,
        input_size=(crop_size, crop_size),
        patch_size=patch_size,
        n_targets=num_pred_masks,
        n_contexts=num_enc_masks,
        context_mask_scale=enc_mask_scale,
        target_mask_scale=pred_mask_scale,
        num_mc_samples=3,
        select_uncertain_ratio=0.75,
        target_overlaps_context=allow_overlap
    )


    # -- make data transforms
    mask_collator = collator


    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter)

    # -- init data-loaders/samplers
    _, unsupervised_loader, unsupervised_sampler = make_imagenet1k(
            transform=transform,
            batch_size=batch_size,
            collator=mask_collator,
            pin_mem=pin_mem,
            training=True,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            copy_data=copy_data,
            drop_last=True)
    globals().update({
        'unsupervised_loader': unsupervised_loader,
    })

    ipe = len(unsupervised_loader)

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)
    # Ensure scaler exists even when CUDA/AMP is unavailable
    # On CPU-only runs, set scaler=None and fall back to .backward()
    if ('scaler' not in locals()) or (scaler is None):
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    globals().update({
        'optimizer': optimizer,
        'scheduler': scheduler,
        'wd_scheduler': wd_scheduler,
        'scaler': scaler,
        'use_bfloat16': use_bfloat16,
    })
    # Set default values for missing variables
    attention_log_freq = 100
    attention_output_dir = './output/attention_maps'
    variance_log_freq = 50
    variance_output_dir = './output/variance_stats'
    log_freq = 10

    encoder = encoder.to(device)
    predictor = predictor.to(device)
    target_encoder = target_encoder.to(device)




    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule (cosine schedule from ema[0] -> ema[1])
    total_iters = int(ipe * num_epochs * ipe_scale)
    def _cosine_schedule_iter(base_value, final_value, total_steps):
        for i in range(total_steps + 1):
            progress = i / total_steps if total_steps > 0 else 1.0
            value = float(final_value + (base_value - final_value) * 0.5 * (1. + np.cos(np.pi * progress)))
            yield value
    momentum_scheduler = iter(_cosine_schedule_iter(ema[0], ema[1], total_iters))

    globals().update({
      'momentum_scheduler': momentum_scheduler,
    })

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    def save_checkpoint(epoch, loss_avg):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Set epoch for data loader sampler
        unsupervised_sampler.set_epoch(epoch)
        
        # Train for one epoch
        loss = train_epoch(epoch)
        
        # Save checkpoint
        if rank == 0 and (epoch % checkpoint_freq == 0 or epoch == num_epochs-1):
            save_checkpoint(epoch, loss)
        
        # Synchronize across processes (only if distributed is initialized)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
    
    # Final checkpoint
    if rank == 0:
        save_checkpoint(epoch=num_epochs, loss_avg=loss)
        logger.info('Training finished.')


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train I-JEPA model')
    parser.add_argument('--config', type=str, default='configs/in1k_vith14_ep300.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # Load config file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    main(config, resume_preempt=args.resume)
