#!/usr/bin/env python3
"""
FlexTok Fine-tuning Training Script

Full fine-tuning script for FlexTok on custom datasets like CelebA-HQ.
Includes flow matching loss, wandb logging, checkpointing, and visualization.

Usage:
    python train_flextok.py --config configs/train_celebahq.yaml
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import math
import time
from omegaconf import DictConfig, OmegaConf
import hydra

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.utils.checkpoint import checkpoint

# Import FlexTok components
from flextok import FlexTokFromHub, model
from flextok.utils.dataloader import create_celeb_dataloader
from flextok.utils.demo import denormalize, batch_to_pil
from flextok.regularizers.quantize_fsq import FSQ
from flextok.model.postprocessors.heads import LinearHead
from flextok.model.preprocessors.linear import LinearLayer

import warnings
warnings.filterwarnings("ignore") 

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not installed. Run 'pip install wandb' for experiment tracking.")

class NonReentrantCheckpointWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self._checkpoint_wrapped_module = module

    def forward(self, *args, **kwargs):
        return checkpoint(
            self._checkpoint_wrapped_module,
            *args,
            use_reentrant=False,  # Critical: allows non-deterministic ops
            **kwargs
        )
                            
class FlexTokTrainer:
    """
    Trainer class for fine-tuning FlexTok with flow matching.

    Args:
        model: FlexTokFromHub model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dictionary
        device: Device to train on (cuda/cpu)
    """

    def __init__(
        self,
        config: DictConfig,
        model: FlexTokFromHub,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Gradient accumulation
        self.grad_accum_steps = config.get('gradient_accumulation_steps', 1)

        # Gradient clipping
        self.max_grad_norm = config.get('max_grad_norm', 1.0)

        # Setup mixed precision training
        self.use_amp = config.get('use_amp', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp and device.type == 'cuda' else None

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup learning rate scheduler (needs grad_accum_steps to be defined)
        self.scheduler = self._create_scheduler()

        # Checkpointing (set to today's date by default)
        self.checkpoint_dir = Path(f'./checkpoints/{config.get("wandb_run_name", "default")}/{time.strftime("%Y%m%d")}')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = config.get('save_every', 1)

        # Wandb logging
        self.use_wandb = config.get('use_wandb', True) and WANDB_AVAILABLE
        self.log_every = config.get('log_every', 10)
        self.vis_every = config.get('visualize_every', 100)

        # EMA (optional)
        self.use_ema = config.get('use_ema', False)
        if self.use_ema:
            self.ema_model = self._create_ema_model()
            self.ema_decay = config.get('ema_decay', 0.9999)

        # REPA Loss (optional)
        self.use_repa = config.get('use_repa', False)
        self.repa_weight = config.get('repa_weight', 1.0)
        if self.use_repa:
            from flextok.model.utils.repa_loss import REPAModule
            from flextok.model.trunks.transformers import FlexTransformer

            # Get decoder dimension from model
            transformer: FlexTransformer = self.model.decoder.module_dict['dec_transformer']
            decoder_dim = transformer.blocks[0].dim if transformer is not None else None

            # Initialize REPA module with frozen encoder
            self.repa_module = REPAModule(
                features_read_key=transformer.intermediate_layer_write_key,  # Key to read decoder features
                images_read_key=self.model.vae.images_read_key,  # Key to read target images
                write_key='repa_projected_features',       # Key to write projected features
                decoder_dim=decoder_dim,
                encoder_type=config.get('repa_encoder_type', 'dinov2_vitl14'),
                encoder_dim=config.get('repa_encoder_dim', 1024),
                target_size=tuple(config.get('repa_target_size', [37, 37])),
            ).to(device)

            print(f"\nInitialized REPA module:")
            print(f"  Encoder: {config.get('repa_encoder_type', 'dinov2_vitl14')}")
            print(f"  Decoder dim: {decoder_dim}")
            print(f"  Encoder dim: {config.get('repa_encoder_dim', 1024)}")
            print(f"  Target size: {config.get('repa_target_size', [37, 37])}")
            print(f"  REPA weight: {self.repa_weight}")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for training."""
        config = self.config

        # Separate parameters: encoder, decoder, and optionally VAE
        train_vae = config.get('train_vae', False)

        param_groups = []

        # Encoder parameters
        if config.get('train_encoder', True):
            param_groups.append({
                'params': self.model.encoder.parameters(),
                'lr': config.get('encoder_lr', config.get('learning_rate', 1e-4)),
                'name': 'encoder'
            })

        # Decoder parameters
        if config.get('train_decoder', True):
            param_groups.append({
                'params': self.model.decoder.parameters(),
                'lr': config.get('decoder_lr', config.get('learning_rate', 1e-4)),
                'name': 'decoder'
            })

        # VAE parameters (optional)
        if train_vae:
            param_groups.append({
                'params': self.model.vae.parameters(),
                'lr': config.get('vae_lr', config.get('learning_rate', 1e-5)),
                'name': 'vae'
            })
        else:
            # Freeze VAE
            for param in self.model.vae.parameters():
                param.requires_grad = False

        # Regularizer is typically kept frozen
        for param in self.model.regularizer.parameters():
            param.requires_grad = False

        # REPA projector parameters (optional)
        if hasattr(self, 'repa_module') and self.use_repa:
            param_groups.append({
                'params': self.repa_module.projector.parameters(),
                'lr': config.get('repa_lr', config.get('learning_rate', 1e-4)),
                'name': 'repa_projector'
            })

        # Create optimizer
        optimizer_type = config.get('optimizer', 'adamw').lower()
        if optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=config.get('learning_rate', 1e-4),
                betas=config.get('betas', (0.9, 0.999)),
                weight_decay=config.get('weight_decay', 0.01),
                eps=config.get('eps', 1e-8),
            )
        elif optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                param_groups,
                lr=config.get('learning_rate', 1e-4),
                betas=config.get('betas', (0.9, 0.999)),
                weight_decay=config.get('weight_decay', 0.0),
                eps=config.get('eps', 1e-8),
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        return optimizer

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_type = self.config.get('scheduler', 'cosine').lower()

        if scheduler_type == 'none':
            return None

        num_epochs = self.config.get('num_epochs', 50)
        num_training_steps = len(self.train_loader) * num_epochs // self.grad_accum_steps
        num_warmup_steps = self.config.get('warmup_steps', 0)

        if scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps - num_warmup_steps,
                eta_min=self.config.get('min_lr', 1e-6),
            )
        elif scheduler_type == 'linear':
            scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=num_training_steps,
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")

        # Wrap with warmup if needed
        if num_warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=num_warmup_steps,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[num_warmup_steps],
            )

        return scheduler

    def _create_ema_model(self):
        """Create EMA model for stabilized inference."""
        from copy import deepcopy
        ema_model = deepcopy(self.model)
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model

    def _update_ema(self):
        """Update EMA model parameters."""
        if not self.use_ema:
            return

        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def compute_loss(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute flow matching loss + REPA loss for a batch of images.

        Args:
            batch: Batch of images, shape (B, 3, H, W), normalized to [-1, 1]

        Returns:
            Dictionary containing loss and metrics
        """
        # split into list of single-image tensors
        images_list = batch.split(1)

        data_dict = {self.model.vae.images_read_key: images_list}

        # Forward pass: encode -> add noise -> decode
        data_dict = self.model(data_dict)

        # Extract predictions and targets
        # see huggingface safetensors for keys but I'm hardcoding for now
        # The decoder outputs reconstructed latents in 'vae_latents_reconst'
        # The target is the original clean latents in 'vae_latents'
        pred_latents_list = data_dict['vae_latents_reconst']
        clean_latents_list = data_dict['vae_latents']
        noise = data_dict[self.model.flow_matching_noise_module.noise_write_key]
        flow_matching_target = [noise[i] - clean_latents_list[i] for i in range(len(clean_latents_list))]

        # Compute MSE loss between predicted and clean latents (Flow Matching loss)
        losses = []
        for pred, target in zip(pred_latents_list, flow_matching_target):
            losses.append(F.mse_loss(pred, target))

        flow_loss = torch.stack(losses).mean()
        loss_std = torch.stack(losses).std() if len(losses) > 1 else torch.tensor(0.0)

        # Compute REPA loss
        repa_loss = self.compute_repa_loss(data_dict, batch)

        # Combined loss
        loss = flow_loss + self.repa_weight * repa_loss

        # Additional metrics for debugging
        with torch.no_grad():
            pred_mean = torch.stack([p.mean() for p in pred_latents_list]).mean()
            pred_std = torch.stack([p.std() for p in pred_latents_list]).mean()
            target_mean = torch.stack([t.mean() for t in clean_latents_list]).mean()
            target_std = torch.stack([t.std() for t in clean_latents_list]).mean()

        del data_dict
        del pred_latents_list
        del clean_latents_list
        del noise
        del flow_matching_target
        del losses
        torch.cuda.empty_cache()

        metrics = {
            'loss': loss,
            'flow_loss': flow_loss,
            'repa_loss': repa_loss,
            'loss_std': loss_std,
            'pred_mean': pred_mean,
            'pred_std': pred_std,
            'target_mean': target_mean,
            'target_std': target_std,
        }

        return metrics

    def compute_repa_loss(self, data_dict: Dict[str, torch.Tensor], original_images: torch.Tensor) -> torch.Tensor:
        """
        Compute REPA loss for a batch of images.

        Args:
            data_dict: Dictionary containing data after forward pass
            original_images: Original input images (B, 3, H, W)
        Returns:
            REPA loss scalar
        """
        if not hasattr(self, 'repa_module') or not self.use_repa:
            return torch.tensor(0.0, device=original_images.device)

        # Extract intermediate layer features from decoder layer 1
        intermediate_features = data_dict.get('dec_packed_seq_repa_layer')

        if intermediate_features is None:
            return torch.tensor(0.0, device=original_images.device)

        # Unpack the features to spatial format using the dec_repa_unpacker
        # The unpacker needs the packing shapes from the seq_packer
        unpacker = self.model.decoder.module_dict['dec_repa_unpacker']

        unpack_dict = {
            unpacker.packed_seq_read_key: intermediate_features,
            unpacker.inner_packed_shapes_read_key: data_dict['dec_ps_inner'],
            unpacker.outer_packed_shapes_read_key: data_dict['dec_ps_outer'],
        }
        unpack_dict = unpacker(unpack_dict)

        # Get the unpacked patches (first element of inner_seq_write_keys is 'dec_patches_repa_layer')
        # This contains the spatial patch features, not the register tokens
        decoder_features = unpack_dict[unpacker.inner_seq_write_keys[0]]

        # Prepare data_dict for REPA module
        repa_data_dict = {
            self.repa_module.features_read_key: decoder_features,
            self.repa_module.images_read_key: original_images,
        }

        # Compute REPA loss
        repa_loss = self.repa_module(repa_data_dict)

        return repa_loss

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_losses = []
        epoch_start = time.time()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.get('num_epochs', 50)}")

        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)

            # Forward pass with mixed precision
            if self.use_amp and self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    metrics = self.compute_loss(batch)
                    loss = metrics['loss'] / self.grad_accum_steps

                # Backward pass
                self.scaler.scale(loss).backward()
            else:
                metrics = self.compute_loss(batch)
                loss = metrics['loss'] / self.grad_accum_steps
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # Gradient clipping
                if self.max_grad_norm > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()

                # Update EMA
                self._update_ema()

                # Logging
                if self.global_step % self.log_every == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']

                    log_dict = {
                        'train/loss': metrics['loss'].item(),
                        'train/flow_loss': metrics['flow_loss'].item(),
                        'train/repa_loss': metrics['repa_loss'].item() if isinstance(metrics['repa_loss'], torch.Tensor) else metrics['repa_loss'],
                        'train/loss_std': metrics['loss_std'].item(),
                        'train/lr': current_lr,
                        'train/epoch': epoch,
                        'train/step': self.global_step,
                        # Additional debugging metrics
                        'debug/pred_mean': metrics['pred_mean'].item(),
                        'debug/pred_std': metrics['pred_std'].item(),
                        'debug/target_mean': metrics['target_mean'].item(),
                        'debug/target_std': metrics['target_std'].item(),
                    }

                    if self.use_wandb:
                        wandb.log(log_dict, step=self.global_step)

                    # Update progress bar with both flow and REPA loss if available
                    postfix_dict = {
                        'loss': f"{metrics['loss'].item():.4f}",
                        'flow': f"{metrics['flow_loss'].item():.4f}",
                        'lr': f"{current_lr:.2e}",
                    }
                    if self.use_repa and isinstance(metrics['repa_loss'], torch.Tensor):
                        postfix_dict['repa'] = f"{metrics['repa_loss'].item():.4f}"

                    pbar.set_postfix(postfix_dict)

                # Visualization
                if self.global_step % self.vis_every == 0 and self.use_wandb:
                    vis_batch = batch[:4].clone().detach()
                    self.visualize_reconstructions(vis_batch)

                epoch_losses.append(metrics['loss'].item())
                self.global_step += 1

        epoch_time = time.time() - epoch_start
        avg_loss = sum(epoch_losses) / len(epoch_losses)

        return {
            'loss': avg_loss,
            'time': epoch_time,
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        # Clear cache before validation
        torch.cuda.empty_cache()

        val_losses = []

        # Optionally limit validation batches to save memory
        max_val_batches = self.config.get('max_val_batches', None)

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validating")):
            if max_val_batches and batch_idx >= max_val_batches:
                break

            batch = batch.to(self.device)

            if self.use_amp and self.device.type == 'cuda':
                with torch.amp.autocast("cuda"):
                    metrics = self.compute_loss(batch)
            else:
                metrics = self.compute_loss(batch)

            val_losses.append(metrics['loss'].item())

            # Delete batch to free memory
            del batch
            del metrics

            # Clear cache every few batches
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0

        # Log validation metrics
        if self.use_wandb:
            wandb.log({
                'val/loss': avg_val_loss,
                'val/epoch': epoch,
            }, step=self.global_step)

        # Clear cache after validation
        torch.cuda.empty_cache()

        return {
            'loss': avg_val_loss,
        }

    @torch.no_grad()
    def visualize_reconstructions(self, batch: torch.Tensor, num_images: int = 4):
        """
        Visualize model reconstructions and log to wandb.

        Creates two types of visualizations:
        1. Full 256-token reconstruction
        2. Flexible-length reconstructions (1, 4, 16, 64, 256 tokens)

        Args:
            batch: Batch of images to reconstruct
            num_images: Number of images to visualize
        """
        if not self.use_wandb:
            return

        self.model.eval()

        batch = batch[:num_images].to(self.device)
        images_list = batch.split(1)

        # Clear GPU cache before visualization to free memory
        torch.cuda.empty_cache()

        # === Get VAE reconstruction (encode + decode without FlexTok) ===
        with torch.no_grad():
            # pass in data_dict instead of images_list to avoid modifying model state
            data_dict = {self.model.vae.images_read_key: images_list}
            data_dict = self.model.vae.encode(data_dict)
            data_dict[self.model.vae.vae_latents_read_key] = data_dict[self.model.vae.vae_latents_write_key]
            data_dict = self.model.vae.decode(data_dict)
            vae_recons = torch.cat(data_dict[self.model.vae.images_reconst_write_key], dim=0).cpu()

        # === Flexible-length reconstructions ===
        # Tokenize images once
        if self.use_amp and self.device.type == 'cuda':
            with torch.amp.autocast("cuda"):
                tokens_list = self.model.tokenize(batch)
        else:
            tokens_list = self.model.tokenize(batch)

        # Token counts to visualize
        k_keep_list = self.config.get('vis_token_counts', [1, 4, 16, 64, 256])

        # Detokenize with different token counts
        flex_reconstructions = {}
        for k_keep in k_keep_list:
            # Truncate token sequences
            subseq_list = [seq[:, :k_keep].clone() for seq in tokens_list]

            # Detokenize
            if self.use_amp and self.device.type == 'cuda':
                with torch.amp.autocast("cuda"):
                    reconst_imgs = self.model.detokenize(
                        subseq_list,
                        timesteps=self.config.get('inference_steps', 20),
                        guidance_scale=self.config.get('guidance_scale', 7.5),
                        perform_norm_guidance=self.config.get('perform_norm_guidance', True),
                        verbose=False,
                    )
            else:
                reconst_imgs = self.model.detokenize(
                    subseq_list,
                    timesteps=self.config.get('inference_steps', 20),
                    guidance_scale=self.config.get('guidance_scale', 7.5),
                    perform_norm_guidance=self.config.get('perform_norm_guidance', True),
                    verbose=False,
                )

            # Move to CPU immediately to free GPU memory
            flex_reconstructions[k_keep] = reconst_imgs.cpu()

            # Clear cache after each detokenization
            torch.cuda.empty_cache()

        # Create grid visualization
        import matplotlib.pyplot as plt
        import matplotlib
        import torchvision.transforms.functional as TF
        matplotlib.use('Agg')  # Non-interactive backend

        nrows = num_images
        ncols = len(k_keep_list) + 2  # VAE + k_keep_list + original

        fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
        if nrows == 1:
            axes = axes.reshape(1, -1)

        # Move batch to CPU for visualization
        batch = batch.cpu()

        # Denormalize images for display (like in the notebook)
        batch_display = denormalize(batch).clamp(0, 1)
        vae_recons_display = denormalize(vae_recons).clamp(0, 1)
        flex_recons_display = {k: denormalize(v).clamp(0, 1) for k, v in flex_reconstructions.items()}

        for img_idx in range(nrows):
            # Column 0: VAE reconstruction
            img_pil = TF.to_pil_image(vae_recons_display[img_idx])
            axes[img_idx, 0].imshow(img_pil)
            axes[img_idx, 0].axis('off')
            if img_idx == 0:
                axes[img_idx, 0].set_title('VAE', fontsize=12)

            # Columns 1 to len(k_keep_list): FlexTok reconstructions
            for col_idx, k_keep in enumerate(k_keep_list):
                img_pil = TF.to_pil_image(flex_recons_display[k_keep][img_idx])
                axes[img_idx, col_idx + 1].imshow(img_pil)
                axes[img_idx, col_idx + 1].axis('off')
                if img_idx == 0:
                    token_word = 'token' if k_keep == 1 else 'tokens'
                    axes[img_idx, col_idx + 1].set_title(f'{k_keep} {token_word}', fontsize=12)

            # Last column: Original
            img_pil = TF.to_pil_image(batch_display[img_idx])
            axes[img_idx, -1].imshow(img_pil)
            axes[img_idx, -1].axis('off')
            if img_idx == 0:
                axes[img_idx, -1].set_title('Original', fontsize=12)

        plt.tight_layout()

        # Log flexible-length visualization
        wandb.log({
            'visualizations/flexible_length': wandb.Image(fig, caption="VAE | Flexible-length | Original"),
        }, step=self.global_step)

        plt.close(fig)

        self.model.train()

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        if self.use_ema:
            checkpoint['ema_model_state_dict'] = self.ema_model.state_dict()

        if self.use_repa and hasattr(self, 'repa_module'):
            checkpoint['repa_module_state_dict'] = self.repa_module.state_dict()

        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, checkpoint_path)
        # artifact = wandb.Artifact('latest', type='model')
        # artifact.add_file(checkpoint_path)
        # wandb.log_artifact(artifact)
        print(f"Saved checkpoint to {checkpoint_path}")

        # Save epoch checkpoint (DON'T DO RN TO SAVE SPACE)
        # if epoch % self.save_every == 0:
        #     epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pt'
        #     torch.save(checkpoint, epoch_path)
        #     # artifact = wandb.Artifact(f'epoch_{epoch:04d}', type='model')
        #     # artifact.add_file(epoch_path)
        #     # wandb.log_artifact(artifact)
        #     print(f"Saved epoch checkpoint to {epoch_path}")

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
            # artifact = wandb.Artifact('latest', type='model')
            # artifact.add_file(best_path)
            # wandb.log_artifact(artifact)
            print(f"Saved best checkpoint to {best_path}")

            if self.use_wandb:
                wandb.run.summary['best_val_loss'] = self.best_val_loss
                wandb.run.summary['best_epoch'] = epoch

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if self.use_ema and checkpoint.get('ema_model_state_dict'):
            self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])

        if self.use_repa and hasattr(self, 'repa_module') and checkpoint.get('repa_module_state_dict'):
            self.repa_module.load_state_dict(checkpoint['repa_module_state_dict'])
            print("Loaded REPA module state from checkpoint")

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")

    def train(self):
        """Main training loop."""
        num_epochs = self.config.get('num_epochs', 50)
        start_epoch = self.current_epoch + 1

        print(f"\nStarting training from epoch {start_epoch} to {num_epochs}")
        print(f"Total training steps: {len(self.train_loader) * num_epochs // self.grad_accum_steps}")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Gradient accumulation steps: {self.grad_accum_steps}")
        print(f"Wandb logging: {self.use_wandb}\n")

        for epoch in range(start_epoch, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            torch.cuda.empty_cache()
            print(f"\nEpoch {epoch}/{num_epochs} - "
                  f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Time: {train_metrics['time']:.2f}s")

            # Validate
            val_metrics = self.validate(epoch)
            print(f"Validation Loss: {val_metrics['loss']:.4f}")

            # Check if best model
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                print(f"New best validation loss: {self.best_val_loss:.4f}")

            # Save checkpoint
            self.save_checkpoint(epoch, is_best=is_best)

            self.current_epoch = epoch

        print("\nTraining completed!")
        if self.use_wandb:
            wandb.finish()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


@hydra.main(version_base=None, config_path="configs/", config_name="train_celebahq")
def main(cfg: DictConfig):
    # Load config
    config = cfg

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    print("Loading FlexTok model...")
    model_name = config.get('model_name', 'EPFL-VILAB/flextok_d18_d28_dfn')
    model = FlexTokFromHub.from_pretrained(model_name)
    print(f"Loaded model: {model_name}")
    if config.get("train_from_scratch", True):
        print("  Training from scratch: reinitializing model weights")

        # Custom initialization function that uses standard PyTorch initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.elementwise_affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        # Apply standard initialization to encoder and decoder
        model.encoder.apply(init_weights)
        model.decoder.apply(init_weights)
        print("  Applied standard PyTorch initialization to encoder and decoder")

    # Handle gradient checkpointing configuration
    gradient_checkpointing = config.get('gradient_checkpointing', True)

    if not gradient_checkpointing:
        print("\nDisabling gradient checkpointing...")

        def unwrap_checkpoint_blocks(transformer_module):
            """Unwrap checkpoint-wrapped blocks in a transformer."""
            if not hasattr(transformer_module, 'blocks'):
                return False

            unwrapped_blocks = []
            for block in transformer_module.blocks:
                # Check if block is wrapped with checkpoint
                if hasattr(block, '_checkpoint_wrapped_module'):
                    # Unwrap the checkpoint wrapper
                    unwrapped_blocks.append(block._checkpoint_wrapped_module)
                else:
                    unwrapped_blocks.append(block)

            # Replace the wrapped blocks with unwrapped ones
            import torch.nn as nn
            transformer_module.blocks = nn.ModuleList(unwrapped_blocks)
            transformer_module.use_act_checkpoint = False
            return True

        # Encoder transformer
        if hasattr(model.encoder, 'module_dict') and 'enc_transformer' in model.encoder.module_dict:
            enc_transformer = model.encoder.module_dict['enc_transformer']
            if unwrap_checkpoint_blocks(enc_transformer):
                print("  Unwrapped checkpointed blocks in encoder transformer")

        # Decoder transformer
        if hasattr(model.decoder, 'module_dict') and 'dec_transformer' in model.decoder.module_dict:
            dec_transformer = model.decoder.module_dict['dec_transformer']
            if unwrap_checkpoint_blocks(dec_transformer):
                print("  Unwrapped checkpointed blocks in decoder transformer")

    else:
        # Fix gradient checkpointing to use use_reentrant=False for compatibility with FlexTok
        # This is necessary because FlexTok's flexible token packing can produce different
        # tensor shapes during forward/backward, which breaks the old reentrant checkpointing
        print("\nFixing gradient checkpointing for FlexTok compatibility...")

        def rewrap_checkpoint_blocks_non_reentrant(transformer_module):
            """Rewrap checkpoint blocks to use use_reentrant=False."""
            if not hasattr(transformer_module, 'blocks'):
                return False

            rewrapped_blocks = []
            for block in transformer_module.blocks:
                # Check if block is wrapped with checkpoint
                if hasattr(block, '_checkpoint_wrapped_module'):
                    # Get the unwrapped module
                    unwrapped = block._checkpoint_wrapped_module
                    # Rewrap with use_reentrant=False
                    rewrapped_blocks.append(NonReentrantCheckpointWrapper(unwrapped))
                else:
                    # Not wrapped, just keep as is
                    rewrapped_blocks.append(block)

            # Replace blocks with rewrapped versions
            import torch.nn as nn
            transformer_module.blocks = nn.ModuleList(rewrapped_blocks)
            return True

        # Fix encoder transformer
        if hasattr(model.encoder, 'module_dict') and 'enc_transformer' in model.encoder.module_dict:
            enc_transformer = model.encoder.module_dict['enc_transformer']
            if rewrap_checkpoint_blocks_non_reentrant(enc_transformer):
                print("  Fixed encoder transformer checkpointing (use_reentrant=False)")

        # Fix decoder transformer
        if hasattr(model.decoder, 'module_dict') and 'dec_transformer' in model.decoder.module_dict:
            dec_transformer = model.decoder.module_dict['dec_transformer']
            if rewrap_checkpoint_blocks_non_reentrant(dec_transformer):
                print("  Fixed decoder transformer checkpointing (use_reentrant=False)")

    # Optionally modify FSQ levels (e.g., to use binary quantization)
    if config.get('fsq_levels', None) is not None:
        from flextok.regularizers.quantize_fsq import FSQ
        new_levels = config['fsq_levels']
        print(f"\nModifying FSQ levels from {model.regularizer._levels.tolist()} to {new_levels}")

        # Get the original FSQ configuration
        old_fsq: FSQ = model.regularizer

        # Check if the encoder output dimension matches the fsq levels length
        old_fsq_output_dim = old_fsq.dim
        if old_fsq_output_dim != len(new_levels):
            # project encoder dim to new fsq dim
            print(f"Adjusting encoder output dimension from {old_fsq_output_dim} to {len(new_levels)}")
            new_enc_linear_head = LinearHead(
                read_key=model.encoder.module_dict["enc_to_latents"].read_key,
                write_key=model.encoder.module_dict["enc_to_latents"].write_key,
                dim=model.encoder.module_dict["enc_to_latents"].dim_in,
                dim_out=len(new_levels),
                use_mup_readout=False,
            )

            model.encoder.module_dict['enc_to_latents'] = new_enc_linear_head

            # project back from fsq to decoder input dim
            print(f"Adjusting decoder input dimension to {len(new_levels)}")
            new_dec_linear_head = LinearLayer(
                read_key=model.decoder.module_dict["dec_from_latents"].read_key,
                write_key=model.decoder.module_dict["dec_from_latents"].write_key,
                dim_in=len(new_levels),
                dim=model.decoder.module_dict["dec_from_latents"].dim_out,
            )
            model.decoder.module_dict['dec_from_latents'] = new_dec_linear_head

        # Create new FSQ with modified levels
        new_fsq = FSQ(
            latents_read_key=old_fsq.latents_read_key,
            quants_write_key=old_fsq.quants_write_key,
            tokens_write_key=old_fsq.tokens_write_key,
            levels=new_levels,
            drop_quant_p=old_fsq.drop_quant_p,
            corrupt_tokens_p=old_fsq.corrupt_tokens_p,
            min_corrupt_tokens_p=old_fsq.min_corrupt_tokens_p,
            apply_corrupt_tokens_p=old_fsq.apply_corrupt_tokens_p,
            packed_call=old_fsq.packed_call,
        )

        # Replace the FSQ module
        model.regularizer = new_fsq

        print(f"New FSQ configuration: {new_fsq}")
        print(f"  Codebook size: {new_fsq.codebook_size} (was {old_fsq.codebook_size})")
        print(f"  Dimensions: {new_fsq.dim} (was {old_fsq.dim})")

    if config.get('use_attention_mask', True):
        print("\nUsing attention mask instead of learned mask...")
        from flextok.model.preprocessors.attention_masked_nested_dropout import AttentionMaskedNestedDropout
        from flextok.model.preprocessors.nested_dropout_seq_packer import NestedDropoutSequencePacker
        from flextok.model.preprocessors.token_dropout import MaskedNestedDropout
        from flextok.model.preprocessors.flex_seq_packing import BlockWiseSequencePacker
        old_dropout: MaskedNestedDropout = model.decoder.module_dict['dec_nested_dropout']
        if old_dropout is None:
            raise ValueError("Model does not have 'dec_nested_dropout' module to replace")
        old_seq_packer: BlockWiseSequencePacker = model.decoder.module_dict['dec_seq_packer']
        if old_seq_packer is None:
            raise ValueError("Model does not have 'dec_seq_packer' module to replace")
        new_dropout = AttentionMaskedNestedDropout(
            read_write_key=old_dropout.read_write_key,
        )
        new_masking = NestedDropoutSequencePacker(
            input_list_read_keys=old_seq_packer.input_list_read_keys,
            packed_seq_write_key=old_seq_packer.packed_seq_write_key,
            inner_packed_shapes_write_key=old_seq_packer.inner_packed_shapes_write_key,
            outer_packed_shapes_write_key=old_seq_packer.outer_packed_shapes_write_key,
            max_seq_len=old_seq_packer.max_seq_len,
            block_mask_write_key=old_seq_packer.block_mask_write_key,
            emb_packing_fn_write_key=old_seq_packer.emb_packing_fn_write_key,
            pad_to_multiple=old_seq_packer.pad_to_multiple,
            compile_block_mask=old_seq_packer.compile_block_mask,
            return_materialized_mask=old_seq_packer.return_materialized_mask,
            per_subseq_embs=old_seq_packer.per_subseq_embs,
        )
        
        model.decoder.module_dict['dec_nested_dropout'] = new_dropout
        model.decoder.module_dict['dec_seq_packer'] = new_masking
        print("  Replaced nested dropout and sequence packer modules in decoder")

    # Create dataloaders
    print("\nCreating dataloaders...")

    # Data augmentation
    train_transforms = None
    if config.get('use_augmentation', True):
        train_transforms = transforms.RandomHorizontalFlip(p=0.5)

    dataset_name = config.get('dataset_name', 'celebahq')
    train_loader = create_celeb_dataloader(
        dataset_type=dataset_name,
        root_dir=config['data_path'],
        img_size=config.get('img_size', 256),
        batch_size=config.get('batch_size', 32),
        split='train',
        num_workers=config.get('num_workers', 4),
        transform=train_transforms,
    )

    val_loader = create_celeb_dataloader(
        dataset_type=dataset_name,
        root_dir=config['data_path'],
        img_size=config.get('img_size', 256),
        batch_size=config.get('val_batch_size', config.get('batch_size', 32)),
        split='val',
        num_workers=config.get('num_workers', 4),
        shuffle=False,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create trainer
    trainer = FlexTokTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    # Resume from checkpoint if provided
    if config.resume:
        trainer.load_checkpoint(config.resume)

    # Initialize wandb
    if config.get('use_wandb', True) and WANDB_AVAILABLE:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            tags=config.wandb_tags,
            config=OmegaConf.to_container(config, resolve=True),
            resume='allow' if config.resume else False,
        )

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
