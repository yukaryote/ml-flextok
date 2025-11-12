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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Import FlexTok components
from flextok import FlexTokFromHub, model
from flextok.utils.dataloader import create_celebahq_dataloader
from flextok.utils.demo import denormalize, batch_to_pil

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not installed. Run 'pip install wandb' for experiment tracking.")


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
        model: FlexTokFromHub,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
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

        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
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
        Compute flow matching loss for a batch of images.

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

        # Compute MSE loss between predicted and clean latents
        losses = []
        for pred, target in zip(pred_latents_list, flow_matching_target):
            losses.append(F.mse_loss(pred, target))

        loss = torch.stack(losses).mean()

        # Additional metrics for debugging
        with torch.no_grad():
            pred_mean = torch.stack([p.mean() for p in pred_latents_list]).mean()
            pred_std = torch.stack([p.std() for p in pred_latents_list]).mean()
            target_mean = torch.stack([t.mean() for t in clean_latents_list]).mean()
            target_std = torch.stack([t.std() for t in clean_latents_list]).mean()

        metrics = {
            'loss': loss,
            'loss_std': torch.stack(losses).std() if len(losses) > 1 else torch.tensor(0.0),
            'pred_mean': pred_mean,
            'pred_std': pred_std,
            'target_mean': target_mean,
            'target_std': target_std,
        }

        return metrics

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

                    pbar.set_postfix({
                        'loss': f"{metrics['loss'].item():.4f}",
                        'lr': f"{current_lr:.2e}",
                    })

                # Visualization
                if self.global_step % self.vis_every == 0 and self.use_wandb:
                    self.visualize_reconstructions(batch[:4])
                    
                self.global_step += 1
            epoch_losses.append(metrics['loss'].item())

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

        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        # Save epoch checkpoint
        if epoch % self.save_every == 0:
            epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pt'
            torch.save(checkpoint, epoch_path)
            print(f"Saved epoch checkpoint to {epoch_path}")

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
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

        # Validate before starting to test oom
        val_metrics = self.validate(0)
        print(f"Validation Loss: {val_metrics['loss']:.4f}")
        torch.cuda.empty_cache()

        for epoch in range(start_epoch, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)

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


def main():
    parser = argparse.ArgumentParser(description="Fine-tune FlexTok on custom datasets")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')

    # Wandb options
    parser.add_argument('--wandb-project', type=str, default=None, help='Wandb project name')
    parser.add_argument('--wandb-name', type=str, default=None, help='Wandb run name')
    parser.add_argument('--wandb-tags', type=str, nargs='+', default=None, help='Wandb tags (space-separated)')

    # Training options
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with CLI args
    if args.no_wandb:
        config['use_wandb'] = False
    if args.wandb_project:
        config['wandb_project'] = args.wandb_project
    if args.wandb_name:
        config['wandb_run_name'] = args.wandb_name
    if args.wandb_tags:
        config['wandb_tags'] = args.wandb_tags
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr
    if args.epochs:
        config['num_epochs'] = args.epochs

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize wandb
    if config.get('use_wandb', True) and WANDB_AVAILABLE:
        wandb.init(
            project=config.get('wandb_project', 'flextok-finetuning'),
            name=config.get('wandb_run_name', None),
            tags=config.get('wandb_tags', None),
            config=config,
            resume='allow' if args.resume else False,
        )

    # Load model
    print("Loading FlexTok model...")
    model_name = config.get('model_name', 'EPFL-VILAB/flextok_d18_d28_dfn')
    model = FlexTokFromHub.from_pretrained(model_name)
    print(f"Loaded model: {model_name}")

    # Create dataloaders
    print("\nCreating dataloaders...")

    # Data augmentation
    train_transforms = None
    if config.get('use_augmentation', True):
        train_transforms = transforms.RandomHorizontalFlip(p=0.5)

    train_loader = create_celebahq_dataloader(
        root_dir=config['data_path'],
        img_size=config.get('img_size', 256),
        batch_size=config.get('batch_size', 32),
        split='train',
        num_workers=config.get('num_workers', 4),
        transform=train_transforms,
    )

    val_loader = create_celebahq_dataloader(
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
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    # Resume from checkpoint if provided
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
