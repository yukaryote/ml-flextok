#!/usr/bin/env python3
"""
Check the loss scale and latent statistics for FlexTok.

This script helps diagnose why the initial loss might be high.

Usage:
    python scripts/check_loss_scale.py --data_path ./data/celeba_hq --model_name mit-han-lab/flextok-dfn-depth-12
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import random

import torch
import torch.nn.functional as F
from flextok import FlexTokFromHub
from flextok.utils.dataloader import create_celebahq_dataloader


def denormalize(tensor):
    """Denormalize from [-1, 1] to [0, 1]."""
    return (tensor + 1) / 2


def plot_flexible_length_reconsts(model, images, losses, k_keep_list, device, save_path='loss_visualization.png'):
    """
    Plot flexible-length reconstructions for given images along with their losses.

    Args:
        model: FlexTok model
        images: Tensor of shape [N, 3, H, W] in range [-1, 1]
        losses: List of loss values for each image
        k_keep_list: List of token counts to visualize (e.g., [1, 4, 16, 64, 256])
        device: torch device
        save_path: Path to save the visualization
    """
    nrows = images.shape[0]
    ncols = len(k_keep_list) + 2  # VAE | k_keep_list | GT

    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        # Move images to device
        images = images.to(device)
        images_list = images.split(1)

        # Get VAE reconstruction
        vae_data_dict = {model.vae.images_read_key: images_list}
        vae_data_dict = model.vae(vae_data_dict)
        vae_reconst = torch.cat(vae_data_dict[model.vae.images_reconst_write_key], dim=0)

        # Tokenize once
        data_dict = {model.vae.images_read_key: images_list}
        data_dict = model.encoder(data_dict)
        tokens_list = data_dict['tokens']  # [B, N]

        # Detokenize for various token counts
        all_reconst = {}
        for k_keep in k_keep_list:
            # Truncate tokens
            subseq_list = [tokens_list[i:i+1, :k_keep].clone() for i in range(len(images_list))]

            # Properly detokenize with denoising pipeline
            reconst_imgs = model.detokenize(
                subseq_list,
                timesteps=25,
                guidance_scale=7.5,
                perform_norm_guidance=True,
                verbose=False,
            )

            all_reconst[k_keep] = reconst_imgs

        # Move everything to CPU for plotting
        images_cpu = images.cpu()
        vae_reconst_cpu = vae_reconst.cpu()
        all_reconst_cpu = {k: v.cpu() for k, v in all_reconst.items()}

        # Denormalize
        images_plot = denormalize(images_cpu).clamp(0, 1)
        vae_reconst_plot = denormalize(vae_reconst_cpu).clamp(0, 1)
        all_reconst_plot = {k: denormalize(v).clamp(0, 1) for k, v in all_reconst_cpu.items()}

    # Plot
    for img_idx in range(nrows):
        col_idx = 0

        # VAE reconstruction
        axes[img_idx][col_idx].imshow(TF.to_pil_image(vae_reconst_plot[img_idx]))
        axes[img_idx][col_idx].axis('off')
        if img_idx == 0:
            axes[img_idx][col_idx].set_title('VAE', fontsize=14)
        col_idx += 1

        # Flexible-length reconstructions
        for k_keep in k_keep_list:
            axes[img_idx][col_idx].imshow(TF.to_pil_image(all_reconst_plot[k_keep][img_idx]))
            axes[img_idx][col_idx].axis('off')
            if img_idx == 0:
                title = f'{k_keep} token' + ('' if k_keep == 1 else 's')
                axes[img_idx][col_idx].set_title(title, fontsize=14)
            col_idx += 1

        # Ground truth
        axes[img_idx][col_idx].imshow(TF.to_pil_image(images_plot[img_idx]))
        axes[img_idx][col_idx].axis('off')
        if img_idx == 0:
            axes[img_idx][col_idx].set_title('Ground Truth', fontsize=14)

        # Add loss as ylabel
        axes[img_idx][0].set_ylabel(f'Loss: {losses[img_idx]:.4f}', fontsize=12, rotation=0,
                                     labelpad=40, va='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {save_path}")
    plt.close()

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='EPFL-VILAB/flextok_d18_d28_dfn')
    parser.add_argument('--num_batches', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_viz_images', type=int, default=4, help='Number of images to visualize')
    parser.add_argument('--viz_token_counts', type=int, nargs='+', default=[1, 16, 64, 256],
                        help='Token counts to visualize')
    parser.add_argument('--output_path', type=str, default='loss_visualization.png',
                        help='Path to save visualization')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("FlexTok Loss Scale Diagnostic")
    print("=" * 80)

    # Load model
    print(f"\nLoading model: {args.model_name}")
    try:
        model = FlexTokFromHub.from_pretrained(args.model_name).to(device).eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative model names...")
        # Try alternative format
        alt_name = args.model_name.replace('_', '-')
        try:
            model = FlexTokFromHub.from_pretrained(alt_name).to(device).eval()
            print(f"Successfully loaded with alternative name: {alt_name}")
        except:
            print(f"Failed to load model. Please check the model name.")
            print("Available models on HuggingFace:")
            print("  - EPFL-VILAB/flextok_d18_d28_dfn")
            print("  - EPFL-VILAB/flextok_d12_d12_in1k")
            sys.exit(1)
    print("✓ Model loaded")

    # Load data
    print(f"\nLoading data from: {args.data_path}")
    dataloader = create_celebahq_dataloader(
        root_dir=args.data_path,
        batch_size=args.batch_size,
        img_size=256,
        split='train',
        num_workers=0,
        shuffle=True,
    )
    print(f"✓ Data loaded ({len(dataloader)} batches)")

    # Analyze losses
    print(f"\nAnalyzing {args.num_batches} batches...")
    print("-" * 80)

    all_losses = []
    all_pred_means = []
    all_pred_stds = []
    all_target_means = []
    all_target_stds = []

    # Store images and losses for visualization
    viz_images = []
    viz_losses = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= args.num_batches:
                break

            batch = batch.to(device)
            images_list = batch.split(1)

            # Forward pass
            data_dict = {model.vae.images_read_key: images_list}
            data_dict = model(data_dict)

            # Extract latents
            pred_latents_list = data_dict['vae_latents_reconst']
            clean_latents_list = data_dict['vae_latents']

            # Compute loss
            losses = []
            for pred, target in zip(pred_latents_list, clean_latents_list):
                loss = F.mse_loss(pred, target).item()
                losses.append(loss)

                # Statistics
                all_pred_means.append(pred.mean().item())
                all_pred_stds.append(pred.std().item())
                all_target_means.append(target.mean().item())
                all_target_stds.append(target.std().item())

            batch_loss = sum(losses) / len(losses)
            all_losses.extend(losses)

            # Store images and losses for visualization
            for img, loss in zip(images_list, losses):
                if len(viz_images) < args.num_viz_images:
                    viz_images.append(img.squeeze(0).cpu())
                    viz_losses.append(loss)

            print(f"Batch {batch_idx + 1}: Loss = {batch_loss:.6f}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    import statistics

    print(f"\nLoss Statistics:")
    print(f"  Mean:   {statistics.mean(all_losses):.6f}")
    print(f"  Median: {statistics.median(all_losses):.6f}")
    print(f"  Std:    {statistics.stdev(all_losses):.6f}")
    print(f"  Min:    {min(all_losses):.6f}")
    print(f"  Max:    {max(all_losses):.6f}")

    print(f"\nPredicted Latents:")
    print(f"  Mean: {statistics.mean(all_pred_means):.4f} ± {statistics.stdev(all_pred_means):.4f}")
    print(f"  Std:  {statistics.mean(all_pred_stds):.4f} ± {statistics.stdev(all_pred_stds):.4f}")

    print(f"\nTarget Latents:")
    print(f"  Mean: {statistics.mean(all_target_means):.4f} ± {statistics.stdev(all_target_means):.4f}")
    print(f"  Std:  {statistics.mean(all_target_stds):.4f} ± {statistics.stdev(all_target_stds):.4f}")

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    avg_loss = statistics.mean(all_losses)

    if avg_loss < 0.01:
        print("✓ Loss is LOW - Model is performing well (expected for pre-trained model)")
    elif avg_loss < 0.1:
        print("⚠ Loss is MODERATE - Model may need fine-tuning")
    elif avg_loss < 1.0:
        print("⚠ Loss is HIGH - Check if:")
        print("  - Weights are properly initialized")
        print("  - Data normalization is correct")
        print("  - Model architecture matches pre-trained checkpoint")
    else:
        print("❌ Loss is VERY HIGH - Something is wrong:")
        print("  - Data may not be normalized correctly")
        print("  - Model weights may not be loaded")
        print("  - Wrong model architecture")

    # Check if predictions are close to targets
    pred_target_diff = abs(statistics.mean(all_pred_means) - statistics.mean(all_target_means))
    if pred_target_diff > 0.5:
        print(f"\n⚠ WARNING: Large mean difference ({pred_target_diff:.4f}) between predictions and targets")
        print("  This suggests the model is not properly initialized or data is incorrectly normalized")

    print("\n" + "=" * 80)

    # Generate visualization
    if viz_images:
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATION")
        print("=" * 80)
        print(f"\nCreating flexible-length reconstruction visualization for {len(viz_images)} images...")
        print(f"Token counts: {args.viz_token_counts}")

        # Stack images into a batch
        viz_batch = torch.stack(viz_images)

        # Generate visualization
        plot_flexible_length_reconsts(
            model=model,
            images=viz_batch,
            losses=viz_losses,
            k_keep_list=args.viz_token_counts,
            device=device,
            save_path=args.output_path
        )

        print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
