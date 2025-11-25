#!/usr/bin/env python3
"""
Example usage of the CelebA dataloader with FlexTok.

This script demonstrates how to:
1. Load the CelebA dataset
2. Create dataloaders for training and validation
3. Iterate through batches
4. Visualize sample images
"""

import sys
from pathlib import Path

# Add flextok to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flextok.utils.dataloader import create_celeba_dataloader, CelebADataset
import torch
import matplotlib.pyplot as plt
import numpy as np


def denormalize(tensor):
    """
    Denormalize image tensor from [-1, 1] to [0, 1] for visualization.

    Args:
        tensor: Image tensor normalized to [-1, 1]

    Returns:
        Denormalized tensor in [0, 1] range
    """
    return (tensor + 1.0) / 2.0


def visualize_batch(images, num_images=8, title="CelebA Samples"):
    """
    Visualize a batch of images.

    Args:
        images: Batch of images (B, C, H, W) in [-1, 1] range
        num_images: Number of images to display
        title: Plot title
    """
    num_images = min(num_images, len(images))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(num_images):
        img = denormalize(images[i]).cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('celeba_samples.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: celeba_samples.png")
    plt.close()


def main():
    # Configuration
    data_dir = Path.home() / "data" / "celeba"
    batch_size = 32
    img_size = 256
    num_workers = 4

    print("=" * 80)
    print("CelebA Dataloader Example")
    print("=" * 80)

    # Create training dataloader
    print("\n1. Creating training dataloader...")
    try:
        train_loader = create_celeba_dataloader(
            root_dir=str(data_dir),
            batch_size=batch_size,
            img_size=img_size,
            split="train",
            num_workers=num_workers,
            download=False,  # Set to True to download if not present
        )
        print(f"   ✓ Training dataloader created")
        print(f"     - Total batches: {len(train_loader)}")
        print(f"     - Batch size: {batch_size}")
        print(f"     - Image size: {img_size}x{img_size}")
    except Exception as e:
        print(f"   ✗ Failed to create training dataloader: {e}")
        print("\n   Please download the dataset first:")
        print(f"     python download_celeba.py --data-dir {data_dir}")
        return

    # Create validation dataloader
    print("\n2. Creating validation dataloader...")
    val_loader = create_celeba_dataloader(
        root_dir=str(data_dir),
        batch_size=batch_size,
        img_size=img_size,
        split="valid",
        shuffle=False,
        num_workers=num_workers,
        download=False,
    )
    print(f"   ✓ Validation dataloader created")
    print(f"     - Total batches: {len(val_loader)}")

    # Load and display a batch
    print("\n3. Loading a sample batch...")
    batch = next(iter(train_loader))
    print(f"   ✓ Batch loaded")
    print(f"     - Shape: {batch.shape}")
    print(f"     - Dtype: {batch.dtype}")
    print(f"     - Range: [{batch.min():.3f}, {batch.max():.3f}]")

    # Visualize samples
    print("\n4. Visualizing samples...")
    visualize_batch(batch, num_images=8, title="CelebA Training Samples")

    # Example: Training loop structure
    print("\n5. Example training loop structure:")
    print("-" * 80)
    print("""
    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, images in enumerate(train_loader):
            # images shape: (batch_size, 3, img_size, img_size)
            # images range: [-1, 1]

            # Move to device
            images = images.to(device)

            # Forward pass through your model
            # outputs = model(images)
            # loss = criterion(outputs, targets)

            # Backward pass
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch}] Batch [{batch_idx}/{len(train_loader)}]')
    """)
    print("-" * 80)

    # Dataset statistics
    print("\n6. Dataset statistics:")
    print("-" * 80)
    print(f"   Data directory: {data_dir}")
    print(f"   Training samples: {len(train_loader.dataset):,}")
    print(f"   Validation samples: {len(val_loader.dataset):,}")
    print(f"   Image dimensions: {img_size}x{img_size}")
    print(f"   Normalization: mean=0.5, std=0.5 (range [-1, 1])")
    print(f"   Original image size: 178x218 (center cropped to 178x178)")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
