#!/usr/bin/env python3
"""
Verify that pre-trained FlexTok weights are loading correctly.

This script:
1. Loads the pre-trained model
2. Runs inference on a test image
3. Checks if reconstructions are reasonable (they should be with pre-trained weights)

Usage:
    python scripts/verify_pretrained_weights.py --data_path ./data/celeba_hq
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt

from flextok import FlexTokFromHub
from flextok.utils.dataloader import create_celebahq_dataloader


def denormalize(tensor):
    """Denormalize from [-1, 1] to [0, 1]."""
    return (tensor + 1) / 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='EPFL-VILAB/flextok_d18_d28_dfn')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("VERIFYING PRE-TRAINED FLEXTOK WEIGHTS")
    print("=" * 80)

    # Load model
    print(f"\nLoading model: {args.model_name}")
    model = FlexTokFromHub.from_pretrained(args.model_name).to(device).eval()
    print("✓ Model loaded")

    # Load a single image
    print(f"\nLoading test image from: {args.data_path}")
    dataloader = create_celebahq_dataloader(
        root_dir=args.data_path,
        batch_size=1,
        img_size=256,
        split='train',
        num_workers=0,
        shuffle=False,
    )

    image = next(iter(dataloader)).to(device)
    print(f"✓ Image loaded: {image.shape}")

    # Test 1: VAE reconstruction
    print("\n" + "=" * 80)
    print("TEST 1: VAE Reconstruction")
    print("=" * 80)

    with torch.no_grad():
        images_list = image.split(1)
        vae_data_dict = {model.vae.images_read_key: images_list}
        vae_data_dict = model.vae(vae_data_dict)
        vae_reconst = vae_data_dict[model.vae.images_reconst_write_key][0]

        # Check reconstruction quality
        mse_loss = torch.nn.functional.mse_loss(image, vae_reconst).item()
        print(f"VAE MSE loss: {mse_loss:.6f}")

        if mse_loss < 0.01:
            print("✓ VAE reconstruction looks good")
        else:
            print("⚠ VAE MSE is higher than expected")

    # Test 2: Full tokenization + detokenization (256 tokens)
    print("\n" + "=" * 80)
    print("TEST 2: Full FlexTok Reconstruction (256 tokens)")
    print("=" * 80)

    with torch.no_grad():
        # Tokenize
        tokens_list = model.tokenize(image)
        print(f"Tokens shape: {tokens_list[0].shape}")

        # Detokenize with proper inference settings
        reconst_imgs = model.detokenize(
            tokens_list,
            timesteps=25,
            guidance_scale=7.5,
            perform_norm_guidance=True,
            verbose=False,
        )

        # Check reconstruction quality
        mse_loss = torch.nn.functional.mse_loss(image, reconst_imgs).item()
        print(f"FlexTok MSE loss: {mse_loss:.6f}")

        if mse_loss < 0.05:
            print("✓ FlexTok reconstruction looks good")
        elif mse_loss < 0.2:
            print("⚠ FlexTok reconstruction is acceptable but not great")
        else:
            print("✗ FlexTok reconstruction is poor - weights may not be loaded correctly!")

    # Test 3: Check the training loss (what you see during training)
    print("\n" + "=" * 80)
    print("TEST 3: Training Loss (Flow Matching)")
    print("=" * 80)

    with torch.no_grad():
        # This mimics what happens during training
        data_dict = {model.vae.images_read_key: images_list}
        data_dict = model(data_dict)

        pred_latents = data_dict['vae_latents_reconst'][0]
        clean_latents = data_dict['vae_latents'][0]

        training_loss = torch.nn.functional.mse_loss(pred_latents, clean_latents).item()
        print(f"Training loss (latent MSE): {training_loss:.6f}")

        print(f"\nPredicted latents - mean: {pred_latents.mean().item():.4f}, std: {pred_latents.std().item():.4f}")
        print(f"Clean latents      - mean: {clean_latents.mean().item():.4f}, std: {clean_latents.std().item():.4f}")

        if training_loss < 0.1:
            print("✓ Training loss is low - model appears to be properly initialized")
        elif training_loss < 1.0:
            print("⚠ Training loss is moderate - may indicate partial loading")
        else:
            print("✗ Training loss is HIGH - weights are likely NOT loaded correctly!")
            print("\nThis explains why your fine-tuning makes things worse:")
            print("  1. Pre-trained weights aren't loading (random initialization)")
            print("  2. Training from scratch in latent space")
            print("  3. Learned latents are incompatible with frozen VAE")
            print("  4. Result: garbage reconstructions despite low loss")

    # Visualize
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATION")
    print("=" * 80)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Original
    img_np = denormalize(image[0]).clamp(0, 1).cpu()
    axes[0].imshow(TF.to_pil_image(img_np))
    axes[0].set_title('Original')
    axes[0].axis('off')

    # VAE reconstruction
    vae_np = denormalize(vae_reconst).clamp(0, 1).cpu()
    axes[1].imshow(TF.to_pil_image(vae_np[0]))
    axes[1].set_title(f'VAE\n(MSE: {torch.nn.functional.mse_loss(image, vae_reconst).item():.4f})')
    axes[1].axis('off')

    # FlexTok reconstruction
    flex_np = denormalize(reconst_imgs[0]).clamp(0, 1).cpu()
    axes[2].imshow(TF.to_pil_image(flex_np))
    axes[2].set_title(f'FlexTok (256 tokens)\n(MSE: {torch.nn.functional.mse_loss(image, reconst_imgs).item():.4f})')
    axes[2].axis('off')

    plt.tight_layout()
    save_path = 'pretrained_verification.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {save_path}")
    plt.close()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nIf the training loss is >1.0 and reconstructions look bad,")
    print("the issue is that pre-trained weights aren't loading correctly.")
    print("\nPossible causes:")
    print("  1. Model architecture mismatch with checkpoint")
    print("  2. Wrong model name/path")
    print("  3. Hugging Face cache issues")
    print("\nTry:")
    print("  - Check model exists: huggingface-cli download EPFL-VILAB/flextok_d18_d28_dfn")
    print("  - Clear cache: rm -rf ~/.cache/huggingface/hub/models--EPFL-VILAB--flextok*")
    print("  - Use alternative model: EPFL-VILAB/flextok_d12_d12_in1k")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
