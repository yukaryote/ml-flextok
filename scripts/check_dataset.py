#!/usr/bin/env python3
"""
Check CelebA-HQ dataset for corrupted or invalid images.

Usage:
    python scripts/check_dataset.py --data_path ./data/celeba_hq
"""

import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import sys

def check_image(img_path):
    """
    Check if an image file is valid.

    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        with Image.open(img_path) as img:
            img.verify()  # Verify it's a valid image

        # Reopen to actually load the image data
        with Image.open(img_path) as img:
            img.load()

        return True, None
    except Exception as e:
        return False, str(e)


def find_images(root_dir, extensions=None):
    """Find all image files in directory."""
    if extensions is None:
        extensions = [".jpg", ".png", ".jpeg"]

    root_dir = Path(root_dir)
    image_paths = []

    # Search in root_dir
    for ext in extensions:
        image_paths.extend(root_dir.glob(f"*{ext}"))

    # Also search in root_dir/images
    images_subdir = root_dir / "images"
    if images_subdir.exists():
        for ext in extensions:
            image_paths.extend(images_subdir.glob(f"*{ext}"))

    # Sort by numeric stem
    def sort_key(path):
        try:
            return int(path.stem)
        except ValueError:
            return path.stem

    return sorted(image_paths, key=sort_key)


def main():
    parser = argparse.ArgumentParser(description="Check dataset for corrupted images")
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--fix', action='store_true', help='Delete corrupted images')

    args = parser.parse_args()

    print(f"Scanning dataset at: {args.data_path}")
    print("=" * 60)

    # Find all images
    image_paths = find_images(args.data_path)
    print(f"Found {len(image_paths)} images\n")

    if len(image_paths) == 0:
        print("ERROR: No images found!")
        sys.exit(1)

    # Check each image
    corrupted = []

    print("Checking images...")
    for img_path in tqdm(image_paths):
        is_valid, error = check_image(img_path)
        if not is_valid:
            corrupted.append((img_path, error))

    # Report results
    print("\n" + "=" * 60)
    print(f"Results:")
    print(f"  Valid images:     {len(image_paths) - len(corrupted)}")
    print(f"  Corrupted images: {len(corrupted)}")
    print("=" * 60)

    if len(corrupted) > 0:
        print("\nCorrupted images:")
        for img_path, error in corrupted:
            print(f"  - {img_path}")
            print(f"    Error: {error}")

        if args.fix:
            print("\n⚠️  Deleting corrupted images...")
            for img_path, _ in corrupted:
                img_path.unlink()
                print(f"  Deleted: {img_path}")
            print(f"✓ Deleted {len(corrupted)} corrupted images")
        else:
            print("\n💡 Run with --fix flag to delete corrupted images")
    else:
        print("\n✓ All images are valid!")


if __name__ == '__main__':
    main()
