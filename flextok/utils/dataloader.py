# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. and EPFL. All Rights Reserved.
"""
CelebA-HQ Dataset Loader for FlexTok Fine-tuning

This module provides a PyTorch Dataset class for loading CelebA-HQ images
with preprocessing suitable for FlexTok training.
"""

from pathlib import Path
from typing import Optional, Tuple, Callable, List
import warnings

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

__all__ = ["CelebAHQDataset", "create_celebahq_dataloader"]


class CelebAHQDataset(Dataset):
    """
    CelebA-HQ Dataset for FlexTok fine-tuning.

    Supports multiple image resolutions and provides preprocessing
    compatible with FlexTok's expected input format (normalized to [-1, 1]).

    Args:
        root_dir (str): Root directory containing CelebA-HQ images.
            Expected structure: root_dir/00000.jpg, root_dir/00001.jpg, etc.
            or root_dir/images/00000.jpg, root_dir/images/00001.jpg, etc.
        img_size (int): Target image size for resizing. Default: 256.
            CelebA-HQ is available at 256x256, 512x512, or 1024x1024.
        split (str): Dataset split. One of "train", "val", or "test".
            Default: "train"
            - train: First 27,000 images (indices 0-26999)
            - val: Next 1,500 images (indices 27000-28499)
            - test: Last 1,500 images (indices 28500-29999)
        transform (Optional[Callable]): Additional transforms to apply after
            default preprocessing. Default: None.
        return_path (bool): If True, return (image, path) tuples instead of
            just images. Useful for debugging. Default: False.
        extensions (List[str]): List of valid image extensions to search for.
            Default: [".jpg", ".png", ".jpeg"]

    Returns:
        torch.Tensor: Image tensor of shape (3, img_size, img_size) normalized
            to [-1, 1] range, or (image, path) tuple if return_path=True.

    Examples:
        >>> # Basic usage
        >>> dataset = CelebAHQDataset(root_dir="/path/to/celeba_hq", img_size=256)
        >>> img = dataset[0]  # Returns tensor of shape (3, 256, 256)

        >>> # With validation split
        >>> val_dataset = CelebAHQDataset(
        ...     root_dir="/path/to/celeba_hq",
        ...     img_size=512,
        ...     split="val"
        ... )

        >>> # With custom transforms
        >>> from torchvision import transforms
        >>> custom_transform = transforms.RandomHorizontalFlip(p=0.5)
        >>> dataset = CelebAHQDataset(
        ...     root_dir="/path/to/celeba_hq",
        ...     img_size=256,
        ...     transform=custom_transform
        ... )
    """

    # Standard CelebA-HQ splits (30,000 images total)
    SPLIT_INDICES = {
        "train": (0, 27000),
        "val": (27000, 28500),
        "test": (28500, 30000),
    }

    def __init__(
        self,
        root_dir: str,
        img_size: int = 256,
        split: str = "train",
        transform: Optional[Callable] = None,
        return_path: bool = False,
        extensions: List[str] = None,
    ):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.split = split
        self.custom_transform = transform
        self.return_path = return_path

        if extensions is None:
            extensions = [".jpg", ".png", ".jpeg"]
        self.extensions = extensions

        # Validate split
        if split not in self.SPLIT_INDICES:
            raise ValueError(
                f"Invalid split '{split}'. Must be one of {list(self.SPLIT_INDICES.keys())}"
            )

        # Find all image files
        self.image_paths = self._find_images()

        if len(self.image_paths) == 0:
            raise ValueError(
                f"No images found in {self.root_dir}. "
                f"Please check the directory structure and file extensions."
            )

        # Apply split
        start_idx, end_idx = self.SPLIT_INDICES[split]
        self.image_paths = self.image_paths[start_idx:end_idx]

        if len(self.image_paths) == 0:
            warnings.warn(
                f"No images found for split '{split}' in range [{start_idx}, {end_idx}). "
                f"Total images found: {len(self.image_paths)}"
            )

        # Define preprocessing transforms
        # FlexTok expects images normalized to [-1, 1] using mean=0.5, std=0.5
        self.base_transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),  # Converts to [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # To [-1, 1]
        ])

    def _find_images(self) -> List[Path]:
        """
        Find all image files in the root directory.

        Searches in both root_dir and root_dir/images subdirectory.
        Sorts files numerically by stem (filename without extension).
        """
        image_paths = []

        # Search in root_dir
        for ext in self.extensions:
            image_paths.extend(self.root_dir.glob(f"*{ext}"))

        # Also search in root_dir/images if it exists
        images_subdir = self.root_dir / "images"
        if images_subdir.exists():
            for ext in self.extensions:
                image_paths.extend(images_subdir.glob(f"*{ext}"))

        # Sort by numeric stem (e.g., "00000.jpg" -> 0, "00001.jpg" -> 1)
        def sort_key(path):
            try:
                return int(path.stem)
            except ValueError:
                return path.stem

        image_paths = sorted(image_paths, key=sort_key)
        return image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor | Tuple[torch.Tensor, str]:
        """
        Load and preprocess an image.

        Args:
            idx: Index of the image to load.

        Returns:
            Image tensor of shape (3, img_size, img_size) normalized to [-1, 1],
            or (image, path) tuple if return_path=True.
        """
        img_path = self.image_paths[idx]

        # Load image with retry logic for corrupted files
        max_retries = 5
        for attempt in range(max_retries):
            try:
                img = Image.open(img_path).convert("RGB")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    # Last attempt failed - try loading a different image
                    warnings.warn(
                        f"Failed to load image {img_path} after {max_retries} attempts: {e}. "
                        f"Skipping to next image."
                    )
                    # Return a random other image from the dataset
                    new_idx = (idx + 1) % len(self.image_paths)
                    if new_idx == idx:
                        # Only one image in dataset, create a blank image
                        img = Image.new('RGB', (self.img_size, self.img_size), color=(128, 128, 128))
                    else:
                        return self.__getitem__(new_idx)
                else:
                    # Retry after a short delay
                    import time
                    time.sleep(0.1)

        # Apply base transforms (resize, crop, normalize)
        img = self.base_transform(img)

        # Apply custom transforms if provided
        if self.custom_transform is not None:
            img = self.custom_transform(img)

        if self.return_path:
            return img, str(img_path)
        return img

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"root_dir={self.root_dir}, "
            f"img_size={self.img_size}, "
            f"split={self.split}, "
            f"num_images={len(self)}"
            f")"
        )


def create_celebahq_dataloader(
    root_dir: str,
    img_size: int = 256,
    batch_size: int = 32,
    split: str = "train",
    shuffle: bool = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    transform: Optional[Callable] = None,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for CelebA-HQ dataset.

    Convenience function to create a DataLoader with sensible defaults
    for FlexTok fine-tuning.

    Args:
        root_dir (str): Root directory containing CelebA-HQ images.
        img_size (int): Target image size. Default: 256.
        batch_size (int): Batch size. Default: 32.
        split (str): Dataset split ("train", "val", or "test"). Default: "train".
        shuffle (bool): Whether to shuffle the data. If None, defaults to True
            for train split and False for val/test splits.
        num_workers (int): Number of worker processes for data loading. Default: 4.
        pin_memory (bool): Whether to pin memory for faster GPU transfer. Default: True.
        transform (Optional[Callable]): Additional transforms to apply. Default: None.
        **kwargs: Additional arguments to pass to DataLoader.

    Returns:
        DataLoader: Configured DataLoader for CelebA-HQ.

    Examples:
        >>> # Basic usage
        >>> train_loader = create_celebahq_dataloader(
        ...     root_dir="/path/to/celeba_hq",
        ...     batch_size=32,
        ...     img_size=256
        ... )
        >>> for batch in train_loader:
        ...     # batch shape: (32, 3, 256, 256)
        ...     pass

        >>> # Validation loader
        >>> val_loader = create_celebahq_dataloader(
        ...     root_dir="/path/to/celeba_hq",
        ...     batch_size=64,
        ...     split="val",
        ...     shuffle=False
        ... )

        >>> # With data augmentation
        >>> from torchvision import transforms
        >>> train_transform = transforms.Compose([
        ...     transforms.RandomHorizontalFlip(p=0.5),
        ...     transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ... ])
        >>> train_loader = create_celebahq_dataloader(
        ...     root_dir="/path/to/celeba_hq",
        ...     transform=train_transform,
        ...     batch_size=32
        ... )
    """
    # Default shuffle behavior: True for train, False for val/test
    if shuffle is None:
        shuffle = (split == "train")

    dataset = CelebAHQDataset(
        root_dir=root_dir,
        img_size=img_size,
        split=split,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs
    )

    return dataloader
