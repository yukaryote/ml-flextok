# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. and EPFL. All Rights Reserved.
"""
CelebA and CelebA-HQ Dataset Loaders for FlexTok Fine-tuning

This module provides PyTorch Dataset classes for loading CelebA and CelebA-HQ images
with preprocessing suitable for FlexTok training.
"""

from pathlib import Path
from typing import Optional, Tuple, Callable, List
import warnings

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

__all__ = ["CelebADataset", "CelebAHQDataset", "create_celeb_dataloader"]


class _BaseCelebADataset(Dataset):
    """
    Base class for CelebA-family datasets with common preprocessing logic.

    This base class provides shared functionality for CelebA and CelebA-HQ datasets,
    including image normalization, transform application, and path handling.
    """
    SPLIT_INDICES = {
        "train": (0, 27000),  # To be defined in subclasses
        "val": (27000, 28500),  # To be defined in subclasses
        "test": (28500, 30000),  # To be defined in subclasses
        "all": (0, 30000),  # To be defined in subclasses
    }

    def __init__(
        self,
        root_dir: str,
        img_size: int,
        split: str,
        transform: Optional[Callable],
        return_path: bool,
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

        # Define preprocessing transforms using base class helper
        # FlexTok expects images normalized to [-1, 1] using mean=0.5, std=0.5
        self.base_transform = self._create_base_transform(
            resize_first=True,  # Resize before crop for HQ images
            center_crop_size=None  # Crop to img_size
        )

    def __len__(self) -> int:
        return len(self.image_paths)
    
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
    
    def _create_base_transform(
        self,
        resize_first: bool = True,
        center_crop_size: Optional[int] = None
    ) -> transforms.Compose:
        """
        Create base transform pipeline for image preprocessing.

        Args:
            resize_first: If True, resize before center crop. Otherwise crop first.
            center_crop_size: Size for center crop. If None, crops to img_size.

        Returns:
            Composed transform pipeline that normalizes to [-1, 1].
        """
        crop_size = center_crop_size if center_crop_size is not None else self.img_size

        if resize_first:
            transform_list = [
                transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.img_size),
            ]
        else:
            transform_list = [
                transforms.CenterCrop(crop_size),
                transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            ]

        # Add tensor conversion and normalization to [-1, 1]
        transform_list.extend([
            transforms.ToTensor(),  # Converts to [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # To [-1, 1]
        ])

        return transforms.Compose(transform_list)

    def _apply_transforms(self, img: Image.Image) -> torch.Tensor:
        """Apply base and custom transforms to an image."""
        img = self.base_transform(img)
        if self.custom_transform is not None:
            img = self.custom_transform(img)
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

        # Apply transforms using base class helper
        img = self._apply_transforms(img)

        if self.return_path:
            return img, str(img_path)
        return img


class CelebADataset(_BaseCelebADataset):
    """
    CelebA (Aligned and Cropped) Dataset for FlexTok fine-tuning.

    This dataset uses torchvision's built-in CelebA dataset loader which handles
    downloading the aligned and cropped CelebA images (178x218) and provides
    preprocessing compatible with FlexTok's expected input format (normalized to [-1, 1]).

    The dataset contains 202,599 face images of celebrities with 40 binary attributes
    per image. This class focuses on the images for generative modeling.

    Args:
        root_dir (str): Root directory where the dataset will be downloaded/stored.
            The images will be stored in root_dir/celeba/img_align_celeba/.
        img_size (int): Target image size for resizing. Default: 256.
            Original images are 178x218 and will be center cropped to square
            then resized to img_size x img_size.
        split (str): Dataset split. One of "train", "valid", "test", or "all".
            Default: "train"
            - train: 162,770 images
            - valid: 19,867 images
            - test: 19,962 images
            - all: All 202,599 images
        transform (Optional[Callable]): Additional transforms to apply after
            default preprocessing. Default: None.
        download (bool): If True, downloads the dataset from the internet.
            Default: True. Requires 'gdown' package to be installed.
        return_path (bool): If True, return (image, path) tuples instead of
            just images. Useful for debugging. Default: False.

    Returns:
        torch.Tensor: Image tensor of shape (3, img_size, img_size) normalized
            to [-1, 1] range, or (image, path) tuple if return_path=True.

    Examples:
        >>> # Basic usage with automatic download
        >>> dataset = CelebADataset(root_dir="/path/to/data", download=True)
        >>> img = dataset[0]  # Returns tensor of shape (3, 256, 256)

        >>> # With validation split
        >>> val_dataset = CelebADataset(
        ...     root_dir="/path/to/data",
        ...     img_size=128,
        ...     split="valid",
        ...     download=False  # Already downloaded
        ... )

        >>> # With custom transforms
        >>> from torchvision import transforms
        >>> custom_transform = transforms.RandomHorizontalFlip(p=0.5)
        >>> dataset = CelebADataset(
        ...     root_dir="/path/to/data",
        ...     transform=custom_transform,
        ...     download=True
        ... )

    Note:
        The dataset download requires the 'gdown' package. Install it with:
        pip install gdown
    """
    SPLIT_INDICES = {
        "train": (0, 162769),
        "val": (162770, 182636),
        "test": (182637, 202598),
        "all": (0, 202598),
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
        # Initialize base class
        super().__init__(root_dir, img_size, split, transform, return_path, extensions)


class CelebAHQDataset(_BaseCelebADataset):
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
        # Initialize base class
        super().__init__(root_dir, img_size, split, transform, return_path, extensions)


class SyntheticFaceDataset(_BaseCelebADataset):
    SPLIT_INDICES = {
        "train": (0, 59999),
        "val": (60000, 67999),
        "test": (68000, 71999),
        "all": (0, 71999),
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
        super().__init__(root_dir, img_size, split, transform, return_path, extensions)


def create_celeb_dataloader(
    dataset_type: str,
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
    Create a DataLoader for CelebA-HQ or CelebA dataset.

    Convenience function to create a DataLoader with sensible defaults
    for FlexTok fine-tuning.

    Args:
        dataset_type (str): Type of dataset ("celeba" or "celebahq").
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
        >>> train_loader = create_celeba_dataloader(
        ...     dataset_type="celebahq",
        ...     root_dir="/path/to/celeba_hq",
        ...     batch_size=32,
        ...     img_size=256
        ... )
        >>> for batch in train_loader:
        ...     # batch shape: (32, 3, 256, 256)
        ...     pass

        >>> # Validation loader
        >>> val_loader = create_celeba_dataloader(
        ...     dataset_type="celebahq",
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
        >>> train_loader = create_celeba_dataloader(
        ...     dataset_type="celebahq",
        ...     root_dir="/path/to/celeba_hq",
        ...     transform=train_transform,
        ...     batch_size=32
        ... )
    """
    # Default shuffle behavior: True for train, False for val/test
    if shuffle is None:
        shuffle = (split == "train")

    if dataset_type.lower() == "celebahq":
        dataset = CelebAHQDataset(
            root_dir=root_dir,
            img_size=img_size,
            split=split,
            transform=transform,
        )
    elif dataset_type.lower() == "celeba":
        dataset = CelebADataset(
            root_dir=root_dir,
            img_size=img_size,
            split=split,
            transform=transform,
        )
    elif dataset_type.lower() == "synth_faces":
        dataset = SyntheticFaceDataset(
            root_dir=root_dir,
            img_size=img_size,
            split=split,
            transform=transform,
        )
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}. Supported types: 'celeba', 'celebahq', 'synth_faces'.")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs
    )

    return dataloader
