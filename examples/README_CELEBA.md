# CelebA Dataset Usage Guide

This guide shows how to use the CelebA aligned and cropped dataset with FlexTok.

## Quick Start

### 1. Download the Dataset

**Option A: Automatic Download (may fail due to rate limits)**

```python
from flextok.utils.dataloader import create_celeba_dataloader

# This will automatically download the dataset
train_loader = create_celeba_dataloader(
    root_dir="~/data/celeba",
    batch_size=32,
    img_size=256,
    download=True  # Automatic download
)
```

**Option B: Using the download script**

```bash
# Run the download helper script
python download_celeba.py --data-dir ~/data/celeba
```

**Option C: Manual Download**

If automatic download fails (Google Drive rate limits), see [CELEBA_DOWNLOAD_GUIDE.md](../CELEBA_DOWNLOAD_GUIDE.md) for manual download instructions.

### 2. Create Dataloaders

```python
from flextok.utils.dataloader import create_celeba_dataloader

# Training dataloader
train_loader = create_celeba_dataloader(
    root_dir="~/data/celeba",
    batch_size=32,
    img_size=256,
    split="train",
    shuffle=True,
    num_workers=4,
    download=False,  # Already downloaded
)

# Validation dataloader
val_loader = create_celeba_dataloader(
    root_dir="~/data/celeba",
    batch_size=64,
    img_size=256,
    split="valid",
    shuffle=False,
    num_workers=4,
    download=False,
)
```

### 3. Use in Training

```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for epoch in range(num_epochs):
    for batch_idx, images in enumerate(train_loader):
        # images shape: (batch_size, 3, 256, 256)
        # images range: [-1, 1]

        images = images.to(device)

        # Your training code here
        # ...
```

## Dataset Information

- **Total images**: 202,599
- **Original size**: 178 x 218 (aligned and cropped faces)
- **Preprocessed**: Center cropped to 178x178, then resized to your specified size
- **Normalization**: [-1, 1] range (compatible with FlexTok)
- **Format**: RGB (3 channels)

### Dataset Splits

| Split | Images | Usage |
|-------|--------|-------|
| train | 162,770 | Training |
| valid | 19,867 | Validation |
| test | 19,962 | Testing |
| all | 202,599 | All images |

## API Reference

### CelebADataset

```python
from flextok.utils.dataloader import CelebADataset

dataset = CelebADataset(
    root_dir="~/data/celeba",          # Where dataset is stored
    img_size=256,                       # Target image size
    split="train",                      # "train", "valid", "test", or "all"
    transform=None,                     # Optional additional transforms
    download=True,                      # Whether to download if missing
    return_path=False,                  # Return (image, path) tuples
)

# Access single image
image = dataset[0]  # Shape: (3, 256, 256), Range: [-1, 1]

# With return_path=True
image, path = dataset[0]
```

### create_celeba_dataloader

```python
from flextok.utils.dataloader import create_celeba_dataloader

dataloader = create_celeba_dataloader(
    root_dir="~/data/celeba",          # Where dataset is stored
    img_size=256,                       # Target image size
    batch_size=32,                      # Batch size
    split="train",                      # Dataset split
    shuffle=None,                       # Auto: True for train, False otherwise
    num_workers=4,                      # Data loading workers
    pin_memory=True,                    # Pin memory for GPU
    transform=None,                     # Optional transforms
    download=True,                      # Auto-download if missing
)
```

## Advanced Usage

### Custom Transforms

Add data augmentation:

```python
from torchvision import transforms

# Define augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
])

# Create dataloader with augmentation
train_loader = create_celeba_dataloader(
    root_dir="~/data/celeba",
    transform=train_transform,
    batch_size=32,
)
```

### Different Image Sizes

```python
# 128x128 images
loader_128 = create_celeba_dataloader(
    root_dir="~/data/celeba",
    img_size=128,
    batch_size=64,
)

# 512x512 images
loader_512 = create_celeba_dataloader(
    root_dir="~/data/celeba",
    img_size=512,
    batch_size=16,
)
```

### Using All Data

```python
# Use entire dataset (no train/val/test split)
all_loader = create_celeba_dataloader(
    root_dir="~/data/celeba",
    split="all",
    batch_size=32,
)
```

## Examples

Run the example script:

```bash
# Make sure the dataset is downloaded first
python examples/celeba_example.py
```

This will:
1. Create train and validation dataloaders
2. Load a sample batch
3. Visualize sample images
4. Show example training loop structure

## Troubleshooting

### "Dataset not found or corrupted"

Make sure you've downloaded the dataset and the directory structure is correct:

```
~/data/celeba/
└── celeba/
    ├── img_align_celeba/
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   └── ... (202,599 images)
    ├── list_attr_celeba.txt
    ├── identity_CelebA.txt
    ├── list_bbox_celeba.txt
    ├── list_landmarks_align_celeba.txt
    └── list_eval_partition.txt
```

### "Too many users have viewed or downloaded this file"

This is Google Drive's rate limit. Solutions:
1. Wait 24 hours and try again
2. Use manual download (see [CELEBA_DOWNLOAD_GUIDE.md](../CELEBA_DOWNLOAD_GUIDE.md))
3. Try alternative sources (Kaggle, official website)

### "gdown not installed"

Install gdown for automatic downloads:

```bash
pip install gdown
```

### Slow data loading

Increase `num_workers`:

```python
train_loader = create_celeba_dataloader(
    root_dir="~/data/celeba",
    num_workers=8,  # Increase workers
    batch_size=32,
)
```

## References

- **Official Website**: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- **Paper**: Liu et al., "Deep Learning Face Attributes in the Wild", ICCV 2015
- **Google Drive**: https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8
- **Kaggle**: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
