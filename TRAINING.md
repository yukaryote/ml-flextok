# FlexTok Fine-tuning Guide

This guide explains how to fine-tune FlexTok on custom datasets like CelebA-HQ.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Monitoring with Weights & Biases](#monitoring-with-weights--biases)
- [Resuming Training](#resuming-training)
- [Configuration](#configuration)
- [Advanced Usage](#advanced-usage)

## Installation

### 1. Install FlexTok

```bash
pip install -e .
```

### 2. Install Additional Training Dependencies

```bash
pip install wandb pyyaml
```

### 3. (Optional) Login to Weights & Biases

```bash
wandb login
```

If you don't want to use wandb, you can disable it with `--no-wandb` flag.

## Quick Start

### 1. Prepare Your Dataset

Download CelebA-HQ and organize it:

```bash
# Option 1: Using Hugging Face datasets
pip install datasets
python -c "
from datasets import load_dataset
import os

dataset = load_dataset('korexyz/celeba-hq-256x256')
os.makedirs('./data/celeba_hq/images', exist_ok=True)

for idx, sample in enumerate(dataset['train']):
    sample['image'].save(f'./data/celeba_hq/{idx:05d}.jpg')
"

# Option 2: Download from Kaggle
# Visit: https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256
# Extract to ./data/celeba_hq/
```

Expected directory structure:
```
data/celeba_hq/
├── 00000.jpg
├── 00001.jpg
├── 00002.jpg
...
└── 29999.jpg
```

### 2. Start Training

```bash
# Full training configuration
python train_flextok.py --config configs/train_celebahq.yaml

# Smaller configuration (for quick experiments or limited hardware)
python train_flextok.py --config configs/train_celebahq_small.yaml
```

### 3. Monitor Training

Training progress will be logged to:
- **Console**: Real-time progress bars and metrics
- **Weights & Biases**: Detailed metrics and visualizations (if enabled)
- **Checkpoints**: Saved to `./checkpoints/celebahq_ft/`

## Dataset Preparation

### CelebA-HQ

CelebA-HQ contains 30,000 high-resolution face images at 1024×1024 resolution.

**Download Options:**

1. **Hugging Face** (Recommended):
   ```python
   from datasets import load_dataset
   dataset = load_dataset("korexyz/celeba-hq-256x256")
   ```

2. **Kaggle**:
   - 256×256: https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256
   - 1024×1024: Search "CelebA-HQ" on Kaggle

3. **Official Source**:
   - https://github.com/tkarras/progressive_growing_of_gans

**Dataset Splits:**

The dataloader automatically splits the dataset:
- **Train**: 27,000 images (indices 0-26,999)
- **Val**: 1,500 images (indices 27,000-28,499)
- **Test**: 1,500 images (indices 28,500-29,999)

### Custom Datasets

To use your own dataset:

1. Organize images in a directory:
   ```
   your_dataset/
   ├── 00000.jpg
   ├── 00001.jpg
   └── ...
   ```

2. Create a custom config file based on `configs/train_celebahq.yaml`

3. Update `data_path` in the config

## Training

### Basic Training

```bash
python train_flextok.py --config configs/train_celebahq.yaml
```

### Training Options

```bash
# Disable wandb logging
python train_flextok.py --config configs/train_celebahq.yaml --no-wandb

# Specify device
python train_flextok.py --config configs/train_celebahq.yaml --device cuda:0

# Resume from checkpoint
python train_flextok.py --config configs/train_celebahq.yaml --resume checkpoints/celebahq_ft/checkpoint_latest.pt
```

### What Gets Trained?

By default:
- ✅ **Encoder**: Fine-tuned on your dataset
- ✅ **Decoder**: Fine-tuned on your dataset
- ❌ **VAE**: Frozen (uses pre-trained Stable Diffusion VAE)
- ❌ **Regularizer (FSQ)**: Frozen

You can modify this in the config file:
```yaml
train_encoder: true
train_decoder: true
train_vae: false  # Set to true to train VAE (not recommended)
```

## Monitoring with Weights & Biases

### Metrics Logged

**Training Metrics:**
- `train/loss`: Flow matching reconstruction loss
- `train/loss_std`: Standard deviation of batch losses
- `train/lr`: Current learning rate
- `train/epoch`: Current epoch
- `train/step`: Global training step

**Validation Metrics:**
- `val/loss`: Validation reconstruction loss
- `val/epoch`: Current epoch

**Visualizations:**
- `visualizations/originals`: Original input images
- `visualizations/reconstructions`: Model reconstructions

### Viewing Results

1. Training starts → Check console for wandb URL
2. Open the URL in your browser
3. View real-time metrics and visualizations

Example wandb URL:
```
https://wandb.ai/your-username/flextok-finetuning/runs/xxxxx
```

## Resuming Training

Training automatically saves checkpoints:
- `checkpoint_latest.pt`: Latest checkpoint (every epoch)
- `checkpoint_best.pt`: Best validation loss checkpoint
- `checkpoint_epoch_XXXX.pt`: Periodic epoch checkpoints

### Resume from Latest

```bash
python train_flextok.py \
  --config configs/train_celebahq.yaml \
  --resume checkpoints/celebahq_ft/checkpoint_latest.pt
```

### Resume from Best

```bash
python train_flextok.py \
  --config configs/train_celebahq.yaml \
  --resume checkpoints/celebahq_ft/checkpoint_best.pt
```

## Configuration

### Key Configuration Parameters

```yaml
# Model
model_name: "mit-han-lab/flextok-dfn-depth-12"  # Pre-trained model

# Data
data_path: "./data/celeba_hq"
img_size: 256
batch_size: 32

# Training
num_epochs: 50
learning_rate: 1.0e-4
optimizer: "adamw"
scheduler: "cosine"

# Hardware
use_amp: true  # Mixed precision (faster, less memory)
gradient_accumulation_steps: 1
num_workers: 4

# Logging
use_wandb: true
log_every: 10
visualize_every: 100
```

### Available Pre-trained Models

| Model | Description | Best For |
|-------|-------------|----------|
| `mit-han-lab/flextok-dfn-depth-12` | Trained on DFN dataset | General images, faces ✅ |
| `mit-han-lab/flextok-in1k-depth-12` | Trained on ImageNet-1K | Natural images |
| `mit-han-lab/flextok-in1k-depth-18` | Deeper model on ImageNet-1K | Higher quality |

**Recommendation**: Use `flextok-dfn-depth-12` for CelebA-HQ and general face datasets.

### Learning Rate Guidelines

| Batch Size | Learning Rate | Notes |
|------------|---------------|-------|
| 16 | 5e-5 | Small batch |
| 32 | 1e-4 | Standard ✅ |
| 64 | 2e-4 | Large batch |
| 128 | 4e-4 | Very large batch |

**Formula**: `lr ≈ base_lr * sqrt(batch_size / 32)`

### Memory Optimization

If you run out of GPU memory:

1. **Reduce batch size**:
   ```yaml
   batch_size: 16  # or 8
   ```

2. **Enable gradient accumulation**:
   ```yaml
   batch_size: 16
   gradient_accumulation_steps: 2  # Effective batch = 32
   ```

3. **Use mixed precision** (should be on by default):
   ```yaml
   use_amp: true
   ```

4. **Reduce image size**:
   ```yaml
   img_size: 128  # Instead of 256
   ```

## Advanced Usage

### Custom Loss Functions

The training script uses flow matching loss (MSE between predicted and clean latents).
To customize the loss, edit the `compute_loss` method in [train_flextok.py](train_flextok.py):

```python
def compute_loss(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
    # ... existing code ...

    # Add your custom loss here
    custom_loss = ...

    total_loss = mse_loss + 0.1 * custom_loss

    return {'loss': total_loss, 'mse_loss': mse_loss, 'custom_loss': custom_loss}
```

### Training Only the Decoder

To freeze the encoder and only train the decoder:

```yaml
train_encoder: false
train_decoder: true
train_vae: false
```

### Using EMA (Exponential Moving Average)

EMA can stabilize training and improve inference quality:

```yaml
use_ema: true
ema_decay: 0.9999
```

The EMA model is automatically saved in checkpoints.

### Multi-GPU Training

The current script doesn't support distributed training out-of-the-box. To add DDP:

1. Wrap the model with `torch.nn.parallel.DistributedDataParallel`
2. Use `torch.distributed.launch` or `torchrun`
3. Modify the training script accordingly

## Troubleshooting

### Issue: Out of Memory

**Solutions:**
- Reduce `batch_size`
- Enable `gradient_accumulation_steps`
- Reduce `img_size`
- Set `num_workers: 0` (may slow down training)

### Issue: Loss Not Decreasing

**Possible causes:**
- Learning rate too high → Try reducing by 10x
- Learning rate too low → Try increasing by 10x
- Incorrect data normalization → Check images are in [-1, 1] range
- Model architecture mismatch → Verify pre-trained model is loaded correctly

### Issue: Training Too Slow

**Solutions:**
- Enable `use_amp: true` (mixed precision)
- Increase `num_workers`
- Increase `batch_size` (if memory allows)
- Use a GPU with more compute (V100, A100, etc.)

### Issue: Wandb Not Logging

**Solutions:**
- Run `wandb login` and enter your API key
- Use `--no-wandb` to disable wandb
- Check internet connection

## Performance Benchmarks

Approximate training speed on different hardware:

| GPU | Batch Size | Images/sec | Time per Epoch (27k images) |
|-----|------------|------------|----------------------------|
| RTX 3090 | 32 | ~150 | ~3 min |
| A100 40GB | 64 | ~300 | ~1.5 min |
| V100 32GB | 32 | ~100 | ~4.5 min |
| RTX 3060 | 16 | ~60 | ~7.5 min |

## Example Training Session

```bash
# 1. Prepare data
mkdir -p data/celeba_hq
# ... download CelebA-HQ ...

# 2. Start training
python train_flextok.py --config configs/train_celebahq.yaml

# Output:
# Loading FlexTok model...
# Loaded model: mit-han-lab/flextok-dfn-depth-12
#
# Creating dataloaders...
# Train batches: 843
# Val batches: 47
#
# Starting training from epoch 1 to 50
# Total training steps: 42150
# Device: cuda
# Mixed precision: True
# Gradient accumulation steps: 1
# Wandb logging: True
#
# Epoch 1/50: 100%|██████| 843/843 [03:12<00:00, 4.38it/s, loss=0.0234, lr=1.0e-4]
#
# Epoch 1/50 - Train Loss: 0.0234, Time: 192.34s
# Validating: 100%|██████| 47/47 [00:12<00:00, 3.92it/s]
# Validation Loss: 0.0198
# New best validation loss: 0.0198
# Saved checkpoint to checkpoints/celebahq_ft/checkpoint_latest.pt
# Saved epoch checkpoint to checkpoints/celebahq_ft/checkpoint_epoch_0001.pt
# Saved best checkpoint to checkpoints/celebahq_ft/checkpoint_best.pt
```

## Citation

If you use this training code, please cite the original FlexTok paper:

```bibtex
@article{flextok2024,
  title={FlexTok: Resampling Images into 1D Token Sequences of Flexible Length},
  author={...},
  journal={arXiv preprint arXiv:...},
  year={2024}
}
```

## Questions?

- Check the [FlexTok paper](https://flextok.epfl.ch)
- Open an issue on GitHub
- Review the [example notebook](notebooks/celebahq_dataloader_demo.ipynb)
