# ArcFace Loss Integration for FlexTok

This document describes the ArcFace identity preservation loss integration for FlexTok training.

## Overview

ArcFace loss has been integrated into FlexTok training to preserve facial identity during image reconstruction. This is particularly useful for face datasets where maintaining identity consistency is critical.

## Architecture

### Components

1. **ArcFaceModule** ([flextok/model/utils/arcface_loss.py](flextok/model/utils/arcface_loss.py))
   - Frozen pretrained ArcFace recognition model from InsightFace
   - Extracts 512-dim embeddings from images
   - Computes cosine similarity loss between original and reconstructed faces
   - Supports optional positive pairs for contrastive learning

2. **Identity-Aware Dataloader** ([flextok/utils/dataloader.py](flextok/utils/dataloader.py))
   - Returns identity labels with images
   - Optional custom collate function for same-identity pair sampling
   - Backward compatible (works without ArcFace enabled)

3. **Trainer Integration** ([train_flextok.py](train_flextok.py))
   - Computes combined loss: `flow_loss + repa_weight * repa_loss + arcface_weight * arcface_loss`
   - Handles both tensor and dictionary batch formats
   - Logs ArcFace metrics to wandb

## Usage

### Basic Configuration

Enable ArcFace loss in your config YAML:

```yaml
# Enable ArcFace identity preservation
use_arcface: true
arcface_weight: 0.5
arcface_model_name: 'buffalo_l'  # or 'buffalo_s', 'antelopev2'
arcface_root: '~/.insightface'
arcface_use_pairs: false
```

### Training

```bash
python train_flextok.py --config configs/train_celebahq.yaml
```

The training script will:
1. Auto-download the InsightFace model on first run (to `~/.insightface/models/`)
2. Load identity mappings from your dataset's `ids/` directory
3. Compute ArcFace loss alongside flow matching and REPA losses
4. Log metrics to wandb

### Dataset Structure

Your dataset should have identity labels available:

```
data/celeba/
├── images/
│   ├── 00000.jpg
│   ├── 00001.jpg
│   └── ...
└── ids/
    └── identity_labels.txt  # Format: "00000.jpg person_0001"
```

The `_BaseCelebADataset` class automatically:
- Reads the identity mapping file
- Creates `id_to_img_paths` and `img_path_to_id` dictionaries
- Returns identity labels when `return_identity=True`

## Configuration Options

### Required Parameters

- `use_arcface` (bool): Enable/disable ArcFace loss (default: `false`)
- `arcface_weight` (float): Loss weight for ArcFace term (default: `0.5`)

### Model Selection

- `arcface_model_name` (str): Choose from:
  - `'buffalo_l'`: Large model, best accuracy (~200MB, 512-dim embeddings)
  - `'buffalo_s'`: Small model, faster (~100MB, 512-dim embeddings)
  - `'antelopev2'`: Alternative model pack

### Advanced Options

- `arcface_root` (str): Directory for model storage (default: `'~/.insightface'`)
- `arcface_embedding_size` (int): Embedding dimension (default: `512`)
- `arcface_use_pairs` (bool): Enable positive pair sampling (default: `false`)

## How It Works

### 1. Forward Pass

```python
# Original images
original_images: (B, 3, 256, 256) in [-1, 1]

# FlexTok forward pass
data_dict = model(data_dict)

# Get predicted latents
pred_latents = data_dict['vae_latents_reconst']

# Decode to reconstructed images
reconstructed_images = vae.decode(pred_latents)
```

### 2. ArcFace Loss Computation

```python
# Extract embeddings (frozen ArcFace model)
original_emb = arcface.get_embeddings(original_images)      # (B, 512)
reconstructed_emb = arcface.get_embeddings(reconstructed)   # (B, 512)

# Compute cosine similarity loss
cos_sim = (original_emb * reconstructed_emb).sum(dim=1)
loss = (1 - cos_sim).mean()
```

### 3. Combined Loss

```python
total_loss = flow_matching_loss +
             repa_weight * repa_loss +
             arcface_weight * arcface_loss
```

## Positive Pairs (Optional)

Enable `arcface_use_pairs: true` for contrastive learning with same-identity pairs:

### How It Works

1. **Batch Collation**: Custom collate function groups images by identity
2. **Pair Sampling**: For each image, sample another image with same identity
3. **Loss Computation**:
   ```python
   # Original vs Reconstructed
   loss_recon = 1 - cos_sim(emb_original, emb_reconstructed)

   # Original vs Same-Identity Pair
   loss_pair = 1 - cos_sim(emb_original, emb_pair)

   # Combined
   arcface_loss = 0.5 * loss_recon + 0.5 * loss_pair
   ```

### When to Use Pairs

- **Enable** if: Dataset has multiple images per identity (e.g., VGGFace2, CASIA-WebFace)
- **Disable** if: Dataset has mostly single images per identity (e.g., CelebA)

## Monitoring

### WandB Metrics

The following metrics are logged:

- `train/arcface_loss`: ArcFace identity preservation loss
- `train/flow_loss`: Flow matching reconstruction loss
- `train/repa_loss`: REPA semantic preservation loss (if enabled)
- `train/loss`: Combined total loss

### Progress Bar

During training, you'll see:
```
Epoch 1/50: 100%|████| loss=0.1234 flow=0.0800 arc=0.0234 lr=1.0e-6
```

## Memory Usage

- **ArcFace Model**: ~200MB GPU memory (buffalo_l)
- **Embedding Computation**: Minimal overhead (~10MB for batch of 32)
- **Total Impact**: +5-10% GPU memory usage

## Troubleshooting

### Model Download Issues

If models don't auto-download:
```bash
# Manually download using insightface
python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_l')"
```

### No Identity Labels

If `ids/` directory doesn't exist, the dataloader will warn but continue:
```
Warning: No ID mapping file found. Continuing without ID mappings.
```

ArcFace loss will return 0 and training continues normally.

### ONNX Runtime Errors

Ensure onnxruntime is installed:
```bash
pip install onnxruntime-gpu  # For CUDA
# or
pip install onnxruntime  # For CPU
```

## Performance Tips

### 1. Loss Weight Tuning

Start with `arcface_weight: 0.5` and adjust based on results:
- **Too high**: Over-preservation of identity, loss of variation
- **Too low**: Identity drift in reconstructions

### 2. Model Selection

- Use `buffalo_l` for best identity preservation
- Use `buffalo_s` if memory or speed is constrained

### 3. Batch Size

ArcFace loss benefits from larger batches (more diverse faces):
- Recommended: 16-32 images per batch
- Minimum: 8 images per batch

## Implementation Files

1. `flextok/model/utils/arcface_loss.py` - ArcFace module
2. `flextok/utils/dataloader.py` - Identity-aware dataloader
3. `train_flextok.py` - Trainer integration
4. `configs/train_celebahq.yaml` - Configuration template

## Citation

If you use ArcFace loss in your work:

```bibtex
@inproceedings{deng2019arcface,
  title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={CVPR},
  year={2019}
}

@article{insightface,
  title={InsightFace: 2D and 3D Face Analysis Project},
  author={Guo, Jia and Deng, Jiankang and others},
  year={2018}
}
```

## Example Configurations

### Minimal (Identity Preservation Only)
```yaml
use_arcface: true
arcface_weight: 0.3
arcface_model_name: 'buffalo_s'
```

### Balanced (With REPA)
```yaml
use_repa: true
repa_weight: 1.0

use_arcface: true
arcface_weight: 0.5
arcface_model_name: 'buffalo_l'
```

### Maximum Identity Preservation
```yaml
use_arcface: true
arcface_weight: 1.0
arcface_model_name: 'buffalo_l'
arcface_use_pairs: true
```

## Next Steps

1. Set `use_arcface: true` in your config
2. Ensure your dataset has identity labels in `ids/` directory
3. Run training and monitor `train/arcface_loss`
4. Adjust `arcface_weight` based on reconstruction quality vs identity preservation trade-off

For questions or issues, refer to the implementation files or create an issue in the repository.
