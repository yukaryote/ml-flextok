# REPA Loss Implementation for FlexTok

This document describes the implementation of REPA (REPresentation Alignment) loss for FlexTok fine-tuning, based on the paper "Representation Alignment for Generation" (https://arxiv.org/pdf/2410.06940).

## Overview

REPA loss is an auxiliary loss that aligns intermediate decoder representations with frozen pretrained vision encoder features (DINOv2). This helps the model learn better semantic representations during training by ensuring that the decoder's intermediate activations are consistent with high-quality pre-trained features.

The total loss is: `L = L_flow_matching + λ * L_REPA`

Where the paper recommends λ = 1.0 for equal weighting.

## Implementation Details

### Architecture

1. **Intermediate Layer Extraction**: The FlexTok decoder already outputs layer 1 features at key `'dec_packed_seq_repa_layer'`

2. **REPA Module Components**:
   - **Frozen Encoder**: DINOv2 (vitl14/vitg14/vitb14) - loads from torch.hub
   - **Trainable Projector**: 3-layer MLP that projects decoder features to encoder dimension
   - **Loss Function**: Negative cosine similarity between projected decoder features and encoder features

3. **Feature Flow**:
   ```
   Input Images (B, 3, H, W)
   ↓
   FlexTok Forward Pass
   ↓
   Extract 'dec_packed_seq_repa_layer' (layer 1 activations)
   ↓
   Unpack using 'dec_repa_unpacker' → Spatial features
   ↓
   Resize to target size (37×37)
   ↓
   Project with MLP → (B, encoder_dim, 37, 37)
   ↓
   ← Extract target features from DINOv2(images)
   ↓
   Normalize both to unit length
   ↓
   Compute -cosine_similarity → REPA loss
   ```

## Files Modified/Created

### Created Files

1. **`flextok/model/utils/repa_loss.py`**
   - `REPAProjector`: 3-layer MLP projector
   - `REPAModule`: Complete module with frozen encoder and trainable projector
   - Handles DINOv2 loading, preprocessing, and loss computation

### Modified Files

1. **`train_flextok.py`**
   - Added REPA module initialization in `__init__` (lines 112-134)
   - Added REPA projector parameters to optimizer (lines 177-183)
   - Modified `compute_loss` to include REPA loss (lines 303-307)
   - Added `compute_repa_loss` method (lines 337-372)
   - Updated logging to track flow_loss and repa_loss separately (lines 427-454)
   - Added REPA state to checkpointing (save: line 672-673, load: lines 721-723)

2. **`configs/train_celebahq.yaml`**
   - Added REPA configuration section (lines 150-172)

## Configuration Parameters

```yaml
# Enable/disable REPA loss
use_repa: false  # Set to true to enable

# Loss weight (paper recommends 1.0)
repa_weight: 1.0

# Encoder selection
repa_encoder_type: 'dinov2_vitl14'  # Options: vitl14, vitg14, vitb14
repa_encoder_dim: 1024  # L:1024, G:1536, B:768

# Feature matching resolution
repa_target_size: [37, 37]  # As per paper

# Learning rate for REPA projector
repa_lr: 1.0e-4
```

## Usage

### Training with REPA

1. **Enable REPA in config**:
   ```yaml
   use_repa: true
   repa_weight: 1.0
   ```

2. **Run training**:
   ```bash
   python train_flextok.py --config configs/train_celebahq.yaml
   ```

3. **Monitor in WandB**:
   - `train/loss`: Combined loss (flow + repa)
   - `train/flow_loss`: Flow matching reconstruction loss
   - `train/repa_loss`: REPA alignment loss

### Training without REPA (baseline)

Simply set `use_repa: false` in the config. The code is fully backward compatible.

## Memory Considerations

- **Frozen DINOv2-L**: ~300MB of frozen parameters
- **Trainable Projector**: ~10-20MB depending on decoder dimension
- **Intermediate Features**: Stored temporarily during forward pass
- **Total overhead**: ~320MB + activation memory

If OOM occurs:
- Reduce batch size slightly
- Use smaller encoder (vitb14 instead of vitl14)
- Reduce `repa_target_size` to [32, 32]

## Expected Behavior

### Loss Values

- **Flow Loss**: Should be similar to baseline (e.g., 0.01-0.1)
- **REPA Loss**: Negative cosine similarity, typically -0.5 to -0.9 (more negative = better alignment)
- **Total Loss**: Flow loss + repa_weight * repa_loss

### Training Dynamics

1. **Initial Phase**: REPA loss may be high (-0.3 to -0.5)
2. **Mid Training**: Should decrease to -0.7 to -0.9
3. **Convergence**: Both losses should stabilize

If REPA loss doesn't decrease:
- Check that intermediate features are being extracted correctly
- Verify unpacker is producing correct spatial format
- Try adjusting `repa_weight` or `repa_lr`

## Debugging

### Check Intermediate Features

```python
# In compute_repa_loss, add:
print(f"Intermediate features shape: {intermediate_features.shape}")
print(f"Decoder features shape: {decoder_features.shape}")
print(f"Target features shape: {target_features.shape}")
```

### Verify REPA Module Loaded

```python
# After trainer initialization:
if trainer.use_repa:
    print(f"REPA module loaded: {hasattr(trainer, 'repa_module')}")
    print(f"Encoder device: {next(trainer.repa_module.encoder.parameters()).device}")
    print(f"Projector trainable params: {sum(p.numel() for p in trainer.repa_module.projector.parameters())}")
```

### Test Forward Pass

```python
# Test with dummy batch:
dummy_batch = torch.randn(2, 3, 256, 256).cuda() * 2 - 1  # [-1, 1]
with torch.no_grad():
    metrics = trainer.compute_loss(dummy_batch)
    print(f"Flow loss: {metrics['flow_loss'].item():.4f}")
    print(f"REPA loss: {metrics['repa_loss'].item():.4f}")
```

## Implementation Notes

### Key Design Decisions

1. **Intermediate Layer Selection**: Uses layer 1 of decoder (first layer after input)
   - Early enough to capture high-level features
   - Before too much generation-specific processing

2. **Target Size 37×37**: Matches paper specification
   - 256÷8 = 32 (VAE downsampling)
   - Additional upsampling to 37×37 for richer spatial resolution

3. **3-Layer MLP**: Matches REPA paper architecture
   - Projects decoder_dim → encoder_dim
   - Uses SiLU activation (smooth, unbounded)

4. **Cosine Similarity**: Measures alignment regardless of magnitude
   - Normalized features ensure scale-invariant comparison
   - Negative loss encourages high similarity

### Potential Issues & Solutions

**Issue**: `KeyError: 'dec_packed_seq_repa_layer'`
- **Solution**: Verify model has intermediate layer output enabled. Check that `intermediate_layer_write_key` is set correctly in decoder transformer.

**Issue**: Dimension mismatch in projector
- **Solution**: Ensure `decoder_dim` matches transformer dimension. Check with:
  ```python
  print(self.model.decoder.module_dict['dec_transformer'].dim)
  ```

**Issue**: REPA loss is always 0
- **Solution**: Check `use_repa=True` and `repa_weight > 0`. Verify intermediate features are not None.

**Issue**: Out of memory
- **Solution**: Reduce batch size, use smaller encoder (vitb14), or disable REPA temporarily.

## Ablation Studies

To evaluate REPA effectiveness:

1. **Baseline**: Train with `use_repa: false`
2. **REPA**: Train with `use_repa: true`
3. **Compare**:
   - Reconstruction quality (FID, LPIPS)
   - Token clustering (silhouette score on first tokens)
   - Visual quality of flexible-length reconstructions

## Future Improvements

1. **Multi-layer REPA**: Align multiple decoder layers instead of just layer 1
2. **Dynamic weighting**: Adjust `repa_weight` during training (warmup/cooldown)
3. **Other encoders**: Support CLIP, MAE, or other vision encoders
4. **Adaptive target size**: Automatically compute optimal spatial size
5. **Feature caching**: Cache encoder features to reduce computation

## References

- REPA Paper: https://arxiv.org/pdf/2410.06940
- REPA Code: /home/iyu/REPA
- FlexTok Paper: https://arxiv.org/abs/2410.01309
- DINOv2: https://github.com/facebookresearch/dinov2

## License

This implementation follows the same license as FlexTok. The REPA loss concept is based on the publicly available research paper.
