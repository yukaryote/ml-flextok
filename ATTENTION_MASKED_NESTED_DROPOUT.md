# Attention-Masked Nested Dropout Implementation

This guide explains how to replace the learnable mask token approach with proper attention masking using FlexAttention.

## Overview

### Original Approach (Learnable Mask Token)
- Tokens beyond `keep_k` are replaced with a **learnable mask token**
- All tokens (including mask tokens) participate in attention
- Model learns to make mask token "semantically neutral"
- Requires extra learnable parameters
- Masked tokens still consume compute during attention

### New Approach (Attention Masking)
- Tokens beyond `keep_k` are **excluded from attention** via masking
- FlexAttention creates efficient block-sparse attention patterns
- No learnable mask token needed
- Masked tokens don't participate in attention at all
- More memory efficient and conceptually cleaner

## Architecture Changes

### 1. Replace `MaskedNestedDropout` with `AttentionMaskedNestedDropout`

**Old module** ([flextok/model/preprocessors/token_dropout.py](flextok/model/preprocessors/token_dropout.py)):
```python
class MaskedNestedDropout(nn.Module):
    def __init__(self, read_write_key, dim, ...):
        self.dropout_mask_token = nn.Parameter(torch.randn(dim))  # Learnable token

    def forward(self, data_dict):
        # Replaces tokens with mask token
        data_dict[self.read_write_key][i][:, keep_k:] = self.dropout_mask_token
```

**New module** ([flextok/model/preprocessors/attention_masked_nested_dropout.py](flextok/model/preprocessors/attention_masked_nested_dropout.py)):
```python
class AttentionMaskedNestedDropout(nn.Module):
    def __init__(self, read_key, keep_k_write_key, ...):
        # No learnable parameters!

    def forward(self, data_dict):
        # Only writes keep_k values, doesn't modify tokens
        data_dict[self.keep_k_write_key] = keep_ks
```

### 2. Replace `BlockWiseSequencePacker` with `NestedDropoutSequencePacker`

**Old packer**: Creates masks based only on sequence IDs (different images can't attend to each other)

**New packer**: Creates masks that combine:
- Block-wise masking (sequence separation)
- Nested dropout masking (tokens beyond keep_k are masked out)

## Implementation Steps

### Step 1: Update Decoder Configuration

Modify your decoder config to use the new modules:

```python
decoder:
  _target_: flextok.model.utils.wrappers.SequentialModuleDictWrapper
  module_dict:
    # ... other modules ...

    # REPLACE THIS:
    # dec_nested_dropout:
    #   _target_: flextok.model.preprocessors.token_dropout.MaskedNestedDropout
    #   read_write_key: dec_quants
    #   dim: 1792
    #   eval_keep_k_read_key: eval_keep_k
    #   train_keep_k_write_key: train_keep_k
    #   size_sampling_mode: uniform

    # WITH THIS:
    dec_nested_dropout:
      _target_: flextok.model.preprocessors.attention_masked_nested_dropout.AttentionMaskedNestedDropout
      read_key: dec_quants
      keep_k_write_key: nested_dropout_keep_k
      eval_keep_k_read_key: eval_keep_k
      train_keep_k_write_key: train_keep_k
      size_sampling_mode: uniform
      block_mask_write_key: nested_dropout_block_mask

    # ... other modules ...

    # REPLACE THIS:
    # dec_seq_packer:
    #   _target_: flextok.model.preprocessors.flex_seq_packing.BlockWiseSequencePacker
    #   input_list_read_keys: [dec_patches, dec_quants]
    #   packed_seq_write_key: dec_packed_seq
    #   block_mask_write_key: dec_sa_block_mask
    #   ...

    # WITH THIS:
    dec_seq_packer:
      _target_: flextok.model.preprocessors.nested_dropout_seq_packer.NestedDropoutSequencePacker
      input_list_read_keys: [dec_patches, dec_quants]
      packed_seq_write_key: dec_packed_seq
      block_mask_write_key: dec_sa_block_mask
      keep_k_read_key: nested_dropout_keep_k  # NEW: Read keep_k values
      inner_packed_shapes_write_key: dec_ps_inner
      outer_packed_shapes_write_key: dec_ps_outer
      mask_mode: full
      pad_to_multiple: 128
      compile_block_mask: true
```

### Step 2: Update Module Order

The module execution order should be:

1. **dec_from_latents**: Project FSQ embeddings to decoder dim
2. **dec_registers_posemb_module**: Add positional embeddings
3. **dec_nested_dropout**: Generate keep_k values (NEW - no token modification!)
4. **dec_latent_dropout**: Null conditioning
5. **dec_noise_channels_to_last**: Channel reordering
6. **dec_noise_patch_emb**: Patch embedding
7. **dec_patches_posemb_module**: Patch positional embeddings
8. **dec_seq_packer**: Pack sequences + create attention mask with nested dropout (MODIFIED)
9. **dec_time_embedder**: Time embeddings
10. **dec_transformer**: Transformer with masked attention
11. **dec_unpacker**: Unpack sequences
12. **dec_to_patches**: Output head

### Step 3: Verify Attention Masking

The new attention mask should:
- Allow tokens 0 to `keep_k-1` to attend to each other
- Prevent tokens `keep_k` to 255 from attending or being attended to
- Maintain block-wise separation between different images

## How It Works

### Training Example

With `keep_k = 64` sampled during training:

```
Packed sequence for one image:
┌─────────────────────┬──────────────────────────────────────┐
│  Image Patches      │       Register Tokens                │
│  (1024 tokens)      │       (256 tokens)                   │
├─────────────────────┼──────────────────┬───────────────────┤
│  p₀, p₁, ..., p₁₀₂₃ │ r₀, r₁, ..., r₆₃│ r₆₄, ..., r₂₅₅   │
├─────────────────────┼──────────────────┼───────────────────┤
│  NEVER MASKED       │  UNMASKED        │  MASKED (✗)       │
│  (always attend)    │  (keep_k=64)     │  (dropout)        │
└─────────────────────┴──────────────────┴───────────────────┘

Mask IDs:  [ 0,  0, ...,  0 ] [ 1,  1, ...,  1 ] [ -2, ..., -2 ]
            ←─ seq ID 0 ───→   ←─ seq ID 1 ──→   ←─ masked ──→

Attention Matrix (simplified):
                    Patches      Regs(0-63)   Regs(64-255)
                    p₀...p₁₀₂₃   r₀...r₆₃     r₆₄...r₂₅₅
    Patches         ✓✓✓✓✓✓✓      ✓✓✓✓✓        ✗✗✗✗✗
    p₀...p₁₀₂₃      ✓✓✓✓✓✓✓      ✓✓✓✓✓        ✗✗✗✗✗

    Registers       ✓✓✓✓✓✓✓      ✓✓✓✓✓        ✗✗✗✗✗
    r₀...r₆₃        ✓✓✓✓✓✓✓      ✓✓✓✓✓        ✗✗✗✗✗

    Masked Regs     ✗✗✗✗✗✗✗      ✗✗✗✗✗        ✗✗✗✗✗
    r₆₄...r₂₅₅      ✗✗✗✗✗✗✗      ✗✗✗✗✗        ✗✗✗✗✗

✓ = Can attend (mask = True)
✗ = Cannot attend (mask = False)

Key insight: Only the REGISTER tokens beyond keep_k are masked!
             The image patches are NEVER masked (always fully attended).
```

### Inference Example

With user-specified `eval_keep_k = 16`:

```
Only the first 16 tokens participate in attention.
Tokens 16-255 are completely ignored.
```

## Important Implementation Detail

**CRITICAL**: The nested dropout masking is applied **ONLY to the register tokens**, NOT to the image patches!

When the decoder processes:
- **Image patches** (e.g., 1024 tokens from 32×32 grid): Always fully attended, never masked
- **Register tokens** (e.g., 256 tokens): Subject to nested dropout based on `keep_k`

This is crucial because:
1. Image patches provide the spatial context from the noisy latents
2. Register tokens provide the compressed image representation from the encoder
3. We want hierarchical compression in the **registers**, not the patches
4. Masking patches would prevent the decoder from seeing the full spatial context

The implementation automatically detects this based on `input_list_read_keys`:
```python
# In dec_seq_packer config:
input_list_read_keys: [dec_patches, dec_quants]
# First = patches (never masked), Last = registers (masked based on keep_k)
```

## Benefits

1. **Memory Efficiency**: Masked register tokens don't participate in attention computation
2. **No Extra Parameters**: No learnable mask token to train
3. **Cleaner Semantics**: Masked = truly absent, not "neutral"
4. **FlexAttention Optimization**: Block-sparse patterns are optimized in PyTorch
5. **Easier Reasoning**: Attention patterns are explicit, not learned
6. **Patches Always Visible**: Full spatial context from noisy latents is always available

## Comparison

| Aspect | Learnable Mask Token | Attention Masking |
|--------|---------------------|-------------------|
| Token modification | Yes (replace with mask) | No (keep original) |
| Attention | Full (all 256 tokens) | Sparse (only keep_k tokens) |
| Parameters | +1792 (mask token) | 0 (no extra params) |
| Compute | O(256²) attention | O(keep_k²) attention |
| Semantics | Learned "null" | True masking |
| Memory | Higher | Lower |

## Testing

To verify the implementation works correctly:

```python
from flextok import FlexTokFromHub

# Load model with new config
model = FlexTokFromHub.from_pretrained("path/to/model/with/new/config")

# Test tokenization/detokenization
images = torch.randn(1, 3, 256, 256)
tokens = model.tokenize(images)

# Test with different keep_k values
for keep_k in [1, 16, 64, 256]:
    truncated_tokens = [t[:, :keep_k] for t in tokens]
    reconstructed = model.detokenize(truncated_tokens, vae_image_sizes=32)
    print(f"keep_k={keep_k}: shape={reconstructed.shape}")
```

## Training Considerations

1. **Gradient Flow**: Only tokens 0 to `keep_k-1` receive gradients during backprop
2. **Earlier Tokens Learn More**: First few tokens get trained more often (smaller keep_k more common)
3. **Hierarchical Encoding**: Model learns to put important info in earlier tokens
4. **Stability**: Should be more stable than learnable mask token (no adversarial mask learning)

## Migration Path

If you have a model trained with learnable mask tokens:

1. You **cannot** directly load old checkpoints (different architecture)
2. You need to **retrain from scratch** with the new approach
3. Or implement a hybrid mode that supports both (not recommended)

The attention-masked approach is cleaner for new training runs!
