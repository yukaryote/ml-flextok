# VLM-Based Perceptual Similarity Scorer

This module provides Vision-Language Model (VLM) based alternatives to DreamSim for computing perceptual similarity between facial images in the twenty questions game.

## Overview

The VLM scorer is designed as a drop-in replacement for DreamSim in the `auto_twenty_q()` function. It uses sophisticated Vision-Language Models to evaluate facial similarity based on identity and geometric features.

## Available Scorers

### 1. VLMQASimilarityScorer (Recommended for Semantic Understanding)

Uses a Vision-Language Model with detailed prompts to analyze facial identity and geometry.

**Features:**
- Analyzes facial identity (same person vs different people)
- Evaluates facial geometry (feature positions, proportions, bone structure)
- Uses detailed prompts for comprehensive comparison
- Returns normalized distance scores (lower = more similar)

**Supported Models:**
- `llava-hf/llava-1.5-7b-hf` (default, good balance)
- `llava-hf/llava-1.5-13b-hf` (better quality, slower)
- `Qwen/Qwen-VL-Chat` (experimental)

**Advantages:**
- Deep semantic understanding of facial features
- Considers both identity and geometry explicitly
- Can distinguish subtle facial differences
- More interpretable reasoning

**Disadvantages:**
- Slower than embedding-based approaches
- Requires more GPU memory
- Response parsing may occasionally fail

### 2. CLIPSimilarityScorer (Fast Alternative)

Uses CLIP image embeddings for rapid similarity computation.

**Features:**
- Fast cosine similarity between image embeddings
- No language model overhead
- Reliable and deterministic

**Supported Models:**
- `ViT-B/16` (default, fast)
- `ViT-L/14` (better quality, slower)
- Any OpenCLIP model

**Advantages:**
- Very fast inference
- Low memory footprint
- Deterministic results
- Similar to DreamSim in approach

**Disadvantages:**
- Less semantic understanding than VLM
- May miss subtle identity differences
- Limited to embedding space similarity

## Usage

### Basic Usage

```python
from twenty_questions.vlm_scorer import VLMQASimilarityScorer, CLIPSimilarityScorer

# Initialize VLM scorer
vlm_scorer = VLMQASimilarityScorer(
    model_name='llava-hf/llava-1.5-7b-hf',
    device='cuda',
    load_in_4bit=True,  # Reduce memory usage
)

# Use in auto_twenty_q
chosen_history, rejected_history = auto_twenty_q(
    flextok_model=model,
    secret_image=secret_image_pil,
    eval_model=vlm_scorer,  # Drop-in replacement for DreamSim
    num_questions=256
)
```

### Advanced Configuration

```python
# VLM with 8-bit quantization (faster, less memory)
vlm_scorer = VLMQASimilarityScorer(
    model_name='llava-hf/llava-1.5-7b-hf',
    device='cuda',
    load_in_8bit=True,
    cache_dir='./models/vlm_cache'  # Custom cache directory
)

# CLIP scorer with larger model
clip_scorer = CLIPSimilarityScorer(
    model_name='ViT-L/14',
    device='cuda'
)

# Test similarity between two images
from PIL import Image
img1 = Image.open('face1.jpg')
img2 = Image.open('face2.jpg')

distance = vlm_scorer(img1, img2)
print(f"Similarity distance: {distance:.4f}")
# Lower values = more similar
```

### Memory Optimization

For limited GPU memory, use quantization:

```python
# 4-bit quantization (lowest memory, ~7GB for LLaVA-7B)
vlm_scorer = VLMQASimilarityScorer(
    model_name='llava-hf/llava-1.5-7b-hf',
    load_in_4bit=True
)

# 8-bit quantization (medium memory, ~14GB for LLaVA-7B)
vlm_scorer = VLMQASimilarityScorer(
    model_name='llava-hf/llava-1.5-7b-hf',
    load_in_8bit=True
)

# Full precision (highest quality, ~28GB for LLaVA-7B)
vlm_scorer = VLMQASimilarityScorer(
    model_name='llava-hf/llava-1.5-7b-hf'
)
```

## The Detailed Prompt

The VLM scorer uses a comprehensive prompt that asks the model to evaluate:

1. **Identity Similarity**: Whether the images show the same person
   - Facial features and bone structure
   - Unique identifying characteristics
   - Facial proportions and symmetry

2. **Facial Geometry**: Spatial arrangement of features
   - Eye shape, size, and spacing
   - Nose shape and position
   - Mouth shape and position
   - Jawline and face shape
   - Cheekbone structure
   - Forehead and chin proportions

3. **Overall Resemblance**: Combined assessment of identity and geometry

The model returns a score from 0-10, which is converted to a distance metric (lower = more similar) to match the DreamSim API.

## API Compatibility

Both scorers implement the same interface as DreamSim:

```python
# DreamSim API
dreamsim_model, preprocess = dreamsim(pretrained=True, device='cuda')
score = dreamsim_model(preprocessed_img1, preprocessed_img2)

# VLM Scorer API (drop-in replacement)
vlm_scorer = VLMQASimilarityScorer(device='cuda')
score = vlm_scorer(img1, img2)  # No preprocessing needed
```

**Key differences:**
- VLM scorer handles preprocessing internally (no separate preprocess function)
- VLM scorer accepts PIL Images or torch Tensors directly
- Score ranges are normalized to be comparable

## Performance Considerations

### Speed Comparison (approximate)

| Model | Speed | Memory | Quality |
|-------|-------|--------|---------|
| DreamSim | ~10ms | ~2GB | Good |
| CLIP ViT-B/16 | ~15ms | ~1GB | Good |
| CLIP ViT-L/14 | ~25ms | ~2GB | Better |
| LLaVA-7B (4-bit) | ~500ms | ~7GB | Best (semantic) |
| LLaVA-7B (full) | ~300ms | ~28GB | Best (semantic) |
| LLaVA-13B (4-bit) | ~800ms | ~13GB | Best+ (semantic) |

### Recommendations

**For speed**: Use `CLIPSimilarityScorer` with `ViT-B/16`
**For quality**: Use `VLMQASimilarityScorer` with `llava-1.5-13b-hf`
**For balance**: Use `VLMQASimilarityScorer` with `llava-1.5-7b-hf` and 4-bit quantization

## Troubleshooting

### Out of Memory

```python
# Use 4-bit quantization
vlm_scorer = VLMQASimilarityScorer(load_in_4bit=True)

# Or switch to CLIP
clip_scorer = CLIPSimilarityScorer()
```

### Slow Inference

```python
# Use CLIP for faster scoring
clip_scorer = CLIPSimilarityScorer(model_name='ViT-B/16')

# Or use smaller VLM model
vlm_scorer = VLMQASimilarityScorer(
    model_name='llava-hf/llava-1.5-7b-hf',  # Not 13b
    load_in_4bit=True
)
```

### Score Parsing Errors

If the VLM returns non-numerical responses:
- Check that the model is properly loaded
- Try a different model (LLaVA models are most reliable)
- Fallback to CLIP scorer if issues persist

### Installation Requirements

```bash
# For VLM scorer
pip install transformers accelerate bitsandbytes

# For CLIP scorer
pip install openai-clip
# or
pip install open-clip-torch
```

## Examples

See `notebooks/search_twenty_questions.ipynb` for complete examples of:
- Loading and testing the VLM scorer
- Running auto_twenty_q with VLM evaluation
- Comparing results with DreamSim

## Implementation Details

### Score Conversion

VLM returns similarity scores 0-10 (higher = more similar).
These are converted to distance: `distance = (10 - score) / 10`
This ensures consistency with DreamSim's distance metric.

### Lazy Loading

Models are loaded on first use to avoid unnecessary initialization:
```python
vlm_scorer = VLMQASimilarityScorer()  # No loading yet
score = vlm_scorer(img1, img2)  # Model loads here
```

### Tensor Handling

The scorer automatically converts between formats:
- PIL Image → used directly
- torch.Tensor (C,H,W) or (B,C,H,W) → converted to PIL
- Assumes tensors are in [0,1] range with ImageNet normalization

## Citation

If you use this VLM scorer in your research, please cite the underlying models:

**LLaVA:**
```bibtex
@misc{liu2023llava,
    title={Visual Instruction Tuning},
    author={Haotian Liu and Chunyuan Li and Qingyang Wu and Yong Jae Lee},
    year={2023},
    eprint={2304.08485},
    archivePrefix={arXiv}
}
```

**CLIP:**
```bibtex
@inproceedings{radford2021learning,
    title={Learning Transferable Visual Models From Natural Language Supervision},
    author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others},
    booktitle={ICML},
    year={2021}
}
```
