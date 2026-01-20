"""
VLM-based perceptual similarity scorers for the twenty questions game.

This module provides Vision-Language Model (VLM) based alternatives to DreamSim
for computing perceptual similarity between images. The scorers are designed to
be drop-in replacements for DreamSim in the auto_twenty_q() function.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Literal
import re
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np


class VLMSimilarityScorer(ABC):
    """
    Base class for VLM-based similarity scoring.

    All scorers should implement the __call__ method to compute similarity
    between two images, returning a scalar score where lower values indicate
    higher similarity (matching DreamSim's distance metric).
    """

    def __init__(self, device: str = 'cuda', cache_dir: Optional[str] = None):
        """
        Initialize the VLM similarity scorer.

        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            cache_dir: Optional directory to cache model weights
        """
        self.device = device
        self.cache_dir = cache_dir

    @abstractmethod
    def __call__(self, image1: Union[Image.Image, torch.Tensor],
                 image2: Union[Image.Image, torch.Tensor]) -> float:
        """
        Compute similarity score between two images.

        Args:
            image1: First image (reference/secret image)
            image2: Second image (candidate image)

        Returns:
            Similarity score (lower = more similar)
        """
        pass

    def _to_pil(self, image: Union[Image.Image, torch.Tensor]) -> Image.Image:
        """Convert tensor to PIL Image if needed."""
        if isinstance(image, torch.Tensor):
            # Assume image is in [0, 1] range, shape (C, H, W) or (B, C, H, W)
            if image.ndim == 4:
                image = image[0]  # Take first image from batch
            image = image.cpu().numpy()
            if image.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                image = np.transpose(image, (1, 2, 0))
            image = (image * 255).astype(np.uint8)
            return Image.fromarray(image)
        return image


class VLMQASimilarityScorer(VLMSimilarityScorer):
    """
    VLM Question-Answering based similarity scorer.

    Uses a Vision-Language Model to analyze and compare images by asking
    detailed questions about facial identity and geometry similarity.
    """

    def __init__(
        self,
        model_name: str = 'llava-hf/llava-1.5-7b-hf',
        device: str = 'cuda',
        cache_dir: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        batch_mode: bool = False,
    ):
        """
        Initialize the VLM QA-based scorer.

        Args:
            model_name: HuggingFace model ID or path. Supported models:
                - 'llava-hf/llava-1.5-7b-hf' (default)
                - 'llava-hf/llava-1.5-13b-hf'
                - 'Qwen/Qwen-VL-Chat'
                - 'THUDM/cogvlm-chat-hf'
            device: Device to run the model on
            cache_dir: Optional directory to cache model weights
            load_in_4bit: Whether to load model in 4-bit quantization
            load_in_8bit: Whether to load model in 8-bit quantization
            batch_mode: Whether to enable batch processing (experimental)
        """
        super().__init__(device, cache_dir)
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.batch_mode = batch_mode

        # Lazy loading - will load on first call
        self.model = None
        self.processor = None
        self._model_type = self._detect_model_type()

    def _detect_model_type(self) -> Literal['llava', 'qwen', 'cogvlm', 'generic']:
        """Detect model type from model name."""
        name_lower = self.model_name.lower()
        if 'llava' in name_lower:
            return 'llava'
        elif 'qwen' in name_lower:
            return 'qwen'
        elif 'cogvlm' in name_lower:
            return 'cogvlm'
        return 'generic'

    def _load_model(self):
        """Lazily load the VLM model and processor."""
        if self.model is not None:
            return

        print(f"Loading VLM model: {self.model_name}...")

        if self._model_type == 'llava':
            self._load_llava()
        elif self._model_type == 'qwen':
            self._load_qwen()
        elif self._model_type == 'cogvlm':
            self._load_cogvlm()
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        print("VLM model loaded successfully!")

    def _load_llava(self):
        """Load LLaVA model."""
        from transformers import LlavaForConditionalGeneration, AutoProcessor

        kwargs = {}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True)
        elif self.load_in_8bit:
            from transformers import BitsAndBytesConfig
            kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs['torch_dtype'] = torch.float16

        if self.cache_dir:
            kwargs['cache_dir'] = self.cache_dir

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            **kwargs
        )

        if not (self.load_in_4bit or self.load_in_8bit):
            self.model = self.model.to(self.device)

        self.model.eval()

    def _load_qwen(self):
        """Load Qwen-VL model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = {'trust_remote_code': True}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True)
        elif self.load_in_8bit:
            from transformers import BitsAndBytesConfig
            kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs['torch_dtype'] = torch.float16

        if self.cache_dir:
            kwargs['cache_dir'] = self.cache_dir

        self.processor = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=self.cache_dir
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **kwargs
        )

        if not (self.load_in_4bit or self.load_in_8bit):
            self.model = self.model.to(self.device)

        self.model.eval()

    def _load_cogvlm(self):
        """Load CogVLM model."""
        raise NotImplementedError("CogVLM support coming soon!")

    def _create_comparison_prompt(self, is_batch: bool = False) -> str:
        """
        Create a detailed prompt for comparing facial identity and geometry.

        Args:
            is_batch: Whether this is for batch processing

        Returns:
            Prompt string
        """
        if is_batch:
            prompt = """You are an expert in facial recognition and geometry analysis. You will be shown a reference face (the "secret" face) and a candidate face.

Your task: Analyze how similar the candidate face is to the reference face in terms of:
1. IDENTITY: Do they appear to be the same person? Consider facial features, bone structure, and unique characteristics.
2. FACIAL GEOMETRY: Compare the spatial arrangement and proportions of facial features (eyes, nose, mouth, jawline, cheekbones, etc.)
3. OVERALL RESEMBLANCE: How closely does the candidate match the reference?

Provide a similarity score from 0 to 10, where:
- 0 = Completely different person with very different facial structure
- 5 = Different people but with some similar facial features or geometry
- 10 = Identical or nearly identical in both identity and facial geometry

IMPORTANT: Respond with ONLY a number between 0 and 10. Do not include any explanation, just the number.

Your score:"""
        else:
            prompt = """You are an expert in facial recognition and geometry analysis. I will show you two face images.

IMAGE 1 is the REFERENCE face (the "secret" face we're trying to match).
IMAGE 2 is a CANDIDATE face (a potential match).

Your task: Analyze how similar IMAGE 2 is to IMAGE 1 in terms of:
1. IDENTITY: Do they appear to be the same person? Consider:
   - Overall facial features and bone structure
   - Unique identifying characteristics
   - Facial proportions and symmetry

2. FACIAL GEOMETRY: Compare the spatial arrangement of features:
   - Eye shape, size, and spacing
   - Nose shape and position
   - Mouth shape and position
   - Jawline and face shape
   - Cheekbone structure
   - Forehead and chin proportions

3. OVERALL RESEMBLANCE: Considering both identity and geometry, how closely does the candidate match the reference?

Provide a similarity score from 0 to 10, where:
- 0 = Completely different person with very different facial structure
- 3 = Different people with notably different features
- 5 = Different people but with some similar facial features or geometry
- 7 = Could be the same person with some differences (angle, expression, etc.)
- 10 = Identical or nearly identical in both identity and facial geometry

IMPORTANT: Respond with ONLY a number between 0 and 10 (can use decimals like 7.5). Do not include any explanation, just the number.

Your similarity score:"""

        return prompt

    def _parse_score(self, response: str) -> float:
        """
        Parse numerical score from VLM response.

        Args:
            response: Raw text response from VLM

        Returns:
            Parsed score as float
        """
        # Try to extract a number from the response
        # Look for patterns like "8.5", "8", "score: 7.5", etc.
        patterns = [
            r'(\d+\.?\d*)\s*(?:/\s*10)?',  # Match "8.5" or "8.5/10"
            r'score:\s*(\d+\.?\d*)',  # Match "score: 8.5"
            r'(\d+\.?\d*)\s*out of',  # Match "8.5 out of"
        ]

        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                try:
                    score = float(match.group(1))
                    # Clamp to [0, 10] range
                    score = max(0.0, min(10.0, score))
                    return score
                except ValueError:
                    continue

        # If no number found, try to find any float/int in the text
        numbers = re.findall(r'\d+\.?\d*', response)
        if numbers:
            try:
                score = float(numbers[0])
                score = max(0.0, min(10.0, score))
                return score
            except ValueError:
                pass

        # Default to middle score if parsing fails
        print(f"Warning: Could not parse score from response: '{response}'. Using default score 5.0")
        return 5.0

    def _query_llava(self, prompt: str, images: list[Image.Image]) -> str:
        """Query LLaVA model with images and prompt."""
        # Prepare conversation format for LLaVA
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        # Process inputs
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )

        # Add images
        inputs['pixel_values'] = self.processor.image_processor(
            images,
            return_tensors="pt"
        )['pixel_values']

        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.0,
            )

        # Decode response
        generated_text = self.processor.decode(output[0], skip_special_tokens=True)

        # Extract just the assistant's response
        if "ASSISTANT:" in generated_text:
            response = generated_text.split("ASSISTANT:")[-1].strip()
        else:
            response = generated_text.strip()

        return response

    def _query_qwen(self, prompt: str, images: list[Image.Image]) -> str:
        """Query Qwen-VL model with images and prompt."""
        # Qwen-VL uses special image tokens
        # For now, simplified implementation - may need adjustment based on Qwen-VL API
        query = f"Picture 1: <img></img>\nPicture 2: <img></img>\n{prompt}"

        inputs = self.processor(
            query,
            return_tensors='pt',
        ).to(self.device)

        # Note: Actual Qwen-VL implementation may need image preprocessing
        # This is a placeholder - adjust based on actual Qwen-VL API

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
            )

        response = self.processor.decode(output[0], skip_special_tokens=True)
        return response.strip()

    def __call__(self, image1: Union[Image.Image, torch.Tensor],
                 image2: Union[Image.Image, torch.Tensor]) -> float:
        """
        Compute similarity score between two images using VLM.

        Args:
            image1: Reference/secret image
            image2: Candidate image to compare

        Returns:
            Similarity distance (lower = more similar, to match DreamSim API)
        """
        # Lazy load model
        self._load_model()

        # Convert to PIL if needed
        img1_pil = self._to_pil(image1)
        img2_pil = self._to_pil(image2)

        # Create prompt
        prompt = self._create_comparison_prompt(is_batch=False)

        # Query model based on type
        if self._model_type == 'llava':
            response = self._query_llava(prompt, [img1_pil, img2_pil])
        elif self._model_type == 'qwen':
            response = self._query_qwen(prompt, [img1_pil, img2_pil])
        else:
            raise NotImplementedError(f"Model type {self._model_type} not implemented")

        # Parse score (0-10, higher = more similar)
        similarity_score = self._parse_score(response)

        # Convert to distance metric (lower = more similar, matching DreamSim)
        # Scale from [0, 10] to approximate DreamSim range
        # DreamSim typically outputs values in [0, 1] or [0, 2] range
        distance = (10.0 - similarity_score) / 10.0

        return distance


class CLIPSimilarityScorer(VLMSimilarityScorer):
    """
    CLIP embedding-based similarity scorer.

    Fast alternative that computes cosine similarity between CLIP image embeddings.
    Much faster than VLM QA approach but may be less semantically aware.
    """

    def __init__(
        self,
        model_name: str = 'ViT-B/16',
        device: str = 'cuda',
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize CLIP-based scorer.

        Args:
            model_name: CLIP model variant ('ViT-B/16', 'ViT-L/14', etc.)
            device: Device to run the model on
            cache_dir: Optional directory to cache model weights
        """
        super().__init__(device, cache_dir)
        self.model_name = model_name

        # Lazy loading
        self.model = None
        self.preprocess = None

    def _load_model(self):
        """Lazily load CLIP model."""
        if self.model is not None:
            return

        print(f"Loading CLIP model: {self.model_name}...")

        try:
            import clip
            self.model, self.preprocess = clip.load(
                self.model_name,
                device=self.device,
                download_root=self.cache_dir
            )
            self.model.eval()
            print("CLIP model loaded successfully!")
        except ImportError:
            print("CLIP not installed. Trying OpenCLIP...")
            import open_clip
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                device=self.device,
                cache_dir=self.cache_dir
            )
            self.model.eval()
            print("OpenCLIP model loaded successfully!")

    def __call__(self, image1: Union[Image.Image, torch.Tensor],
                 image2: Union[Image.Image, torch.Tensor]) -> float:
        """
        Compute similarity using CLIP embeddings.

        Args:
            image1: Reference/secret image
            image2: Candidate image to compare

        Returns:
            Similarity distance (lower = more similar)
        """
        # Lazy load model
        self._load_model()

        # Convert to PIL if needed
        img1_pil = self._to_pil(image1)
        img2_pil = self._to_pil(image2)

        # Preprocess images
        img1_tensor = self.preprocess(img1_pil).unsqueeze(0).to(self.device)
        img2_tensor = self.preprocess(img2_pil).unsqueeze(0).to(self.device)

        # Get embeddings
        with torch.no_grad():
            emb1 = self.model.encode_image(img1_tensor)
            emb2 = self.model.encode_image(img2_tensor)

            # Normalize embeddings
            emb1 = F.normalize(emb1, dim=-1)
            emb2 = F.normalize(emb2, dim=-1)

            # Cosine similarity
            similarity = (emb1 * emb2).sum(dim=-1).item()

        # Convert similarity [-1, 1] to distance [0, 2]
        # Higher similarity -> lower distance
        distance = 1.0 - similarity

        return distance
