"""
ArcFace Identity Preservation Loss Module

This module provides identity-preserving loss using pretrained ArcFace models
from InsightFace. It computes embeddings for original and reconstructed images
and uses cosine similarity to ensure identity consistency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import os
import sys
from pathlib import Path


class ArcFaceModule(nn.Module):
    """
    Complete ArcFace module with frozen pretrained recognition model.

    Uses InsightFace's pretrained ArcFace models via ONNX runtime for
    extracting face embeddings and computing identity preservation loss.
    """

    def __init__(
        self,
        model_name: str = 'buffalo_l',
        root: str = '~/.insightface',
        embedding_size: int = 512,
    ):
        """
        Initialize ArcFace module with pretrained model.

        Args:
            model_name: Name of insightface model pack ('buffalo_l', 'buffalo_s', 'antelopev2')
            root: Root directory for insightface models
            embedding_size: Size of face embeddings (typically 512)
        """
        super().__init__()
        self.model_name = model_name
        self.root = os.path.expanduser(root)
        self.embedding_size = embedding_size

        # Load frozen ArcFace model
        self.recognition_model = self._load_recognition_model()

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def _load_recognition_model(self) -> nn.Module:
        """
        Load pretrained ArcFace recognition model from InsightFace.

        Returns:
            Loaded recognition model (frozen)
        """
        try:
            # Add insightface to path if needed
            insightface_path = Path(__file__).parent.parent.parent.parent.parent / 'insightface' / 'python-package'
            if insightface_path.exists() and str(insightface_path) not in sys.path:
                sys.path.insert(0, str(insightface_path))

            from insightface.app import FaceAnalysis

            # Initialize FaceAnalysis with only recognition module
            app = FaceAnalysis(
                name=self.model_name,
                root=self.root,
                allowed_modules=['recognition']
            )
            app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)

            # Extract the recognition model
            if 'recognition' not in app.models:
                raise ValueError(f"Recognition model not found in {self.model_name}")

            recognition_model = app.models['recognition']

            print(f"Loaded ArcFace recognition model: {self.model_name}")
            print(f"  Input size: {recognition_model.input_size}")
            print(f"  Embedding size: {recognition_model.output_shape}")

            return recognition_model

        except Exception as e:
            raise RuntimeError(
                f"Failed to load ArcFace model '{self.model_name}' from {self.root}. "
                f"Error: {e}\n"
                f"Please ensure insightface is installed and models are downloaded."
            )

    def preprocess_images(self, images: torch.Tensor) -> np.ndarray:
        """
        Preprocess images for ArcFace model.

        Args:
            images: Tensor of shape (B, 3, H, W) normalized to [-1, 1]

        Returns:
            Numpy array ready for ArcFace input (B, 3, 112, 112) in BGR format
        """
        # Denormalize from [-1, 1] to [0, 255]
        images = (images + 1.0) * 127.5
        images = images.clamp(0, 255)

        # Resize to ArcFace input size (typically 112x112)
        target_size = self.recognition_model.input_size  # (112, 112)
        images = F.interpolate(
            images,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )

        # Convert to numpy and BGR format (ArcFace expects BGR)
        images_np = images.cpu().numpy()
        images_np = images_np[:, [2, 1, 0], :, :]  # RGB to BGR
        images_np = images_np.transpose(0, 2, 3, 1)  # (B, 3, H, W) -> (B, H, W, 3)

        return images_np.astype(np.uint8)

    @torch.no_grad()
    def get_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract ArcFace embeddings from images.

        Args:
            images: Tensor of shape (B, 3, H, W) normalized to [-1, 1]

        Returns:
            Embeddings of shape (B, embedding_size), L2-normalized
        """
        # Preprocess images
        images_np = self.preprocess_images(images)

        # Extract embeddings for each image
        embeddings = []
        for img_np in images_np:
            # Get features using ArcFace model
            # Note: get_feat expects images in proper format
            feat = self.recognition_model.get_feat(img_np)
            embeddings.append(feat)

        # Stack and convert to torch tensor
        embeddings = np.stack(embeddings, axis=0)
        embeddings = torch.from_numpy(embeddings).to(images.device)

        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def forward(
        self,
        original_images: torch.Tensor,
        reconstructed_images: torch.Tensor,
        positive_pairs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute ArcFace identity preservation loss.

        Args:
            original_images: Original input images (B, 3, H, W), range [-1, 1]
            reconstructed_images: Reconstructed images (B, 3, H, W), range [-1, 1]
            positive_pairs: Optional same-identity pairs (B, 3, H, W), range [-1, 1]

        Returns:
            ArcFace loss scalar (1 - cosine_similarity)
        """
        self.eval()

        with torch.no_grad():
            # Extract embeddings from original images
            original_embeddings = self.get_embeddings(original_images)

            # Extract embeddings from reconstructed images
            reconstructed_embeddings = self.get_embeddings(reconstructed_images)

            # Compute cosine similarity (embeddings are already normalized)
            cos_sim = (original_embeddings * reconstructed_embeddings).sum(dim=1)

            # Loss is 1 - cosine_similarity (range [0, 2], ideal is 0)
            loss = (1.0 - cos_sim).mean()

            # Optional: If positive pairs are provided, add contrastive loss
            if positive_pairs is not None:
                pair_embeddings = self.get_embeddings(positive_pairs)

                # Positive pair similarity (should be high)
                pos_sim = (original_embeddings * pair_embeddings).sum(dim=1)
                pair_loss = (1.0 - pos_sim).mean()

                # Combine losses
                loss = 0.5 * loss + 0.5 * pair_loss

        return loss

    def compute_similarity(
        self,
        images1: torch.Tensor,
        images2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pairwise cosine similarity between two sets of images.

        Args:
            images1: First set of images (B, 3, H, W)
            images2: Second set of images (B, 3, H, W)

        Returns:
            Cosine similarities (B,)
        """
        with torch.no_grad():
            emb1 = self.get_embeddings(images1)
            emb2 = self.get_embeddings(images2)
            similarities = (emb1 * emb2).sum(dim=1)
        return similarities
