"""
Advanced neural network architectures for FlexTok RL agents.

Provides multiple architecture options optimized for the multi-modal
observation space: position, token/reward history, and images.
"""
import torch
import torch.nn as nn
import gymnasium as gym
from typing import Dict, Tuple
import numpy as np


class TransformerHistoryEncoder(nn.Module):
    """
    Transformer-based encoder for token and reward history.

    Better than MLP for capturing sequential dependencies and
    long-range patterns in token choices.
    """

    def __init__(
        self,
        token_history_shape: Tuple[int, int],
        reward_history_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        output_dim: int = 256,
    ):
        super().__init__()

        self.history_length = token_history_shape[0]
        token_dim = np.prod(token_history_shape[1:])

        # Project token history to d_model
        self.token_proj = nn.Linear(token_dim, d_model)

        # Project rewards to d_model
        self.reward_proj = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.history_length, d_model)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

    def forward(self, token_history: torch.Tensor, reward_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_history: (batch, history_length, token_dim)
            reward_history: (batch, history_length)

        Returns:
            Encoded history: (batch, output_dim)
        """
        batch_size = token_history.shape[0]

        # Flatten token history if needed
        if len(token_history.shape) > 2:
            token_history = token_history.reshape(batch_size, self.history_length, -1)

        # Project to d_model
        token_features = self.token_proj(token_history)  # (B, H, d_model)
        reward_features = self.reward_proj(reward_history.unsqueeze(-1))  # (B, H, d_model)

        # Combine token and reward features
        combined = token_features + reward_features

        # Add positional encoding
        combined = combined + self.pos_encoding

        # Apply transformer
        transformed = self.transformer(combined)  # (B, H, d_model)

        # Pool over sequence (use CLS token approach or mean pooling)
        pooled = transformed.mean(dim=1)  # (B, d_model)

        # Project to output
        output = self.output_proj(pooled)  # (B, output_dim)

        return output


class GRUHistoryEncoder(nn.Module):
    """
    GRU-based encoder for token and reward history.

    Lighter than Transformer, good for capturing sequential patterns.
    """

    def __init__(
        self,
        token_history_shape: Tuple[int, int],
        reward_history_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 256,
    ):
        super().__init__()

        self.history_length = token_history_shape[0]
        token_dim = np.prod(token_history_shape[1:])

        # Input dimension: token + reward
        input_dim = token_dim + 1

        # GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, token_history: torch.Tensor, reward_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_history: (batch, history_length, token_dim)
            reward_history: (batch, history_length)

        Returns:
            Encoded history: (batch, output_dim)
        """
        batch_size = token_history.shape[0]

        # Flatten token history if needed
        if len(token_history.shape) > 2:
            token_history = token_history.reshape(batch_size, self.history_length, -1)

        # Concatenate token and reward
        combined = torch.cat([
            token_history,
            reward_history.unsqueeze(-1)
        ], dim=-1)  # (B, H, token_dim + 1)

        # Apply GRU
        output, hidden = self.gru(combined)  # output: (B, H, hidden_dim)

        # Use last hidden state
        last_hidden = output[:, -1, :]  # (B, hidden_dim)

        # Project to output
        output = self.output_proj(last_hidden)  # (B, output_dim)

        return output


class EfficientImageEncoder(nn.Module):
    """
    Efficient CNN encoder for images.

    Uses smaller architecture than ResNet for faster training.
    """

    def __init__(self, output_dim: int = 512):
        super().__init__()

        self.encoder = nn.Sequential(
            # 256x256x3 -> 128x128x32
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 128x128x32 -> 64x64x64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 64x64x64 -> 32x32x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 32x32x128 -> 16x16x256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # 16x16x256 -> 8x8x512
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # Flatten
            nn.AdaptiveAvgPool2d(1),  # -> 1x1x512
            nn.Flatten(),
        )

        self.output_proj = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, 256, 256) normalized to [0, 1]

        Returns:
            features: (batch, output_dim)
        """
        features = self.encoder(x)  # (B, 512)
        output = self.output_proj(features)  # (B, output_dim)
        return output


class ResNetImageEncoder(nn.Module):
    """
    ResNet-based image encoder.

    Higher quality but slower than EfficientImageEncoder.
    """

    def __init__(self, output_dim: int = 512, pretrained: bool = False):
        super().__init__()

        from torchvision.models import resnet18, ResNet18_Weights

        if pretrained:
            self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.resnet = resnet18(weights=None)

        # Remove final FC layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])

        # Project to output dim
        self.output_proj = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, 256, 256) normalized to [0, 1]

        Returns:
            features: (batch, output_dim)
        """
        features = self.features(x)  # (B, 512, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (B, 512)
        output = self.output_proj(features)  # (B, output_dim)
        return output


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention between image and history features.

    Allows history to attend to image features and vice versa.
    """

    def __init__(self, dim: int = 256, num_heads: int = 4):
        super().__init__()

        self.history_to_image = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.image_to_history = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
        )

    def forward(self, history_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history_features: (batch, dim)
            image_features: (batch, dim)

        Returns:
            fused_features: (batch, dim)
        """
        # Add sequence dimension for attention
        history = history_features.unsqueeze(1)  # (B, 1, dim)
        image = image_features.unsqueeze(1)  # (B, 1, dim)

        # Cross attention
        history_attended, _ = self.history_to_image(history, image, image)  # (B, 1, dim)
        image_attended, _ = self.image_to_history(image, history, history)  # (B, 1, dim)

        # Remove sequence dimension
        history_attended = history_attended.squeeze(1)  # (B, dim)
        image_attended = image_attended.squeeze(1)  # (B, dim)

        # Fuse
        fused = torch.cat([history_attended, image_attended], dim=-1)  # (B, 2*dim)
        output = self.fusion(fused)  # (B, dim)

        return output


class AdvancedFeatureExtractor(nn.Module):
    """
    Advanced multi-modal feature extractor with multiple architecture options.

    Architecture options:
    - 'transformer': Transformer for history + CNN for images + cross-attention fusion
    - 'gru': GRU for history + CNN for images + concatenation
    - 'efficient': Efficient CNN + simple MLP for history
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 512,
        use_image: bool = True,
        architecture: str = 'transformer',  # 'transformer', 'gru', or 'efficient'
        image_encoder: str = 'efficient',  # 'efficient' or 'resnet'
    ):
        super().__init__()

        self.use_image = use_image
        self.architecture = architecture
        self._features_dim = features_dim

        # Get observation space dimensions
        position_dim = observation_space['position'].shape[0]
        token_history_shape = observation_space['token_history'].shape
        reward_history_dim = observation_space['reward_history'].shape[0]

        # Position encoder (simple MLP)
        self.position_encoder = nn.Sequential(
            nn.Linear(position_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        position_out_dim = 64

        # History encoder (Transformer or GRU)
        if architecture == 'transformer':
            self.history_encoder = TransformerHistoryEncoder(
                token_history_shape=token_history_shape,
                reward_history_dim=reward_history_dim,
                d_model=128,
                nhead=4,
                num_layers=2,
                output_dim=256,
            )
            history_out_dim = 256
        elif architecture == 'gru':
            self.history_encoder = GRUHistoryEncoder(
                token_history_shape=token_history_shape,
                reward_history_dim=reward_history_dim,
                hidden_dim=128,
                num_layers=2,
                output_dim=256,
            )
            history_out_dim = 256
        else:  # 'efficient' or default
            # Simple MLP
            token_history_dim = np.prod(token_history_shape)
            self.history_encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(token_history_dim + reward_history_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
            )
            history_out_dim = 256

        # Image encoder (if using images)
        if use_image and 'image' in observation_space.spaces:
            if image_encoder == 'resnet':
                self.image_encoder = ResNetImageEncoder(output_dim=512)
            else:  # 'efficient'
                self.image_encoder = EfficientImageEncoder(output_dim=512)
            image_out_dim = 512

            # Fusion mechanism
            if architecture == 'transformer':
                # Use cross-attention fusion
                # First project to same dimension
                self.history_proj = nn.Linear(history_out_dim, 256)
                self.image_proj = nn.Linear(image_out_dim, 256)
                self.fusion = CrossAttentionFusion(dim=256, num_heads=4)
                combined_dim = position_out_dim + 256
            else:
                # Simple concatenation
                combined_dim = position_out_dim + history_out_dim + image_out_dim
        else:
            self.image_encoder = None
            combined_dim = position_out_dim + history_out_dim

        # Final combiner
        self.combiner = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from observations."""
        # Encode position
        position_features = self.position_encoder(observations['position'])

        # Encode history
        if isinstance(self.history_encoder, (TransformerHistoryEncoder, GRUHistoryEncoder)):
            history_features = self.history_encoder(
                observations['token_history'],
                observations['reward_history']
            )
        else:
            # Simple MLP: concatenate token and reward history
            token_flat = observations['token_history'].flatten(start_dim=1)
            reward_flat = observations['reward_history']
            combined_history = torch.cat([token_flat, reward_flat], dim=-1)
            history_features = self.history_encoder(combined_history)

        # Encode image (if available)
        features_list = [position_features]

        if self.use_image and self.image_encoder is not None and 'image' in observations:
            # Normalize image to [0, 1]
            images = observations['image'].float() / 255.0
            # Permute to (B, C, H, W)
            images = images.permute(0, 3, 1, 2)
            image_features = self.image_encoder(images)

            # Fusion
            if self.architecture == 'transformer':
                # Cross-attention fusion
                history_proj = self.history_proj(history_features)
                image_proj = self.image_proj(image_features)
                fused = self.fusion(history_proj, image_proj)
                features_list.append(fused)
            else:
                # Simple concatenation
                features_list.extend([history_features, image_features])
        else:
            features_list.append(history_features)

        # Combine all features
        combined = torch.cat(features_list, dim=-1)
        output = self.combiner(combined)

        return output


def create_feature_extractor(
    observation_space: gym.spaces.Dict,
    features_dim: int = 512,
    use_image: bool = True,
    architecture: str = 'transformer',
) -> AdvancedFeatureExtractor:
    """
    Factory function to create feature extractor.

    Args:
        observation_space: Observation space
        features_dim: Output feature dimension
        use_image: Whether to use image observations
        architecture: 'transformer', 'gru', or 'efficient'

    Returns:
        Feature extractor
    """
    return AdvancedFeatureExtractor(
        observation_space=observation_space,
        features_dim=features_dim,
        use_image=use_image,
        architecture=architecture,
    )
