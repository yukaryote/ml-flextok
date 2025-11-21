import torch
from flextok.model.postprocessors.heads import MLPHead
from flextok.model.preprocessors.linear import LinearLayer
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torchvision import transforms
import einops


class REPAProjector(MLPHead):
    """
    A projector module that maps input features to a specified dimension using an MLP head.
    Typically used in conjunction with REPA loss for representation learning.
    """

    def __init__(self, features_read_key: str, write_key: str, input_dim: int, proj_dim: int, num_layers: int = 3):
        super(REPAProjector, self).__init__(
            read_key=features_read_key,
            write_key=write_key,
            dim=input_dim,
            dim_out=proj_dim,
            num_layers=num_layers,
            use_mup_readout=False,
            norm_layer=None,
        )

class REPAModule(nn.Module):
    """
    Complete REPA module with frozen encoder and trainable projector.
    """
    def __init__(
        self,
        features_read_key: str,
        images_read_key: str, 
        write_key: str,
        decoder_dim: int,
        encoder_type: str = 'dinov2_vitl14',
        encoder_dim: int = 1024,
        target_size: Tuple[int, int] = (37, 37),
    ):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        self.target_size = target_size
        self.features_read_key = features_read_key
        self.images_read_key = images_read_key
        self.write_key = write_key
        
        # Load frozen encoder
        self.encoder = self._load_encoder(encoder_type)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Trainable projector
        # REPAProjector expects input_dim and proj_dim as arguments
        self.projector = REPAProjector(
            features_read_key=features_read_key,
            write_key=write_key,
            input_dim=decoder_dim,
            proj_dim=encoder_dim,
            num_layers=3,
        )
        
        # Preprocessing for encoder (ImageNet normalization)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def _load_encoder(self, encoder_type: str) -> nn.Module:
        """Load pretrained encoder (DINOv2)."""
        if 'dinov2' in encoder_type:
            # Extract model size: vitl14, vitg14, vitb14
            model_size = encoder_type.split('_')[-1]
            encoder = torch.hub.load(
                'facebookresearch/dinov2', 
                f'dinov2_{model_size}'
            )
            # Remove classification head
            if hasattr(encoder, 'head'):
                encoder.head = nn.Identity()
            return encoder
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
    
    @torch.no_grad()
    def extract_target_features(
        self, 
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract features from frozen encoder.
        
        Args:
            images: Input images (B, 3, H, W), range [-1, 1]
        Returns:
            Encoder features (B, C, H', W')
        """
        # Denormalize from [-1, 1] to [0, 1]
        images = (images + 1.0) / 2.0
        
        # Apply ImageNet normalization
        images = self.normalize(images)
        
        # Resize to 224x224 for DINOv2 (or appropriate size)
        images = F.interpolate(
            images, 
            size=224, 
            mode='bicubic', 
            align_corners=False
        )
        
        # Extract features
        self.encoder.eval()
        features_dict = self.encoder.forward_features(images)

        # DINOv2 returns dict with keys:
        # - 'x_norm_patchtokens': (B, num_patches, embed_dim) - spatial patch features
        # - 'x_norm_clstoken': (B, embed_dim) - global CLS token
        # - 'x_norm_regtokens': (B, num_registers, embed_dim) - register tokens
        # We want the spatial patch tokens for REPA
        patch_features = features_dict["x_norm_patchtokens"]  # (B, N, C)

        B, N, C = patch_features.shape
        # Reshape to spatial grid: (B, N, C) -> (B, H, W, C) -> (B, C, H, W)
        H = W = int(N ** 0.5)
        if H * W != N:
            raise ValueError(f"Cannot reshape {N} patches into square grid. Expected {H}x{W}={H*W}, got {N}")

        features = patch_features.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        features = F.interpolate(
            features,
            size=self.target_size,
            mode='bilinear',
            align_corners=False
        )

        return features
    
    def forward(
        self,
        data_dict: dict
    ) -> torch.Tensor:
        """
        Compute REPA loss.

        Args:
            data_dict: Input data dictionary consisting of images and other info
        Returns:
            REPA loss scalar
        """
        decoder_features = data_dict[self.features_read_key]  # B-length List of (1, H, W, 1152)
        target_images = data_dict[self.images_read_key]  # (B, 3, H, W), range [-1, 1]

        # Handle list format (unpacker returns list of tensors)
        if isinstance(decoder_features, list):
            # Stack list into (B, H, W, C)
            decoder_features = torch.cat(decoder_features, dim=0)

        # Reshape to (B * H * W, C)
        B, H, W, C = decoder_features.shape
        decoder_features = einops.rearrange(
            decoder_features, 
            'b h w c -> (b h w) c', 
            b=B, h=H, w=W, c=C
        )
        # Project decoder features using MLPHead
        proj_dict = {self.features_read_key: decoder_features.split(1, dim=0)}
        proj_dict = self.projector(proj_dict)
        projected_features = torch.cat(proj_dict[self.write_key], dim=0)
        # turn back into (B, C, H, W)
        projected_features = einops.rearrange(
            projected_features, 
            '(b h w) c -> b c h w', 
            b=B, h=H, w=W
        )

        # Extract target features from encoder (B, C, H', W')
        with torch.no_grad():
            target_features = self.extract_target_features(target_images)
        
        # Resize decoder features to target size (B, C, H', W')
        projected_features = F.interpolate(
            projected_features,
            size=self.target_size,
            mode='bilinear',
            align_corners=False
        )

        # Normalize both to unit length
        projected_features = F.normalize(projected_features, dim=1)
        target_features = F.normalize(target_features, dim=1)
        
        # Compute negative cosine similarity (loss)
        # Cosine similarity per spatial location, then average
        cosine_sim = (projected_features * target_features).sum(dim=1)
        loss = -cosine_sim.mean()
        
        return loss