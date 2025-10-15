"""
Attention Bending: Apply spatial/frequency transformations to cross-attention maps.

Inspired by Terence Broad's Network Bending applied to GANs, this module enables
manipulation of attention scores in Diffusion Transformers to explore semantic control.

The core idea: Transform attention maps BEFORE they're used to compute attention output,
potentially affecting where and how strongly text tokens influence the generated image.

Theory:
    Standard: output = softmax(Q @ K^T / sqrt(d)) @ V
    Bending:  output = transform(softmax(Q @ K^T / sqrt(d))) @ V
    
    Where transform() applies per-token spatial/frequency manipulations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BendingMode(Enum):
    """Types of attention bending transformations."""
    AMPLIFY = "amplify"  # Amplify/dampen attention weights 
    SCALE = "scale"  # Scale in spatial domain - zoom in/out 
    ROTATE = "rotate"  # Rotate attention pattern
    TRANSLATE = "translate"  # Shift attention pattern spatially
    FLIP = "flip"  # Mirror attention pattern
    BLUR = "blur"  # Blur attention (low-pass filter)
    SHARPEN = "sharpen"  # Sharpen attention (high-pass filter)
    REGIONAL_MASK = "regional_mask"  # Confine token to spatial region
    FREQUENCY_FILTER = "frequency_filter"  # Custom frequency domain filter


@dataclass
class BendingConfig:
    """Configuration for a single token's attention bending."""
    token: str  # Token to apply bending to (e.g., "kiss")
    mode: BendingMode
    
    # Amplify parameters (for AMPLIFY mode - simple multiplier)
    amplify_factor: float = 1.0  # For AMPLIFY mode 
    
    # Spatial transformation parameters
    angle: float = 0.0  # Rotation angle in degrees
    translate_x: float = 0.0  # Translation in x (normalized -1 to 1)
    translate_y: float = 0.0  # Translation in y (normalized -1 to 1)
    flip_horizontal: bool = False
    flip_vertical: bool = False
    scale_factor: float = 1.0  # For SCALE mode - spatial zoom 
    
    # Blur/sharpen parameters
    kernel_size: int = 3  # Kernel size for blur/sharpen
    sigma: float = 1.0  # Gaussian blur sigma
    sharpen_amount: float = 1.0  # Sharpen intensity
    
    # Regional mask parameters
    region: Optional[Tuple[float, float, float, float]] = None  # (x1, y1, x2, y2) normalized 0-1
    region_feather: float = 0.1  # Soft edge blending
    
    # Control parameters
    strength: float = 1.0  # Blend factor: 0=original, 1=fully bent
    apply_to_layers: Optional[List[int]] = None  # Which transformer layers (None=all)
    apply_to_timesteps: Optional[Tuple[int, int]] = None  # (start, end) timestep range
    
    # Affine transformation padding mode (for scale, rotate, translate)
    padding_mode: str = 'zeros'  # How to handle out-of-canvas areas:
                                 # 'zeros': out-of-canvas becomes zero (transparent/empty - standard graphics)
                                 # 'border': replicate edge pixels (extends image boundaries)
                                 # 'reflection': mirror reflection at boundaries
    
    # Stability
    renormalize: bool = True  # Re-normalize after transformation
    preserve_sparsity: bool = False  # Try to maintain attention sparsity pattern


class AttentionBender:
    """
    Main class for applying attention bending transformations.
    
    This integrates with the existing attention storage pathway to intercept
    and manipulate attention probabilities before they're used in the attention output.
    """
    
    def __init__(self, 
                 bending_configs: List[BendingConfig],
                 token_to_index_map: Optional[Dict[str, int]] = None,
                 device: str = "cuda"):
        """
        Initialize the attention bender.
        
        Args:
            bending_configs: List of bending configurations for different tokens
            token_to_index_map: Mapping from token strings to indices in attention map
            device: Device for tensor operations
        """
        self.bending_configs = bending_configs
        self.token_to_index_map = token_to_index_map or {}
        self.device = device
        
        # Statistics tracking
        self.stats = {
            "applications": 0,
            "tokens_bent": {},
            "norm_changes": [],
        }
        
        logger.info(f"AttentionBender initialized with {len(bending_configs)} configs")
        for config in bending_configs:
            logger.info(f"  - Token '{config.token}': {config.mode.value} (strength={config.strength})")
    
    def update_token_map(self, token_to_index_map: Dict[str, int]):
        """Update the token-to-index mapping (called per generation with actual tokens)."""
        self.token_to_index_map = token_to_index_map
        logger.debug(f"Updated token map: {token_to_index_map}")
    
    def should_apply(self, config: BendingConfig, layer_idx: Optional[int], timestep: Optional[int]) -> bool:
        """Check if bending should be applied based on layer/timestep constraints."""
        if config.apply_to_layers is not None and layer_idx is not None:
            if layer_idx not in config.apply_to_layers:
                return False
        
        if config.apply_to_timesteps is not None and timestep is not None:
            start, end = config.apply_to_timesteps
            if not (start <= timestep <= end):
                return False
        
        return True
    
    def bend_attention(self,
                      attention_probs: torch.Tensor,
                      layer_idx: Optional[int] = None,
                      timestep: Optional[int] = None,
                      spatial_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Apply attention bending transformations.
        
        Args:
            attention_probs: Attention probability tensor [B, num_heads, H*W, seq_len]
                           or [B*num_heads, H*W, seq_len]
            layer_idx: Current transformer layer index
            timestep: Current denoising timestep
            spatial_shape: (height, width) of spatial dimensions if known
            
        Returns:
            Transformed attention probabilities with same shape as input
        """
        if not self.bending_configs:
            return attention_probs
        
        logger.info(f"🎨 === ATTENTION BENDING START ===")
        logger.info(f"   Layer: {layer_idx}, Timestep: {timestep}")
        logger.info(f"   Input shape: {attention_probs.shape}")
        logger.info(f"   Device: {attention_probs.device}, Dtype: {attention_probs.dtype}")
        logger.info(f"   Number of configs: {len(self.bending_configs)}")
        logger.info(f"   Token map: {self.token_to_index_map}")
        
        # Handle different input shapes
        original_shape = attention_probs.shape
        if len(original_shape) == 4:
            # [B, heads, spatial, seq] -> [B*heads, spatial, seq]
            B, H, S, T = original_shape
            attention_probs = attention_probs.reshape(B * H, S, T)
            logger.info(f"   Reshaped from {original_shape} to {attention_probs.shape}")
        elif len(original_shape) != 3:
            logger.warning(f"Unexpected attention shape: {original_shape}")
            return attention_probs
        
        batch_heads, spatial_tokens, seq_len = attention_probs.shape
        
        # Infer spatial shape if not provided
        if spatial_shape is None:
            spatial_shape = self._infer_spatial_shape(spatial_tokens)
        logger.info(f"   Spatial shape: {spatial_shape}")
        
        # Clone to avoid modifying original
        bent_attention = attention_probs.clone()
        
        # Apply each bending config
        configs_applied = 0
        for config in self.bending_configs:
            if not self.should_apply(config, layer_idx, timestep):
                logger.info(f"   ⏭️  Skipping config for '{config.token}' (conditions not met)")
                continue
            
            # Find token index
            token_idx = self.token_to_index_map.get(config.token.lower())
            if token_idx is None:
                logger.warning(f"   ❌ Token '{config.token}' not found in map: {self.token_to_index_map}")
                continue
            
            logger.info(f"   ✅ Applying {config.mode.value} to token '{config.token}' (idx={token_idx})")
            
            if token_idx >= seq_len:
                logger.warning(f"   ❌ Token index {token_idx} >= seq_len {seq_len}, skipping")
                continue
            
            # Extract attention for this token [batch_heads, spatial_tokens]
            token_attention = bent_attention[:, :, token_idx]
            logger.info(f"      Token attention shape: {token_attention.shape}, mean: {token_attention.mean():.4f}, max: {token_attention.max():.4f}")
            
            # Apply transformation
            transformed = self._apply_transformation(
                token_attention,
                config,
                spatial_shape
            )
            logger.info(f"      After transform: mean: {transformed.mean():.4f}, max: {transformed.max():.4f}")
            
            # Blend with original based on strength
            blended = (1 - config.strength) * token_attention + config.strength * transformed
            logger.info(f"      After blend (strength={config.strength}): mean: {blended.mean():.4f}, max: {blended.max():.4f}")
            
            # Re-normalize if requested
            if config.renormalize:
                # Normalize across spatial dimension
                blended = self._safe_normalize(blended, dim=1)
                logger.info(f"      After renormalize: mean: {blended.mean():.4f}, max: {blended.max():.4f}")
            
            # Update bent attention
            bent_attention[:, :, token_idx] = blended
            configs_applied += 1
            
            # Track stats
            self.stats["tokens_bent"][config.token] = self.stats["tokens_bent"].get(config.token, 0) + 1
        
        self.stats["applications"] += 1
        logger.info(f"   📊 Applied {configs_applied}/{len(self.bending_configs)} configs")
        logger.info(f"🎨 === ATTENTION BENDING END ===")

        
        # Restore original shape if needed
        if len(original_shape) == 4:
            bent_attention = bent_attention.reshape(original_shape)
        
        return bent_attention
    
    def _apply_transformation(self,
                            attention_map: torch.Tensor,
                            config: BendingConfig,
                            spatial_shape: Tuple[int, int]) -> torch.Tensor:
        """Apply the specific transformation defined by config."""
        H, W = spatial_shape
        batch_heads = attention_map.shape[0]
        
        # Reshape to spatial grid [batch_heads, H, W]
        attention_2d = attention_map.reshape(batch_heads, H, W)
        
        if config.mode == BendingMode.AMPLIFY:
            return self._apply_amplify(attention_2d, config).reshape(batch_heads, -1)
        
        elif config.mode == BendingMode.SCALE:
            return self._apply_scale(attention_2d, config).reshape(batch_heads, -1)
        
        elif config.mode == BendingMode.ROTATE:
            return self._apply_rotation(attention_2d, config).reshape(batch_heads, -1)
        
        elif config.mode == BendingMode.TRANSLATE:
            return self._apply_translation(attention_2d, config).reshape(batch_heads, -1)
        
        elif config.mode == BendingMode.FLIP:
            return self._apply_flip(attention_2d, config).reshape(batch_heads, -1)
        
        elif config.mode == BendingMode.BLUR:
            return self._apply_blur(attention_2d, config).reshape(batch_heads, -1)
        
        elif config.mode == BendingMode.SHARPEN:
            return self._apply_sharpen(attention_2d, config).reshape(batch_heads, -1)
        
        elif config.mode == BendingMode.REGIONAL_MASK:
            return self._apply_regional_mask(attention_2d, config).reshape(batch_heads, -1)
        
        elif config.mode == BendingMode.FREQUENCY_FILTER:
            return self._apply_frequency_filter(attention_2d, config).reshape(batch_heads, -1)
        
        else:
            logger.warning(f"Unknown bending mode: {config.mode}")
            return attention_map
    
    def _apply_amplify(self, attention: torch.Tensor, config: BendingConfig) -> torch.Tensor:
        """Simple multiplicative scaling (amplify/dampen attention weights)."""
        return attention * config.amplify_factor
    
    def _apply_scale(self, attention: torch.Tensor, config: BendingConfig) -> torch.Tensor:
        """
        Scale (zoom) attention pattern using affine transformation matrix.
        
        This uses the same transformation matrix approach as rotation and translation,
        keeping the canvas size fixed while transforming the attention "image" content.
        
        - scale_factor > 1: Features get LARGER (zoom in, content scaled up on fixed canvas)
        - scale_factor < 1: Features get SMALLER (zoom out, content scaled down on fixed canvas)
        
        The transformation is centered at the origin, so scaling happens from the center.
        """
        if config.scale_factor == 1.0:
            return attention
        
        batch_heads, H, W = attention.shape
        
        # Create scaling transformation matrix
        # In affine transformations, scaling by s means dividing coordinates by s
        # (inverse transformation for grid_sample)
        s = 1.0 / config.scale_factor
        
        theta = torch.tensor([
            [s, 0, 0],
            [0, s, 0]
        ], dtype=attention.dtype, device=attention.device).unsqueeze(0).repeat(batch_heads, 1, 1)
        
        # Generate sampling grid and apply transformation
        grid = F.affine_grid(theta, attention.unsqueeze(1).size(), align_corners=False)
        scaled = F.grid_sample(
            attention.unsqueeze(1), 
            grid, 
            mode='bilinear',
            padding_mode=config.padding_mode,  # Configurable: 'zeros', 'border', 'reflection'
            align_corners=False
        )
        
        return scaled.squeeze(1)
    
    def _apply_rotation(self, attention: torch.Tensor, config: BendingConfig) -> torch.Tensor:
        """
        Rotate attention pattern using affine transformation matrix.
        
        Canvas size remains fixed. Rotation is centered at origin.
        Areas that rotate outside the canvas become zero (transparent).
        """
        if config.angle == 0:
            return attention
        
        batch_heads, H, W = attention.shape
        angle_rad = np.deg2rad(config.angle)
        
        # Create rotation matrix (standard 2D rotation)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=attention.dtype, device=attention.device).unsqueeze(0).repeat(batch_heads, 1, 1)
        
        grid = F.affine_grid(theta, attention.unsqueeze(1).size(), align_corners=False)
        rotated = F.grid_sample(
            attention.unsqueeze(1), 
            grid, 
            mode='bilinear',
            padding_mode=config.padding_mode,  # Configurable: 'zeros', 'border', 'reflection'
            align_corners=False
        )
        
        return rotated.squeeze(1)
    
    def _apply_translation(self, attention: torch.Tensor, config: BendingConfig) -> torch.Tensor:
        """
        Translate (shift) attention pattern using affine transformation matrix.
        
        Canvas size remains fixed. Translation values are normalized (-1 to 1).
        Areas that shift outside the canvas become zero (transparent).
        """
        if config.translate_x == 0 and config.translate_y == 0:
            return attention
        
        batch_heads, H, W = attention.shape
        
        # Create translation matrix (standard 2D translation)
        theta = torch.tensor([
            [1, 0, config.translate_x],
            [0, 1, config.translate_y]
        ], dtype=attention.dtype, device=attention.device).unsqueeze(0).repeat(batch_heads, 1, 1)
        
        grid = F.affine_grid(theta, attention.unsqueeze(1).size(), align_corners=False)
        translated = F.grid_sample(
            attention.unsqueeze(1), 
            grid, 
            mode='bilinear',
            padding_mode=config.padding_mode,  # Configurable: 'zeros', 'border', 'reflection'
            align_corners=False
        )
        
        return translated.squeeze(1)
    
    def _apply_flip(self, attention: torch.Tensor, config: BendingConfig) -> torch.Tensor:
        """
        Flip (mirror) attention pattern.
        
        Canvas size remains fixed. This is a simple pixel reordering operation.
        Unlike affine transformations, flip operates directly on the tensor without sampling.
        """
        result = attention
        if config.flip_horizontal:
            result = torch.flip(result, dims=[2])  # Flip along width dimension
        if config.flip_vertical:
            result = torch.flip(result, dims=[1])  # Flip along height dimension
        return result
    
    def _apply_blur(self, attention: torch.Tensor, config: BendingConfig) -> torch.Tensor:
        """Apply Gaussian blur to attention."""
        kernel_size = config.kernel_size
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd
        
        # Create Gaussian kernel
        sigma = config.sigma
        kernel_1d = self._gaussian_kernel_1d(kernel_size, sigma, attention.device, attention.dtype)
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
        
        # Apply convolution
        batch_heads = attention.shape[0]
        padding = kernel_size // 2
        blurred = F.conv2d(attention.unsqueeze(1), kernel_2d, padding=padding, groups=1)
        
        return blurred.squeeze(1)
    
    def _apply_sharpen(self, attention: torch.Tensor, config: BendingConfig) -> torch.Tensor:
        """Apply sharpening to attention (unsharp mask)."""
        # First blur
        blurred = self._apply_blur(attention, config)
        
        # Unsharp mask: original + amount * (original - blurred)
        sharpened = attention + config.sharpen_amount * (attention - blurred)
        
        return sharpened
    
    def _apply_regional_mask(self, attention: torch.Tensor, config: BendingConfig) -> torch.Tensor:
        """Confine attention to a spatial region."""
        if config.region is None:
            return attention
        
        batch_heads, H, W = attention.shape
        x1, y1, x2, y2 = config.region
        
        # Create mask
        mask = torch.zeros_like(attention)
        
        # Convert normalized coordinates to pixel coordinates
        x1_px = int(x1 * W)
        x2_px = int(x2 * W)
        y1_px = int(y1 * H)
        y2_px = int(y2 * H)
        
        # Set region to 1
        mask[:, y1_px:y2_px, x1_px:x2_px] = 1.0
        
        # Apply feathering if requested
        if config.region_feather > 0:
            feather_px = max(1, int(config.region_feather * min(H, W)))
            mask = self._apply_blur(mask, BendingConfig(
                token="", mode=BendingMode.BLUR,
                kernel_size=feather_px * 2 + 1,
                sigma=feather_px / 2
            ))
            # Normalize mask
            mask = mask / (mask.max() + 1e-8)
        
        return attention * mask
    
    def _apply_frequency_filter(self, attention: torch.Tensor, config: BendingConfig) -> torch.Tensor:
        """Apply frequency domain filtering (FFT-based)."""
        # This is a placeholder for more sophisticated frequency filtering
        # For now, just use blur/sharpen as proxy
        if config.sigma < 1.0:
            return self._apply_sharpen(attention, config)
        else:
            return self._apply_blur(attention, config)
    
    def _gaussian_kernel_1d(self, kernel_size: int, sigma: float, 
                           device: str, dtype: torch.dtype) -> torch.Tensor:
        """Create 1D Gaussian kernel."""
        x = torch.arange(kernel_size, device=device, dtype=dtype)
        x = x - kernel_size // 2
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        return kernel
    
    def _safe_normalize(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """Safely normalize tensor along dimension."""
        sum_vals = tensor.sum(dim=dim, keepdim=True)
        # Avoid division by zero
        sum_vals = torch.where(sum_vals > 1e-8, sum_vals, torch.ones_like(sum_vals))
        return tensor / sum_vals
    
    def _infer_spatial_shape(self, num_tokens: int) -> Tuple[int, int]:
        """Infer spatial H, W from number of tokens."""
        # Assume square or close to square
        h = int(np.sqrt(num_tokens))
        w = num_tokens // h
        if h * w != num_tokens:
            # Try to find best factorization
            for h_try in range(int(np.sqrt(num_tokens)), 0, -1):
                if num_tokens % h_try == 0:
                    h = h_try
                    w = num_tokens // h
                    break
        
        return (h, w)
    
    def get_stats(self) -> Dict:
        """Get statistics about bending applications."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics tracking."""
        self.stats = {
            "applications": 0,
            "tokens_bent": {},
            "norm_changes": [],
        }


def create_bending_from_config(config_dict: Dict) -> AttentionBender:
    """
    Create AttentionBender from configuration dictionary.
    
    Example config:
    {
        "bending_configs": [
            {
                "token": "kiss",
                "mode": "scale",
                "scale_factor": 2.0,
                "strength": 0.8,
                "apply_to_timesteps": [0, 25]
            },
            {
                "token": "sunset",
                "mode": "regional_mask",
                "region": [0.0, 0.0, 1.0, 0.5],  # Top half
                "strength": 1.0
            }
        ]
    }
    """
    configs = []
    for cfg_dict in config_dict.get("bending_configs", []):
        mode = BendingMode(cfg_dict["mode"])
        config = BendingConfig(
            token=cfg_dict["token"],
            mode=mode,
            **{k: v for k, v in cfg_dict.items() if k not in ["token", "mode"]}
        )
        configs.append(config)
    
    return AttentionBender(
        bending_configs=configs,
        device=config_dict.get("device", "cuda")
    )


# Example usage configurations
EXAMPLE_CONFIGS = {
    "amplify_kiss": BendingConfig(
        token="kiss",
        mode=BendingMode.AMPLIFY,
        amplify_factor=2.0,
        strength=0.8,
        renormalize=True
    ),
    
    "rotate_sunset": BendingConfig(
        token="sunset",
        mode=BendingMode.ROTATE,
        angle=45,
        strength=0.5
    ),
    
    "localize_people_top": BendingConfig(
        token="people",
        mode=BendingMode.REGIONAL_MASK,
        region=(0.0, 0.0, 1.0, 0.5),  # Top half
        region_feather=0.15,
        strength=1.0
    ),
    
    "blur_background": BendingConfig(
        token="background",
        mode=BendingMode.BLUR,
        kernel_size=5,
        sigma=2.0,
        strength=0.7
    ),
    
    "sharpen_face": BendingConfig(
        token="face",
        mode=BendingMode.SHARPEN,
        sharpen_amount=1.5,
        strength=0.6
    ),
}
