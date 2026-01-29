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
from pathlib import Path
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
    """Configuration for a single token's attention bending.
    
    The 'token' field supports:
    - Specific tokens: "kiss", "person", etc.
    - Wildcard tokens: "ALL", "*", "ALLTOKENS" - applies transformation to ALL tokens
    """
    token: str  # Token to apply bending to (e.g., "kiss", or "ALL" for all tokens)
    mode: BendingMode
    
    # Amplify parameters (for AMPLIFY mode - simple multiplier)
    amplify_factor: float = 1.0  # For AMPLIFY mode 
    
    # Spatial transformation parameters
    angle: float = 0.0  # Rotation angle in degrees
    crop_rotated: bool = True  # If True, crop rotated content to original canvas (default). If False, stretch to fit.
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
    padding_mode: str = 'border'  # How to handle out-of-canvas areas:
                                 # 'zeros': out-of-canvas becomes zero (transparent/empty - standard graphics)
                                 # 'border': replicate edge pixels (extends image boundaries)
                                 # 'reflection': mirror reflection at boundaries
    
    # Stability
    renormalize: bool = False  # Re-normalize after transformation
    preserve_sparsity: bool = False  # Try to maintain attention sparsity pattern

    should_debug_visualize: bool = False  # Whether to generate debug visualizations


class AttentionBender:
    """
    Main class for applying attention bending transformations.
    
    This integrates with the existing attention storage pathway to intercept
    and manipulate attention probabilities before they're used in the attention output.
    """
    
    def __init__(self, 
                 bending_configs: List[BendingConfig],
                 token_to_index_map: Optional[Dict[str, int]] = None,
                 device: str = "cuda",
                 apply_before_softmax: bool = False):
        """
        Initialize the attention bender.
        
        Args:
            bending_configs: List of bending configurations for different tokens
            token_to_index_map: Mapping from token strings to indices in attention map
            device: Device for tensor operations
            apply_before_softmax: [NOT IMPLEMENTED] If True, bend attention scores BEFORE softmax.
                                 Currently IGNORED - bending always happens post-softmax.
                                 Pre-softmax bending would require pipeline changes to intercept
                                 attention scores before softmax is applied.
        
        IMPORTANT: Current implementation bends POST-softmax attention probabilities, which has
        minimal effect for amplification because values already sum to 1. For effects similar to
        prompt weighting, we would need to bend pre-softmax scores (not currently supported).
        """
        self.bending_configs = bending_configs
        logger.info(f"INITIAL TOKEN TO INDEX MAP: {token_to_index_map}")    
        self.token_to_index_map = token_to_index_map or {}
        self.current_prompt = None  # Store prompt for comma-separated token filtering
        self.device = device
        self.apply_before_softmax = apply_before_softmax
        
        if apply_before_softmax:
            logger.warning(
                "⚠️  apply_before_softmax=True is NOT FULLY IMPLEMENTED.\n"
                "   Current behavior: softmax → bend → log → softmax (not true pre-softmax bending)\n"
                "   For strong amplification effects similar to prompt weighting, use prompt weighting instead: (token:2.0)\n"
                "   Post-softmax amplification has minimal effect because attention already sums to 1."
            )
        
        # Statistics tracking
        self.stats = {
            "applications": 0,
            "tokens_bent": {},
            "norm_changes": [],
        }
        
        # Debug visualization tracking - group all steps from same run
        self._debug_batch_dir = None  # Will be initialized on first use
        
        logger.info(f"AttentionBender initialized with {len(bending_configs)} configs")
        logger.info(f"  Apply before softmax: {apply_before_softmax}")
        for config in bending_configs:
            logger.info(f"  - Token '{config.token}': {config.mode.value} (strength={config.strength})")
    
    def update_token_map(self, token_to_index_map: Dict[str, int], prompt: str = None):
        """Update the token-to-index mapping (called per generation with actual tokens).
        
        Args:
            token_to_index_map: Mapping from token strings to indices
            prompt: The prompt text (used for filtering comma-separated tokens)
        """
        self.token_to_index_map = token_to_index_map
        self.current_prompt = prompt
        logger.debug(f"Updated token map: {token_to_index_map}")
        if prompt:
            logger.debug(f"Updated prompt: {prompt[:100]}...")
    
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
        
        # logger.info(f"🎨 === ATTENTION BENDING START ===")
        # logger.info(f"   Layer: {layer_idx}, Timestep: {timestep}")
        # logger.info(f"   Input shape: {attention_probs.shape}")
        # logger.info(f"   Device: {attention_probs.device}, Dtype: {attention_probs.dtype}")
        # logger.info(f"   Number of configs: {len(self.bending_configs)}")
        # logger.info(f"   Token map: {self.token_to_index_map}")
        
        # Handle different input shapes
        original_shape = attention_probs.shape
        if len(original_shape) == 4:
            # TODO: maybe just squeeze to 3 dims?
            # [B, heads, spatial, seq] -> [B*heads, spatial, seq]
            B, H, S, T = original_shape
            attention_probs = attention_probs.reshape(B * H, S, T)
            # logger.info(f"   Reshaped from {original_shape} to {attention_probs.shape}")
        elif len(original_shape) != 3:
            logger.warning(f"Unexpected attention shape: {original_shape}")
            return attention_probs
        
        batch_heads, spatial_tokens, seq_len = attention_probs.shape
        
        # Infer spatial shape if not provided
        if spatial_shape is None:
            # TODO: probably just fail here...
            spatial_shape = self._infer_spatial_shape(spatial_tokens)
        # logger.info(f"   Spatial shape: {spatial_shape}")
        
        # Clone to avoid modifying original
        bent_attention = attention_probs.clone()
        
        # DIAGNOSTIC: Check normalization state BEFORE bending
        if logger.isEnabledFor(logging.DEBUG):
            # For cross-attention, each token's attention should sum to 1 across spatial dimension (dim=-1)?
            # OR each spatial position's attention should sum to 1 across tokens (dim=2)?
            spatial_sums = bent_attention.sum(dim=-1)  # Sum across spatial dimension for each token
            token_sums = bent_attention.sum(dim=2, keepdim=True)  # Sum across tokens for each spatial position
            logger.debug(f"   📊 PRE-BENDING normalization check:")
            logger.debug(f"      Spatial sums (per token): range [{spatial_sums.min():.6f}, {spatial_sums.max():.6f}], mean={spatial_sums.mean():.6f}")
            logger.debug(f"      Token sums (per spatial pos): range [{token_sums.min():.6f}, {token_sums.max():.6f}], mean={token_sums.mean():.6f}")
        
        # Apply each bending config
        configs_applied = 0
        for config in self.bending_configs:
            if not self.should_apply(config, layer_idx, timestep):
                # logger.info(f"   ⏭️  Skipping config for '{config.token}' (conditions not met)")
                continue
            
            # Check if this config applies to ALL tokens (special marker)
            is_all_tokens_config = config.token.upper() in ["__ALL_TOKENS__", "ALL_TOKENS", "*"]
            
            if is_all_tokens_config:
                # STRESS TEST MODE: Apply this config to ALL tokens
                logger.info(f"   🔥 STRESS TEST: Applying '{config.mode.value}' to ALL {seq_len} tokens (token='{config.token}')")
                
                # Apply transformation to every token
                for token_idx in range(seq_len):
                    token_attention = bent_attention[:, :, token_idx]
                    transformed = self._apply_transformation(
                        token_attention,
                        config,
                        spatial_shape
                    )
                    
                    # Blend with strength
                    if config.strength < 1.0:
                        transformed = config.strength * transformed + (1 - config.strength) * token_attention
                    
                    # Update token in full tensor
                    bent_attention[:, :, token_idx] = transformed
                
                # Optionally renormalize entire attention tensor across token dimension ONCE
                # This restores the softmax property: sum across tokens = 1 at each spatial position
                if config.renormalize:
                    # DIAGNOSTIC: Check sum BEFORE renormalization
                    if logger.isEnabledFor(logging.DEBUG):
                        pre_renorm_sums = bent_attention.sum(dim=2, keepdim=True)
                        logger.debug(f"      PRE-renorm token sums: range [{pre_renorm_sums.min():.6f}, {pre_renorm_sums.max():.6f}], mean={pre_renorm_sums.mean():.6f}")
                    
                    bent_attention = bent_attention / (bent_attention.sum(dim=2, keepdim=True) + 1e-10)
                    
                    # DIAGNOSTIC: Check sum AFTER renormalization
                    if logger.isEnabledFor(logging.DEBUG):
                        post_renorm_sums = bent_attention.sum(dim=2, keepdim=True)
                        logger.debug(f"      POST-renorm token sums: range [{post_renorm_sums.min():.6f}, {post_renorm_sums.max():.6f}], mean={post_renorm_sums.mean():.6f}")
                
                logger.info(f"   ✅ Applied to all {seq_len} tokens")
                configs_applied += 1
                
                # Continue to next config (allows multiple all-tokens configs with special marker)
                continue
        
        # Normal mode: Apply configs to specific tokens
        for config in self.bending_configs:
            if not self.should_apply(config, layer_idx, timestep):
                continue
            
            # Check if this is a wildcard token (apply to all tokens)
            is_wildcard = config.token.upper() in ['ALL', '*', 'ALLTOKENS']
            
            if is_wildcard:
                # Apply to all tokens in the sequence
                # logger.info(f"   ✅ Applying {config.mode.value} to ALL TOKENS (wildcard: {config.token})")
                for token_idx in range(seq_len):
                    # Extract attention for this token [batch_heads, spatial_tokens]
                    token_attention = bent_attention[:, :, token_idx]
                    
                    # Apply transformation
                    transformed = self._apply_transformation(
                        token_attention,
                        config,
                        spatial_shape
                    )
                    
                    # Blend with original based on strength
                    blended = (
                        config.strength * transformed +
                        (1 - config.strength) * token_attention
                    )
                    
                    # Update token in full tensor
                    bent_attention[:, :, token_idx] = blended
                
                # Optionally renormalize entire attention tensor across token dimension ONCE
                if config.renormalize:
                    bent_attention = bent_attention / (bent_attention.sum(dim=2, keepdim=True) + 1e-10)
            else:
                # NEW: Handle comma-separated token groups like "kiss, rose, ship"
                # Split by comma and filter to tokens present in BOTH prompt text AND token map
                if ',' in config.token:
                    # Parse comma-separated tokens
                    candidate_tokens = [t.strip().lower() for t in config.token.split(',')]
                    
                    # Filter to tokens present in prompt text (case-insensitive)
                    prompt_lower = self.current_prompt.lower() if self.current_prompt else ""
                    tokens_in_prompt = [t for t in candidate_tokens if t in prompt_lower]
                    
                    # Further filter to only tokens that have been tracked (in token map)
                    # NOTE: For bending to work, tokens must be in parentheses in prompt for tracking
                    active_tokens = [t for t in tokens_in_prompt if t in self.token_to_index_map]
                    
                    if active_tokens:
                        logger.info(f"   📋 Comma-separated group '{config.token}':")
                        logger.info(f"      In prompt: {len(tokens_in_prompt)}/{len(candidate_tokens)} tokens")
                        logger.info(f"      In token map: {len(active_tokens)}/{len(tokens_in_prompt)} tokens: {active_tokens}")
                        
                        # Apply bending to each active token
                        for token_name in active_tokens:
                            token_idx = self.token_to_index_map[token_name]
                            
                            if token_idx >= seq_len:
                                logger.warning(f"      ❌ Token '{token_name}' index {token_idx} >= seq_len {seq_len}, skipping")
                                continue
                            
                            # Extract attention for this token
                            token_attention = bent_attention[:, :, token_idx]
                            
                            # Apply transformation
                            transformed = self._apply_transformation(
                                token_attention,
                                config,
                                spatial_shape
                            )
                            
                            # Blend with original based on strength
                            blended = (
                                config.strength * transformed +
                                (1 - config.strength) * token_attention
                            )
                            
                            # Update token in full tensor
                            bent_attention[:, :, token_idx] = blended
                            
                            logger.info(f"      ✅ Applied {config.mode.value} to '{token_name}' (idx={token_idx})")
                        
                        # Optionally renormalize entire attention tensor across token dimension ONCE
                        if config.renormalize:
                            bent_attention = bent_attention / (bent_attention.sum(dim=2, keepdim=True) + 1e-10)
                    else:
                        if tokens_in_prompt:
                            logger.warning(f"   ⚠️  Comma-separated group '{config.token}': {len(tokens_in_prompt)} tokens in prompt but not tracked (wrap in parentheses)")
                        else:
                            logger.debug(f"   ⏭️  Comma-separated group '{config.token}': no tokens found in prompt")
                    
                    # Continue to next config (already processed all active tokens in this group)
                    # TODO: Handle multi-token words like "cinematic" → ["cinema", "tic"]
                    # For now, user must use exact token strings that appear in token map
                    continue
                
                # Standard single token lookup
                token_idx = self.token_to_index_map.get(config.token.lower())
                if token_idx is None:
                    logger.warning(f"   ❌ Token '{config.token}' not found in map: {self.token_to_index_map}")
                    continue
                
                # logger.info(f"   ✅ Applying {config.mode.value} to token '{config.token}' (idx={token_idx})")
                # logger.info(f"   ✅ Applying {config.mode.value} to token '{config.token}' (idx={token_idx})")
                
                if token_idx >= seq_len:
                    logger.warning(f"   ❌ Token index {token_idx} >= seq_len {seq_len}, skipping")
                    continue
                
                # Extract attention for this token [batch_heads, spatial_tokens]
                token_attention = bent_attention[:, :, token_idx]
                # logger.info(f"      Token attention shape: {token_attention.shape}, mean: {token_attention.mean():.4f}, max: {token_attention.max():.4f}")
                
                # Apply transformation
                transformed = self._apply_transformation(
                    token_attention,
                    config,
                    spatial_shape
                )
                # logger.info(f"      After transform: mean: {transformed.mean():.4f}, max: {transformed.max():.4f}")
                
                # Blend with original based on strength
                blended = (
                    config.strength * transformed +
                    (1 - config.strength) * token_attention
                )
                
                # Update token in full tensor
                bent_attention[:, :, token_idx] = blended
                
                # Optionally renormalize entire attention tensor across token dimension
                if config.renormalize:
                    bent_attention = bent_attention / (bent_attention.sum(dim=2, keepdim=True) + 1e-10)
            
            # DEBUG: Visualize original vs transformed attention
            # if config.should_debug_visualize:
            #     try:
            #         from datetime import datetime
            #         from ..visualization.attention_visualizer import visualize_attention_tensor
                    
            #         # Initialize batch directory on first use
            #         if self._debug_batch_dir is None:
            #             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            #             self._debug_batch_dir = Path("debug_bent_attention") / f"batch_{timestamp}"
            #             self._debug_batch_dir.mkdir(parents=True, exist_ok=True)
            #             # logger.info(f"      🎨 DEBUG: Created batch directory: {self._debug_batch_dir}")
                    
            #         # Create subdirectory for this step and token
            #         debug_subdir = self._debug_batch_dir / f"step_{timestep:03d}-layer_{layer_idx}-token_{config.token}"
            #         debug_subdir.mkdir(parents=True, exist_ok=True)
                    
            #         # logger.info(f"      🔍 DEBUG: Visualizing attention to {debug_subdir}")
                    
            #         # Determine target size from video dimensions if available
            #         target_size = None  # Will use latent size by default
                    
            #         # Visualize original attention
            #         orig_results = visualize_attention_tensor(
            #             attention_tensor=token_attention,
            #             spatial_shape=spatial_shape,
            #             output_path=debug_subdir / "original",
            #             format='both',
            #             colormap='jet',
            #             fps=15,
            #             aggregate_heads=True,
            #             target_size=target_size,
            #             title=f"Original - Token '{config.token}' Step {timestep}"
            #         )
            #         logger.info(f"      ✅ Original visualization: {orig_results}")
                    
            #         # Visualize transformed attention
            #         transformed_results = visualize_attention_tensor(
            #             attention_tensor=transformed,
            #             spatial_shape=spatial_shape,
            #             output_path=debug_subdir / "transformed",
            #             format='both',
            #             colormap='jet',
            #             fps=15,
            #             aggregate_heads=True,
            #             target_size=target_size,
            #             title=f"Transformed - Token '{config.token}' Step {timestep} ({config.mode.value})"
            #         )
            #         logger.info(f"      ✅ Transformed visualization: {transformed_results}")
                    
            #     except Exception as viz_error:
            #         logger.warning(f"      ⚠️  Debug visualization failed: {viz_error}")
            #         import traceback
            #         traceback.print_exc()
            
            # Blend with original based on strength
            blended = (1 - config.strength) * token_attention + config.strength * transformed
            # logger.info(f"      After blend (strength={config.strength}): mean: {blended.mean():.4f}, max: {blended.max():.4f}")
            
            # Update bent attention BEFORE renormalization
            bent_attention[:, :, token_idx] = blended
            configs_applied += 1
            
            # Track stats
            self.stats["tokens_bent"][config.token] = self.stats["tokens_bent"].get(config.token, 0) + 1
        
        # NOTE: Renormalization is now done per-token immediately after transformation
        # This allows different tokens to have different renormalize settings
        # and prevents renormalization from affecting tokens that weren't bent
        
        self.stats["applications"] += 1
        # logger.info(f"   📊 Applied {configs_applied}/{len(self.bending_configs)} configs")
        # logger.info(f"🎨 === ATTENTION BENDING END ===")

        
        # Restore original shape if needed
        if len(original_shape) == 4:
            bent_attention = bent_attention.reshape(original_shape)
        
        return bent_attention
    
    def _apply_transformation(self,
                            attention_map: torch.Tensor,
                            config: BendingConfig,
                            spatial_shape: Union[Tuple[int, int], Tuple[int, int, int]]) -> torch.Tensor:
        """
        Apply the specific transformation defined by config.
        
        Args:
            attention_map: Attention tensor [batch_heads, spatial_tokens]
            config: Bending configuration
            spatial_shape: Either (H, W) for images or (F, H, W) for videos
        """
        batch_heads = attention_map.shape[0]
        
        # Check if we have 3D (video) or 2D (image) spatial shape
        is_video = len(spatial_shape) == 3
        
        if is_video:
            F, H, W = spatial_shape
            # logger.info(f"      🔧 _apply_transformation: mode={config.mode}, spatial_shape=({F}f, {H}h, {W}w) [VIDEO]")
            # logger.info(f"         Input: shape={attention_map.shape}, mean={attention_map.mean():.6f}")
            
            # Reshape to 4D video format: [batch_heads, F, H, W]
            attention_4d = attention_map.reshape(batch_heads, F, H, W)
            # logger.info(f"         After reshape to 4D: shape={attention_4d.shape}")
            
            # OPTIMIZED: Batch-process all frames in parallel instead of frame-by-frame
            # Reshape [batch_heads, F, H, W] → [batch_heads*F, H, W]
            # This treats each frame as a separate batch item for parallel processing
            attention_batched = attention_4d.reshape(batch_heads * F, H, W)
            # logger.info(f"         📹 Batch-processing ALL {F} frames in parallel: shape={attention_batched.shape}")
            
            # Apply transformation to all frames at once
            transformed_batched = self._apply_2d_transformation(attention_batched, config)
            
            # Reshape back [batch_heads*F, H, W] → [batch_heads, F, H, W]
            transformed_4d = transformed_batched.reshape(batch_heads, F, H, W)
            result = transformed_4d.reshape(batch_heads, -1)
            
            # logger.info(f"         After parallel transform: shape={result.shape}, mean={result.mean():.6f}")
            return result
            
        else:
            # 2D image case
            H, W = spatial_shape
            # logger.info(f"      🔧 _apply_transformation: mode={config.mode}, spatial_shape=({H}, {W}) [IMAGE]")
            # logger.info(f"         Input: shape={attention_map.shape}, mean={attention_map.mean():.6f}")
            
            # Reshape to spatial grid [batch_heads, H, W]
            attention_2d = attention_map.reshape(batch_heads, H, W)
            # logger.info(f"         After reshape to 2D: shape={attention_2d.shape}")
            
            result = self._apply_2d_transformation(attention_2d, config).reshape(batch_heads, -1)
            # logger.info(f"         After transform+reshape: shape={result.shape}, mean={result.mean():.6f}")
            return result
    
    def _apply_2d_transformation(self,
                                attention_2d: torch.Tensor,
                                config: BendingConfig) -> torch.Tensor:
        """Apply transformation to 2D attention map [batch_heads, H, W]."""
        if config.mode == BendingMode.AMPLIFY:
            return self._apply_amplify(attention_2d, config)
        
        elif config.mode == BendingMode.SCALE:
            return self._apply_scale(attention_2d, config)
        
        elif config.mode == BendingMode.ROTATE:
            return self._apply_rotation(attention_2d, config)
        
        elif config.mode == BendingMode.TRANSLATE:
            return self._apply_translation(attention_2d, config)
        
        elif config.mode == BendingMode.FLIP:
            return self._apply_flip(attention_2d, config)
        
        elif config.mode == BendingMode.BLUR:
            return self._apply_blur(attention_2d, config)
        
        elif config.mode == BendingMode.SHARPEN:
            return self._apply_sharpen(attention_2d, config)
        
        elif config.mode == BendingMode.REGIONAL_MASK:
            return self._apply_regional_mask(attention_2d, config)
        
        elif config.mode == BendingMode.FREQUENCY_FILTER:
            return self._apply_frequency_filter(attention_2d, config)
        
        else:
            logger.warning(f"Unknown bending mode: {config.mode}")
            return attention_2d
    
    def _apply_amplify(self, attention: torch.Tensor, config: BendingConfig) -> torch.Tensor:
        """Simple multiplicative scaling (amplify/dampen attention weights).
        
        NOTE: This multiplies POST-softmax attention probabilities, which has minimal effect
        because values already sum to 1. Amplifying by 2.0 then renormalizing just slightly
        redistributes probabilities.
        
        For stronger effects similar to prompt weighting, we would need to amplify BEFORE
        softmax (which requires modifying the pipeline to bend attention scores, not probs).
        """
        # Diagnostic: Check if attention is actually normalized
        if logger.isEnabledFor(logging.DEBUG):
            # Sum across token dimension (should be ~1.0 if normalized)
            token_sums = attention.sum(dim=-1)  # Sum across last dim
            logger.debug(f"   🔍 AMPLIFY diagnostic: token_sums range [{token_sums.min():.6f}, {token_sums.max():.6f}], mean={token_sums.mean():.6f}")
            if token_sums.mean() < 0.99 or token_sums.mean() > 1.01:
                logger.warning(f"   ⚠️  Attention does NOT sum to 1! Mean sum: {token_sums.mean():.6f}")
        
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
        # logger.info(f"      🔍 _apply_scale called: scale_factor={config.scale_factor}")
        
        if config.scale_factor == 1.0:
            # logger.info(f"         Scale factor is 1.0, returning unchanged")
            return attention
        
        batch_heads, H, W = attention.shape
        # logger.info(f"         Input shape: {attention.shape} ({batch_heads} heads × {H}h × {W}w)")
        # logger.info(f"         ✨ Applying scale to ALL {batch_heads} heads in parallel via batch processing")
        # logger.info(f"         Statistics: mean={attention.mean():.6f}, max={attention.max():.6f}")
        
        # Check if attention has any spatial structure
        # spatial_std = attention.std()
        # spatial_range = attention.max() - attention.min()
        # logger.info(f"         ⚠️  Spatial variation: std={spatial_std:.6f}, range={spatial_range:.6f}")
        # if spatial_range < 0.001:
            # logger.warning(f"         ⚠️  Attention is nearly UNIFORM (range={spatial_range:.6f}) - spatial transform may have no visible effect!")
        
        # Create scaling transformation matrix
        # In affine transformations, scaling by s means dividing coordinates by s
        # (inverse transformation for grid_sample)
        s = 1.0 / config.scale_factor
        # logger.info(f"         Scaling matrix s = {s:.4f} (1/{config.scale_factor})")
        
        theta = torch.tensor([
            [s, 0, 0],
            [0, s, 0]
        ], dtype=attention.dtype, device=attention.device).unsqueeze(0).repeat(batch_heads, 1, 1)
        
        # logger.info(f"         Theta shape: {theta.shape}, theta[0]:\n{theta[0]}")
        
        # Generate sampling grid and apply transformation
        input_for_grid = attention.unsqueeze(1)
        # logger.info(f"         Input for grid_sample: shape={input_for_grid.shape}")
        
        grid = F.affine_grid(theta, input_for_grid.size(), align_corners=False)
        # logger.info(f"         Grid shape: {grid.shape}, grid range: [{grid.min():.4f}, {grid.max():.4f}]")
        # logger.info(f"         Grid center sample (first head): {grid[0, H//2, W//2]}")
        
        scaled = F.grid_sample(
            input_for_grid, 
            grid, 
            mode='bilinear',
            padding_mode=config.padding_mode,  # Configurable: 'zeros', 'border', 'reflection'
            align_corners=False
        )
        
        # logger.info(f"         After grid_sample: shape={scaled.shape}")
        # logger.info(f"         Attention statistics - Input: min={attention.min():.6f}, max={attention.max():.6f}, std={attention.std():.6f}")
        # logger.info(f"         Attention statistics - Output: min={scaled.squeeze(1).min():.6f}, max={scaled.squeeze(1).max():.6f}, std={scaled.squeeze(1).std():.6f}")
        
        # Show spatial statistics to verify transformation is affecting the pattern
        result = scaled.squeeze(1)
        
        # Compare center vs edge regions to see if zoom is working
        # center_h_start, center_h_end = H // 4, 3 * H // 4
        # center_w_start, center_w_end = W // 4, 3 * W // 4
        
        # input_center_mean = attention[:, center_h_start:center_h_end, center_w_start:center_w_end].mean()
        # output_center_mean = result[:, center_h_start:center_h_end, center_w_start:center_w_end].mean()
        
        # input_edge_mean = attention[:, [0, -1], :].mean()  # Top and bottom edges
        # output_edge_mean = result[:, [0, -1], :].mean()
        
        # logger.info(f"         📊 Spatial region analysis:")
        # logger.info(f"            Input  - Center: {input_center_mean:.6f}, Edge: {input_edge_mean:.6f}")
        # logger.info(f"            Output - Center: {output_center_mean:.6f}, Edge: {output_edge_mean:.6f}")
        # logger.info(f"            Change - Center: {output_center_mean - input_center_mean:.6f}, Edge: {output_edge_mean - input_edge_mean:.6f}")
        
        # logger.info(f"         Output shape: {result.shape}, mean: {result.mean():.6f}, max: {result.max():.6f}")
        # logger.info(f"         Change: Δmean={result.mean() - attention.mean():.6f}, Δmax={result.max() - attention.max():.6f}")
        
        return result
    
    def _apply_rotation(self, attention: torch.Tensor, config: BendingConfig) -> torch.Tensor:
        """
        Rotate attention pattern around the center.
        
        Two modes:
        - crop_rotated=True (default): Pure rotation. Rotate the image on a fixed canvas.
          Parts outside canvas are clipped, empty areas filled with padding_mode.
          No stretching - this is standard image rotation behavior.
          
        - crop_rotated=False: Rotate and inscribe. After rotation, zoom out just enough
          so the entire original image fits in the canvas (inscribed rectangle).
        """
        if config.angle == 0:
            return attention
        
        batch_heads, H, W = attention.shape
        angle_rad = np.deg2rad(config.angle)
        
        # Create coordinate grids for the output
        # These are in pixel coordinates [0, H-1] and [0, W-1]
        y = torch.arange(H, dtype=attention.dtype, device=attention.device)
        x = torch.arange(W, dtype=attention.dtype, device=attention.device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # Center coordinates (rotation center)
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        
        # Translate to origin
        xx_centered = xx - cx
        yy_centered = yy - cy
        
        if config.crop_rotated:
            # Pure rotation: identity scale
            scale = 1.0
        else:
            # Inscribed rectangle: zoom out to fit rotated content
            # For a WxH rectangle rotated by θ, the bounding box is:
            # width' = W|cos(θ)| + H|sin(θ)|
            # height' = W|sin(θ)| + H|cos(θ)|
            # We want to scale so this fits in original WxH
            cos_a = abs(np.cos(angle_rad))
            sin_a = abs(np.sin(angle_rad))
            scale_w = W / (W * cos_a + H * sin_a)
            scale_h = H / (W * sin_a + H * cos_a)
            scale = min(scale_w, scale_h)
        
        # Apply inverse rotation (we're finding where each output pixel came from)
        # R^-1(θ) = R(-θ) for rotation matrices
        cos_a = np.cos(-angle_rad) * scale
        sin_a = np.sin(-angle_rad) * scale
        
        # Rotate coordinates
        xx_src = xx_centered * cos_a - yy_centered * sin_a + cx
        yy_src = xx_centered * sin_a + yy_centered * cos_a + cy
        
        # Normalize to [-1, 1] for grid_sample
        xx_norm = 2.0 * xx_src / (W - 1) - 1.0
        yy_norm = 2.0 * yy_src / (H - 1) - 1.0
        
        # Stack to create grid [H, W, 2] where last dim is (x, y)
        grid = torch.stack([xx_norm, yy_norm], dim=-1)  # [H, W, 2]
        
        # Expand for batch: grid_sample expects [N, H, W, 2] for [N, C, H, W] input
        grid = grid.unsqueeze(0).expand(batch_heads, -1, -1, -1)  # [batch_heads, H, W, 2]
        
        # Apply grid sampling
        # Input: [batch_heads, H, W] → add channel dim → [batch_heads, 1, H, W]
        # Grid: [batch_heads, H, W, 2]
        rotated = F.grid_sample(
            attention.unsqueeze(1),  # [batch_heads, 1, H, W]
            grid,  # [batch_heads, H, W, 2]
            mode='bilinear',
            padding_mode=config.padding_mode,
            align_corners=True
        )
        
        return rotated.squeeze(1)  # [batch_heads, H, W]
    
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
            print("Flipping horizontally")
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
        # TODO: Just SOFTMAX? OR... Do transform then softmax?
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
        
        logger.info(f"📐 Inferred spatial shape: ({h}, {w}) from {num_tokens} tokens")
        logger.info(f"   Factorization: {h} × {w} = {h*w}")
        return (h, w)
    
    def get_stats(self) -> Dict:
        """Get statistics about bending applications."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics tracking and debug batch directory for new run."""
        self.stats = {
            "applications": 0,
            "tokens_bent": {},
            "norm_changes": [],
        }
        # Reset batch directory so next run gets a new timestamp
        self._debug_batch_dir = None


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
        device=config_dict.get("device", "cuda"),
        apply_before_softmax=config_dict.get("apply_before_softmax", False)
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
