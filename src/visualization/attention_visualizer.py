#!/usr/bin/env python3
"""
Comprehensive attention map visualization library with video generation and overlay capabilities.
"""

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import imageio
from pathlib import Path
from typing import Union, Optional, Tuple, List, Dict, Any
from enum import Enum
import logging
from dataclasses import dataclass
try:
    from scipy.ndimage import zoom
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .attention_analyzer import AttentionAnalyzer, AttentionMapInfo


class FusionMethod(Enum):
    """Methods for fusing attention across dimensions."""
    MEAN = "mean"
    MAX = "max" 
    MIN = "min"
    MEDIAN = "median"


class ColorMap(Enum):
    """Available color maps for attention visualization."""
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    JET = "jet"
    HOT = "hot"
    COOL = "cool"
    TURBO = "turbo"


@dataclass
class VideoConfig:
    """Configuration for video generation."""
    fps: int = 15
    codec: str = "libx264"
    quality: int = 8
    interpolation_factor: int = 4  # How many frames to interpolate between latent frames
    upscale_factor: int = 16  # How much to upscale latent resolution


@dataclass
class OverlayConfig:
    """Configuration for attention overlay on videos."""
    alpha: float = 0.75  # Blend strength (0=only source, 1=only attention)
    colormap: ColorMap = ColorMap.JET
    normalize_per_frame: bool = False  # Whether to normalize attention per frame
    threshold: Optional[float] = None  # Optional threshold for attention values
    invert_values: bool = False  # Whether to invert attention values 


class AttentionVisualizer:
    """Comprehensive attention map visualization with video generation capabilities."""
    
    def __init__(self, 
                 analyzer: AttentionAnalyzer, 
                 output_dir: Union[str, Path] = "outputs/visualizations",
                 figsize: Tuple[int, int] = (12, 8),
                 colormap: str = "hot",
                 fps: int = 10,
                 overlay_alpha: float = 0.6,
                 interpolation_steps: int = 2,
                 include_colorbar: bool = True):
        self.analyzer = analyzer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Visualization parameters
        self.figsize = figsize
        self.colormap = colormap
        self.fps = fps
        self.overlay_alpha = overlay_alpha
        self.interpolation_steps = interpolation_steps
        self.include_colorbar = include_colorbar
        
        # Set up matplotlib for non-interactive use
        plt.switch_backend('Agg')
    
    def _apply_fusion(self, tensor: torch.Tensor, method: FusionMethod, dim: int) -> torch.Tensor:
        """Apply fusion method along specified dimension."""
        if method == FusionMethod.MEAN:
            return tensor.mean(dim=dim)
        elif method == FusionMethod.MAX:
            return tensor.max(dim=dim).values
        elif method == FusionMethod.MIN:
            return tensor.min(dim=dim).values
        elif method == FusionMethod.MEDIAN:
            return tensor.median(dim=dim).values
        else:
            raise ValueError(f"Unknown fusion method: {method}")
    
    def _normalize_attention(self, attention: np.ndarray, 
                           normalize_per_frame: bool = False,
                           threshold: Optional[float] = None,
                           invert_values: bool = False) -> np.ndarray:
        """Normalize attention values to [0, 255] range."""
        
        # Invert values if needed (for when low values should represent high attention)
        if invert_values:
            # Find the range and invert: low becomes high, high becomes low
            global_min, global_max = attention.min(), attention.max()
            attention = global_max - attention + global_min
        
        if threshold is not None:
            attention = np.where(attention > threshold, attention, 0)
        
        if normalize_per_frame and attention.ndim > 2:
            # Normalize each frame independently
            normalized = np.zeros_like(attention)
            for i in range(attention.shape[0]):
                frame = attention[i]
                frame_min, frame_max = frame.min(), frame.max()
                if frame_max > frame_min:
                    normalized[i] = 255 * (frame - frame_min) / (frame_max - frame_min)
                else:
                    normalized[i] = frame
        else:
            # Global normalization
            global_min, global_max = attention.min(), attention.max()
            if global_max > global_min:
                normalized = 255 * (attention - global_min) / (global_max - global_min)
            else:
                normalized = attention
        
        return normalized.astype(np.uint8)
    
    def _get_colormap(self, colormap: ColorMap) -> Any:
        """Get matplotlib colormap."""
        if colormap == ColorMap.JET:
            return cv2.COLORMAP_JET
        elif colormap == ColorMap.HOT:
            return cv2.COLORMAP_HOT
        elif colormap == ColorMap.COOL:
            return cv2.COLORMAP_COOL
        elif colormap == ColorMap.VIRIDIS:
            return cv2.COLORMAP_VIRIDIS
        elif colormap == ColorMap.PLASMA:
            return cv2.COLORMAP_PLASMA
        elif colormap == ColorMap.INFERNO:
            return cv2.COLORMAP_INFERNO
        elif colormap == ColorMap.MAGMA:
            return cv2.COLORMAP_MAGMA
        elif colormap == ColorMap.TURBO:
            return cv2.COLORMAP_TURBO
        else:
            return cv2.COLORMAP_JET
    
    def _interpolate_frames(self, attention_frames: np.ndarray, 
                          interpolation_factor: int) -> np.ndarray:
        """Interpolate between attention frames for smoother video."""
        if interpolation_factor <= 1:
            return attention_frames
        
        num_frames, height, width = attention_frames.shape
        new_num_frames = (num_frames - 1) * interpolation_factor + 1
        interpolated = np.zeros((new_num_frames, height, width), dtype=attention_frames.dtype)
        
        for i in range(new_num_frames):
            original_idx = i / interpolation_factor
            lower_idx = int(np.floor(original_idx))
            upper_idx = min(lower_idx + 1, num_frames - 1)
            alpha = original_idx - lower_idx
            
            interpolated[i] = (1 - alpha) * attention_frames[lower_idx] + alpha * attention_frames[upper_idx]
        
        return interpolated.astype(attention_frames.dtype)
    
    def _upscale_frames(self, frames: np.ndarray, upscale_factor: int) -> np.ndarray:
        """Upscale attention frames using interpolation."""
        if upscale_factor <= 1:
            return frames
        
        num_frames, height, width = frames.shape
        new_height, new_width = height * upscale_factor, width * upscale_factor
        upscaled = np.zeros((num_frames, new_height, new_width), dtype=frames.dtype)
        
        for i in range(num_frames):
            upscaled[i] = cv2.resize(frames[i], (new_width, new_height), 
                                   interpolation=cv2.INTER_LINEAR)
        
        return upscaled
    
    def visualize_single_frame(self, video_id: str, token_word: str, step: int,
                             frame_idx: int = 0,
                             fusion_method: FusionMethod = FusionMethod.MEAN,
                             colormap: ColorMap = ColorMap.VIRIDIS,
                             save_path: Optional[str] = None,
                             show_plot: bool = False) -> Tuple[plt.Figure, AttentionMapInfo]:
        """
        Visualize a single attention frame as a static plot.
        
        Args:
            video_id: Video identifier
            token_word: Token to visualize
            step: Diffusion step
            frame_idx: Which frame to show (if attention has multiple frames)
            fusion_method: How to fuse blocks/heads
            colormap: Color scheme
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            (matplotlib_figure, attention_metadata)
        """
        # Load attention map and metadata
        spatial_attention, metadata = self.analyzer.get_spatial_attention_map(
            video_id, token_word, step, aggregate_all=True
        )
        
        # Convert to numpy
        attention_np = spatial_attention.cpu().numpy()
        
        # Handle frame selection
        if attention_np.ndim == 3:  # [frames, height, width]
            frame_attention = attention_np[frame_idx]
        elif attention_np.ndim == 2:  # [height, width]
            frame_attention = attention_np
        else:
            raise ValueError(f"Unexpected attention shape: {attention_np.shape}")
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        im = ax.imshow(frame_attention, cmap=colormap.value, aspect='auto')
        ax.set_title(f"Attention Map - {token_word} (Step {step}, Frame {frame_idx})\n"
                    f"Video: {video_id} | Method: {metadata.aggregation_method}")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label="Attention Weight")
        
        # Add metadata text
        info_text = (f"Shape: {metadata.attention_shape}\n"
                    f"Video Size: {metadata.video_width}x{metadata.video_height}x{metadata.video_frames}\n"
                    f"Blocks: {metadata.num_blocks}, Heads: {metadata.num_heads}\n"
                    f"Seed: {metadata.seed}, CFG: {metadata.cfg_scale}")
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            # self.logger.info(f"Saved attention frame plot to: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig, metadata
    
    def generate_attention_video(self, video_id: str, token_word: str,
                               step: Optional[int] = None,
                               video_config: VideoConfig = VideoConfig(),
                               overlay_config: OverlayConfig = OverlayConfig(),
                               fusion_method: FusionMethod = FusionMethod.MEAN,
                               output_filename: Optional[str] = None,
                               source_video_path: Optional[str] = None) -> str:
        """
        Generate attention video that matches the exact dimensions and frames of the original video.
        
        Args:
            video_id: Video identifier
            token_word: Token to visualize
            step: Specific diffusion step to visualize (if None, uses aggregated attention)
            video_config: Video generation settings
            overlay_config: Overlay settings
            fusion_method: How to fuse attention dimensions
            output_filename: Custom filename for output
            source_video_path: Optional source video to overlay attention on
            
        Returns:
            Path to generated video file
        """
        # Load attention map for specific step or aggregated
        if step is not None:
            # Load specific step
            self.logger.info(f"Loading specific step {step} for {video_id}:{token_word}")
            spatial_attention, metadata = self.analyzer.get_spatial_attention_map(
                video_id, token_word, step, aggregate_all=True
            )
            # self.logger.info(f"Loaded step {step}: spatial_attention.shape={spatial_attention.shape}")
        else:
            # Use aggregated attention across all steps
            # self.logger.info(f"Loading aggregated attention for {video_id}:{token_word}")
            aggregated_attention = self.generate_aggregated_attention_map(
                video_id, token_word, aggregate_steps=True, fusion_method=fusion_method
            )
            # self.logger.info(f"Generated aggregated attention: shape={aggregated_attention.shape}")
            spatial_attention = aggregated_attention.squeeze(-1)  # Remove token dimension
            # self.logger.info(f"After squeeze: spatial_attention.shape={spatial_attention.shape}")
            
            # Get metadata from first available step
            _, metadata_list = self.analyzer.get_temporal_evolution(
                video_id, token_word, aggregate_blocks=True, aggregate_heads=True
            )
            metadata = metadata_list[0]
            # self.logger.info(f"Using metadata from first step: {metadata.step}")
        
        # self.logger.info(f"Final spatial_attention: shape={spatial_attention.shape}, dtype={spatial_attention.dtype}")
        
        if not (metadata.video_frames and metadata.video_height and metadata.video_width):
            raise ValueError("Video dimensions not available in metadata")
        
        # Target video dimensions
        target_frames = metadata.video_frames
        target_height = metadata.video_height  
        target_width = metadata.video_width
        
        # Calculate latent dimensions based on VAE downsampling
        latent_frames = (target_frames - 1) // 4 + 1
        latent_height = target_height // 16
        latent_width = target_width // 16
        
        # Reshape spatial attention to latent video format: [frames, height, width]
        spatial_size = spatial_attention.shape[0]
        expected_spatial_size = latent_frames * latent_height * latent_width
        
        # self.logger.info(f"Reshaping attention: spatial_size={spatial_size}, expected={expected_spatial_size}")
        # self.logger.info(f"Original spatial_attention shape: {spatial_attention.shape}")
        # self.logger.info(f"Latent dimensions: frames={latent_frames}, height={latent_height}, width={latent_width}")
        
        if spatial_size == expected_spatial_size:
            # Perfect match - reshape to latent dimensions
            attention_latent = spatial_attention.view(latent_frames, latent_height, latent_width)
            # self.logger.info(f"Perfect reshape match: {attention_latent.shape}")
        else:
            # Try to infer dimensions
            self.logger.warning(f"Spatial size mismatch: got {spatial_size}, expected {expected_spatial_size}")
            # Assume temporal dimension is correct and spatial is flattened
            attention_latent = spatial_attention.view(latent_frames, -1)
            spatial_per_frame = attention_latent.shape[1]
            # self.logger.info(f"Fallback reshape: {attention_latent.shape}, spatial_per_frame={spatial_per_frame}")
            
            # Try to make it square-ish
            frame_dim = int(np.sqrt(spatial_per_frame))
            if frame_dim * frame_dim == spatial_per_frame:
                attention_latent = attention_latent.view(latent_frames, frame_dim, frame_dim)
                latent_height, latent_width = frame_dim, frame_dim
                # self.logger.info(f"Square reshape: {attention_latent.shape}")
            else:
                # Fallback: use original latent dimensions and pad/crop
                attention_latent = attention_latent.view(latent_frames, latent_height, latent_width)
                # self.logger.info(f"Forced reshape: {attention_latent.shape}")
        
        # Convert to numpy
        attention_np = attention_latent.cpu().numpy()
        # self.logger.info(f"Converted to numpy: shape={attention_np.shape}, dtype={attention_np.dtype}")
        
        # self.logger.info(f"Attention latent shape: {attention_np.shape} -> Target: {target_frames}×{target_height}×{target_width}")
        
        # Resize each frame to match target video dimensions with interpolation
        # Calculate interpolated frames similar to the original approach
        expected_interpolated_frames = (latent_frames - 1) * 4 + 1
        use_interpolation = target_frames >= expected_interpolated_frames or target_frames > latent_frames
        
        attention_frames = np.zeros((target_frames, target_height, target_width), dtype=np.float32)
        
        # First, create interpolated latent frames if we have fewer latent frames than target frames
        if target_frames > latent_frames:
            # Interpolate between latent frames
            interpolated_latent = np.zeros((target_frames, latent_height, latent_width), dtype=np.float32)
            for i in range(target_frames):
                original_idx = i * (latent_frames - 1) / (target_frames - 1) if target_frames > 1 else 0
                lower_idx = int(np.floor(original_idx))
                upper_idx = min(lower_idx + 1, latent_frames - 1)
                alpha = original_idx - lower_idx
                
                if lower_idx == upper_idx:
                    interpolated_latent[i] = attention_np[lower_idx]
                else:
                    interpolated_latent[i] = (1 - alpha) * attention_np[lower_idx] + alpha * attention_np[upper_idx]
        else:
            interpolated_latent = attention_np
        
        # Now upscale each frame to target dimensions
        for frame_idx in range(target_frames):
            # Get the interpolated latent attention for this frame
            latent_attention = interpolated_latent[frame_idx] if target_frames > latent_frames else attention_np[min(frame_idx, latent_frames - 1)]
            
            # self.logger.debug(f"Processing frame {frame_idx}")
            # self.logger.debug(f"  latent_attention shape: {latent_attention.shape}")
            # self.logger.debug(f"  latent_attention dtype: {latent_attention.dtype}")
            # self.logger.debug(f"  latent_attention range: [{latent_attention.min():.6f}, {latent_attention.max():.6f}]")
            # self.logger.debug(f"  has NaN: {np.any(np.isnan(latent_attention))}")
            # self.logger.debug(f"  has inf: {np.any(np.isinf(latent_attention))}")
            # self.logger.debug(f"  is contiguous: {latent_attention.flags.c_contiguous}")
            
            # Debug: Check for invalid values
            if np.any(np.isnan(latent_attention)) or np.any(np.isinf(latent_attention)):
                self.logger.warning(f"Frame {frame_idx}: Found NaN or inf values in attention map")
                latent_attention = np.nan_to_num(latent_attention, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Apply attention threshold for visualization if specified
            if overlay_config.threshold is not None:
                pre_threshold_range = f"[{latent_attention.min():.6f}, {latent_attention.max():.6f}]"
                latent_attention = np.where(latent_attention > overlay_config.threshold, latent_attention, 0)
                post_threshold_range = f"[{latent_attention.min():.6f}, {latent_attention.max():.6f}]"
                self.logger.debug(f"  Applied threshold {overlay_config.threshold}: {pre_threshold_range} -> {post_threshold_range}")
            
            # Ensure proper dtype for OpenCV (float32)
            if latent_attention.dtype != np.float32:
                self.logger.debug(f"Converting dtype from {latent_attention.dtype} to float32")
                latent_attention = latent_attention.astype(np.float32)
            
            # Ensure the array is contiguous
            if not latent_attention.flags.c_contiguous:
                self.logger.debug("Making array contiguous")
                latent_attention = np.ascontiguousarray(latent_attention)
            
            self.logger.debug(f"  Final before resize: shape={latent_attention.shape}, "
                            f"dtype={latent_attention.dtype}, contiguous={latent_attention.flags.c_contiguous}")
            self.logger.debug(f"  Target resize: {latent_attention.shape} -> ({target_width}, {target_height})")
            
            # Resize from latent to target dimensions
            try:
                resized_attention = cv2.resize(
                    latent_attention, 
                    (target_width, target_height), 
                    interpolation=cv2.INTER_LINEAR
                )
                self.logger.debug(f"  Resize successful: {resized_attention.shape}")
            except cv2.error as e:
                self.logger.error(f"OpenCV resize failed for frame {frame_idx}: {e}")
                self.logger.error(f"  Input shape: {latent_attention.shape}")
                self.logger.error(f"  Input dtype: {latent_attention.dtype}")
                self.logger.error(f"  Input contiguous: {latent_attention.flags.c_contiguous}")
                self.logger.error(f"  Target dimensions: ({target_width}, {target_height})")
                self.logger.error(f"  Input value range: [{latent_attention.min():.6f}, {latent_attention.max():.6f}]")
                
                # Fallback: use scipy zoom
                self.logger.info("Attempting fallback with scipy zoom")
                from scipy.ndimage import zoom
                scale_h = target_height / latent_attention.shape[0]
                scale_w = target_width / latent_attention.shape[1]
                resized_attention = zoom(latent_attention, (scale_h, scale_w), order=1)
                resized_attention = resized_attention.astype(np.float32)
                self.logger.info(f"  Scipy fallback successful: {resized_attention.shape}")
            
            attention_frames[frame_idx] = resized_attention
        
        # Normalize attention values
        attention_normalized = self._normalize_attention(
            attention_frames,
            normalize_per_frame=overlay_config.normalize_per_frame,
            threshold=overlay_config.threshold,
            invert_values=overlay_config.invert_values
        )
        
        # Generate output filename with descriptive names matching attention maps structure
        if output_filename is None:
            if step is not None:
                # For specific steps: step_000_overlay.mp4 or step_000_attention.mp4
                base_name = f"step_{step:03d}"
            else:
                # For aggregated: aggregate_overlay.mp4 or aggregate_attention.mp4
                base_name = "aggregate"
            
            if source_video_path:
                output_filename = f"{base_name}_overlay.mp4"
            else:
                output_filename = f"{base_name}_attention.mp4"
        
        self.logger.info(f"AttentionVisualizer received output_filename: '{output_filename}'")
        self.logger.info(f"AttentionVisualizer self.output_dir: '{self.output_dir}'")
        
        # Create nested directory structure: prompt_000/vid_000/token_flower/
        # Parse video_id to extract prompt and video parts
        if "_vid" in video_id:
            prompt_part, vid_part = video_id.rsplit("_vid", 1)
            # Use the output directory directly (it's already set to attention_videos)
            # Add token directory to match attention maps structure
            video_output_dir = self.output_dir / prompt_part / f"vid{vid_part}" / f"token_{token_word}"
            self.logger.info(f"Creating attention video directory: {video_output_dir}")
        else:
            # Fallback for video IDs that don't follow the expected pattern
            video_output_dir = self.output_dir / video_id / f"token_{token_word}"
            self.logger.info(f"Creating fallback attention video directory: {video_output_dir}")
        
        video_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = video_output_dir / output_filename
        self.logger.info(f"Final output path: {output_path}")
        
        # Get colormap
        cv_colormap = self._get_colormap(overlay_config.colormap)
        
        # Load source video if provided
        source_frames = None
        if source_video_path and Path(source_video_path).exists():
            source_reader = imageio.get_reader(source_video_path)
            source_frames = []
            try:
                for frame in source_reader:
                    source_frames.append(frame)
            except Exception as e:
                self.logger.warning(f"Could not read source video: {e}")
                source_frames = None
            finally:
                source_reader.close()
        
        # Generate video with exact same frame count as original
        with imageio.get_writer(str(output_path), fps=self.fps, 
                              codec=video_config.codec, quality=video_config.quality) as writer:
            
            for frame_idx in range(target_frames):
                # Apply colormap to attention
                attention_colored = cv2.applyColorMap(attention_normalized[frame_idx], cv_colormap)
                attention_colored = cv2.cvtColor(attention_colored, cv2.COLOR_BGR2RGB)
                
                # Post-process resize if needed (for video_scale parameter)
                # This is a no-op at full resolution, but enables scaling output
                if metadata and (attention_colored.shape[0] != target_height or attention_colored.shape[1] != target_width):
                    attention_colored = cv2.resize(
                        attention_colored,
                        (target_width, target_height),
                        interpolation=cv2.INTER_LINEAR
                    )
                
                # Overlay on source video if available
                if source_frames and frame_idx < len(source_frames):
                    source_frame = source_frames[frame_idx]
                    
                    # Ensure source frame matches attention frame dimensions
                    if source_frame.shape[:2] != attention_colored.shape[:2]:
                        source_frame = cv2.resize(source_frame, 
                                                (attention_colored.shape[1], attention_colored.shape[0]))
                    
                    # Blend frames
                    blended = cv2.addWeighted(
                        attention_colored, overlay_config.alpha,
                        source_frame, 1 - overlay_config.alpha,
                        0
                    )
                    writer.append_data(blended)
                else:
                    writer.append_data(attention_colored)
        
        self.logger.info(f"Generated attention video: {output_path} ({target_frames} frames, {target_height}×{target_width})")
        return str(output_path)
    
    def _generate_attention_video_from_data(self, 
                                           spatial_attention: torch.Tensor,
                                           video_id: str, 
                                           token_word: str,
                                           step: Optional[int] = None,
                                           video_config: VideoConfig = VideoConfig(),
                                           overlay_config: OverlayConfig = OverlayConfig(),
                                           output_filename: Optional[str] = None,
                                           source_video_path: Optional[str] = None,
                                           metadata: Optional[Dict[str, Any]] = None,
                                           output_width: Optional[int] = None,
                                           output_height: Optional[int] = None) -> str:
        """
        Generate attention video from pre-loaded attention data.
        
        This is a simplified version of generate_attention_video that works with
        already loaded attention data instead of loading from the analyzer.
        
        Args:
            spatial_attention: Pre-loaded attention tensor
            video_id: Video identifier 
            token_word: Token name
            step: Step number (if specific step)
            video_config: Video generation settings
            overlay_config: Overlay settings
            output_filename: Output filename
            source_video_path: Optional source video for overlay
            metadata: Optional metadata dict with video dimensions (reflects stored data, not output)
            output_width: Optional output video width (overrides metadata for scaling)
            output_height: Optional output video height (overrides metadata for scaling)
            
        Returns:
            Path to generated video file
        """
        # Convert to numpy if needed
        if isinstance(spatial_attention, torch.Tensor):
            attention_np = spatial_attention.cpu().numpy()
        else:
            attention_np = spatial_attention
        
        self.logger.debug(f"Received attention tensor: shape={attention_np.shape}, ndim={attention_np.ndim}")
        
        # Get target dimensions from metadata if available
        if metadata:
            target_frames = metadata.get('video_frames', 25)
            target_height = metadata.get('video_height', 480)
            target_width = metadata.get('video_width', 848)
            
            # Calculate latent dimensions based on VAE downsampling (typical 16x spatial, 4x temporal)
            latent_frames = (target_frames - 1) // 4 + 1
            latent_height = target_height // 16
            latent_width = target_width // 16
            
            # self.logger.info(f"Using metadata dimensions: target={target_frames}×{target_height}×{target_width}, latent={latent_frames}×{latent_height}×{latent_width}")
        else:
            # No metadata - will determine dimensions from tensor factorization
            latent_frames = None
            latent_height = None 
            latent_width = None
            target_height = 480
            target_width = 848
            target_frames = 25
            self.logger.warning("No metadata provided, will determine dimensions from tensor")
        
        # Handle different dimensionalities
        if attention_np.ndim == 1:
            spatial_size = attention_np.shape[0]
            
            if metadata and latent_frames and latent_height and latent_width:
                # Try to use metadata dimensions first
                expected_latent_size = latent_frames * latent_height * latent_width
                if spatial_size == expected_latent_size:
                    attention_np = attention_np.reshape(latent_frames, latent_height, latent_width)
                    self.logger.debug(f"Reshaped 1D attention {spatial_size} -> {latent_frames}×{latent_height}×{latent_width} (metadata match)")
                else:
                    self.logger.warning(f"Metadata size mismatch: tensor={spatial_size}, expected={expected_latent_size}")
                    # Fall back to factorization
                    factors = []
                    for i in range(1, int(np.sqrt(spatial_size)) + 1):
                        if spatial_size % i == 0:
                            factors.append((i, spatial_size // i))
                    
                    if factors:
                        best_factor = min(factors, key=lambda x: abs(x[0] - x[1]))
                        h, w = best_factor
                        attention_np = attention_np.reshape(h, w)
                        # self.logger.info(f"Reshaped 1D attention {spatial_size} -> {h}×{w} (spatial fallback)")
                    else:
                        raise ValueError(f"Cannot factorize attention size {spatial_size}")
            else:
                # No metadata - find best factorization
                factors = []
                for i in range(1, int(np.sqrt(spatial_size)) + 1):
                    if spatial_size % i == 0:
                        factors.append((i, spatial_size // i))
                
                if factors:
                    best_factor = min(factors, key=lambda x: abs(x[0] - x[1]))
                    h, w = best_factor
                    attention_np = attention_np.reshape(h, w)
                    # self.logger.info(f"Reshaped 1D attention {spatial_size} -> {h}×{w} (no metadata)")
                else:
                    raise ValueError(f"Cannot factorize attention size {spatial_size}")
                
        elif attention_np.ndim == 2:
            # Already in 2D format (height, width)
            self.logger.info(f"Using 2D attention as-is: {attention_np.shape}")
            
        elif attention_np.ndim == 3:
            # 3D format (frames, height, width) - already in correct format
            self.logger.info(f"Using 3D attention as-is: {attention_np.shape}")
            
        elif attention_np.ndim == 4:
            # 4D format - squeeze out singleton dimensions and handle remaining
            original_shape = attention_np.shape
            attention_np = attention_np.squeeze()
            # self.logger.info(f"Squeezed 4D attention {original_shape} -> {attention_np.shape}")
            
            # Apply same logic as above for the squeezed result
            if attention_np.ndim == 1:
                spatial_size = attention_np.shape[0]
                
                if metadata and latent_frames and latent_height and latent_width:
                    expected_latent_size = latent_frames * latent_height * latent_width
                    if spatial_size == expected_latent_size:
                        attention_np = attention_np.reshape(latent_frames, latent_height, latent_width)
                        # self.logger.info(f"Reshaped squeezed 1D attention {spatial_size} -> {latent_frames}×{latent_height}×{latent_width}")
                    else:
                        factors = []
                        for i in range(1, int(np.sqrt(spatial_size)) + 1):
                            if spatial_size % i == 0:
                                factors.append((i, spatial_size // i))
                        
                        if factors:
                            best_factor = min(factors, key=lambda x: abs(x[0] - x[1]))
                            h, w = best_factor
                            attention_np = attention_np.reshape(h, w)
                            # self.logger.info(f"Reshaped squeezed 1D attention {spatial_size} -> {h}×{w}")
                else:
                    factors = []
                    for i in range(1, int(np.sqrt(spatial_size)) + 1):
                        if spatial_size % i == 0:
                            factors.append((i, spatial_size // i))
                    
                    if factors:
                        best_factor = min(factors, key=lambda x: abs(x[0] - x[1]))
                        h, w = best_factor
                        attention_np = attention_np.reshape(h, w)
                        # self.logger.info(f"Reshaped squeezed 1D attention {spatial_size} -> {h}×{w}")
        
        # Ensure we have at least 2D data
        if attention_np.ndim == 1:
            raise ValueError(f"Unable to convert attention data to usable format. Final shape: {attention_np.shape}")
        
        # Add temporal dimension if needed (convert 2D to 3D)
        if attention_np.ndim == 2:
            attention_np = attention_np[np.newaxis, ...]
            
        # Now attention_np should be in format [frames, height, width]
        if attention_np.ndim != 3:
            raise ValueError(f"Unable to convert attention data to 3D format. Final shape: {attention_np.shape}")
        
        # Get actual dimensions from final tensor
        actual_frames, actual_height, actual_width = attention_np.shape
        # self.logger.info(f"Final attention tensor: {actual_frames}×{actual_height}×{actual_width}")
        
        # Use metadata dimensions for all processing (reflects actual stored data)
        # output_width/output_height will only be used at the very end for final resize
        if metadata:
            target_frames = metadata.get('video_frames', actual_frames * 4)
            target_height = metadata.get('video_height', actual_height * 16)
            target_width = metadata.get('video_width', actual_width * 16)
        else:
            # Default upscaling
            target_frames = max(actual_frames * 4, 25) if actual_frames > 1 else 25
            target_height = actual_height * 16
            target_width = actual_width * 16
        
        # If we have source video, get its dimensions
        if source_video_path and Path(source_video_path).exists():
            try:
                import imageio
                with imageio.get_reader(source_video_path) as reader:
                    source_meta = reader.get_meta_data()
                    if 'fps' in source_meta:
                        video_config.fps = source_meta['fps']
                    # Get actual frame count
                    target_frames = reader.count_frames()
                    # Get frame dimensions from first frame
                    first_frame = reader.get_data(0)
                    target_height, target_width = first_frame.shape[:2]
            except Exception as e:
                self.logger.warning(f"Could not get video metadata: {e}")
        
        # Create video frames by interpolating and upscaling attention data
        attention_video_frames = np.zeros((target_frames, target_height, target_width), dtype=np.float32)
        
        # Interpolate attention frames to match target frame count
        for frame_idx in range(target_frames):
            # Map target frame to latent frame
            if actual_frames > 1:
                latent_frame_idx = frame_idx * (actual_frames - 1) / (target_frames - 1) if target_frames > 1 else 0
                lower_idx = int(np.floor(latent_frame_idx))
                upper_idx = min(lower_idx + 1, actual_frames - 1)
                alpha = latent_frame_idx - lower_idx
                
                if lower_idx == upper_idx:
                    latent_attention = attention_np[lower_idx]
                else:
                    latent_attention = (1 - alpha) * attention_np[lower_idx] + alpha * attention_np[upper_idx]
            else:
                # Single frame, replicate across all target frames
                latent_attention = attention_np[0]
            
            # Resize from latent to target dimensions
            resized_attention = cv2.resize(
                latent_attention.astype(np.float32), 
                (target_width, target_height), 
                interpolation=cv2.INTER_LINEAR
            )
            
            attention_video_frames[frame_idx] = resized_attention
        
        # Normalize attention values
        attention_normalized = self._normalize_attention(
            attention_video_frames,
            normalize_per_frame=overlay_config.normalize_per_frame,
            threshold=overlay_config.threshold,
            invert_values=overlay_config.invert_values
        )
        
        # Generate output filename
        if output_filename is None:
            if step is not None:
                base_name = f"step_{step:03d}"
            else:
                base_name = "attention"
            
            if source_video_path:
                output_filename = f"{base_name}_overlay.mp4"
            else:
                output_filename = f"{base_name}_attention.mp4"
        
        # Use the output directory directly
        output_path = self.output_dir / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get colormap
        cv_colormap = self._get_colormap(overlay_config.colormap)
        
        # Load source video if provided
        source_frames = None
        if source_video_path and Path(source_video_path).exists():
            import imageio
            source_reader = imageio.get_reader(source_video_path)
            source_frames = []
            try:
                for frame in source_reader:
                    source_frames.append(frame)
            except Exception as e:
                self.logger.warning(f"Could not read source video: {e}")
                source_frames = None
            finally:
                source_reader.close()
        
        # Generate video
        import imageio
        with imageio.get_writer(str(output_path), fps=video_config.fps, 
                              codec=video_config.codec, quality=video_config.quality) as writer:
            
            for frame_idx in range(target_frames):
                # Apply colormap to attention
                attention_colored = cv2.applyColorMap(attention_normalized[frame_idx], cv_colormap)
                attention_colored = cv2.cvtColor(attention_colored, cv2.COLOR_BGR2RGB)
                
                # Post-process resize ONLY if explicit output dimensions provided (for scaling)
                # This is the final step before writing to video
                if output_width is not None and output_height is not None:
                    if attention_colored.shape[0] != output_height or attention_colored.shape[1] != output_width:
                        attention_colored = cv2.resize(
                            attention_colored,
                            (output_width, output_height),
                            interpolation=cv2.INTER_LINEAR
                        )
                
                # Overlay on source video if available
                if source_frames and frame_idx < len(source_frames):
                    source_frame = source_frames[frame_idx]
                    
                    # Ensure source frame matches attention frame dimensions
                    if source_frame.shape[:2] != attention_colored.shape[:2]:
                        source_frame = cv2.resize(source_frame, 
                                                (attention_colored.shape[1], attention_colored.shape[0]))
                    
                    # Blend frames
                    blended = cv2.addWeighted(
                        attention_colored, overlay_config.alpha,
                        source_frame, 1 - overlay_config.alpha,
                        0
                    )
                    writer.append_data(blended)
                else:
                    writer.append_data(attention_colored)
        
        # self.logger.info(f"Generated attention video from data: {output_path} ({target_frames} frames, {target_height}×{target_width})")
        return str(output_path)
    
    def generate_temporal_comparison(self, video_id: str, token_words: List[str],
                                   steps: Optional[List[int]] = None,
                                   fusion_method: FusionMethod = FusionMethod.MEAN,
                                   colormap: ColorMap = ColorMap.VIRIDIS,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Generate comparison plot of attention across multiple tokens and/or steps.
        
        Args:
            video_id: Video identifier
            token_words: List of tokens to compare
            steps: Optional list of specific steps to show (default: all steps)
            fusion_method: How to fuse attention dimensions
            colormap: Color scheme
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib figure
        """
        # Determine grid layout
        num_tokens = len(token_words)
        cols = min(num_tokens, 4)
        rows = (num_tokens + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, token_word in enumerate(token_words):
            ax = axes[i] if i < len(axes) else None
            if ax is None:
                break
            
            try:
                # Get temporal evolution
                temporal_attention, metadata_list = self.analyzer.get_temporal_evolution(
                    video_id, token_word, aggregate_blocks=True, aggregate_heads=True
                )
                
                # Average across tokens and spatial dimensions for plotting
                attention_over_time = temporal_attention.mean(dim=[1, 2]).cpu().numpy()
                
                # Plot temporal evolution
                timesteps = [meta.timestep for meta in metadata_list]
                ax.plot(timesteps, attention_over_time, marker='o', linewidth=2, markersize=4)
                ax.set_title(f"Attention: {token_word}")
                ax.set_xlabel("Timestep")
                ax.set_ylabel("Mean Attention")
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading\n{token_word}:\n{str(e)}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"Error: {token_word}")
        
        # Hide unused subplots
        for i in range(num_tokens, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            # self.logger.info(f"Saved temporal comparison to: {save_path}")
        
        return fig
    
    def generate_aggregated_attention_map(self, video_id: str, token_word: str,
                                        aggregate_steps: bool = True,
                                        fusion_method: FusionMethod = FusionMethod.MEAN) -> torch.Tensor:
        """
        Generate aggregated attention map averaged across all steps.
        
        Args:
            video_id: Video identifier
            token_word: Token to analyze
            aggregate_steps: Whether to average across diffusion steps
            fusion_method: How to fuse dimensions
            
        Returns:
            Aggregated attention tensor
        """
        # Get all steps
        temporal_attention, metadata_list = self.analyzer.get_temporal_evolution(
            video_id, token_word, aggregate_blocks=True, aggregate_heads=True
        )
        
        if aggregate_steps:
            # Average across steps: [steps, spatial, tokens] -> [spatial, tokens]
            aggregated = temporal_attention.mean(dim=0)
        else:
            aggregated = temporal_attention
        
        return aggregated
    
    def create_static_video_from_aggregated(self, 
                                           aggregated_file: Path, 
                                           output_path: Path,
                                           duration: float = 3.0) -> Path:
        """Create a static video from aggregated attention data."""
        try:
            # Load aggregated attention
            data = np.load(aggregated_file)
            if 'attention' in data:
                attention_map = data['attention']
            else:
                attention_map = data[data.files[0]]
            
            # Create visualization
            fig, ax = plt.subplots(figsize=self.figsize, facecolor='black')
            ax.set_facecolor('black')
            
            # Display attention map
            im = ax.imshow(attention_map, cmap=self.colormap, interpolation='bilinear')
            ax.set_title(f"Aggregated Attention", color='white', fontsize=14, fontweight='bold')
            ax.axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            
            plt.tight_layout()
            
            # Create temporary image
            temp_img = output_path.parent / f"temp_{output_path.stem}.png"
            plt.savefig(temp_img, facecolor='black', bbox_inches='tight', dpi=150)
            plt.close()
            
            # Convert to video by repeating the frame
            fps = 30
            total_frames = int(duration * fps)
            
            # Load the image
            img = cv2.imread(str(temp_img))
            height, width = img.shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            # Write frames
            for _ in range(total_frames):
                writer.write(img)
            
            writer.release()
            
            # Clean up
            temp_img.unlink(missing_ok=True)
            
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"Error creating static video from aggregated attention: {e}")


# ============================================================================
# Static Helper Methods for Direct Tensor Visualization
# ============================================================================

@staticmethod
def prepare_attention_for_visualization(
    attention_tensor: Union[torch.Tensor, np.ndarray],
    spatial_shape: Union[Tuple[int, int], Tuple[int, int, int]],
    aggregate_heads: bool = True
) -> np.ndarray:
    """
    Prepare attention tensor for visualization by reshaping and aggregating.
    
    This is a core helper used by both class methods and external callers.
    
    Args:
        attention_tensor: Attention tensor in various formats:
            - [batch_heads, spatial_tokens]: Multi-head attention (needs aggregation)
            - [spatial_tokens]: Single aggregated attention
            - [F, H, W]: Video format (already spatial)
            - [H, W]: Image format (already spatial)
        spatial_shape: Expected spatial dimensions:
            - (F, H, W) for videos: frames × height × width
            - (H, W) for images: height × width
        aggregate_heads: If True, average across heads dimension
        
    Returns:
        np.ndarray in format [F, H, W] for video or [H, W] for image
    """
    logger = logging.getLogger(__name__)
    
    # Convert to numpy if needed
    if isinstance(attention_tensor, torch.Tensor):
        # Convert bfloat16 to float32 first (numpy doesn't support bfloat16)
        if attention_tensor.dtype == torch.bfloat16:
            attention_np = attention_tensor.detach().cpu().float().numpy()
        else:
            attention_np = attention_tensor.detach().cpu().numpy()
    else:
        attention_np = attention_tensor
    
    is_video = len(spatial_shape) == 3
    
    # Reshape based on input format
    if attention_np.ndim == 1:
        # [spatial_tokens] -> reshape to spatial format
        if is_video:
            F, H, W = spatial_shape
            attention_np = attention_np.reshape(F, H, W)
        else:
            H, W = spatial_shape
            attention_np = attention_np.reshape(H, W)
            
    elif attention_np.ndim == 2:
        # Could be [batch_heads, spatial_tokens] or [H, W]
        expected_spatial = np.prod(spatial_shape)
        
        if attention_np.shape[1] == expected_spatial and attention_np.shape[0] > 1:
            # Looks like [batch_heads, spatial_tokens]
            if aggregate_heads:
                attention_np = attention_np.mean(axis=0)  # -> [spatial_tokens]
            else:
                attention_np = attention_np[0]  # Take first head
            
            # Now reshape to spatial format
            if is_video:
                F, H, W = spatial_shape
                attention_np = attention_np.reshape(F, H, W)
            else:
                H, W = spatial_shape
                attention_np = attention_np.reshape(H, W)
        else:
            # Already [H, W] format
            if is_video:
                # Add frame dimension
                attention_np = attention_np[np.newaxis, :, :]
                
    elif attention_np.ndim == 3:
        # Could be [F, H, W] for video or [batch_heads, H, W]
        if is_video and attention_np.shape == spatial_shape:
            pass  # Already correct format
        elif attention_np.shape[0] > 1 and attention_np.shape[1:] == spatial_shape[1:]:
            # Looks like [batch_heads, H, W]
            if aggregate_heads:
                attention_np = attention_np.mean(axis=0)  # -> [H, W]
            else:
                attention_np = attention_np[0]
            
            # Add frame dimension if video
            if is_video:
                attention_np = attention_np[np.newaxis, :, :]
    
    logger.debug(f"prepare_attention_for_visualization: {attention_tensor.shape if hasattr(attention_tensor, 'shape') else 'array'} -> {attention_np.shape}")
    
    return attention_np


@staticmethod
def render_attention_frame(
    attention_frame: np.ndarray,
    colormap: Union[ColorMap, int] = ColorMap.JET,
    normalize: bool = True,
    target_size: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None
) -> np.ndarray:
    """
    Render a single attention frame as a colored image.
    
    This is a core helper used by both class methods and external callers.
    
    Args:
        attention_frame: 2D attention map [H, W]
        colormap: ColorMap enum or cv2 colormap constant
        normalize: Whether to normalize to [0, 255] range
        target_size: Optional (width, height) to resize to
        title: Optional title text to overlay
        
    Returns:
        RGB image as np.ndarray [H, W, 3]
    """
    logger = logging.getLogger(__name__)
    
    # Normalize if requested
    if normalize:
        att_min = attention_frame.min()
        att_max = attention_frame.max()
        if att_max > att_min:
            attention_normalized = ((attention_frame - att_min) / (att_max - att_min) * 255).astype(np.uint8)
        else:
            attention_normalized = np.zeros_like(attention_frame, dtype=np.uint8)
    else:
        attention_normalized = attention_frame.astype(np.uint8)
    
    # Get colormap
    if isinstance(colormap, ColorMap):
        if colormap == ColorMap.JET:
            cv_colormap = cv2.COLORMAP_JET
        elif colormap == ColorMap.HOT:
            cv_colormap = cv2.COLORMAP_HOT
        elif colormap == ColorMap.VIRIDIS:
            cv_colormap = cv2.COLORMAP_VIRIDIS
        elif colormap == ColorMap.PLASMA:
            cv_colormap = cv2.COLORMAP_PLASMA
        elif colormap == ColorMap.INFERNO:
            cv_colormap = cv2.COLORMAP_INFERNO
        elif colormap == ColorMap.MAGMA:
            cv_colormap = cv2.COLORMAP_MAGMA
        elif colormap == ColorMap.TURBO:
            cv_colormap = cv2.COLORMAP_TURBO
        elif colormap == ColorMap.COOL:
            cv_colormap = cv2.COLORMAP_COOL
        else:
            cv_colormap = cv2.COLORMAP_JET
    else:
        cv_colormap = colormap
    
    # Apply colormap
    colored_frame = cv2.applyColorMap(attention_normalized, cv_colormap)
    
    # Resize if requested
    if target_size is not None:
        width, height = target_size
        colored_frame = cv2.resize(colored_frame, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # Add title if provided
    if title:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
        text_x = (colored_frame.shape[1] - text_size[0]) // 2
        text_y = 30
        cv2.putText(colored_frame, title, (text_x, text_y), font, font_scale, 
                   (255, 255, 255), thickness, cv2.LINE_AA)
    
    return colored_frame


def visualize_attention_tensor(
    attention_tensor: Union[torch.Tensor, np.ndarray],
    spatial_shape: Union[Tuple[int, int], Tuple[int, int, int]],
    output_path: Union[str, Path],
    format: str = 'both',  # 'image', 'video', or 'both'
    colormap: Union[ColorMap, str] = ColorMap.JET,
    fps: int = 15,
    quality: int = 8,
    aggregate_heads: bool = True,
    target_size: Optional[Tuple[int, int]] = None,  # (width, height) - FULL VIDEO DIMENSIONS
    title: Optional[str] = None
) -> Dict[str, Path]:
    """
    Visualize attention tensor directly without requiring AttentionAnalyzer.
    
    This function follows the same upscaling logic as the main class:
    1. Reshape attention to LATENT dimensions (e.g., [13, 30, 53])
    2. Interpolate latent frames: (F-1)*4+1 (e.g., 13 → 49 frames)
    3. Upscale spatial dimensions: H*16, W*16 (e.g., 30×53 → 480×848)
    
    Example flow for WAN video:
        Input: [12, 20670] (heads × spatial_tokens)
        → Reshape: [13, 30, 53] (latent frames × height × width)
        → Interpolate: [49, 30, 53] (real frames at latent resolution)
        → Upscale: [49, 480, 848] (real frames at full resolution)
    
    Args:
        attention_tensor: Attention tensor (various formats supported)
        spatial_shape: LATENT spatial dimensions - (F, H, W) for video or (H, W) for image
        output_path: Where to save (without extension)
        format: 'image' (first frame), 'video' (all frames), or 'both'
        colormap: Color scheme (ColorMap enum or string)
        fps: Frame rate for video
        quality: Video quality (0-10, lower=higher quality)
        aggregate_heads: If True, average across heads
        target_size: (width, height) of FULL VIDEO (defaults to spatial_shape * 16)
        title: Optional title text
        
    Returns:
        Dictionary with paths to generated files
    """
    logger = logging.getLogger(__name__)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert colormap string to enum if needed
    if isinstance(colormap, str):
        try:
            colormap = ColorMap(colormap.lower())
        except ValueError:
            logger.warning(f"Unknown colormap '{colormap}', using JET")
            colormap = ColorMap.JET
    
    # Prepare attention using static helper - this gives us latent dimensions
    attention_latent = prepare_attention_for_visualization(
        attention_tensor=attention_tensor,
        spatial_shape=spatial_shape,
        aggregate_heads=aggregate_heads
    )
    
    logger.info(f"visualize_attention_tensor: latent shape={attention_latent.shape}")
    
    is_video = attention_latent.ndim == 3
    
    # Get latent dimensions from spatial_shape parameter
    if is_video:
        if len(spatial_shape) != 3:
            raise ValueError(f"Video format requires 3D spatial_shape (F, H, W), got {spatial_shape}")
        latent_frames, latent_h, latent_w = spatial_shape
        # Verify reshaped tensor matches expected latent dimensions
        if attention_latent.shape != spatial_shape:
            logger.warning(f"Attention shape mismatch: got {attention_latent.shape}, expected {spatial_shape}")
    else:
        if len(spatial_shape) != 2:
            raise ValueError(f"Image format requires 2D spatial_shape (H, W), got {spatial_shape}")
        latent_h, latent_w = spatial_shape
    
    # Default target size if not provided (upscale by 16x for spatial, 4x for temporal)
    if target_size is None:
        target_width = latent_w * 16
        target_height = latent_h * 16
        logger.info(f"No target_size provided, using 16x spatial upscaling: {latent_w}×{latent_h} → {target_width}×{target_height}")
    else:
        target_width, target_height = target_size
        logger.info(f"Using provided target_size: {target_width}×{target_height}")
    
    # For video: interpolate frames and upscale to full resolution
    # This matches the logic in _generate_attention_video_from_data()
    if is_video:
        latent_frames, latent_h, latent_w = attention_latent.shape
        
        # Calculate target frame count: (latent_frames - 1) * 4 + 1
        # Example: 13 latent frames → (13-1)*4+1 = 49 frames
        target_frames = (latent_frames - 1) * 4 + 1
        logger.info(f"Interpolating frames: {latent_frames} latent → {target_frames} real")
        
        # Create upscaled video frames
        attention_upscaled = np.zeros((target_frames, target_height, target_width), dtype=np.float32)
        
        # Interpolate and upscale each target frame
        for frame_idx in range(target_frames):
            # Map target frame to latent frame (with interpolation)
            if latent_frames > 1:
                latent_frame_idx = frame_idx * (latent_frames - 1) / (target_frames - 1) if target_frames > 1 else 0
                lower_idx = int(np.floor(latent_frame_idx))
                upper_idx = min(lower_idx + 1, latent_frames - 1)
                alpha = latent_frame_idx - lower_idx
                
                if lower_idx == upper_idx:
                    latent_attention = attention_latent[lower_idx]
                else:
                    latent_attention = (1 - alpha) * attention_latent[lower_idx] + alpha * attention_latent[upper_idx]
            else:
                latent_attention = attention_latent[0]
            
            # Upscale from latent to full resolution
            upscaled_frame = cv2.resize(
                latent_attention.astype(np.float32),
                (target_width, target_height),
                interpolation=cv2.INTER_LINEAR
            )
            attention_upscaled[frame_idx] = upscaled_frame
        
        attention_spatial = attention_upscaled
        logger.info(f"Upscaled to full resolution: {attention_spatial.shape}")
        
    else:
        # For single image: just upscale
        attention_spatial = cv2.resize(
            attention_latent.astype(np.float32),
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR
        )
        logger.info(f"Upscaled single frame: {attention_latent.shape} → {attention_spatial.shape}")
    
    results = {}
    
    # Generate image (first frame)
    if format in ['image', 'both']:
        if is_video:
            first_frame = attention_spatial[0]
        else:
            first_frame = attention_spatial
        
        # Render using static helper (no additional resize needed - already at target size)
        colored_frame = render_attention_frame(
            attention_frame=first_frame,
            colormap=colormap,
            normalize=True,
            target_size=None,  # Already at correct size
            title=title
        )
        
        image_path = output_path.with_suffix('.png')
        cv2.imwrite(str(image_path), colored_frame)
        results['image'] = image_path
        # logger.info(f"   ✅ Saved image: {image_path} ({target_height}×{target_width})")
    
    # Generate video (all frames)
    if format in ['video', 'both'] and is_video:
        num_frames = attention_spatial.shape[0]
        video_path = output_path.with_suffix('.mp4')
        
        # Create video writer
        with imageio.get_writer(str(video_path), fps=fps, codec='libx264', quality=quality) as writer:
            for frame_idx in range(num_frames):
                frame = attention_spatial[frame_idx]
                
                # Render using static helper (no additional resize needed)
                colored_frame = render_attention_frame(
                    attention_frame=frame,
                    colormap=colormap,
                    normalize=True,
                    target_size=None,  # Already at correct size
                    title=title if frame_idx == 0 else None  # Title only on first frame
                )
                
                # Convert BGR to RGB for imageio
                colored_frame_rgb = cv2.cvtColor(colored_frame, cv2.COLOR_BGR2RGB)
                writer.append_data(colored_frame_rgb)
        
        results['video'] = video_path
        logger.info(f"   ✅ Saved video: {video_path} ({num_frames} frames @ {target_height}×{target_width})")
    
    return results


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Use command line argument or find latest attention maps
    if len(sys.argv) > 1:
        attention_dir = sys.argv[1]
    else:
        latest_dirs = sorted(Path("outputs").glob("*attention*"), key=lambda x: x.stat().st_mtime, reverse=True)
        if latest_dirs:
            attention_dir = latest_dirs[0] / "attention_maps"
        else:
            print("No attention maps directory found. Please specify one as an argument.")
            sys.exit(1)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print(f"Testing AttentionVisualizer with: {attention_dir}")
    
    try:
        # Initialize
        analyzer = AttentionAnalyzer(attention_dir)
        visualizer = AttentionVisualizer(analyzer, "outputs/test_visualizations")
        
        # Get test data
        videos = analyzer.get_available_videos()
        if not videos:
            print("No videos found")
            sys.exit(1)
        
        video_id = videos[0]
        tokens = analyzer.get_available_tokens(video_id)
        if not tokens:
            print("No tokens found")
            sys.exit(1)
        
        token = tokens[0]
        
        print(f"Testing with video: {video_id}, token: {token}")
        
        # Test single frame visualization
        fig, metadata = visualizer.visualize_single_frame(
            video_id, token, step=0,
            save_path="outputs/test_visualizations/test_frame.png"
        )
        print(f"✅ Single frame visualization - shape: {metadata.attention_shape}")
        plt.close(fig)
        
        # Test video generation
        video_path = visualizer.generate_attention_video(
            video_id, token,
            output_filename="test_attention_video.mp4"
        )
        print(f"✅ Attention video generated: {video_path}")
        
        # Test temporal comparison
        comparison_fig = visualizer.generate_temporal_comparison(
            video_id, [token],
            save_path="outputs/test_visualizations/test_comparison.png"
        )
        print(f"✅ Temporal comparison generated")
        plt.close(comparison_fig)
        
        # Test aggregated attention
        aggregated = visualizer.generate_aggregated_attention_map(video_id, token)
        print(f"✅ Aggregated attention shape: {aggregated.shape}")
        
        print("\n🎉 AttentionVisualizer tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
