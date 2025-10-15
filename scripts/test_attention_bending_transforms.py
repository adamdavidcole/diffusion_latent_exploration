#!/usr/bin/env python3
"""
Test script for attention bending transformations.

This script loads raw attention maps and applies transformations directly,
then visualizes the results to verify they're working correctly.

Usage:
    python scripts/test_attention_bending_transforms.py <attention_map_file> [--mode scale] [--scale-factor 5.0]
"""

import sys
import argparse
import gzip
import numpy as np
import torch
from pathlib import Path
import logging

from typing import Dict, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.attention_bending import AttentionBender, BendingConfig, BendingMode

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_attention_map(file_path: Path) -> np.ndarray:
    """Load a compressed attention map."""
    logger.info(f"Loading attention map from {file_path}")
    with gzip.open(file_path, 'rb') as f:
        data = np.load(f)
    logger.info(f"Loaded shape: {data.shape}, dtype: {data.dtype}")
    logger.info(f"Stats: min={data.min():.6f}, max={data.max():.6f}, mean={data.mean():.6f}, std={data.std():.6f}")
    return data


def save_attention_map(data: np.ndarray, output_path: Path):
    """Save attention map as compressed numpy."""
    logger.info(f"Saving to {output_path}")
    with gzip.open(output_path, 'wb') as f:
        np.save(f, data)


def visualize_comparison(original: np.ndarray, transformed: np.ndarray, output_dir: Path, prefix: str = "comparison", metadata: Optional[Dict] = None):
    """Create comparison visualization of original vs transformed attention."""
    import matplotlib.pyplot as plt
    
    # Handle different input shapes - convert both to [H, W]
    def extract_2d(data, is_transformed=False):
        """Extract a 2D spatial map from various input formats."""
        # For video data, extract middle frame
        if metadata is not None and 'video_frames' in metadata:
            frames = metadata['video_frames']
            height = metadata['video_height']
            width = metadata['video_width']
            
            F = (frames - 1) // 4 + 1
            H = height // 16
            W = width // 16
            
            if data.ndim == 4:
                if is_transformed and data.shape == (1, F, H, W):
                    # Transformed video format [B, F, H, W]
                    frame_idx = F // 2
                    return data[0, frame_idx, :, :]  # [H, W]
                else:
                    # Original storage format [B, heads, spatial, tokens]
                    spatial_data = data[0, 0, :, 0]  # [spatial]
                    spatial_3d = spatial_data.reshape(F, H, W)
                    frame_idx = F // 2
                    return spatial_3d[frame_idx, :, :]  # [H, W]
        
        # Generic handling for non-video data
        if data.ndim == 4:  # [B, heads, spatial, tokens]
            # Extract first batch, first head, all spatial, first token
            data = data[0, 0, :, 0]
            # Reshape spatial to 2D
            spatial = data.shape[0]
            H = int(np.sqrt(spatial))
            W = spatial // H
            if H * W != spatial:
                for h_try in range(int(np.sqrt(spatial)), 0, -1):
                    if spatial % h_try == 0:
                        H = h_try
                        W = spatial // h_try
                        break
            data = data.reshape(H, W)
        elif data.ndim == 3:  # [heads, H, W]
            data = data[0]  # First head
        elif data.ndim == 1:  # [spatial]
            spatial = data.shape[0]
            H = int(np.sqrt(spatial))
            W = spatial // H
            if H * W != spatial:
                for h_try in range(int(np.sqrt(spatial)), 0, -1):
                    if spatial % h_try == 0:
                        H = h_try
                        W = spatial // h_try
                        break
            data = data.reshape(H, W)
        return data
    
    original_vis = extract_2d(original, is_transformed=False)
    transformed_vis = extract_2d(transformed, is_transformed=True)
    
    logger.info(f"Visualization: original shape {original_vis.shape}, transformed shape {transformed_vis.shape}")
    
    H, W = original_vis.shape
    aspect_ratio = W / H
    
    # Create figure with proper aspect ratio for each subplot
    fig_height = 5
    fig_width = fig_height * aspect_ratio * 3 + 2  # 3 subplots + space for colorbars
    
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))
    
    # Original
    im0 = axes[0].imshow(original_vis, cmap='jet', aspect='auto')
    axes[0].set_title(f'Original\n{H}x{W}', fontsize=12, fontweight='bold')
    axes[0].set_xlabel(f'Width ({W}px)')
    axes[0].set_ylabel(f'Height ({H}px)')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Transformed
    im1 = axes[1].imshow(transformed_vis, cmap='jet', aspect='auto')
    axes[1].set_title(f'Transformed\n{H}x{W}', fontsize=12, fontweight='bold')
    axes[1].set_xlabel(f'Width ({W}px)')
    axes[1].set_ylabel(f'Height ({H}px)')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Difference (use symmetric colormap centered at 0)
    diff = transformed_vis - original_vis
    max_abs_diff = max(abs(diff.min()), abs(diff.max()))
    if max_abs_diff > 0:
        im2 = axes[2].imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-max_abs_diff, vmax=max_abs_diff)
    else:
        im2 = axes[2].imshow(diff, cmap='RdBu_r', aspect='auto')
    axes[2].set_title(f'Difference\n(Transformed - Original)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel(f'Width ({W}px)')
    axes[2].set_ylabel(f'Height ({H}px)')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    comparison_path = output_dir / f"{prefix}_comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved comparison to {comparison_path}")
    plt.close()
    
    # Also save a single large visualization of just the transformed result
    fig_single = plt.figure(figsize=(fig_height * aspect_ratio, fig_height))
    plt.imshow(transformed_vis, cmap='jet', aspect='auto')
    plt.title(f'Transformed Attention Map\n{H}x{W}', fontsize=14, fontweight='bold')
    plt.xlabel(f'Width ({W}px)')
    plt.ylabel(f'Height ({H}px)')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    single_path = output_dir / f"{prefix}_transformed_only.png"
    plt.savefig(single_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved single transformed view to {single_path}")
    plt.close()


def test_transform(attention_map: np.ndarray, mode: str, metadata: Optional[Dict] = None, **kwargs) -> np.ndarray:
    """
    Test a transformation on attention map.
    
    Args:
        attention_map: Attention data - can be [B, heads, spatial, tokens] or [heads, H, W] or [H, W]
        mode: Transformation mode
        metadata: Metadata dict with video dimensions
        **kwargs: Parameters for the transformation
        
    Returns:
        Transformed attention map
    """
    # Convert to torch
    attention_torch = torch.from_numpy(attention_map).float()
    
    logger.info(f"Input shape from file: {attention_torch.shape}")
    
    # Handle different storage formats
    if attention_torch.ndim == 4:  # [B, heads, spatial, tokens] - need to extract single token
        # Take first batch, first token (or average across tokens)
        attention_torch = attention_torch[0, :, :, 0]  # [heads, spatial]
        logger.info(f"Extracted from 4D to: {attention_torch.shape}")
    
    # Now we should have [heads, spatial] - need to reshape to [heads, F, H, W] for video
    if attention_torch.ndim == 2:  # [heads, spatial]
        heads, spatial = attention_torch.shape
        
        # Use metadata to get correct video dimensions
        if metadata and 'video_frames' in metadata and 'video_height' in metadata and 'video_width' in metadata:
            target_frames = metadata['video_frames']
            target_height = metadata['video_height']
            target_width = metadata['video_width']
            
            # Calculate latent dimensions (VAE downsamples 16x spatial, 4x temporal)
            F = (target_frames - 1) // 4 + 1  # Latent frames
            H = target_height // 16  # Latent height
            W = target_width // 16  # Latent width
            
            expected_spatial = F * H * W
            logger.info(f"Metadata: video={target_frames}×{target_height}×{target_width} → latent={F}×{H}×{W} = {expected_spatial}")
            
            if spatial == expected_spatial:
                # Reshape to [heads, F, H, W]
                attention_torch = attention_torch.reshape(heads, F, H, W)
                logger.info(f"✅ Reshaped to [heads={heads}, F={F}, H={H}, W={W}] using metadata")
            else:
                logger.warning(f"⚠️  Spatial size mismatch: got {spatial}, expected {expected_spatial}")
                # Fall back to 2D spatial
                H = int(np.sqrt(spatial))
                W = spatial // H
                attention_torch = attention_torch.reshape(heads, H, W)
                logger.warning(f"⚠️  Falling back to 2D: [heads={heads}, H={H}, W={W}]")
        else:
            # No metadata - try to infer
            logger.warning("No metadata provided, inferring spatial dimensions")
            H = int(np.sqrt(spatial))
            W = spatial // H
            if H * W != spatial:
                for h_try in range(int(np.sqrt(spatial)), 0, -1):
                    if spatial % h_try == 0:
                        H = h_try
                        W = spatial // h_try
                        break
            attention_torch = attention_torch.reshape(heads, H, W)
            logger.info(f"Reshaped from [heads={heads}, spatial={spatial}] to [heads={heads}, H={H}, W={W}]")
    elif attention_torch.ndim == 1:  # Just spatial
        spatial = attention_torch.shape[0]
        H = int(np.sqrt(spatial))
        W = spatial // H
        if H * W != spatial:
            for h_try in range(int(np.sqrt(spatial)), 0, -1):
                if spatial % h_try == 0:
                    H = h_try
                    W = spatial // h_try
                    break
        attention_torch = attention_torch.reshape(1, H, W)
        logger.info(f"Reshaped from [spatial={spatial}] to [heads=1, H={H}, W={W}]")
    
    # Now attention_torch is either [heads, H, W] or [heads, F, H, W]
    logger.info(f"Final input shape for transformation: {attention_torch.shape}")
    
    # For video (4D), we need to apply transformation per frame
    if attention_torch.ndim == 4:
        heads, F, H, W = attention_torch.shape
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {mode.upper()} transformation on VIDEO")
        logger.info(f"Input shape: [heads={heads}, frames={F}, H={H}, W={W}]")
        logger.info(f"Input stats: min={attention_torch.min():.6f}, max={attention_torch.max():.6f}, mean={attention_torch.mean():.6f}")
        
        # Create bending config
        bending_mode = BendingMode(mode)
        # Extract renormalize from kwargs to avoid duplicate
        renormalize = kwargs.pop('renormalize', False)
        config = BendingConfig(
            token="test",
            mode=bending_mode,
            strength=1.0,
            renormalize=renormalize,
            **kwargs
        )
        
        # Create bender
        bender = AttentionBender(
            bending_configs=[config],
            device='cpu'
        )
        
        # Apply transformation frame by frame
        transformed_frames = []
        for frame_idx in range(F):
            frame = attention_torch[:, frame_idx, :, :]  # [heads, H, W]
            
            # Apply transformation
            if mode == 'amplify':
                transformed_frame = bender._apply_amplify(frame, config)
            elif mode == 'scale':
                transformed_frame = bender._apply_scale(frame, config)
            elif mode == 'rotate':
                transformed_frame = bender._apply_rotation(frame, config)
            elif mode == 'translate':
                transformed_frame = bender._apply_translation(frame, config)
            elif mode == 'flip':
                transformed_frame = bender._apply_flip(frame, config)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            transformed_frames.append(transformed_frame)
        
        transformed = torch.stack(transformed_frames, dim=1)  # [heads, F, H, W]
        logger.info(f"Transformed each of {F} frames")
        
    else:  # 3D: [heads, H, W]
        heads, H, W = attention_torch.shape
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {mode.upper()} transformation on 2D SPATIAL")
        logger.info(f"Input shape: [heads={heads}, H={H}, W={W}]")
        logger.info(f"Input stats: min={attention_torch.min():.6f}, max={attention_torch.max():.6f}, mean={attention_torch.mean():.6f}")
        
        # Create bending config
        bending_mode = BendingMode(mode)
        # Extract renormalize from kwargs to avoid duplicate
        renormalize = kwargs.pop('renormalize', False)
        config = BendingConfig(
            token="test",
            mode=bending_mode,
            strength=1.0,
            renormalize=renormalize,
            **kwargs
        )
        
        # Create bender
        bender = AttentionBender(
            bending_configs=[config],
            device='cpu'
        )
        
        # Apply transformation directly
        if mode == 'amplify':
            transformed = bender._apply_amplify(attention_torch, config)
        elif mode == 'scale':
            transformed = bender._apply_scale(attention_torch, config)
        elif mode == 'rotate':
            transformed = bender._apply_rotation(attention_torch, config)
        elif mode == 'translate':
            transformed = bender._apply_translation(attention_torch, config)
        elif mode == 'flip':
            transformed = bender._apply_flip(attention_torch, config)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    logger.info(f"Output shape: {transformed.shape}")
    logger.info(f"Output stats: min={transformed.min():.6f}, max={transformed.max():.6f}, mean={transformed.mean():.6f}")
    logger.info(f"Change: Δmean={transformed.mean() - attention_torch.mean():.6f}, Δmax={transformed.max() - attention_torch.max():.6f}")
    logger.info(f"{'='*60}\n")
    
    return transformed.numpy()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {mode.upper()} transformation")
    logger.info(f"Input shape: {attention_torch.shape} [heads={batch_heads}, H={H}, W={W}]")
    logger.info(f"Spatial dimensions: {H} rows × {W} cols (aspect ratio: {W/H:.2f})")
    logger.info(f"Input stats: min={attention_torch.min():.6f}, max={attention_torch.max():.6f}, mean={attention_torch.mean():.6f}")
    
    # Create bending config
    bending_mode = BendingMode(mode)
    config = BendingConfig(
        token="test",
        mode=bending_mode,
        strength=1.0,
        renormalize=False,  # Don't renormalize for testing
        **kwargs
    )
    
    # Create bender
    bender = AttentionBender(
        bending_configs=[config],
        device='cpu'
    )
    
    # Apply transformation directly
    if mode == 'amplify':
        transformed = bender._apply_amplify(attention_torch, config)
    elif mode == 'scale':
        transformed = bender._apply_scale(attention_torch, config)
    elif mode == 'rotate':
        transformed = bender._apply_rotation(attention_torch, config)
    elif mode == 'translate':
        transformed = bender._apply_translation(attention_torch, config)
    elif mode == 'flip':
        transformed = bender._apply_flip(attention_torch, config)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    logger.info(f"Output shape: {transformed.shape} [heads={transformed.shape[0]}, H={transformed.shape[1]}, W={transformed.shape[2]}]")
    logger.info(f"✅ Canvas size preserved: {transformed.shape[1]} rows × {transformed.shape[2]} cols")
    logger.info(f"Output stats: min={transformed.min():.6f}, max={transformed.max():.6f}, mean={transformed.mean():.6f}")
    logger.info(f"Change: Δmean={transformed.mean() - attention_torch.mean():.6f}, Δmax={transformed.max() - attention_torch.max():.6f}")
    logger.info(f"{'='*60}\n")
    
    return transformed.numpy()


def main():
    parser = argparse.ArgumentParser(description="Test attention bending transformations")
    parser.add_argument("attention_file", type=Path, help="Path to attention map .npy.gz file")
    parser.add_argument("--mode", choices=['amplify', 'scale', 'rotate', 'translate', 'flip'], 
                       default='scale', help="Transformation mode")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: same as input)")
    
    # Transformation parameters
    parser.add_argument("--amplify-factor", type=float, default=10.0, help="Amplify factor")
    parser.add_argument("--scale-factor", type=float, default=5.0, help="Scale factor (zoom)")
    parser.add_argument("--angle", type=float, default=45.0, help="Rotation angle in degrees")
    parser.add_argument("--translate-x", type=float, default=0.3, help="Translation in x (-1 to 1)")
    parser.add_argument("--translate-y", type=float, default=0.0, help="Translation in y (-1 to 1)")
    parser.add_argument("--flip-horizontal", action='store_true', help="Flip horizontally")
    parser.add_argument("--flip-vertical", action='store_true', help="Flip vertically")
    parser.add_argument("--padding-mode", type=str, default='border', 
                       choices=['zeros', 'border', 'reflection'],
                       help="Padding mode for spatial transformations (default: border)")
    parser.add_argument("--renormalize", action='store_true', 
                       help="Renormalize attention after transformation (default: False)")
    
    parser.add_argument("--visualize", action='store_true', help="Create comparison visualization")
    parser.add_argument("--save-transformed", action='store_true', help="Save transformed attention map")
    parser.add_argument("--frame-to-show", type=int, default=0, help="Which frame to visualize (for video data)")
    
    args = parser.parse_args()
    
    # Load attention map
    attention_data = load_attention_map(args.attention_file)
    
    # Load metadata if available
    metadata_file = args.attention_file.parent / f"{args.attention_file.stem.replace('.npy', '')}_metadata.json"
    metadata = None
    if metadata_file.exists():
        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata from {metadata_file}")
        logger.info(f"  Video: {metadata.get('video_frames')}×{metadata.get('video_height')}×{metadata.get('video_width')}")
    else:
        logger.warning(f"No metadata file found at {metadata_file}")
    
    # Prepare kwargs based on mode
    kwargs = {}
    if args.mode == 'amplify':
        kwargs['amplify_factor'] = args.amplify_factor
        kwargs['renormalize'] = args.renormalize
    elif args.mode == 'scale':
        kwargs['scale_factor'] = args.scale_factor
        kwargs['padding_mode'] = args.padding_mode
        kwargs['renormalize'] = args.renormalize
    elif args.mode == 'rotate':
        kwargs['angle'] = args.angle
        kwargs['padding_mode'] = args.padding_mode
        kwargs['renormalize'] = args.renormalize
    elif args.mode == 'translate':
        kwargs['translate_x'] = args.translate_x
        kwargs['translate_y'] = args.translate_y
        kwargs['padding_mode'] = args.padding_mode
        kwargs['renormalize'] = args.renormalize
    elif args.mode == 'flip':
        kwargs['flip_horizontal'] = args.flip_horizontal
        kwargs['flip_vertical'] = args.flip_vertical
        kwargs['renormalize'] = args.renormalize
    
    # Apply transformation
    transformed_data = test_transform(attention_data, args.mode, metadata=metadata, **kwargs)
    
    # Output directory
    output_dir = args.output_dir if args.output_dir else args.attention_file.parent / "test_transforms"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save transformed if requested
    if args.save_transformed:
        output_file = output_dir / f"{args.attention_file.stem}_{args.mode}_transformed.npy.gz"
        save_attention_map(transformed_data, output_file)
    
    # Visualize if requested
    if args.visualize:
        visualize_comparison(attention_data, transformed_data, output_dir, prefix=args.mode, metadata=metadata)
        # Note: visualize_comparison already extracts and displays a single frame for video data
        # No need for additional frame visualization
    
    logger.info(f"\n✅ Test complete! Mode: {args.mode}")
    logger.info(f"   Original: mean={attention_data.mean():.6f}, max={attention_data.max():.6f}")
    logger.info(f"   Transformed: mean={transformed_data.mean():.6f}, max={transformed_data.max():.6f}")
    logger.info(f"   Output directory: {output_dir}")


if __name__ == "__main__":
    main()
