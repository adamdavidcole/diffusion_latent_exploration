#!/usr/bin/env python3
"""
Command-line script for generating attention visualization videos.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.attention_analyzer import AttentionAnalyzer
from src.visualization.attention_visualizer import (
    AttentionVisualizer, VideoConfig, OverlayConfig, 
    FusionMethod, ColorMap
)


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate attention visualization videos from stored attention maps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate attention video for a specific token
  python scripts/generate_attention_video.py \\
    --attention-dir outputs/experiment/attention_maps \\
    --video-id prompt_000_vid001 \\
    --token flower

  # Generate with overlay on source video
  python scripts/generate_attention_video.py \\
    --attention-dir outputs/experiment/attention_maps \\
    --video-id prompt_000_vid001 \\
    --token flower \\
    --source-video outputs/experiment/prompt_000/prompt_000_001.mp4 \\
    --overlay-alpha 0.6

  # List available videos and tokens
  python scripts/generate_attention_video.py \\
    --attention-dir outputs/experiment/attention_maps \\
    --list-available
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--attention-dir", 
        type=str, 
        required=True,
        help="Directory containing attention maps"
    )
    
    # Video/token selection
    parser.add_argument(
        "--video-id", 
        type=str,
        help="Video ID to process (e.g., prompt_000_vid001)"
    )
    
    parser.add_argument(
        "--token", 
        type=str,
        help="Token to visualize (e.g., flower, tree)"
    )
    
    parser.add_argument(
        "--list-available", 
        action="store_true",
        help="List available videos and tokens, then exit"
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="outputs/attention_videos",
        help="Output directory for generated videos (default: outputs/attention_videos)"
    )
    
    parser.add_argument(
        "--output-filename", 
        type=str,
        help="Custom filename for output video (default: auto-generated)"
    )
    
    # Source video overlay
    parser.add_argument(
        "--source-video", 
        type=str,
        help="Path to source video for overlay"
    )
    
    parser.add_argument(
        "--overlay-alpha", 
        type=float, 
        default=0.5,
        help="Overlay blend strength (0.0=only source, 1.0=only attention, default: 0.5)"
    )
    
    # Visualization settings
    parser.add_argument(
        "--colormap", 
        type=str, 
        choices=[c.value for c in ColorMap],
        default=ColorMap.JET.value,
        help="Color scheme for attention visualization (default: jet)"
    )
    
    parser.add_argument(
        "--fusion-method", 
        type=str,
        choices=[f.value for f in FusionMethod],
        default=FusionMethod.MEAN.value,
        help="Method for fusing attention dimensions (default: mean)"
    )
    
    parser.add_argument(
        "--normalize-per-frame", 
        action="store_true",
        help="Normalize attention values per frame instead of globally"
    )
    
    parser.add_argument(
        "--threshold", 
        type=float,
        help="Threshold for attention values (values below are set to 0)"
    )
    
    parser.add_argument(
        "--invert-values",
        action="store_true",
        default=True,
        help="Invert attention values (default: True for WAN models where low=high attention)"
    )
    
    parser.add_argument(
        "--no-invert-values",
        dest="invert_values",
        action="store_false",
        help="Don't invert attention values"
    )
    
    # Video generation settings
    parser.add_argument(
        "--fps", 
        type=int, 
        default=15,
        help="Frames per second for output video (default: 15)"
    )
    
    parser.add_argument(
        "--interpolation-factor", 
        type=int, 
        default=4,
        help="Frame interpolation factor between diffusion steps (default: 4)"
    )
    
    parser.add_argument(
        "--upscale-factor", 
        type=int, 
        default=16,
        help="Upscaling factor for latent resolution (default: 16)"
    )
    
    parser.add_argument(
        "--codec", 
        type=str, 
        default="libx264",
        help="Video codec (default: libx264)"
    )
    
    parser.add_argument(
        "--quality", 
        type=int, 
        default=8,
        help="Video quality (lower is better, default: 8)"
    )
    
    # Utility
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate attention directory
    attention_dir = Path(args.attention_dir)
    if not attention_dir.exists():
        logger.error(f"Attention directory does not exist: {attention_dir}")
        return 1
    
    try:
        # Initialize analyzer and visualizer
        analyzer = AttentionAnalyzer(attention_dir)
        visualizer = AttentionVisualizer(analyzer, args.output_dir)
        
        # List available videos and tokens if requested
        if args.list_available:
            videos = analyzer.get_available_videos()
            logger.info(f"Found {len(videos)} videos in {attention_dir}")
            
            for video_id in videos:
                tokens = analyzer.get_available_tokens(video_id)
                logger.info(f"  {video_id}: {len(tokens)} tokens")
                for token in tokens:
                    logger.info(f"    - {token}")
            
            return 0
        
        # Validate required arguments
        if not args.video_id:
            logger.error("--video-id is required (use --list-available to see options)")
            return 1
        
        if not args.token:
            logger.error("--token is required (use --list-available to see options)")
            return 1
        
        # Validate video and token exist
        available_videos = analyzer.get_available_videos()
        if args.video_id not in available_videos:
            logger.error(f"Video ID '{args.video_id}' not found. Available: {available_videos}")
            return 1
        
        available_tokens = analyzer.get_available_tokens(args.video_id)
        if args.token not in available_tokens:
            logger.error(f"Token '{args.token}' not found for video '{args.video_id}'. Available: {available_tokens}")
            return 1
        
        # Validate source video if provided
        source_video_path = None
        if args.source_video:
            source_video_path = Path(args.source_video)
            if not source_video_path.exists():
                logger.warning(f"Source video not found: {source_video_path}. Generating attention-only video.")
                source_video_path = None
            else:
                source_video_path = str(source_video_path)
        
        # Create configuration objects
        video_config = VideoConfig(
            fps=args.fps,
            codec=args.codec,
            quality=args.quality,
            interpolation_factor=args.interpolation_factor,
            upscale_factor=args.upscale_factor
        )
        
        overlay_config = OverlayConfig(
            alpha=args.overlay_alpha,
            colormap=ColorMap(args.colormap),
            normalize_per_frame=args.normalize_per_frame,
            threshold=args.threshold,
            invert_values=args.invert_values
        )
        
        fusion_method = FusionMethod(args.fusion_method)
        
        # Generate the video
        logger.info(f"Generating attention video for {args.video_id}/{args.token}")
        logger.info(f"Video config: fps={video_config.fps}, interpolation={video_config.interpolation_factor}, upscale={video_config.upscale_factor}")
        logger.info(f"Overlay config: alpha={overlay_config.alpha}, colormap={overlay_config.colormap.value}, normalize_per_frame={overlay_config.normalize_per_frame}")
        
        output_path = visualizer.generate_attention_video(
            video_id=args.video_id,
            token_word=args.token,
            video_config=video_config,
            overlay_config=overlay_config,
            fusion_method=fusion_method,
            output_filename=args.output_filename,
            source_video_path=source_video_path
        )
        
        logger.info(f"✅ Successfully generated attention video: {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Error generating attention video: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
