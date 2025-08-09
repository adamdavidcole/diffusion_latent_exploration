#!/usr/bin/env python3
"""
Batch Image Grid Generator

This script creates a comprehensive image grid visualization for a batch directory,
showing all video thumbnails organized by prompt groups (rows) and seed variations (columns).
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.visualization.batch_grid import create_batch_image_grid


def setup_logging(verbose: bool = False):
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a comprehensive image grid visualization for a batch directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/generate_batch_grid.py outputs/my_batch_20250809_123456
  
  # Custom output location and dimensions
  python scripts/generate_batch_grid.py outputs/my_batch_20250809_123456 \\
    --output-path visualizations/my_batch_grid.png \\
    --max-width 2560 --max-height 1440
  
  # Custom thumbnail size
  python scripts/generate_batch_grid.py outputs/my_batch_20250809_123456 \\
    --thumbnail-width 200 --thumbnail-height 120
        """
    )
    
    parser.add_argument(
        "batch_path",
        type=str,
        help="Path to the batch directory containing videos/ and configs/ subdirectories"
    )
    
    parser.add_argument(
        "--output-path", "-o",
        type=str,
        default=None,
        help="Output path for the generated image (default: <batch_path>/<batch_name>_grid.png)"
    )
    
    parser.add_argument(
        "--max-width",
        type=int,
        default=1920,
        help="Maximum width of the output image in pixels (default: 1920)"
    )
    
    parser.add_argument(
        "--max-height",
        type=int,
        default=1080,
        help="Maximum height of the output image in pixels (default: 1080)"
    )
    
    parser.add_argument(
        "--thumbnail-width",
        type=int,
        default=None,
        help="Fixed width for thumbnails in pixels (if not set, will be calculated automatically)"
    )
    
    parser.add_argument(
        "--thumbnail-height",
        type=int,
        default=None,
        help="Fixed height for thumbnails in pixels (if not set, will be calculated automatically)"
    )
    
    parser.add_argument(
        "--margin",
        type=int,
        default=10,
        help="Margin between grid cells in pixels (default: 10)"
    )
    
    parser.add_argument(
        "--outer-padding",
        type=int,
        default=20,
        help="Padding around the entire content area in pixels (default: 20)"
    )
    
    parser.add_argument(
        "--header-height",
        type=int,
        default=120,
        help="Height reserved for header text in pixels (default: 120)"
    )
    
    parser.add_argument(
        "--row-label-width",
        type=int,
        default=200,
        help="Width reserved for row labels in pixels (default: 200)"
    )
    
    parser.add_argument(
        "--col-label-height",
        type=int,
        default=30,
        help="Height reserved for column labels in pixels (default: 30)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def validate_batch_path(batch_path: str) -> Path:
    """Validate that the batch path exists and has the required structure."""
    batch_dir = Path(batch_path)
    
    if not batch_dir.exists():
        raise ValueError(f"Batch directory does not exist: {batch_path}")
    
    if not batch_dir.is_dir():
        raise ValueError(f"Batch path is not a directory: {batch_path}")
    
    videos_dir = batch_dir / "videos"
    if not videos_dir.exists():
        raise ValueError(f"Videos directory not found: {videos_dir}")
    
    configs_dir = batch_dir / "configs"
    if not configs_dir.exists():
        raise ValueError(f"Configs directory not found: {configs_dir}")
    
    return batch_dir


def main():
    """Main entry point."""
    try:
        args = parse_arguments()
        setup_logging(args.verbose)
        
        logger = logging.getLogger(__name__)
        logger.info("🎨 Starting batch image grid generation")
        logger.info("=" * 60)
        
        # Validate batch path
        batch_dir = validate_batch_path(args.batch_path)
        logger.info(f"📁 Batch directory: {batch_dir}")
        
        # Prepare thumbnail size
        thumbnail_size = None
        if args.thumbnail_width is not None and args.thumbnail_height is not None:
            thumbnail_size = (args.thumbnail_width, args.thumbnail_height)
            logger.info(f"🖼️ Using fixed thumbnail size: {thumbnail_size}")
        elif args.thumbnail_width is not None or args.thumbnail_height is not None:
            logger.warning("⚠️ Both thumbnail-width and thumbnail-height must be specified together")
            logger.warning("   Using automatic thumbnail sizing instead")
        
        # Generate the grid
        output_path = create_batch_image_grid(
            batch_path=str(batch_dir),
            output_path=args.output_path,
            max_width=args.max_width,
            max_height=args.max_height,
            thumbnail_size=thumbnail_size,
            margin=args.margin,
            outer_padding=args.outer_padding,
            header_height=args.header_height,
            row_label_width=args.row_label_width,
            col_label_height=args.col_label_height
        )
        
        logger.info("=" * 60)
        logger.info(f"✅ Successfully generated batch grid: {output_path}")
        
    except KeyboardInterrupt:
        print("\\n⏹️  Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"❌ Error generating batch grid: {e}")
        if args.verbose:
            logger.exception("Full error traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
