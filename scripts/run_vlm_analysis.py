#!/usr/bin/env python3
"""
Command-line interface for VLM video analysis.
Supports both single video and batch processing.
"""

import argparse
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.vlm_analysis.vlm_batch_processor import VLMBatchProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run VLM analysis on video batches or single videos")
    
    # Required arguments
    parser.add_argument(
        '--batch-path', 
        type=str, 
        required=True,
        help='Path to batch directory (for batch processing) or video file (for single video)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--schema-path',
        type=str,
        default='src/analysis/vlm_analysis/vlm_analysis_schema.json',
        help='Path to VLM analysis schema JSON file'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to local VLM model (uses default if not specified)'
    )
    
    parser.add_argument(
        '--prompt-groups',
        type=str,
        nargs='*',
        help='Specific prompt groups to process (defaults to all)'
    )
    
    parser.add_argument(
        '--single-video',
        action='store_true',
        help='Process a single video file instead of a batch'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        help='Output path for single video analysis (required with --single-video)'
    )
    
    parser.add_argument(
        '--max-retries',
        type=int,
        default=1,
        help='Maximum retries for failed VLM responses'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Validate arguments
    batch_path = Path(args.batch_path)
    schema_path = Path(args.schema_path)
    
    if not schema_path.exists():
        logger.error(f"Schema file not found: {schema_path}")
        return 1
        
    if args.single_video:
        if not batch_path.exists() or not batch_path.is_file():
            logger.error(f"Video file not found: {batch_path}")
            return 1
        if not args.output_path:
            logger.error("--output-path is required when using --single-video")
            return 1
    else:
        if not batch_path.exists() or not batch_path.is_dir():
            logger.error(f"Batch directory not found: {batch_path}")
            return 1
            
    try:
        # Initialize processor
        processor = VLMBatchProcessor(
            schema_path=str(schema_path),
            max_retries=args.max_retries
        )
        
        # Initialize VLM model
        logger.info("Initializing VLM model...")
        processor.initialize()
        
        if args.single_video:
            # Single video processing
            logger.info(f"Processing single video: {batch_path}")
            
            output_path = Path(args.output_path)
            result = processor.analyze_single_video(
                video_path=batch_path,
                output_path=output_path,
                prompt_text=""
            )
            
            if result.get("ok", False):
                logger.info("✅ Single video analysis completed successfully")
                return 0
            else:
                logger.error("❌ Single video analysis failed")
                logger.error(f"Errors: {result.get('errors', [])}")
                return 1
                
        else:
            # Batch processing
            logger.info(f"Processing batch: {batch_path}")
            
            result = processor.process_batch(
                batch_path=batch_path,
                prompt_groups=args.prompt_groups
            )
            
            summary = result["summary"]
            logger.info(f"✅ Batch processing completed:")
            logger.info(f"  Total videos: {summary['total_videos']}")
            logger.info(f"  Successful: {summary['successful_analyses']}")
            logger.info(f"  Failed: {summary['failed_analyses']}")
            
            if summary["failed_analyses"] > 0:
                logger.warning(f"Some analyses failed. Check logs and output files.")
                return 1
            else:
                return 0
                
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1
        
    finally:
        # Cleanup
        try:
            processor.cleanup()
        except:
            pass

if __name__ == "__main__":
    sys.exit(main())
