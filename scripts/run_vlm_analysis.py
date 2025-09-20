#!/usr/bin/env python3
"""
Command-line interface for VLM video analysis.
Supports both single video and batch processing using conversation-based analysis.
"""

import argparse
import json
import logging
import time
from pathlib import Path
import sys
import traceback
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.vlm_analysis.conversation_vlm_processor import ConversationVLMProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_videos_in_batch(batch_path: Path) -> List[Path]:
    """Find all video files in batch structure."""
    videos = []
    videos_dir = batch_path / "videos"
    
    if not videos_dir.exists():
        logger.warning(f"No videos directory found in {batch_path}")
        return videos
    
    # Look for video files in prompt directories
    for prompt_dir in videos_dir.iterdir():
        if prompt_dir.is_dir() and prompt_dir.name.startswith('prompt_'):
            for video_file in prompt_dir.glob('*.mp4'):
                videos.append(video_file)
                
    return videos

def is_vlm_analysis_complete(video_path: Path, batch_path: Path) -> bool:
    """Check if VLM analysis is already complete for a video."""
    # Determine the expected output path
    rel_path = video_path.relative_to(batch_path / "videos")
    output_path = batch_path / "vlm_analysis" / rel_path.with_suffix('.json')
    
    if not output_path.exists():
        return False
    
    try:
        # Check if the JSON file contains valid analysis results
        with open(output_path, 'r') as f:
            data = json.loads(f.read())
        
        # Check for success indicators
        # Look for either "ok": true (new format) or "analysis_success": true (old format)
        is_successful = (
            data.get("ok", False) or 
            data.get("analysis_success", False) or
            (data.get("stages_completed") and len(data.get("stages_completed", [])) > 0)
        )
        
        return is_successful
        
    except (json.JSONDecodeError, KeyError, IOError):
        # If we can't read/parse the file, assume incomplete
        return False

def process_batch_videos(processor: ConversationVLMProcessor, batch_path: Path, resume: bool = False) -> Dict[str, Any]:
    """Process all videos in a batch directory."""
    
    videos = find_videos_in_batch(batch_path)
    
    if not videos:
        logger.error(f"No videos found in batch: {batch_path}")
        return {"success": False, "error": "No videos found"}
    
    logger.info(f"Found {len(videos)} videos to process")
    
    # Filter out completed videos if resuming
    skipped_videos = []
    if resume:
        pending_videos = []
        for video_path in videos:
            if is_vlm_analysis_complete(video_path, batch_path):
                skipped_videos.append(video_path)
                logger.info(f"⏭️  Skipping already completed: {video_path.relative_to(batch_path / 'videos')}")
            else:
                pending_videos.append(video_path)
        videos = pending_videos
        
        if skipped_videos:
            logger.info(f"📋 Resume mode: Skipping {len(skipped_videos)} completed videos, processing {len(videos)} remaining")
        
        if not videos:
            logger.info("✅ All videos already completed - nothing to process")
            return {
                "success": True, 
                "total_videos": len(skipped_videos),
                "successful": len(skipped_videos),
                "failed": 0,
                "errors": [],
                "processed_videos": [],
                "skipped_videos": len(skipped_videos),
                "resumed": True
            }
    
    # Create output directory structure
    output_base = batch_path / "vlm_analysis"
    output_base.mkdir(exist_ok=True)
    
    results = {
        "success": True,
        "total_videos": len(videos) + len(skipped_videos),
        "successful": len(skipped_videos),  # Count already completed as successful
        "failed": 0,
        "errors": [],
        "processed_videos": [],
        "skipped_videos": len(skipped_videos),
        "resumed": resume
    }
    
    for video_path in videos:
        try:
            # Determine relative path structure
            rel_path = video_path.relative_to(batch_path / "videos")
            
            # Create output path with same structure but .json extension
            output_path = output_base / rel_path.with_suffix('.json')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Processing: {rel_path}")
            
            # Analyze video
            video_results = processor.analyze_video(
                video_path=video_path,
                output_path=output_path
            )
            
            if video_results.get("analysis_success", False):
                results["successful"] += 1
                logger.info(f"  ✅ Success: {output_path}")
            else:
                results["failed"] += 1
                error_msg = f"Analysis failed for {rel_path}"
                results["errors"].append(error_msg)
                logger.error(f"  ❌ {error_msg}")
                logger.error(traceback.format_exc())

            results["processed_videos"].append({
                "video_path": str(rel_path),
                "output_path": str(output_path.relative_to(batch_path)),
                "success": video_results.get("analysis_success", False),
                "stages_completed": video_results.get("stages_completed", []),
                "errors": video_results.get("errors", [])
            })
            
        except Exception as e:
            results["failed"] += 1
            error_msg = f"Exception processing {video_path.name}: {e}"
            results["errors"].append(error_msg)
            logger.error(f"  💥 {error_msg}")
            logger.error(traceback.format_exc())
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run VLM conversation-based analysis on videos")
    
    # Main arguments - one of these is required
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--video-path', 
        type=str,
        help='Path to single video file for analysis'
    )
    group.add_argument(
        '--batch-path',
        type=str, 
        help='Path to batch directory containing videos/ folder'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-path',
        type=str,
        help='Output path for single video analysis (required with --video-path)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--schema-path',
        type=str,
        default='src/analysis/vlm_analysis/vlm_analysis_schema_new.json',
        help='Path to VLM analysis schema JSON file (defaults to new flexible schema)'
    )
    
    parser.add_argument(
        '--model-id',
        type=str,
        default='Qwen/Qwen2.5-VL-32B-Instruct',
        help='Hugging Face model ID for VLM'
    )
    
    parser.add_argument(
        '--max-retries',
        type=int,
        default=2,
        help='Maximum retries for failed conversation stages'
    )
    
    parser.add_argument(
        '--disable-conversation-log',
        action='store_true',
        help='Disable conversation logging for debugging'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume batch processing by skipping videos that already have completed VLM analysis'
    )

    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Validate arguments
    schema_path = Path(args.schema_path)
    
    if not schema_path.exists():
        # Try fallback to old schema
        old_schema_path = Path('src/analysis/vlm_analysis/vlm_analysis_schema.json')
        if old_schema_path.exists():
            logger.warning(f"New schema not found at {schema_path}, using old schema: {old_schema_path}")
            schema_path = old_schema_path
        else:
            logger.error(f"Schema file not found: {schema_path}")
            return 1
    
    # Determine processing mode
    if args.video_path:
        # Single video mode
        video_path = Path(args.video_path)
        if not video_path.exists() or not video_path.is_file():
            logger.error(f"Video file not found: {video_path}")
            return 1
            
        if not args.output_path:
            logger.error("--output-path is required when using --video-path")
            return 1
            
        output_path = Path(args.output_path)
        
        logger.info(f"🎬 Single video mode")
        logger.info(f"  Video: {video_path}")
        logger.info(f"  Output: {output_path}")
        
    else:
        # Batch processing mode
        batch_path = Path(args.batch_path)
        if not batch_path.exists() or not batch_path.is_dir():
            logger.error(f"Batch directory not found: {batch_path}")
            return 1
            
        logger.info(f"📂 Batch processing mode")
        logger.info(f"  Batch: {batch_path}")
        logger.info(f"  Output will be created at: {batch_path}/vlm_analysis")
        
    try:
        # Initialize conversation processor
        logger.info(f"🔄 Initializing conversation-based VLM processor...")
        logger.info(f"  Schema: {schema_path}")
        logger.info(f"  Model: {args.model_id}")
        
        processor = ConversationVLMProcessor(
            schema_path=str(schema_path),
            max_retries=args.max_retries,
            enable_conversation_log=not args.disable_conversation_log
        )
        
        # Initialize VLM model
        logger.info("⚡ Loading VLM model...")
        processor.initialize()
        
        start_time = time.time()
        
        if args.video_path:
            # Single video processing
            logger.info(f"� Processing single video...")
            
            results = processor.analyze_video(
                video_path=video_path,
                output_path=output_path
            )
            
            elapsed_time = time.time() - start_time
            
            # Check results
            if results.get("analysis_success", False):
                logger.info(f"✅ Analysis completed successfully in {elapsed_time:.1f}s")
                logger.info(f"   Stages completed: {len(results.get('stages_completed', []))}")
                logger.info(f"   Output saved: {output_path}")
                
                # Show stage summary
                for stage in results.get("stages_completed", []):
                    logger.info(f"     ✓ {stage}")
                    
                return 0
            else:
                logger.error(f"❌ Analysis failed after {elapsed_time:.1f}s")
                errors = results.get("errors", [])
                if errors:
                    logger.error("Errors encountered:")
                    for error in errors:
                        logger.error(f"  - {error}")
                return 1
        else:
            # Batch processing
            logger.info(f"🚀 Starting batch processing...")
            
            batch_results = process_batch_videos(processor, batch_path, resume=args.resume)
            
            elapsed_time = time.time() - start_time
            
            if batch_results["success"]:
                logger.info(f"✅ Batch processing completed in {elapsed_time:.1f}s")
                logger.info(f"   Total videos: {batch_results['total_videos']}")
                logger.info(f"   Successful: {batch_results['successful']}")
                logger.info(f"   Failed: {batch_results['failed']}")
                
                # Add resume-specific logging
                if batch_results.get('resumed', False):
                    skipped = batch_results.get('skipped_videos', 0)
                    newly_processed = batch_results['successful'] - skipped
                    logger.info(f"   📋 Resume mode: {skipped} previously completed, {newly_processed} newly processed")
                
                if batch_results["failed"] > 0:
                    logger.warning("Some videos failed processing:")
                    for error in batch_results["errors"]:
                        logger.warning(f"  - {error}")
                    return 1
                else:
                    return 0
            else:
                logger.error(f"❌ Batch processing failed: {batch_results.get('error', 'Unknown error')}")
                return 1
                
    except Exception as e:
        logger.error(f"Analysis failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Cleanup
        try:
            processor.cleanup()
            logger.info("🧹 Cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")


if __name__ == "__main__":
    sys.exit(main())
