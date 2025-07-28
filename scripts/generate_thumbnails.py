#!/usr/bin/env python3
"""
Script to retroactively generate thumbnails for all existing videos.
Creates a .jpg thumbnail from the first frame of each .mp4 video.
"""
import os
import sys
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import List, Tuple
import argparse

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)

def generate_thumbnail(video_path: Path, thumbnail_path: Path, logger: logging.Logger) -> Tuple[bool, str]:
    """
    Generate a thumbnail from the first frame of a video using ffmpeg.
    
    Args:
        video_path: Path to the video file
        thumbnail_path: Path where thumbnail should be saved
        logger: Logger instance
    
    Returns:
        Tuple of (success, message)
    """
    try:
        # Check if video exists
        if not video_path.exists():
            return False, f"Video not found: {video_path}"
        
        # Skip if thumbnail already exists (unless forced)
        if thumbnail_path.exists():
            logger.debug(f"Thumbnail already exists: {thumbnail_path}")
            return True, "Already exists"
        
        # Create thumbnail directory if it doesn't exist
        thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use ffmpeg to extract first frame
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vf', 'select=eq(n\\,0)',  # Select first frame
            '-vframes', '1',            # Extract only 1 frame
            '-q:v', '2',               # High quality
            '-y',                      # Overwrite output file
            str(thumbnail_path)
        ]
        
        # Run ffmpeg command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )
        
        if result.returncode == 0:
            logger.debug(f"Generated thumbnail: {thumbnail_path}")
            return True, "Generated successfully"
        else:
            error_msg = result.stderr.strip()
            logger.error(f"FFmpeg error for {video_path}: {error_msg}")
            return False, f"FFmpeg error: {error_msg}"
            
    except subprocess.TimeoutExpired:
        return False, "Timeout generating thumbnail"
    except Exception as e:
        return False, f"Error: {str(e)}"

def find_videos(outputs_dir: Path, logger: logging.Logger) -> List[Path]:
    """
    Find all video files in the outputs directory.
    
    Args:
        outputs_dir: Path to outputs directory
        logger: Logger instance
    
    Returns:
        List of video file paths
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    videos = []
    
    for video_file in outputs_dir.rglob('*'):
        if video_file.suffix.lower() in video_extensions:
            videos.append(video_file)
    
    logger.info(f"Found {len(videos)} video files")
    return videos

def generate_thumbnails_batch(videos: List[Path], 
                            max_workers: int = 4, 
                            force: bool = False,
                            logger: logging.Logger = None) -> None:
    """
    Generate thumbnails for a batch of videos using ThreadPoolExecutor.
    
    Args:
        videos: List of video file paths
        max_workers: Maximum number of worker threads
        force: Force regeneration of existing thumbnails
        logger: Logger instance
    """
    if not videos:
        logger.info("No videos to process")
        return
    
    # Prepare tasks
    tasks = []
    for video_path in videos:
        thumbnail_path = video_path.with_suffix('.jpg')
        
        # Skip if thumbnail exists and not forcing
        if not force and thumbnail_path.exists():
            continue
            
        tasks.append((video_path, thumbnail_path))
    
    if not tasks:
        logger.info("All thumbnails already exist (use --force to regenerate)")
        return
    
    logger.info(f"Processing {len(tasks)} videos with {max_workers} workers...")
    
    # Track progress
    completed = 0
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_video = {
            executor.submit(generate_thumbnail, video_path, thumb_path, logger): video_path
            for video_path, thumb_path in tasks
        }
        
        # Process completed tasks
        for future in as_completed(future_to_video):
            video_path = future_to_video[future]
            completed += 1
            
            try:
                success, message = future.result()
                if success:
                    successful += 1
                    if logger.level <= logging.DEBUG:
                        logger.debug(f"[{completed}/{len(tasks)}] ✓ {video_path.name}: {message}")
                else:
                    failed += 1
                    logger.error(f"[{completed}/{len(tasks)}] ✗ {video_path.name}: {message}")
                    
                # Progress update every 10 videos or on completion
                if completed % 10 == 0 or completed == len(tasks):
                    logger.info(f"Progress: {completed}/{len(tasks)} ({successful} success, {failed} failed)")
                    
            except Exception as e:
                failed += 1
                logger.error(f"[{completed}/{len(tasks)}] ✗ {video_path.name}: Unexpected error: {e}")
    
    logger.info(f"Completed! {successful} successful, {failed} failed out of {len(tasks)} total")

def check_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate thumbnails for all videos in outputs directory")
    parser.add_argument('--outputs-dir', '-o', 
                       default='outputs',
                       help='Path to outputs directory (default: outputs)')
    parser.add_argument('--max-workers', '-w', 
                       type=int, default=4,
                       help='Maximum number of worker threads (default: 4)')
    parser.add_argument('--force', '-f', 
                       action='store_true',
                       help='Force regeneration of existing thumbnails')
    parser.add_argument('--verbose', '-v', 
                       action='store_true',
                       help='Verbose output')
    parser.add_argument('--dry-run', 
                       action='store_true',
                       help='Show what would be done without actually doing it')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Check if ffmpeg is available
    if not check_ffmpeg():
        logger.error("ffmpeg is not available. Please install ffmpeg:")
        logger.error("  Ubuntu/Debian: sudo apt install ffmpeg")
        logger.error("  macOS: brew install ffmpeg")
        logger.error("  Windows: Download from https://ffmpeg.org/download.html")
        sys.exit(1)
    
    # Check outputs directory
    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists():
        logger.error(f"Outputs directory not found: {outputs_dir}")
        sys.exit(1)
    
    logger.info(f"Scanning for videos in: {outputs_dir.absolute()}")
    
    # Find all videos
    videos = find_videos(outputs_dir, logger)
    
    if args.dry_run:
        logger.info("DRY RUN MODE - no thumbnails will be generated")
        count = 0
        for video_path in videos:
            thumbnail_path = video_path.with_suffix('.jpg')
            if args.force or not thumbnail_path.exists():
                logger.info(f"Would generate: {thumbnail_path}")
                count += 1
        logger.info(f"Would generate {count} thumbnails")
        return
    
    # Generate thumbnails
    generate_thumbnails_batch(
        videos=videos,
        max_workers=args.max_workers,
        force=args.force,
        logger=logger
    )

if __name__ == "__main__":
    main()
