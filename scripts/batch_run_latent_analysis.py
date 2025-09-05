#!/usr/bin/env python3
"""
Batch Latent Trajectory Analysis Runner

This script runs latent trajectory analysis on multiple batch folders either from:
1. A provided list of batch paths
2. Auto-discovery by searching through outputs/ for experiment folders missing analysis

Usage examples:
  # Run on specific batches
  python batch_run_latent_analysis.py --batch-paths path1,path2,path3
  
  # Auto-discover missing analyses in outputs/
  python batch_run_latent_analysis.py --auto-discover
  
  # Auto-discover with custom search directory
  python batch_run_latent_analysis.py --auto-discover --search-dir /path/to/experiments
  
  # Run with visualizations enabled
  python batch_run_latent_analysis.py --auto-discover --no-skip-visualization
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple
import concurrent.futures
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def setup_logging(log_file=None, verbose=False):
    """Configure logging for batch analysis."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def is_complete_experiment_folder(folder_path: Path) -> bool:
    """
    Check if a folder is a complete experiment with videos/ and latents/ subdirectories
    that contain matching prompt folders with actual data.
    
    Args:
        folder_path: Path to check
        
    Returns:
        True if folder has both videos/ and latents/ with matching prompt folders containing data
    """
    if not (folder_path.is_dir() and 
            (folder_path / "videos").exists() and 
            (folder_path / "latents").exists()):
        return False
    
    videos_dir = folder_path / "videos"
    latents_dir = folder_path / "latents"
    
    # Get prompt folders from both directories
    video_prompt_folders = set()
    latent_prompt_folders = set()
    
    # Check videos directory for prompt_* folders
    for item in videos_dir.iterdir():
        if item.is_dir() and item.name.startswith("prompt_"):
            video_prompt_folders.add(item.name)
    
    # Check latents directory for prompt_* folders
    for item in latents_dir.iterdir():
        if item.is_dir() and item.name.startswith("prompt_"):
            latent_prompt_folders.add(item.name)
    
    # Must have at least one prompt folder and they must match
    if not video_prompt_folders or video_prompt_folders != latent_prompt_folders:
        return False
    
    # Check that each latent prompt folder has vid_* subdirectories with .npz files
    for prompt_folder in latent_prompt_folders:
        prompt_latents_path = latents_dir / prompt_folder
        
        # Look for vid_* folders
        vid_folders = [item for item in prompt_latents_path.iterdir() 
                      if item.is_dir() and item.name.startswith("vid_")]
        
        if not vid_folders:
            return False
        
        # Check that at least one vid folder contains .npz files
        has_npz_data = False
        for vid_folder in vid_folders:
            npz_files = list(vid_folder.glob("*.npz"))
            if npz_files:
                has_npz_data = True
                break
        
        if not has_npz_data:
            return False
    
    return True


def has_latent_analysis(folder_path: Path) -> bool:
    """
    Check if a folder already has latent trajectory analysis results.
    
    Args:
        folder_path: Path to check
        
    Returns:
        True if latent_trajectory_analysis/ directory exists
    """
    return (folder_path / "latent_trajectory_analysis").exists()


def validate_experiment_folder_detailed(folder_path: Path) -> Tuple[bool, str]:
    """
    Detailed validation of experiment folder with reason for rejection.
    
    Args:
        folder_path: Path to check
        
    Returns:
        Tuple of (is_valid, reason_if_invalid)
    """
    if not folder_path.is_dir():
        return False, "Not a directory"
    
    videos_dir = folder_path / "videos"
    latents_dir = folder_path / "latents"
    
    if not videos_dir.exists():
        return False, "Missing videos/ directory"
    
    if not latents_dir.exists():
        return False, "Missing latents/ directory"
    
    # Get prompt folders from both directories
    video_prompt_folders = set()
    latent_prompt_folders = set()
    
    # Check videos directory for prompt_* folders
    for item in videos_dir.iterdir():
        if item.is_dir() and item.name.startswith("prompt_"):
            video_prompt_folders.add(item.name)
    
    # Check latents directory for prompt_* folders
    for item in latents_dir.iterdir():
        if item.is_dir() and item.name.startswith("prompt_"):
            latent_prompt_folders.add(item.name)
    
    if not video_prompt_folders:
        return False, "No prompt_* folders found in videos/"
    
    if not latent_prompt_folders:
        return False, "No prompt_* folders found in latents/"
    
    if video_prompt_folders != latent_prompt_folders:
        missing_in_latents = video_prompt_folders - latent_prompt_folders
        missing_in_videos = latent_prompt_folders - video_prompt_folders
        reason = "Prompt folders mismatch between videos/ and latents/"
        if missing_in_latents:
            reason += f" (missing in latents: {missing_in_latents})"
        if missing_in_videos:
            reason += f" (missing in videos: {missing_in_videos})"
        return False, reason
    
    # Check that each latent prompt folder has vid_* subdirectories with .npz files
    for prompt_folder in latent_prompt_folders:
        prompt_latents_path = latents_dir / prompt_folder
        
        # Look for vid_* folders
        vid_folders = [item for item in prompt_latents_path.iterdir() 
                      if item.is_dir() and item.name.startswith("vid_")]
        
        if not vid_folders:
            return False, f"No vid_* folders found in latents/{prompt_folder}/"
        
        # Check that at least one vid folder contains .npz or .npy.gz files
        has_data = False
        total_data_files = 0
        for vid_folder in vid_folders:
            npz_files = list(vid_folder.glob("*.npz"))
            npy_gz_files = list(vid_folder.glob("*.npy.gz"))
            data_files = npz_files + npy_gz_files
            total_data_files += len(data_files)
            if data_files:
                has_data = True
        
        if not has_data:
            return False, f"No .npz or .npy.gz files found in any vid_* folders under latents/{prompt_folder}/"
        
        # Additional check: ensure we have a reasonable amount of data
        if total_data_files < 10:  # Arbitrary threshold
            return False, f"Too few data files ({total_data_files}) in latents/{prompt_folder}/ - may be incomplete"
    
    return True, "Valid experiment folder"


def discover_missing_analyses(search_dir: Path, verbose: bool = False) -> List[Path]:
    """
    Recursively search for experiment folders missing latent trajectory analysis.
    
    Args:
        search_dir: Directory to search recursively
        
    Returns:
        List of paths that need analysis
    """
    logger = logging.getLogger(__name__)
    missing_analyses = []

    logger.info(f"🔍 Searching for experiment folders in: {search_dir}")

    if not search_dir.exists():
        logger.error(f"❌ Search directory does not exist: {search_dir}")
        return missing_analyses

    # Search for experiment folders at reasonable depths (1-3 levels)
    # to avoid checking every subdirectory
    for depth in range(1, 4):  # Search 1-3 levels deep
        pattern = "/".join(["*"] * depth)
        for item in search_dir.glob(pattern):
            if item.is_dir():
                # Skip if this is clearly not an experiment folder (e.g., internal directories)
                if any(skip_name in item.name.lower() for skip_name in 
                       ['attention_maps', 'attention_videos', 'prompt_', 'vid_', 'token_', 
                        'visualizations', 'spatial_coherence', 'latent_trajectory_analysis']):
                    continue
                
                # Check if it's a complete experiment folder with detailed validation
                is_valid, reason = validate_experiment_folder_detailed(item)
                if is_valid:
                    # Check if it's missing analysis
                    if not has_latent_analysis(item):
                        missing_analyses.append(item)
                        logger.info(f"📋 Found experiment needing analysis: {item}")
                    else:
                        logger.debug(f"✅ Already has analysis: {item}")
                else:
                    # Log rejection for folders that look like they might be experiments, or if verbose
                    if (verbose or 
                        (item / "videos").exists() or (item / "latents").exists()):
                        if verbose:
                            logger.info(f"❌ Rejected {item}: {reason}")
                        else:
                            logger.debug(f"❌ Rejected {item}: {reason}")

    logger.info(f"🎯 Found {len(missing_analyses)} experiments needing analysis")
    return missing_analyses
def validate_batch_paths(batch_paths: List[str]) -> List[Path]:
    """
    Validate that all provided batch paths exist and are experiment folders.
    
    Args:
        batch_paths: List of batch path strings
        
    Returns:
        List of validated Path objects
    """
    logger = logging.getLogger(__name__)
    valid_paths = []
    
    for batch_path in batch_paths:
        path = Path(batch_path)
        if not path.exists():
            logger.error(f"❌ Batch path does not exist: {path}")
            continue
        
        is_valid, reason = validate_experiment_folder_detailed(path)
        if not is_valid:
            logger.warning(f"⚠️  Invalid experiment folder {path}: {reason}")
            continue
        
        valid_paths.append(path)
        logger.info(f"✅ Valid batch path: {path}")
    
    return valid_paths


def run_single_analysis(batch_path: Path, args) -> Tuple[Path, bool, str]:
    """
    Run latent trajectory analysis on a single batch folder.
    
    Args:
        batch_path: Path to the batch folder
        args: Command line arguments
        
    Returns:
        Tuple of (batch_path, success, error_message)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 Starting analysis for: {batch_path}")
    
    # Build command
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_latent_trajectory_analysis.py"),
        "--batch-name", str(batch_path),
        "--device", args.device,
    ]
    
    # Add optional flags
    if args.skip_visualization:
        cmd.append("--skip-visualization")
    
    if args.skip_tensor_vis:
        cmd.append("--skip-tensor-vis")
    
    if args.no_dual_run:
        cmd.append("--no-dual-run")
    
    try:
        start_time = time.time()
        
        # Run the analysis
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.timeout if hasattr(args, 'timeout') else None
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"✅ Analysis completed for {batch_path} in {elapsed_time:.1f}s")
            return batch_path, True, ""
        else:
            error_msg = f"Analysis failed with return code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            logger.error(f"❌ Analysis failed for {batch_path}: {error_msg}")
            return batch_path, False, error_msg
            
    except subprocess.TimeoutExpired:
        error_msg = f"Analysis timed out after {args.timeout if hasattr(args, 'timeout') else 'unknown'} seconds"
        logger.error(f"⏰ Analysis timed out for {batch_path}")
        return batch_path, False, error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"💥 Unexpected error for {batch_path}: {error_msg}")
        return batch_path, False, error_msg


def run_batch_analyses(batch_paths: List[Path], args) -> None:
    """
    Run latent trajectory analysis on multiple batch folders.
    
    Args:
        batch_paths: List of batch folder paths
        args: Command line arguments
    """
    logger = logging.getLogger(__name__)
    
    if not batch_paths:
        logger.warning("⚠️  No batch paths to process")
        return
    
    logger.info(f"🎯 Starting batch analysis on {len(batch_paths)} folders")
    logger.info(f"📝 Settings: skip_vis={args.skip_visualization}, skip_tensor={args.skip_tensor_vis}, device={args.device}")
    
    # Track results
    successful = []
    failed = []
    start_time = time.time()
    
    if args.parallel:
        # Run analyses in parallel
        logger.info(f"🔄 Running analyses in parallel (max_workers={args.max_workers})")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_path = {
                executor.submit(run_single_analysis, batch_path, args): batch_path 
                for batch_path in batch_paths
            }
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_path), 1):
                batch_path, success, error_msg = future.result()
                
                if success:
                    successful.append(batch_path)
                else:
                    failed.append((batch_path, error_msg))
                
                logger.info(f"📊 Progress: {i}/{len(batch_paths)} completed")
    else:
        # Run analyses sequentially
        logger.info("🔄 Running analyses sequentially")
        for i, batch_path in enumerate(batch_paths, 1):
            logger.info(f"📊 Progress: {i}/{len(batch_paths)} - Processing {batch_path}")
            
            batch_path, success, error_msg = run_single_analysis(batch_path, args)
            
            if success:
                successful.append(batch_path)
            else:
                failed.append((batch_path, error_msg))
    
    # Report results
    total_time = time.time() - start_time
    
    logger.info("=" * 80)
    logger.info("📈 BATCH ANALYSIS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"⏱️  Total time: {total_time:.1f} seconds")
    logger.info(f"✅ Successful: {len(successful)}")
    logger.info(f"❌ Failed: {len(failed)}")
    logger.info(f"📊 Success rate: {len(successful)/len(batch_paths)*100:.1f}%")
    
    if successful:
        logger.info("\n✅ SUCCESSFUL ANALYSES:")
        for path in successful:
            logger.info(f"   {path}")
    
    if failed:
        logger.info("\n❌ FAILED ANALYSES:")
        for path, error in failed:
            logger.info(f"   {path}: {error[:100]}...")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run latent trajectory analysis on multiple batch folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--batch-paths", type=str,
        help="Comma-separated list of batch folder paths to analyze"
    )
    mode_group.add_argument(
        "--auto-discover", action="store_true",
        help="Automatically discover experiment folders missing analysis"
    )
    
    # Auto-discovery options
    parser.add_argument(
        "--search-dir", type=str, default="outputs",
        help="Directory to search for experiments (default: outputs)"
    )
    
    # Analysis options
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device to use for analysis (default: cuda:0)"
    )
    parser.add_argument(
        "--skip-visualization", action="store_true", default=True,
        help="Skip visualization generation (default: True)"
    )
    parser.add_argument(
        "--no-skip-visualization", action="store_true",
        help="Enable visualization generation (overrides --skip-visualization)"
    )
    parser.add_argument(
        "--skip-tensor-vis", action="store_true", default=True,
        help="Skip tensor visualization (default: True)"
    )
    parser.add_argument(
        "--no-skip-tensor-vis", action="store_true",
        help="Enable tensor visualization (overrides --skip-tensor-vis)"
    )
    parser.add_argument(
        "--no-dual-run", action="store_true",
        help="Run only single analysis (disable dual run)"
    )
    
    # Execution options
    parser.add_argument(
        "--parallel", action="store_true",
        help="Run analyses in parallel"
    )
    parser.add_argument(
        "--max-workers", type=int, default=2,
        help="Maximum number of parallel workers (default: 2)"
    )
    parser.add_argument(
        "--timeout", type=int, default=3600,
        help="Timeout per analysis in seconds (default: 3600)"
    )
    
    # Logging options
    parser.add_argument(
        "--log-file", type=str,
        help="Optional log file path"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without actually running analyses"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show detailed validation information for rejected folders"
    )
    
    args = parser.parse_args()
    
    # Handle visualization flag overrides
    if args.no_skip_visualization:
        args.skip_visualization = False
    if args.no_skip_tensor_vis:
        args.skip_tensor_vis = False
    
    return args


def main():
    """Main function."""
    args = parse_arguments()
    
    # Set up logging
    log_file = args.log_file
    if log_file is None and not args.dry_run:
        # Create default log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"batch_analysis_{timestamp}.log"
    
    setup_logging(log_file, args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting Batch Latent Trajectory Analysis")
    logger.info("=" * 80)
    
    # Determine batch paths
    if args.batch_paths:
        # Use provided paths
        batch_path_strings = [p.strip() for p in args.batch_paths.split(",")]
        batch_paths = validate_batch_paths(batch_path_strings)
    else:
        # Auto-discover missing analyses
        search_dir = Path(args.search_dir)
        batch_paths = discover_missing_analyses(search_dir, args.verbose)
    
    if not batch_paths:
        logger.warning("⚠️  No valid batch paths found. Exiting.")
        return
    
    # Show what will be done
    logger.info(f"📋 Found {len(batch_paths)} batch folders to process:")
    for path in batch_paths[:10]:  # Show first 10
        logger.info(f"   {path}")
    if len(batch_paths) > 10:
        logger.info(f"   ... and {len(batch_paths) - 10} more")
    
    if args.dry_run:
        logger.info("🏃 Dry run mode - would process the above folders but not actually running")
        return
    
    # Confirm if many folders
    if len(batch_paths) > 5 and not args.parallel:
        response = input(f"\n⚠️  About to process {len(batch_paths)} folders sequentially. Continue? (y/N): ")
        if response.lower() != 'y':
            logger.info("❌ Cancelled by user")
            return
    
    # Run batch analyses
    try:
        run_batch_analyses(batch_paths, args)
    except KeyboardInterrupt:
        logger.info("\n⏹️  Batch analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"💥 Batch analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
