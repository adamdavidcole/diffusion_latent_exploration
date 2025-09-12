#!/usr/bin/env python3
"""
Latent Step Decoder Script

This script decodes stored latent tensors back into video files using the model's VAE.
It reads latent steps from experiment directories and generates corresponding video files.

Usage:
    python scripts/decode_latent_steps.py <experiment_dir> [options]

Examples:
    # Decode all latents in an experiment
    python scripts/decode_latent_steps.py outputs/MyExperiment_20250901_120000/
    
    # Decode only specific prompt/video
    python scripts/decode_latent_steps.py outputs/MyExperiment_20250901_120000/ --prompt-filter prompt_000 --video-filter vid_001
    
    # Decode only specific steps
    python scripts/decode_latent_steps.py outputs/MyExperiment_20250901_120000/ --step-filter step_000,step_010,step_019
    
    # Use specific device
    python scripts/decode_latent_steps.py outputs/MyExperiment_20250901_120000/ --device cuda:1
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add progress bar support
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Install with 'pip install tqdm' for progress bars.")

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.visualization.latent_visualizer import ExperimentLatentDecoder, create_decode_summary_report


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def decode_experiment_with_progress(decoder, output_dir=None, prompt_filter=None, 
                                   video_filter=None, step_filter=None, fps=None,
                                   quality=3.0, scale_factor=1.0):
    """
    Decode experiment with progress tracking.
    
    Args:
        decoder: ExperimentLatentDecoder instance
        output_dir: Output directory for decoded videos
        prompt_filter: Filter for prompt directories
        video_filter: Filter for video directories  
        step_filter: Filter for step files
        fps: Video FPS (None = use config default)
        quality: Video quality (0-10, lower = smaller file)
        scale_factor: Resolution scale factor (0.5 = half size)
        
    Returns:
        Nested dictionary: {video_path: {step_name: result}}
    """
    if decoder.decoder is None:
        decoder.initialize_decoder()
    
    video_dirs = decoder.find_latent_directories()
    
    # Apply filters
    if prompt_filter:
        video_dirs = [d for d in video_dirs if prompt_filter in str(d)]
    if video_filter:
        video_dirs = [d for d in video_dirs if video_filter in str(d)]
    
    if not video_dirs:
        logging.warning("No video directories found after filtering")
        return {}
    
    # Count total steps for overall progress
    total_steps = 0
    video_step_counts = {}
    for video_dir in video_dirs:
        step_files = decoder.find_latent_steps(video_dir)
        
        if step_filter:
            step_filters = [f.strip() for f in step_filter.split(',')]
            step_files = [f for f in step_files 
                         if any(f.stem.replace('.npy', '') == sf for sf in step_filters)]
        
        video_step_counts[str(video_dir)] = len(step_files)
        total_steps += len(step_files)
    
    logging.info(f"Decoding {len(video_dirs)} video directories with {total_steps} total steps")
    
    all_results = {}
    
    # Create overall progress bar if tqdm is available
    if TQDM_AVAILABLE:
        video_pbar = tqdm(video_dirs, desc="Videos", unit="video")
        step_pbar = tqdm(total=total_steps, desc="Steps", unit="step", leave=False)
    else:
        video_pbar = video_dirs
        step_pbar = None
    
    try:
        for video_idx, video_dir in enumerate(video_pbar):
            # Update video progress description
            if TQDM_AVAILABLE:
                video_name = f"{video_dir.parent.name}/{video_dir.name}"
                video_pbar.set_description(f"Processing {video_name}")
            else:
                logging.info(f"Processing video directory ({video_idx + 1}/{len(video_dirs)}): {video_dir}")
            
            try:
                # Get steps for this video
                step_files = decoder.find_latent_steps(video_dir)
                
                if step_filter:
                    step_filters = [f.strip() for f in step_filter.split(',')]
                    step_files = [f for f in step_files 
                                 if any(f.stem.replace('.npy', '') == sf for sf in step_filters)]
                
                # Setup output directory for this video
                if output_dir is None:
                    output_base = decoder.experiment_dir / "latents_videos"
                else:
                    output_base = output_dir
                
                relative_path = video_dir.relative_to(decoder.experiment_dir / "latents")
                video_output_dir = output_base / relative_path
                video_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Get FPS from config or parameter
                if fps is None:
                    fps = decoder.get_video_settings().get('fps', 12)
                
                results = {}
                
                for step_file in step_files:
                    step_name = step_file.stem.replace('.npy', '')  # e.g., "step_000"
                    output_path = video_output_dir / f"{step_name}.mp4"
                    
                    # Update step progress description
                    if TQDM_AVAILABLE:
                        step_pbar.set_description(f"Decoding {step_name}")
                    
                    result = decoder.decoder.decode_latent_step_to_video(
                        latent_path=step_file,
                        output_path=output_path,
                        fps=fps,
                        quality=quality,
                        scale_factor=scale_factor
                    )
                    
                    results[step_name] = result
                    
                    if result.success:
                        if not TQDM_AVAILABLE:
                            print(f"✅ Decoded {video_dir.parent.name}/{video_dir.name}/{step_name} in {result.decode_time:.2f}s")
                    else:
                        if not TQDM_AVAILABLE:
                            print(f"❌ Failed {video_dir.parent.name}/{video_dir.name}/{step_name}: {result.error_message}")
                        else:
                            logging.error(f"❌ Failed to decode {step_name}: {result.error_message}")
                    
                    # Update step progress
                    if TQDM_AVAILABLE:
                        step_pbar.update(1)
                
                all_results[str(video_dir)] = results
                
            except Exception as e:
                logging.error(f"Failed to process {video_dir}: {e}")
                all_results[str(video_dir)] = {}
                
                # Still update progress for skipped steps
                if TQDM_AVAILABLE:
                    step_pbar.update(video_step_counts[str(video_dir)])
    
    finally:
        # Close progress bars
        if TQDM_AVAILABLE:
            video_pbar.close()
            if step_pbar:
                step_pbar.close()
    
    return all_results


def parse_filter_list(filter_str: str) -> list:
    """Parse comma-separated filter string into list."""
    if not filter_str:
        return []
    return [item.strip() for item in filter_str.split(',')]


def main():
    parser = argparse.ArgumentParser(
        description="Decode latent tensors back to videos using model VAE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Path to experiment directory containing configs/ and latents/"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for decoded videos (default: experiment_dir/latents_videos/)"
    )
    
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use for decoding (auto, cuda:0, cpu, etc.) [default: auto]"
    )
    
    parser.add_argument(
        "--prompt-filter",
        help="Filter for prompt directories (e.g., 'prompt_000' or 'prompt_000,prompt_001')"
    )
    
    parser.add_argument(
        "--video-filter", 
        help="Filter for video directories (e.g., 'vid_001' or 'vid_001,vid_002')"
    )
    
    parser.add_argument(
        "--step-filter",
        help="Filter for step files (e.g., 'step_000' or 'step_000,step_010,step_019')"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        help="Output video FPS (default: from experiment config)"
    )
    
    parser.add_argument(
        "--quality",
        type=float,
        default=3.0,
        help="Video quality (0-10, lower = smaller file) [default: 3.0]"
    )
    
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Resolution scale factor (0.5 = half size, 1.0 = full size) [default: 1.0]"
    )
    
    parser.add_argument(
        "--compress-preset",
        choices=["high-quality", "balanced", "small-file", "tiny"],
        help="Compression preset: high-quality (q=7.0,s=1.0), balanced (q=3.0,s=1.0), small-file (q=1.0,s=0.75), tiny (q=0.5,s=0.5)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be decoded without actually doing it"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Handle compression presets
    quality = args.quality
    scale = args.scale
    
    if args.compress_preset:
        if args.compress_preset == "high-quality":
            quality, scale = 7.0, 1.0
        elif args.compress_preset == "balanced":
            quality, scale = 3.0, 1.0
        elif args.compress_preset == "small-file":
            quality, scale = 1.0, 0.75
        elif args.compress_preset == "tiny":
            quality, scale = 0.5, 0.5
        
        logging.info(f"Using {args.compress_preset} preset: quality={quality}, scale={scale}")
    else:
        logging.info(f"Using custom settings: quality={quality}, scale={scale}")
    
    # Validate experiment directory
    if not args.experiment_dir.exists():
        logging.error(f"Experiment directory not found: {args.experiment_dir}")
        return 1
    
    if not (args.experiment_dir / "configs" / "generation_config.yaml").exists():
        logging.error(f"Configuration file not found: {args.experiment_dir}/configs/generation_config.yaml")
        return 1
    
    if not (args.experiment_dir / "latents").exists():
        logging.error(f"Latents directory not found: {args.experiment_dir}/latents")
        return 1
    
    try:
        # Initialize decoder
        logging.info(f"Initializing decoder for experiment: {args.experiment_dir}")
        decoder = ExperimentLatentDecoder(args.experiment_dir, args.device)
        
        # Show experiment info
        model_id = decoder.get_model_id()
        video_settings = decoder.get_video_settings()
        logging.info(f"Model: {model_id}")
        logging.info(f"Video settings: {video_settings}")
        
        # Find latent directories
        video_dirs = decoder.find_latent_directories()
        logging.info(f"Found {len(video_dirs)} video directories")
        
        # Apply filters
        filtered_dirs = video_dirs.copy()
        
        if args.prompt_filter:
            prompt_filters = parse_filter_list(args.prompt_filter)
            filtered_dirs = [d for d in filtered_dirs 
                           if any(pf in str(d) for pf in prompt_filters)]
            logging.info(f"Prompt filter '{args.prompt_filter}' -> {len(filtered_dirs)} directories")
        
        if args.video_filter:
            video_filters = parse_filter_list(args.video_filter)
            filtered_dirs = [d for d in filtered_dirs 
                           if any(vf in str(d) for vf in video_filters)]
            logging.info(f"Video filter '{args.video_filter}' -> {len(filtered_dirs)} directories")
        
        if not filtered_dirs:
            logging.warning("No directories match the specified filters")
            return 1
        
        # Count steps that would be processed
        total_steps = 0
        for video_dir in filtered_dirs:
            step_files = decoder.find_latent_steps(video_dir)
            
            if args.step_filter:
                step_filters = parse_filter_list(args.step_filter)
                step_files = [f for f in step_files 
                            if any(sf in f.stem for sf in step_filters)]
            
            total_steps += len(step_files)
            
            if args.dry_run:
                logging.info(f"Would decode {len(step_files)} steps from {video_dir}")
        
        logging.info(f"Total steps to decode: {total_steps}")
        
        if args.dry_run:
            logging.info("Dry run complete - no actual decoding performed")
            return 0
        
        if total_steps == 0:
            logging.warning("No steps found to decode")
            return 1
        
        # Perform decoding with progress tracking
        start_time = time.time()
        
        results = decode_experiment_with_progress(
            decoder,
            output_dir=args.output_dir,
            prompt_filter=args.prompt_filter,
            video_filter=args.video_filter,
            step_filter=args.step_filter,
            fps=args.fps,
            quality=quality,
            scale_factor=scale
        )
        
        total_time = time.time() - start_time
        
        # Generate summary report
        output_dir = args.output_dir or (args.experiment_dir / "latents_videos")
        report_path = output_dir / "decode_summary.json"
        
        summary = create_decode_summary_report(results, report_path)
        
        # Log final results
        logging.info("=" * 60)
        logging.info("DECODING COMPLETE")
        logging.info("=" * 60)
        logging.info(f"Total steps processed: {summary['total_steps']}")
        logging.info(f"Successful: {summary['successful_steps']}")
        logging.info(f"Failed: {summary['failed_steps']}")
        logging.info(f"Success rate: {summary['success_rate']:.1%}")
        logging.info(f"Total time: {total_time:.2f}s")
        logging.info(f"Average time per step: {summary['avg_decode_time']:.2f}s")
        logging.info(f"Output directory: {output_dir}")
        logging.info(f"Summary report: {report_path}")
        
        # Show any failures
        failed_count = 0
        for video_path, video_results in results.items():
            for step_name, result in video_results.items():
                if not result.success:
                    failed_count += 1
                    if failed_count <= 5:  # Show first 5 failures
                        logging.error(f"Failed: {video_path}/{step_name} - {result.error_message}")
        
        if failed_count > 5:
            logging.error(f"... and {failed_count - 5} more failures (see report for details)")
        
        return 0 if summary['failed_steps'] == 0 else 1
        
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        return 130
    
    except Exception as e:
        logging.error(f"Error during decoding: {e}")
        if args.verbose:
            import traceback
            logging.error(traceback.format_exc())
        return 1
    
    finally:
        # Cleanup
        if 'decoder' in locals():
            decoder.cleanup()


if __name__ == "__main__":
    sys.exit(main())
