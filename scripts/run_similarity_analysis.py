#!/usr/bin/env python3
"""
Command-line interface for Video Similarity Analysis

Analyzes visual similarity between video prompts in diffusion experiments
for bias detection and ranking analysis.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import traceback
import time

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.similarity_analysis import VideoSimilarityAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_weights(weights_str: str) -> dict:
    """Parse weights string into dictionary."""
    if not weights_str:
        return {}
        
    weights = {}
    for pair in weights_str.split(','):
        if '=' in pair:
            key, value = pair.split('=', 1)
            try:
                weights[key.strip()] = float(value.strip())
            except ValueError:
                logger.warning(f"Invalid weight value: {pair}")
                
    return weights


def find_incomplete_experiments(outputs_dir: Path) -> list[Path]:
    """Find all experiment directories that don't have similarity analysis completed."""
    incomplete_experiments = []
    
    # Search for experiment directories with videos/ folder
    for category_dir in outputs_dir.iterdir():
        if not category_dir.is_dir():
            continue
            
        for experiment_dir in category_dir.iterdir():
            if not experiment_dir.is_dir():
                continue
                
            videos_dir = experiment_dir / "videos"
            if not videos_dir.exists():
                continue
                
            # Check if similarity analysis exists
            similarity_dir = experiment_dir / "similarity_analysis"
            if not similarity_dir.exists():
                incomplete_experiments.append(experiment_dir)
                continue
                
            # Check if any analysis files exist
            analysis_files = list(similarity_dir.glob("similarity_analysis_*.json"))
            if not analysis_files:
                incomplete_experiments.append(experiment_dir)
                
    return incomplete_experiments


def validate_experiment_path(experiment_path: Path) -> bool:
    """Validate that experiment path contains videos directory."""
    if not experiment_path.exists():
        logger.error(f"Experiment path not found: {experiment_path}")
        return False
        
    videos_dir = experiment_path / "videos"
    if not videos_dir.exists():
        logger.error(f"Videos directory not found: {videos_dir}")
        return False
        
    return True


def run_single_analysis(experiment_path: Path, args) -> bool:
    """Run similarity analysis on a single experiment directory."""
    if not validate_experiment_path(experiment_path):
        return False
        
    # Parse configuration
    metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]
    weights = parse_weights(args.weights)
    
    # Set up cache directory
    cache_dir = None
    if not args.no_cache:
        if args.cache_dir:
            cache_dir = Path(args.cache_dir)
        else:
            # Place cache in similarity_analysis folder
            similarity_dir = experiment_path / "similarity_analysis"
            similarity_dir.mkdir(exist_ok=True)
            cache_dir = similarity_dir / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
    # Set up output path
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        # Create similarity_analysis folder within experiment
        similarity_dir = experiment_path / "similarity_analysis"
        similarity_dir.mkdir(exist_ok=True)
        
        # Generate filename based on configuration
        config_suffix = f"{args.fps}fps"
        if args.metrics != 'clip,lpips,ssim':
            metrics_str = args.metrics.replace(',', '-')
            config_suffix += f"_{metrics_str}"
        if args.enable_drift_correction:
            config_suffix += "_drift"
            
        output_path = similarity_dir / f"similarity_analysis_{config_suffix}.json"
        
    # Log configuration
    logger.info(f"🎬 Analyzing: {experiment_path.name}")
    logger.info(f"   Output: {output_path}")
    logger.info(f"   Baseline: {args.baseline_prompt or 'auto-detect'}")
    
    try:
        # Initialize analyzer
        analyzer = VideoSimilarityAnalyzer(
            fps_sampling=args.fps,
            enable_drift_correction=args.enable_drift_correction,
            drift_search_frames=args.drift_search_frames,
            cache_dir=cache_dir,
            device=args.device,
            metrics=metrics,
            weights=weights
        )
        
        # Run analysis
        start_time = time.time()
        
        results = analyzer.analyze_experiment(
            experiment_path=experiment_path,
            baseline_prompt=args.baseline_prompt
        )
        
        # Save results
        analyzer.save_results(results, output_path)
        
        # Log summary
        elapsed_time = time.time() - start_time
        
        logger.info(f"✅ Completed in {elapsed_time:.1f}s - {results['prompt_groups_analyzed']} prompts, {results['total_videos_processed']} videos")
        
        # Cleanup
        analyzer.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to analyze {experiment_path.name}: {e}")
        if args.verbose:
            logger.error("Full traceback:")
            traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Analyze visual similarity between video prompts for bias detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with default settings
  python scripts/run_similarity_analysis.py --batch-path outputs/Kiss/14b_kiss_25frame_weighted_20250726_084749

  # Custom baseline and sampling rate
  python scripts/run_similarity_analysis.py --batch-path outputs/Kiss/experiment --baseline-prompt prompt_002 --fps 1.0

  # Process multiple experiments
  python scripts/run_similarity_analysis.py --batch-paths outputs/Kiss/exp1 outputs/Touch/exp2 "outputs/Hero Action/exp3"

  # Process all incomplete experiments in outputs directory
  python scripts/run_similarity_analysis.py --outputs-dir outputs/

  # Custom metrics and weights
  python scripts/run_similarity_analysis.py --batch-path outputs/Kiss/experiment --metrics clip,lpips,ssim --weights clip=0.6,lpips=0.3,ssim=0.1

  # Enable drift correction with custom search window
  python scripts/run_similarity_analysis.py --batch-path outputs/Kiss/experiment --enable-drift-correction --drift-search-frames 2
        """
    )
    
    # Required arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--batch-path',
        type=str,
        help='Path to single experiment directory containing videos/ folder'
    )
    
    group.add_argument(
        '--batch-paths',
        nargs='+',
        help='Multiple experiment directory paths to process'
    )
    
    group.add_argument(
        '--outputs-dir',
        type=str,
        help='Path to outputs directory - will process all experiments without similarity analysis'
    )
    
    # Output arguments  
    parser.add_argument(
        '--output-path',
        type=str,
        help='Output path for analysis results JSON (default: batch_path/similarity_analysis/similarity_analysis_CONFIG.json)'
    )
    
    # Analysis configuration
    parser.add_argument(
        '--baseline-prompt',
        type=str,
        help='Baseline prompt directory name (default: auto-detect first alphabetically)'
    )
    
    parser.add_argument(
        '--fps',
        type=float,
        default=2.0,
        help='Frames per second to extract from videos (default: 2.0)'
    )
    
    parser.add_argument(
        '--metrics',
        type=str,
        default='clip,lpips,ssim',
        help='Comma-separated list of metrics to compute (default: clip,lpips,ssim)'
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        default='clip=0.5,lpips=0.3,ssim=0.1,mse=0.05,phash=0.05',
        help='Comma-separated metric weights (default: clip=0.5,lpips=0.3,ssim=0.1,mse=0.05,phash=0.05)'
    )
    
    # Drift correction options
    parser.add_argument(
        '--enable-drift-correction',
        action='store_true',
        help='Enable temporal drift correction for frame alignment'
    )
    
    parser.add_argument(
        '--drift-search-frames',
        type=int,
        default=1,
        help='Number of frames to search ±1 for drift correction (default: 1)'
    )
    
    # Technical options
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device for computation (cuda/cpu, default: cuda)'
    )
    
    parser.add_argument(
        '--cache-dir',
        type=str,
        help='Directory for caching embeddings and features (default: batch_path/.cache)'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable embedding caching'
    )
    
    # Logging options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce logging output'
    )

    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
        
    # Determine experiment paths to process
    experiment_paths = []
    
    if args.batch_path:
        experiment_paths = [Path(args.batch_path)]
        
    elif args.batch_paths:
        experiment_paths = [Path(p) for p in args.batch_paths]
        
    elif args.outputs_dir:
        outputs_dir = Path(args.outputs_dir)
        if not outputs_dir.exists():
            logger.error(f"Outputs directory not found: {outputs_dir}")
            return 1
            
        experiment_paths = find_incomplete_experiments(outputs_dir)
        if not experiment_paths:
            logger.info("✅ All experiments already have similarity analysis completed")
            return 0
            
        logger.info(f"📋 Found {len(experiment_paths)} experiments without similarity analysis:")
        for exp_path in experiment_paths:
            logger.info(f"   {exp_path}")
    
    # Validate we have something to process
    if not experiment_paths:
        logger.error("No experiment paths specified")
        return 1
        
    # Log batch configuration
    logger.info(f"🎬 Video Similarity Batch Analysis")
    logger.info(f"   Experiments to process: {len(experiment_paths)}")
    logger.info(f"   Baseline: {args.baseline_prompt or 'auto-detect'}")
    logger.info(f"   FPS Sampling: {args.fps}")
    logger.info(f"   Metrics: {[m.strip() for m in args.metrics.split(',') if m.strip()]}")
    logger.info(f"   Weights: {parse_weights(args.weights)}")
    logger.info(f"   Device: {args.device}")
    
    if args.enable_drift_correction:
        logger.info(f"   Drift Correction: enabled (search ±{args.drift_search_frames} frames)")
    
    try:
        # Process each experiment
        start_time = time.time()
        successful = 0
        failed = 0
        
        for i, experiment_path in enumerate(experiment_paths, 1):
            logger.info(f"🚀 Processing {i}/{len(experiment_paths)}: {experiment_path.name}")
            
            try:
                success = run_single_analysis(experiment_path, args)
                if success:
                    successful += 1
                else:
                    failed += 1
                    
            except KeyboardInterrupt:
                logger.info("⚠️  Analysis interrupted by user")
                return 130
                
        # Log batch summary
        elapsed_time = time.time() - start_time
        
        logger.info(f"✅ Batch analysis completed in {elapsed_time:.1f}s")
        logger.info(f"📊 Batch Summary:")
        logger.info(f"   Total experiments: {len(experiment_paths)}")
        logger.info(f"   Successful: {successful}")
        logger.info(f"   Failed: {failed}")
        
        return 0 if failed == 0 else 1
        
    except Exception as e:
        logger.error(f"❌ Batch analysis failed: {e}")
        if args.verbose:
            logger.error("Full traceback:")
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())