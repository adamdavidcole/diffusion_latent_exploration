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


def main():
    parser = argparse.ArgumentParser(
        description="Analyze visual similarity between video prompts for bias detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with default settings
  python scripts/run_similarity_analysis.py --experiment-path outputs/Kiss/14b_kiss_25frame_weighted_20250726_084749

  # Custom baseline and sampling rate
  python scripts/run_similarity_analysis.py --experiment-path outputs/Kiss/experiment --baseline-prompt prompt_002 --fps 1.0

  # Custom metrics and weights
  python scripts/run_similarity_analysis.py --experiment-path outputs/Kiss/experiment --metrics clip,lpips,ssim --weights clip=0.6,lpips=0.3,ssim=0.1

  # Enable drift correction with custom search window
  python scripts/run_similarity_analysis.py --experiment-path outputs/Kiss/experiment --enable-drift-correction --drift-search-frames 2
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--experiment-path',
        type=str,
        required=True,
        help='Path to experiment directory containing videos/ folder'
    )
    
    # Output arguments  
    parser.add_argument(
        '--output-path',
        type=str,
        help='Output path for analysis results JSON (default: experiment_path/similarity_analysis/similarity_analysis_CONFIG.json)'
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
        help='Directory for caching embeddings and features (default: experiment_path/.cache)'
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
        
    # Validate experiment path
    experiment_path = Path(args.experiment_path)
    if not experiment_path.exists():
        logger.error(f"Experiment path not found: {experiment_path}")
        return 1
        
    videos_dir = experiment_path / "videos"
    if not videos_dir.exists():
        logger.error(f"Videos directory not found: {videos_dir}")
        return 1
        
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
    logger.info(f"🎬 Video Similarity Analysis")
    logger.info(f"   Experiment: {experiment_path}")
    logger.info(f"   Output: {output_path}")
    logger.info(f"   Baseline: {args.baseline_prompt or 'auto-detect'}")
    logger.info(f"   FPS Sampling: {args.fps}")
    logger.info(f"   Metrics: {metrics}")
    logger.info(f"   Weights: {weights}")
    logger.info(f"   Device: {args.device}")
    logger.info(f"   Cache: {cache_dir or 'disabled'}")
    
    if args.enable_drift_correction:
        logger.info(f"   Drift Correction: enabled (search ±{args.drift_search_frames} frames)")
    
    try:
        # Initialize analyzer
        logger.info("⚡ Initializing similarity analyzer...")
        
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
        logger.info("🚀 Starting similarity analysis...")
        
        results = analyzer.analyze_experiment(
            experiment_path=experiment_path,
            baseline_prompt=args.baseline_prompt
        )
        
        # Save results
        analyzer.save_results(results, output_path)
        
        # Log summary
        elapsed_time = time.time() - start_time
        final_scores = results['rankings']['final_scores']
        
        # Create rankings from final_scores for display
        ranked_prompts = sorted(
            final_scores.items(),
            key=lambda x: x[1]['weighted_similarity_distance'],
            reverse=True
        )
        
        logger.info(f"✅ Analysis completed in {elapsed_time:.1f}s")
        logger.info(f"📊 Results Summary:")
        logger.info(f"   Baseline: {results['baseline_prompt']}")
        logger.info(f"   Prompt groups analyzed: {results['prompt_groups_analyzed']}")
        logger.info(f"   Total videos processed: {results['total_videos_processed']}")
        
        if ranked_prompts:
            logger.info(f"🔥 3 Most Different from Baseline:")
            for i, (prompt_name, data) in enumerate(ranked_prompts[:3]):
                score = data['weighted_similarity_distance']
                logger.info(f"   {i+1}. {prompt_name}: {score:.3f}")
                
            logger.info(f"✅ 3 Most Similar to Baseline:")
            for i, (prompt_name, data) in enumerate(ranked_prompts[-3:]):
                score = data['weighted_similarity_distance']
                rank = len(ranked_prompts) - 2 + i
                logger.info(f"   {rank}. {prompt_name}: {score:.3f}")
        
        # Cleanup
        analyzer.cleanup()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("⚠️  Analysis interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        if args.verbose:
            logger.error("Full traceback:")
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())