#!/usr/bin/env python3
"""
GPU-Optimized Latent Analysis Runner

This script runs the GPU-accelerated version of the structure-aware latent analysis,
with command-line flexibility for specifying batch name, device, and prompt groups.
"""

import argparse
import json
import logging
import sys
import time
import torch
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.latent_trajectory_analyzer import LatentTrajectoryAnalyzer
from src.visualization.latent_trajectory_visualizer import LatentTrajectoryVisualizer
from src.utils.prompt_utils import load_prompt_metadata

from src.analysis.data_structures import LatentTrajectoryAnalysis


def setup_logging():
    """Configure logging for analysis."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('gpu_structure_analysis.log')
        ]
    )


def check_gpu_availability():
    """Check and report GPU availability."""
    logger = logging.getLogger(__name__)
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        memory_gb = torch.cuda.get_device_properties(current_device).total_memory / 1e9
        
        logger.info(f"🚀 GPU acceleration available!")
        logger.info(f"   Device count: {device_count}")
        logger.info(f"   Current device: {current_device} ({device_name})")
        logger.info(f"   Memory: {memory_gb:.1f} GB")
        
        return f"cuda:{current_device}"
    else:
        logger.warning("⚠️ GPU not available, using CPU")
        logger.warning("   Consider installing CUDA-enabled PyTorch for significant speedup")
        return "cpu"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run GPU-optimized latent trajectory analysis.")
    parser.add_argument(
        "--batch-name", type=str, required=True,
        help="Name of the batch. The script will look in <batch_name>/latents/"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (e.g., 'cpu', 'cuda', 'cuda:1'). Defaults to auto-detect."
    )
    parser.add_argument(
        "--prompt-groups", type=str, default=None,
        help="Comma-separated list of prompt group names (e.g., 'prompt_000,prompt_001'). If not passed, all subdirs in latents/ are used."
    )
    parser.add_argument(
        '--no-dual-run',
        action='store_true',
        dest='no_dual_run', # The parsed attribute will be named 'dual_run'
        help="Use this flag to run only single analysis (disables dual run)."
    )

    parser.add_argument(
        '--results-file-path',
        type=str, 
        default=None,
        help="Path to the file where results are already saved (will skip analysis and only do visualization)."
    )


    # Hull performance/accuracy controls
    parser.add_argument("--hull-mode", type=str, default="auto", choices=["auto", "exact", "approx", "proxy", "off"],
                        help="Convex hull computation mode: auto chooses approx for high-dim; proxy uses ellipsoidal proxy only; off disables.")
    parser.add_argument("--hull-max-dim-exact", type=int, default=8, help="Max dimension for exact hull computation.")
    parser.add_argument("--hull-max-points-exact", type=int, default=500, help="Max points for exact hull computation.")
    parser.add_argument("--hull-rp-dim", type=int, default=8, help="Random projection target dimension for approx hull.")
    parser.add_argument("--hull-rp-projections", type=int, default=12, help="Number of random projections to average for approx hull.")
    parser.add_argument("--hull-sample-points", type=int, default=2000, help="Max points to sample for pairwise distances and proxies.")
    parser.add_argument("--hull-sample-features", type=int, default=8192, help="Max feature dimensions to subsample before projections for memory/time safety.")
    parser.add_argument("--hull-time-budget-ms", type=int, default=3000, help="Soft time budget per group for hull analysis in milliseconds.")
    return parser.parse_args()


def get_prompt_groups(latents_dir, user_specified=None):
    if user_specified:
        return [g.strip() for g in user_specified.split(",")]
    else:
        return sorted([p.name for p in Path(latents_dir).iterdir() if p.is_dir()])


def run_gpu_optimized_analysis(batch_name, device, prompt_groups, args=None):
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🔬 Starting GPU-Optimized Structure-Aware Latent Analysis")
    logger.info("=" * 80)

    latents_dir = Path(batch_name) / "latents"
    if not latents_dir.exists():
        logger.error(f"❌ Latents directory not found: {latents_dir}")
        sys.exit(1)

    visualizations_dir = Path(batch_name) / "latent_trajectory_analysis_visualization"

    if device is None:
        device = check_gpu_availability()

    logger.info(f"Device selected: {device}")
    logger.info(f"Latents directory: {latents_dir}")
    logger.info(f"Visualizations directory: {visualizations_dir}")
    logger.info(f"Prompt groups: {prompt_groups}")

    # Performance configuration
    if device.startswith("cuda"):
        batch_size = 64
        enable_mixed_precision = True
    else:
        batch_size = 8
        enable_mixed_precision = False

    try:
        logger.info("🔧 Initializing analyzer...")
        start_time = time.time()


        # Normalization configuration: if dual run, it will setup normalization later
        # If not dual run, use specific normalization strategy
        norm_cfg = None

        if args.no_dual_run:
            norm_cfg = {
                "per_step_whiten": False,
                "per_channel_standardize": False,
                "snr_normalize": True,
            }
        
        analyzer = LatentTrajectoryAnalyzer(
            latents_dir=str(latents_dir),
            device=device,
            enable_mixed_precision=enable_mixed_precision,
            batch_size=batch_size,
            norm_cfg=norm_cfg,
            # Hull config from CLI
            hull_mode=args.hull_mode if args else "auto",
            hull_max_dim_exact=args.hull_max_dim_exact if args else 8,
            hull_max_points_exact=args.hull_max_points_exact if args else 500,
            hull_rp_dim=args.hull_rp_dim if args else 8,
            hull_rp_projections=args.hull_rp_projections if args else 12,
            hull_sample_points=args.hull_sample_points if args else 2000,
            hull_sample_features=args.hull_sample_features if args else 8192,
            hull_time_budget_ms=args.hull_time_budget_ms if args else 3000,
        )

        visualizer = LatentTrajectoryVisualizer(
            batch_dir=batch_name,
            output_dir=visualizations_dir
        )
        visualizer.visualize()
        
        init_time = time.time() - start_time
        logger.info(f"✅ Analyzer initialized in {init_time:.2f} seconds")

        logger.info(f"🚀 Starting analysis on groups: {prompt_groups}")
        analysis_start = time.time()
        
        # Load prompt metadata from batch configuration
        logger.info("📋 Loading prompt metadata...")
        prompt_metadata = load_prompt_metadata(batch_name, prompt_groups)

        results = None

        if args.results_file_path:
            # Load results from the specified file
            with open(args.results_file_path, 'r') as f:
                json_data = json.load(f)
            results = LatentTrajectoryAnalysis.from_dict(json_data)
            logger.info(f"📁 Loaded existing results from: {args.results_file_path}")
        else:
            if args.no_dual_run:
                # Run single track analysis
                print(f"🔬 Running single analysis (no dual tracks) with {norm_cfg}")
                results = analyzer.analyze_prompt_groups(prompt_groups, prompt_metadata)
            
            else:
                # Run dual tracks analysis
                print(f"🔬 Running dual tracks analysis with differenct norm configs")
                results = analyzer.run_dual_tracks(prompt_groups, prompt_metadata)
                # TODO: handle dual analysis run
        
        # TODO: visualizer
        visualizer.create_comprehensive_visualizations(results)
            

        
        # analysis_time = time.time() - analysis_start
        
        # logger.info(f"📊 Analysis completed in {analysis_time:.2f} seconds")
        # logger.info(f"📁 Results saved to: {analyzer.output_dir}")
        # return results

    except Exception as e:
        logger.exception(f"❌ Analysis failed: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    try:
        args = parse_arguments()
        prompt_groups = get_prompt_groups(
            Path(args.batch_name) / "latents", args.prompt_groups
        )
        run_gpu_optimized_analysis(
            batch_name=args.batch_name,
            device=args.device,
            prompt_groups=prompt_groups,
            args=args
        )
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Analysis failed: {e}")
        traceback.print_exc()
        sys.exit(1)
