#!/usr/bin/env python3
"""
GPU-Optimized Latent Analysis Runner

This script runs the GPU-accelerated version of the structure-aware latent analysis,
with command-line flexibility for specifying batch name, device, and prompt groups.
"""

import argparse
import logging
import sys
import time
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.latent_trajectory_analyzer import LatentTrajectoryAnalyzer


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
    return parser.parse_args()


def get_prompt_groups(latents_dir, user_specified=None):
    if user_specified:
        return [g.strip() for g in user_specified.split(",")]
    else:
        return sorted([p.name for p in Path(latents_dir).iterdir() if p.is_dir()])


def run_gpu_optimized_analysis(batch_name, device, prompt_groups):
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🔬 Starting GPU-Optimized Structure-Aware Latent Analysis")
    logger.info("=" * 80)

    latents_dir = Path(batch_name) / "latents"
    if not latents_dir.exists():
        logger.error(f"❌ Latents directory not found: {latents_dir}")
        sys.exit(1)

    if device is None:
        device = check_gpu_availability()

    logger.info(f"Device selected: {device}")
    logger.info(f"Latents directory: {latents_dir}")
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
        
        analyzer = LatentTrajectoryAnalyzer(
            latents_dir=str(latents_dir),
            device=device,
            enable_mixed_precision=enable_mixed_precision,
            batch_size=batch_size
        )
        
        init_time = time.time() - start_time
        logger.info(f"✅ Analyzer initialized in {init_time:.2f} seconds")

        logger.info(f"🚀 Starting analysis on groups: {prompt_groups}")
        analysis_start = time.time()
        
        # Descriptions can be empty or generated dynamically later
        prompt_descriptions = [f"Description for {g}" for g in prompt_groups]

        results = analyzer.analyze_prompt_groups(prompt_groups, prompt_descriptions)
        analysis_time = time.time() - analysis_start
        
        logger.info(f"📊 Analysis completed in {analysis_time:.2f} seconds")
        logger.info(f"📁 Results saved to: {analyzer.output_dir}")
        return results

    except Exception as e:
        logger.exception(f"❌ Analysis failed: {e}")
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
            prompt_groups=prompt_groups
        )
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Analysis failed: {e}")
        sys.exit(1)
