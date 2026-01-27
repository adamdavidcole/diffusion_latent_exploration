#!/usr/bin/env python3
"""
Meta-Script for Decoding Latent and Attention Steps

This script orchestrates the decoding process for experiments, handling both latent
and attention map decoding in sequence with support for:
- Recursive experiment discovery in directory trees
- Multi-GPU parallel processing for latent decoding
- Multi-process parallel processing for attention decoding
- Parameter pass-through to underlying decoder scripts
- Unified progress tracking and reporting

Usage:
    python scripts/decode_steps.py <directory> [options]

Examples:
    # Decode all experiments in a parent directory
    python scripts/decode_steps.py outputs/ParentDir/
    
    # Decode single experiment
    python scripts/decode_steps.py outputs/MyExperiment_20250901_120000/
    
    # Use specific GPUs for latent decoding
    python scripts/decode_steps.py outputs/ParentDir/ --gpus 0,1
    
    # Control parallel workers for attention decoding
    python scripts/decode_steps.py outputs/ParentDir/ --attention-workers 4
    
    # Skip attention decoding (latents only)
    python scripts/decode_steps.py outputs/ParentDir/ --latents-only
    
    # Skip latent decoding (attention only)
    python scripts/decode_steps.py outputs/ParentDir/ --attention-only
    
    # Pass through parameters to underlying scripts
    python scripts/decode_steps.py outputs/ParentDir/ --quality 5.0 --overlay-alpha 0.7

Requirements:
    - decode_latent_steps.py and decode_attention_steps.py in same directory
    - All dependencies of the underlying decoder scripts
"""

import argparse
import logging
import sys
import time
import subprocess
import json
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# Add progress bar support
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Install with 'pip install tqdm' for progress bars.")


@dataclass
class ExperimentInfo:
    """Information about a discovered experiment."""
    path: Path
    has_latents: bool
    has_attention: bool
    has_config: bool
    latent_count: int = 0
    attention_count: int = 0
    
    def is_valid(self) -> bool:
        """Check if experiment has minimum required structure."""
        return self.has_config and (self.has_latents or self.has_attention)


@dataclass
class DecodeJobResult:
    """Result of a decode job."""
    experiment_path: Path
    job_type: str  # 'latent' or 'attention'
    success: bool
    duration: float
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    error_message: str = ""


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


def find_experiments(root_dir: Path, recursive: bool = True) -> List[ExperimentInfo]:
    """
    Find all experiment directories in the given path.
    
    An experiment directory is identified by having:
    - configs/generation_config.yaml (required)
    - latents/ directory (optional)
    - attention_maps/ directory (optional)
    
    Args:
        root_dir: Root directory to search
        recursive: If True, search recursively for experiments
        
    Returns:
        List of ExperimentInfo objects
    """
    experiments = []
    
    def check_directory(path: Path) -> Optional[ExperimentInfo]:
        """Check if a directory is an experiment."""
        config_path = path / "configs" / "generation_config.yaml"
        latents_path = path / "latents"
        attention_path = path / "attention_maps"
        
        if not config_path.exists():
            return None
        
        # Count latent and attention items
        latent_count = 0
        attention_count = 0
        
        if latents_path.exists():
            # Count video directories in latents
            for prompt_dir in latents_path.iterdir():
                if prompt_dir.is_dir():
                    for video_dir in prompt_dir.iterdir():
                        if video_dir.is_dir():
                            latent_count += 1
        
        if attention_path.exists():
            # Count token directories in attention_maps
            for prompt_dir in attention_path.iterdir():
                if prompt_dir.is_dir():
                    for video_dir in prompt_dir.iterdir():
                        if video_dir.is_dir():
                            for token_dir in video_dir.iterdir():
                                if token_dir.is_dir() and token_dir.name.startswith('token_'):
                                    attention_count += 1
        
        return ExperimentInfo(
            path=path,
            has_latents=latents_path.exists() and latent_count > 0,
            has_attention=attention_path.exists() and attention_count > 0,
            has_config=config_path.exists(),
            latent_count=latent_count,
            attention_count=attention_count
        )
    
    # Check if root_dir itself is an experiment
    exp_info = check_directory(root_dir)
    if exp_info and exp_info.is_valid():
        experiments.append(exp_info)
        return experiments
    
    # Search for experiments
    if recursive:
        # Recursively search all subdirectories
        for subdir in root_dir.rglob("*"):
            if subdir.is_dir():
                exp_info = check_directory(subdir)
                if exp_info and exp_info.is_valid():
                    experiments.append(exp_info)
    else:
        # Only search immediate subdirectories
        for subdir in root_dir.iterdir():
            if subdir.is_dir():
                exp_info = check_directory(subdir)
                if exp_info and exp_info.is_valid():
                    experiments.append(exp_info)
    
    return experiments


def get_available_gpus() -> List[int]:
    """Detect available CUDA GPUs."""
    try:
        import torch
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        return []
    except ImportError:
        logging.warning("PyTorch not available, cannot detect GPUs")
        return []


def parse_gpu_list(gpu_str: str) -> List[int]:
    """Parse comma-separated GPU list."""
    if not gpu_str:
        return []
    try:
        return [int(gpu.strip()) for gpu in gpu_str.split(',')]
    except ValueError:
        logging.error(f"Invalid GPU list: {gpu_str}")
        return []


def get_video_directories(experiment_path: Path) -> List[str]:
    """
    Get list of video directory paths for an experiment.
    Returns relative paths like "prompt_000/vid_001"
    """
    latents_path = experiment_path / "latents"
    if not latents_path.exists():
        return []
    
    video_dirs = []
    for prompt_dir in latents_path.iterdir():
        if prompt_dir.is_dir() and prompt_dir.name.startswith('prompt_'):
            for video_dir in prompt_dir.iterdir():
                if video_dir.is_dir():
                    # Store as "prompt_XXX/vid_XXX" format
                    rel_path = f"{prompt_dir.name}/{video_dir.name}"
                    video_dirs.append(rel_path)
    
    return sorted(video_dirs)


def distribute_video_dirs_across_gpus(video_dirs: List[str], gpus: List[int]) -> Dict[int, List[str]]:
    """
    Distribute video directories across GPUs for balanced workload.
    
    Args:
        video_dirs: List of video directory paths
        gpus: List of GPU indices
        
    Returns:
        Dictionary mapping GPU index to list of video directories
    """
    if not gpus:
        return {}
    
    distribution = {gpu: [] for gpu in gpus}
    
    # Round-robin distribution
    for idx, video_dir in enumerate(video_dirs):
        gpu_idx = gpus[idx % len(gpus)]
        distribution[gpu_idx].append(video_dir)
    
    return distribution


def run_latent_decode(
    experiment_path: Path,
    gpu: int,
    prompt_filter: Optional[str] = None,
    video_filter: Optional[str] = None,
    script_args: Dict = None
) -> DecodeJobResult:
    """
    Run latent decoding for an experiment on a specific GPU.
    
    Args:
        experiment_path: Path to experiment directory
        gpu: GPU index to use
        prompt_filter: Optional filter for prompt directories (e.g., 'prompt_000,prompt_001')
        video_filter: Optional filter for video directories (e.g., 'vid_001,vid_002')
        script_args: Additional arguments to pass to decode script
        
    Returns:
        DecodeJobResult with execution details
    """
    script_path = Path(__file__).parent / "decode_latent_steps.py"
    
    if not script_path.exists():
        return DecodeJobResult(
            experiment_path=experiment_path,
            job_type='latent',
            success=False,
            duration=0.0,
            error_message=f"Script not found: {script_path}"
        )
    
    # Build command
    cmd = [
        sys.executable,
        str(script_path),
        str(experiment_path),
        f"--device=cuda:{gpu}"
    ]
    
    # Add prompt filter if specified
    if prompt_filter:
        cmd.append(f"--prompt-filter={prompt_filter}")
    
    # Add video filter if specified
    if video_filter:
        cmd.append(f"--video-filter={video_filter}")
    
    # Add pass-through arguments
    if script_args:
        for key, value in script_args.items():
            if value is not None:
                if isinstance(value, bool):
                    if value:
                        cmd.append(f"--{key}")
                else:
                    cmd.append(f"--{key}={value}")
    
    # Log the command for debugging
    logging.debug(f"GPU {gpu} command: {' '.join(cmd)}")
    
    # Execute with real-time output streaming
    start_time = time.time()
    process = None
    stdout_lines = []
    stderr_lines = []
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout for real-time streaming
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Stream output in real-time
        logging.info(f"GPU {gpu} starting decode for {experiment_path.name}...")
        for line in process.stdout:
            line = line.rstrip()
            if line:
                # Log important lines, store all lines
                stdout_lines.append(line)
                # Show progress lines and important messages
                # Be more permissive - show anything with progress indicators, key messages, or non-INFO lines
                if any(keyword in line for keyword in ['Processing', 'Decoding', 'ERROR', 'WARNING', 'Successful', 'Failed', 'Total', 'video/s', 'step/s', '%', '|']):
                    logging.info(f"  [GPU {gpu}] {line}")
                # Also log DEBUG and above level messages
                elif any(level in line for level in ['DEBUG:', 'ERROR:', 'WARNING:', 'CRITICAL:']):
                    logging.info(f"  [GPU {gpu}] {line}")
        
        process.wait(timeout=3600 * 4)  # 4 hour timeout
        duration = time.time() - start_time
        
        stdout = '\n'.join(stdout_lines)
        stderr = '\n'.join(stderr_lines)
        
        return DecodeJobResult(
            experiment_path=experiment_path,
            job_type='latent',
            success=process.returncode == 0,
            duration=duration,
            stdout=stdout,
            stderr=stderr,
            return_code=process.returncode
        )
        
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        if process:
            process.kill()
            process.wait()
        logging.error(f"GPU {gpu} timeout after 4 hours")
        return DecodeJobResult(
            experiment_path=experiment_path,
            job_type='latent',
            success=False,
            duration=duration,
            error_message="Timeout after 4 hours"
        )
    except KeyboardInterrupt:
        # Ensure process is killed on interrupt
        if process:
            logging.warning(f"Killing latent decode process for {experiment_path.name} on GPU {gpu}")
            process.kill()
            process.wait()
        raise
    except Exception as e:
        duration = time.time() - start_time
        logging.error(f"GPU {gpu} exception: {e}")
        if process:
            process.kill()
            process.wait()
        return DecodeJobResult(
            experiment_path=experiment_path,
            job_type='latent',
            success=False,
            duration=duration,
            error_message=str(e)
        )


def run_attention_decode(
    experiment_path: Path,
    script_args: Dict = None
) -> DecodeJobResult:
    """
    Run attention decoding for an experiment.
    
    Args:
        experiment_path: Path to experiment directory
        script_args: Additional arguments to pass to decode script
        
    Returns:
        DecodeJobResult with execution details
    """
    script_path = Path(__file__).parent / "decode_attention_steps.py"
    
    if not script_path.exists():
        return DecodeJobResult(
            experiment_path=experiment_path,
            job_type='attention',
            success=False,
            duration=0.0,
            error_message=f"Script not found: {script_path}"
        )
    
    # Build command
    cmd = [
        sys.executable,
        str(script_path),
        str(experiment_path)
    ]
    
    # Add pass-through arguments
    if script_args:
        for key, value in script_args.items():
            if value is not None:
                if isinstance(value, bool):
                    if value:
                        cmd.append(f"--{key}")
                else:
                    cmd.append(f"--{key}={value}")
    
    # Execute
    start_time = time.time()
    process = None
    stdout_lines = []
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Stream output in real-time
        logging.info(f"Starting attention decode for {experiment_path.name}...")
        for line in process.stdout:
            line = line.rstrip()
            if line:
                stdout_lines.append(line)
                # Show progress lines and important messages
                if any(keyword in line for keyword in ['Processing', 'Decoding', 'ERROR', 'WARNING', 'Successful', 'Failed', 'Total', 'video/s', 'step/s', 'token/s', '%', '|']):
                    logging.info(f"  [ATTN] {line}")
                elif any(level in line for level in ['DEBUG:', 'ERROR:', 'WARNING:', 'CRITICAL:']):
                    logging.info(f"  [ATTN] {line}")
        
        process.wait(timeout=3600 * 4)  # 4 hour timeout
        duration = time.time() - start_time
        
        stdout = '\n'.join(stdout_lines)
        
        return DecodeJobResult(
            experiment_path=experiment_path,
            job_type='attention',
            success=process.returncode == 0,
            duration=duration,
            stdout=stdout,
            stderr='',
            return_code=process.returncode
        )
        
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        if process:
            process.kill()
            process.wait()
        logging.error(f"Attention decode timeout after 4 hours for {experiment_path.name}")
        return DecodeJobResult(
            experiment_path=experiment_path,
            job_type='attention',
            success=False,
            duration=duration,
            error_message="Timeout after 4 hours"
        )
    except KeyboardInterrupt:
        # Ensure process is killed on interrupt
        if process:
            logging.warning(f"Killing attention decode process for {experiment_path.name}")
            process.kill()
            process.wait()
        raise
    except Exception as e:
        duration = time.time() - start_time
        logging.error(f"Attention decode exception for {experiment_path.name}: {e}")
        if process:
            process.kill()
            process.wait()
        return DecodeJobResult(
            experiment_path=experiment_path,
            job_type='attention',
            success=False,
            duration=duration,
            error_message=str(e)
        )


def decode_experiment_latents_parallel(
    experiment: ExperimentInfo,
    gpus: List[int],
    script_args: Dict
) -> List[DecodeJobResult]:
    """
    Decode latents for a single experiment across multiple GPUs.
    
    Args:
        experiment: Experiment information
        gpus: List of GPU indices to use
        script_args: Arguments to pass to decode script
        
    Returns:
        List of DecodeJobResults (one per GPU used)
    """
    if not gpus:
        logging.warning(f"No GPUs available for latent decoding: {experiment.path}")
        return []
    
    # Get video directories for this experiment
    video_dirs = get_video_directories(experiment.path)
    
    if not video_dirs:
        logging.warning(f"No video directories found in {experiment.path}")
        return []
    
    # Distribute across GPUs
    distribution = distribute_video_dirs_across_gpus(video_dirs, gpus)
    
    logging.info(f"Distributing {len(video_dirs)} video directories across {len(gpus)} GPUs for {experiment.path.name}")
    for gpu, dirs in distribution.items():
        logging.info(f"  GPU {gpu}: {len(dirs)} directories - {dirs[:3]}{'...' if len(dirs) > 3 else ''}")
    
    # Run parallel decode jobs
    results = []
    
    # Use ThreadPoolExecutor since we're just spawning subprocesses
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {}
        
        for gpu, dirs in distribution.items():
            if not dirs:
                continue
            
            # Extract unique prompt directories and video IDs
            # dirs is list like ["prompt_000/vid_001", "prompt_000/vid_002", "prompt_001/vid_000"]
            prompts = set()
            videos = set()
            for path_str in dirs:
                parts = path_str.split('/')
                if len(parts) == 2:
                    prompts.add(parts[0])
                    videos.add(parts[1])
            
            # Create filter strings (comma-separated)
            prompt_filter = ','.join(sorted(prompts)) if prompts else None
            video_filter = ','.join(sorted(videos)) if videos else None
            
            future = executor.submit(
                run_latent_decode,
                experiment.path,
                gpu,
                prompt_filter,
                video_filter,
                script_args
            )
            futures[future] = gpu
        
        # Collect results with progress tracking
        if TQDM_AVAILABLE:
            pbar = tqdm(total=len(futures), desc=f"Latent decode {experiment.path.name}", unit="GPU")
        
        for future in as_completed(futures):
            gpu = futures[future]
            try:
                result = future.result()
                results.append(result)
                
                if result.success:
                    # Extract key metrics from output
                    steps_processed = 0
                    for line in result.stdout.split('\n') if result.stdout else []:
                        if 'Total steps processed:' in line:
                            try:
                                steps_processed = int(line.split(':')[1].strip())
                            except:
                                pass
                    
                    logging.info(f"✅ GPU {gpu} completed latent decoding for {experiment.path.name} in {result.duration:.1f}s ({steps_processed} steps)")
                else:
                    error_detail = result.error_message or result.stderr or "Unknown error"
                    logging.error(f"❌ GPU {gpu} failed latent decoding for {experiment.path.name}:")
                    if result.stderr:
                        logging.error(f"   stderr: {result.stderr}")
                    if result.error_message:
                        logging.error(f"   error: {result.error_message}")
                    logging.error(f"   return_code: {result.return_code}")
                    
            except Exception as e:
                logging.error(f"❌ Exception on GPU {gpu} for {experiment.path.name}: {e}")
                results.append(DecodeJobResult(
                    experiment_path=experiment.path,
                    job_type='latent',
                    success=False,
                    duration=0.0,
                    error_message=str(e)
                ))
            
            if TQDM_AVAILABLE:
                pbar.update(1)
        
        if TQDM_AVAILABLE:
            pbar.close()
    
    return results


def decode_all_experiments_latents(
    experiments: List[ExperimentInfo],
    gpus: List[int],
    script_args: Dict
) -> Dict[Path, List[DecodeJobResult]]:
    """
    Decode latents for all experiments sequentially, but with parallel GPU processing within each.
    
    Args:
        experiments: List of experiments to process
        gpus: List of GPU indices to use
        script_args: Arguments to pass to decode scripts
        
    Returns:
        Dictionary mapping experiment path to list of results
    """
    all_results = {}
    
    # Filter experiments that have latents
    experiments_with_latents = [exp for exp in experiments if exp.has_latents]
    
    logging.info(f"Processing latent decoding for {len(experiments_with_latents)} experiments")
    
    for exp in experiments_with_latents:
        logging.info(f"Starting latent decode for experiment: {exp.path}")
        results = decode_experiment_latents_parallel(exp, gpus, script_args)
        all_results[exp.path] = results
    
    return all_results


def decode_all_experiments_attention(
    experiments: List[ExperimentInfo],
    num_workers: int,
    script_args: Dict
) -> Dict[Path, DecodeJobResult]:
    """
    Decode attention maps for all experiments in parallel using process pool.
    
    Args:
        experiments: List of experiments to process
        num_workers: Number of parallel workers
        script_args: Arguments to pass to decode scripts
        
    Returns:
        Dictionary mapping experiment path to result
    """
    all_results = {}
    
    # Filter experiments that have attention maps
    experiments_with_attention = [exp for exp in experiments if exp.has_attention]
    
    logging.info(f"Processing attention decoding for {len(experiments_with_attention)} experiments with {num_workers} workers")
    
    # Use ProcessPoolExecutor for CPU parallelization
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = {}
    pbar = None
    
    try:
        for exp in experiments_with_attention:
            future = executor.submit(run_attention_decode, exp.path, script_args)
            futures[future] = exp.path
        
        # Collect results with progress tracking
        if TQDM_AVAILABLE:
            pbar = tqdm(total=len(futures), desc="Attention decode", unit="exp")
        
        for future in as_completed(futures):
            exp_path = futures[future]
            try:
                result = future.result()
                all_results[exp_path] = result
                
                if result.success:
                    logging.info(f"✅ Completed attention decoding for {exp_path.name} in {result.duration:.1f}s")
                else:
                    error_detail = result.error_message or result.stderr or "Unknown error"
                    logging.error(f"❌ Failed attention decoding for {exp_path.name}:")
                    if result.stderr:
                        logging.error(f"   stderr: {result.stderr}")
                    if result.error_message:
                        logging.error(f"   error: {result.error_message}")
                    logging.error(f"   return_code: {result.return_code}")
                    
            except Exception as e:
                logging.error(f"❌ Exception decoding {exp_path.name}: {e}")
                all_results[exp_path] = DecodeJobResult(
                    experiment_path=exp_path,
                    job_type='attention',
                    success=False,
                    duration=0.0,
                    error_message=str(e)
                )
            
            if TQDM_AVAILABLE:
                pbar.update(1)
    
    except KeyboardInterrupt:
        logging.warning("Keyboard interrupt detected - terminating attention decoding workers...")
        # Cancel pending futures
        for future in futures:
            future.cancel()
        raise
    
    finally:
        if TQDM_AVAILABLE and pbar:
            pbar.close()
        # Shutdown executor and kill any running processes
        executor.shutdown(wait=False, cancel_futures=True)
    
    return all_results


def create_summary_report(
    experiments: List[ExperimentInfo],
    latent_results: Dict[Path, List[DecodeJobResult]],
    attention_results: Dict[Path, DecodeJobResult],
    output_path: Path
):
    """Create comprehensive summary report."""
    
    summary = {
        'total_experiments': len(experiments),
        'experiments_with_latents': sum(1 for exp in experiments if exp.has_latents),
        'experiments_with_attention': sum(1 for exp in experiments if exp.has_attention),
        'latent_decoding': {
            'total_jobs': sum(len(results) for results in latent_results.values()),
            'successful_jobs': sum(1 for results in latent_results.values() for r in results if r.success),
            'failed_jobs': sum(1 for results in latent_results.values() for r in results if not r.success),
            'total_duration': sum(r.duration for results in latent_results.values() for r in results)
        },
        'attention_decoding': {
            'total_jobs': len(attention_results),
            'successful_jobs': sum(1 for r in attention_results.values() if r.success),
            'failed_jobs': sum(1 for r in attention_results.values() if not r.success),
            'total_duration': sum(r.duration for r in attention_results.values())
        },
        'experiments': []
    }
    
    # Per-experiment details
    for exp in experiments:
        exp_data = {
            'path': str(exp.path),
            'has_latents': exp.has_latents,
            'has_attention': exp.has_attention,
            'latent_count': exp.latent_count,
            'attention_count': exp.attention_count,
            'latent_results': [],
            'attention_result': None
        }
        
        # Add latent results
        if exp.path in latent_results:
            for result in latent_results[exp.path]:
                exp_data['latent_results'].append({
                    'success': result.success,
                    'duration': result.duration,
                    'return_code': result.return_code,
                    'error_message': result.error_message
                })
        
        # Add attention result
        if exp.path in attention_results:
            result = attention_results[exp.path]
            exp_data['attention_result'] = {
                'success': result.success,
                'duration': result.duration,
                'return_code': result.return_code,
                'error_message': result.error_message
            }
        
        summary['experiments'].append(exp_data)
    
    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Meta-script for decoding latent and attention steps across multiple experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing experiments (can be parent dir or single experiment)"
    )
    
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively search for experiments (default: True)"
    )
    
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive search"
    )
    
    # GPU/parallelization options
    parser.add_argument(
        "--gpus",
        help="Comma-separated list of GPU indices to use for latent decoding (default: auto-detect all)"
    )
    
    parser.add_argument(
        "--attention-workers",
        type=int,
        default=16,
        help="Number of parallel workers for attention decoding (default: CPU count)"
    )
    
    # Processing mode
    parser.add_argument(
        "--latents-only",
        action="store_true",
        help="Only decode latents, skip attention maps"
    )
    
    parser.add_argument(
        "--attention-only",
        action="store_true",
        help="Only decode attention maps, skip latents"
    )
    
    # Pass-through arguments for latent decoding
    parser.add_argument(
        "--quality",
        type=float,
        help="Video quality (0-10, lower = smaller file) [pass-through to decoders]"
    )
    
    parser.add_argument(
        "--scale",
        type=float,
        help="Resolution scale factor [pass-through to decoders]"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        help="Output video FPS [pass-through to decoders]"
    )
    
    parser.add_argument(
        "--compress-preset",
        choices=["high-quality", "balanced", "small-file", "tiny"],
        help="Compression preset [pass-through to latent decoder]"
    )
    
    # Pass-through arguments for attention decoding
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        help="Overlay alpha/transparency [pass-through to attention decoder]"
    )
    
    parser.add_argument(
        "--colormap",
        choices=["viridis", "plasma", "inferno", "magma", "jet", "hot", "cool", "turbo"],
        help="Color scheme for attention visualization [pass-through to attention decoder]"
    )
    
    parser.add_argument(
        "--thumbnail-frame",
        type=float,
        help="Frame position for thumbnail (0.0-1.0) [pass-through to attention decoder]"
    )
    
    # Filters (apply to both decoders)
    parser.add_argument(
        "--prompt-filter",
        help="Filter for prompt directories [pass-through to decoders]"
    )
    
    parser.add_argument(
        "--step-filter",
        help="Filter for step files [pass-through to decoders]"
    )
    
    parser.add_argument(
        "--token-filter",
        help="Filter for token directories [pass-through to attention decoder]"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Base output directory for all decoded videos"
    )
    
    parser.add_argument(
        "--report-path",
        type=Path,
        help="Path for summary report (default: <directory>/decode_steps_summary.json)"
    )
    
    # General options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually decoding"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate directory
    if not args.directory.exists():
        logging.error(f"Directory not found: {args.directory}")
        return 1
    
    # Determine recursive mode
    recursive = args.recursive and not args.no_recursive
    
    # Find experiments
    logging.info(f"Searching for experiments in: {args.directory} (recursive={recursive})")
    experiments = find_experiments(args.directory, recursive=recursive)
    
    if not experiments:
        logging.error(f"No valid experiments found in {args.directory}")
        return 1
    
    logging.info(f"Found {len(experiments)} experiment(s):")
    for exp in experiments:
        logging.info(f"  {exp.path}")
        logging.info(f"    Latents: {exp.latent_count if exp.has_latents else 'none'}")
        logging.info(f"    Attention: {exp.attention_count if exp.has_attention else 'none'}")
    
    # Determine processing mode
    process_latents = not args.attention_only
    process_attention = not args.latents_only
    
    # Setup GPUs for latent decoding
    gpus = []
    if process_latents:
        if args.gpus:
            gpus = parse_gpu_list(args.gpus)
        else:
            gpus = get_available_gpus()
        
        if not gpus:
            logging.warning("No GPUs available for latent decoding. Latent decoding will be skipped.")
            process_latents = False
        else:
            logging.info(f"Using GPUs for latent decoding: {gpus}")
    
    # Setup workers for attention decoding
    attention_workers = args.attention_workers or mp.cpu_count()
    if process_attention:
        logging.info(f"Using {attention_workers} workers for attention decoding")
    
    # Build script arguments for pass-through
    latent_script_args = {}
    attention_script_args = {}
    
    # Common arguments
    if args.quality is not None:
        latent_script_args['quality'] = args.quality
        attention_script_args['quality'] = args.quality
    
    if args.scale is not None:
        latent_script_args['scale'] = args.scale
        attention_script_args['scale'] = args.scale
    
    if args.fps is not None:
        latent_script_args['fps'] = args.fps
        attention_script_args['fps'] = args.fps
    
    if args.prompt_filter:
        latent_script_args['prompt-filter'] = args.prompt_filter
        attention_script_args['prompt-filter'] = args.prompt_filter
    
    if args.step_filter:
        latent_script_args['step-filter'] = args.step_filter
        attention_script_args['step-filter'] = args.step_filter
    
    # Latent-specific arguments
    if args.compress_preset:
        latent_script_args['compress-preset'] = args.compress_preset
    
    # Attention-specific arguments
    if args.overlay_alpha is not None:
        attention_script_args['overlay-alpha'] = args.overlay_alpha
    
    if args.colormap:
        attention_script_args['colormap'] = args.colormap
    
    if args.token_filter:
        attention_script_args['token-filter'] = args.token_filter
    
    if args.thumbnail_frame is not None:
        attention_script_args['thumbnail-frame'] = args.thumbnail_frame
    
    if args.verbose:
        latent_script_args['verbose'] = True
        attention_script_args['verbose'] = True
    
    # Dry run mode
    if args.dry_run:
        logging.info("=" * 60)
        logging.info("DRY RUN MODE - No actual decoding will be performed")
        logging.info("=" * 60)
        
        if process_latents:
            logging.info(f"\nWould decode latents for {sum(1 for e in experiments if e.has_latents)} experiments")
            logging.info(f"GPU distribution: {gpus}")
            logging.info(f"Latent script args: {latent_script_args}")
        
        if process_attention:
            logging.info(f"\nWould decode attention for {sum(1 for e in experiments if e.has_attention)} experiments")
            logging.info(f"Workers: {attention_workers}")
            logging.info(f"Attention script args: {attention_script_args}")
        
        return 0
    
    # Execute decoding pipeline
    total_start_time = time.time()
    
    latent_results = {}
    attention_results = {}
    
    try:
        # Step 1: Decode latents (if enabled)
        if process_latents:
            logging.info("=" * 60)
            logging.info("STEP 1: LATENT DECODING")
            logging.info("=" * 60)
            latent_start = time.time()
            latent_results = decode_all_experiments_latents(experiments, gpus, latent_script_args)
            latent_duration = time.time() - latent_start
            logging.info(f"Latent decoding completed in {latent_duration:.1f}s")
        
        # Step 2: Decode attention (if enabled)
        if process_attention:
            logging.info("=" * 60)
            logging.info("STEP 2: ATTENTION DECODING")
            logging.info("=" * 60)
            attention_start = time.time()
            attention_results = decode_all_experiments_attention(experiments, attention_workers, attention_script_args)
            attention_duration = time.time() - attention_start
            logging.info(f"Attention decoding completed in {attention_duration:.1f}s")
        
        total_duration = time.time() - total_start_time
        
        # Generate summary report
        report_path = args.report_path or (args.directory / "decode_steps_summary.json")
        summary = create_summary_report(experiments, latent_results, attention_results, report_path)
        
        # Print final summary
        logging.info("=" * 60)
        logging.info("DECODING COMPLETE")
        logging.info("=" * 60)
        logging.info(f"Total experiments: {summary['total_experiments']}")
        logging.info(f"Total time: {total_duration:.1f}s")
        
        if process_latents:
            logging.info(f"\nLatent Decoding:")
            logging.info(f"  Jobs: {summary['latent_decoding']['total_jobs']}")
            logging.info(f"  Successful: {summary['latent_decoding']['successful_jobs']}")
            logging.info(f"  Failed: {summary['latent_decoding']['failed_jobs']}")
            logging.info(f"  Duration: {summary['latent_decoding']['total_duration']:.1f}s")
        
        if process_attention:
            logging.info(f"\nAttention Decoding:")
            logging.info(f"  Jobs: {summary['attention_decoding']['total_jobs']}")
            logging.info(f"  Successful: {summary['attention_decoding']['successful_jobs']}")
            logging.info(f"  Failed: {summary['attention_decoding']['failed_jobs']}")
            logging.info(f"  Duration: {summary['attention_decoding']['total_duration']:.1f}s")
        
        logging.info(f"\nSummary report: {report_path}")
        
        # Determine exit code
        total_failures = (
            summary['latent_decoding']['failed_jobs'] +
            summary['attention_decoding']['failed_jobs']
        )
        
        return 0 if total_failures == 0 else 1
        
    except KeyboardInterrupt:
        logging.info("\nInterrupted by user")
        return 130
    
    except Exception as e:
        logging.error(f"Error during decoding pipeline: {e}")
        if args.verbose:
            import traceback
            logging.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())