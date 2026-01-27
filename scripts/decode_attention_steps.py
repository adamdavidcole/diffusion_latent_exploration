#!/usr/bin/env python3
"""
Attention Step Decoder Script

Decode attention map steps to videos with optional latent video overlays.

This script processes attention map data from diffusion model experiments and generates
video files showing attention patterns over time. Each step corresponds to a diffusion timestep,
with accompanying thumbnail images (frame position configurable).

The script processes attention maps from the attention_maps/ directory and overlays them
on corresponding latent videos from latents_videos/ directory. If latent videos are not 
available, it generates pure attention map videos.

Usage:
    python scripts/decode_attention_steps.py <experiment_dir> [options]

Examples:
    # Decode all attention maps in an experiment
    python scripts/decode_attention_steps.py outputs/MyExperiment_20250901_120000/
    
    # Decode only specific prompt/video/token
    python scripts/decode_attention_steps.py outputs/MyExperiment_20250901_120000/ --prompt-filter prompt_000 --video-filter vid_001 --token-filter car
    
    # Decode only specific steps
    python scripts/decode_attention_steps.py outputs/MyExperiment_20250901_120000/ --step-filter step_000,step_010,step_019
    
    # Use specific overlay settings
    python scripts/decode_attention_steps.py outputs/MyExperiment_20250901_120000/ --overlay-alpha 0.7 --colormap hot

Requirements:
    - ffmpeg (for thumbnail generation) - optional but recommended
"""

import argparse
import logging
import sys
import time
import subprocess
import json
import gzip
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add progress bar support
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Install with 'pip install tqdm' for progress bars.")

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.visualization.attention_visualizer import AttentionVisualizer, VideoConfig, OverlayConfig, ColorMap
from src.visualization.attention_analyzer import AttentionAnalyzer


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


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available for thumbnail generation."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def generate_thumbnail_for_video(video_path: Path, thumbnail_path: Path = None, frame_position: float = 0.5) -> bool:
    """
    Generate a thumbnail from a specific frame of a video using ffmpeg.
    
    Args:
        video_path: Path to the video file
        thumbnail_path: Path where thumbnail should be saved (default: video_path with .jpg extension)
        frame_position: Position in video to extract frame from (0.0=start, 0.5=middle, 1.0=end)
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Skip if ffmpeg is not available
    if not check_ffmpeg():
        return False
        
    if thumbnail_path is None:
        thumbnail_path = video_path.with_suffix('.jpg')
    
    try:
        # Check if video exists
        if not video_path.exists():
            logging.warning(f"Video not found for thumbnail generation: {video_path}")
            return False
        
        # Create thumbnail directory if it doesn't exist
        thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Clamp frame position to valid range [0.0, 1.0]
        frame_position = max(0.0, min(1.0, frame_position))
        
        # Use ffmpeg to extract frame at specified position
        if frame_position == 0.0:
            # For first frame, use select filter for precise frame selection
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-vf', 'select=eq(n\\,0)',     # Select first frame
                '-vframes', '1',               # Extract only 1 frame
                '-q:v', '2',                   # High quality
                '-y',                          # Overwrite output file
                str(thumbnail_path)
            ]
        else:
            # For other positions, first get video duration and calculate seek time in seconds
            # Get video duration using ffprobe
            try:
                duration_cmd = [
                    'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                    '-show_entries', 'format=duration', str(video_path)
                ]
                duration_result = subprocess.run(duration_cmd, capture_output=True, text=True, timeout=10)
                
                if duration_result.returncode == 0:
                    import json
                    duration_data = json.loads(duration_result.stdout)
                    duration_seconds = float(duration_data['format']['duration'])
                    seek_seconds = duration_seconds * frame_position
                    
                    cmd = [
                        'ffmpeg',
                        '-ss', str(seek_seconds),      # Seek to position in seconds
                        '-i', str(video_path),
                        '-vframes', '1',               # Extract only 1 frame
                        '-q:v', '2',                   # High quality
                        '-y',                          # Overwrite output file
                        str(thumbnail_path)
                    ]
                else:
                    # Fallback: use first frame if duration detection fails
                    logging.warning(f"Could not determine video duration for {video_path}, using first frame")
                    cmd = [
                        'ffmpeg',
                        '-i', str(video_path),
                        '-vf', 'select=eq(n\\,0)',     # Select first frame
                        '-vframes', '1',               # Extract only 1 frame
                        '-q:v', '2',                   # High quality
                        '-y',                          # Overwrite output file
                        str(thumbnail_path)
                    ]
            except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError) as e:
                # Fallback: use first frame if duration detection fails
                logging.warning(f"Error determining video duration for {video_path}: {e}, using first frame")
                cmd = [
                    'ffmpeg',
                    '-i', str(video_path),
                    '-vf', 'select=eq(n\\,0)',         # Select first frame
                    '-vframes', '1',                   # Extract only 1 frame
                    '-q:v', '2',                       # High quality
                    '-y',                              # Overwrite output file
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
            logging.debug(f"Generated thumbnail: {thumbnail_path}")
            return True
        else:
            error_msg = result.stderr.strip()
            logging.warning(f"FFmpeg error generating thumbnail for {video_path}: {error_msg}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.warning(f"Timeout generating thumbnail for {video_path}")
        return False
    except Exception as e:
        logging.warning(f"Error generating thumbnail for {video_path}: {e}")
        return False


def find_attention_directories(attention_maps_dir: Path, prompt_filter=None, video_filter=None, token_filter=None) -> List[Tuple[Path, str, str, str]]:
    """
    Find all attention token directories in the attention_maps structure.
    
    Supports multiple directory structures:
    - New nested format: attention_maps/prompt_000/p000_b001_s000/token_car/
    - Old nested format: attention_maps/prompt_000/vid001/token_car/
    - Flat format: attention_maps/p000_b001_s000/token_car/ (legacy, still supported)
    
    Args:
        attention_maps_dir: Path to attention_maps directory
        prompt_filter: Filter for prompt directories  
        video_filter: Filter for video directories (applies to video_id)
        token_filter: Filter for token directories
        
    Returns:
        List of tuples: (token_dir_path, prompt_id, video_id, token_name)
    """
    if not attention_maps_dir.exists():
        return []
    
    token_dirs = []
    
    # Iterate through all prompt directories
    for prompt_dir in attention_maps_dir.iterdir():
        if not prompt_dir.is_dir():
            continue
        
        # Handle prompt_000 directories (new nested and old nested formats)
        if prompt_dir.name.startswith('prompt_'):
            if prompt_filter and prompt_filter not in prompt_dir.name:
                continue
            
            prompt_id = prompt_dir.name
            
            # Look for video subdirectories
            for video_dir in prompt_dir.iterdir():
                if not video_dir.is_dir():
                    continue
                
                video_id = video_dir.name
                
                # Apply video filter
                if video_filter and video_filter not in video_id:
                    continue
                
                # Look for token directories
                for token_dir in video_dir.iterdir():
                    if not token_dir.is_dir() or not token_dir.name.startswith('token_'):
                        continue
                    
                    token_name = token_dir.name.replace('token_', '')
                    if token_filter and token_filter not in token_name:
                        continue
                    
                    token_dirs.append((token_dir, prompt_id, video_id, token_name))
        
        # Handle flat format p000_b001_s000 directories (legacy)
        elif prompt_dir.name.startswith('p') and '_b' in prompt_dir.name and '_s' in prompt_dir.name:
            video_id = prompt_dir.name
            
            # Extract prompt from video_id (p000 part)
            try:
                prompt_part = video_id.split('_')[0]  # p000
                prompt_id = f"prompt_{prompt_part[1:]}"  # prompt_000
            except:
                prompt_id = "prompt_000"  # fallback
            
            # Apply filters
            if prompt_filter and prompt_filter not in prompt_id:
                continue
            if video_filter and video_filter not in video_id:
                continue
            
            # Look for token directories
            for token_dir in prompt_dir.iterdir():
                if not token_dir.is_dir() or not token_dir.name.startswith('token_'):
                    continue
                
                token_name = token_dir.name.replace('token_', '')
                if token_filter and token_filter not in token_name:
                    continue
                
                token_dirs.append((token_dir, prompt_id, video_id, token_name))
    
    return token_dirs


def find_attention_steps(token_dir: Path, step_filter=None) -> List[Path]:
    """
    Find all attention step files in a token directory.
    
    Args:
        token_dir: Path to token directory
        step_filter: Optional filter for step files
        
    Returns:
        List of step file paths
    """
    step_files = []
    
    for file_path in token_dir.iterdir():
        if file_path.is_file() and file_path.name.startswith('step_') and file_path.name.endswith('.npy.gz'):
            # Skip metadata files
            if file_path.name.endswith('_metadata.json'):
                continue
                
            step_name = file_path.name.replace('.npy.gz', '')
            
            if step_filter:
                step_filters = [f.strip() for f in step_filter.split(',')]
                if not any(sf == step_name for sf in step_filters):
                    continue
            
            step_files.append(file_path)
    
    return sorted(step_files)


def get_corresponding_latent_video(experiment_dir: Path, prompt_id: str, video_id: str, step_name: str) -> Optional[Path]:
    """
    Find the corresponding latent video for overlay.
    
    Args:
        experiment_dir: Experiment directory
        prompt_id: Prompt identifier (e.g., 'prompt_000')
        video_id: Video identifier from attention_maps (e.g., 'vid001')
        step_name: Step name (e.g., 'step_003')
        
    Returns:
        Path to latent video if it exists, None otherwise
    """
    # Convert video_id to consistent format for latents_videos lookup
    consistent_video_id = video_id.replace('vid', 'vid_') if not video_id.startswith('vid_') else video_id
    latent_video_path = experiment_dir / "latents_videos" / prompt_id / consistent_video_id / f"{step_name}.mp4"
    return latent_video_path if latent_video_path.exists() else None


class AttentionDecodeResult:
    """Result of decoding a single attention step."""
    def __init__(self):
        self.success = False
        self.error_message = ""
        self.decode_time = 0.0
        self.thumbnail_generated = False
        self.thumbnail_path = None
        self.output_path = None
        self.has_overlay = False


def decode_attention_step_to_video(attention_file: Path, 
                                 output_path: Path,
                                 latent_video_path: Optional[Path] = None,
                                 overlay_alpha: float = 0.6,
                                 colormap: str = "jet",
                                 fps: int = 15,
                                 quality: float = 8.0,
                                 scale_factor: float = 1.0) -> AttentionDecodeResult:
    """
    Decode a single attention step file to video.
    
    Args:
        attention_file: Path to attention .npy.gz file
        output_path: Where to save the video
        latent_video_path: Optional path to latent video for overlay
        overlay_alpha: Overlay transparency (0-1)
        colormap: Color scheme for attention visualization
        fps: Video frame rate
        quality: Video quality (0-10)
        scale_factor: Resolution scaling
        
    Returns:
        AttentionDecodeResult with success status and metadata
    """
    result = AttentionDecodeResult()
    start_time = time.time()
    
    try:
        # Load attention map
        logging.debug(f"Loading attention data from {attention_file}")
        with gzip.open(attention_file, 'rb') as f:
            attention_data = np.load(f)
        logging.debug(f"Loaded attention data shape: {attention_data.shape}")
        
        # Load corresponding metadata
        # Extract step name from filename (e.g., step_000.npy.gz -> step_000)
        step_name = attention_file.name.replace('.npy.gz', '')
        metadata_file = attention_file.parent / f"{step_name}_metadata.json"
        metadata = None
        if metadata_file.exists():
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            logging.debug(f"Loaded metadata: video_width={metadata.get('video_width')}, video_height={metadata.get('video_height')}, video_frames={metadata.get('video_frames')}")
        else:
            logging.warning(f"No metadata file found at {metadata_file}")
        
        # Create video config
        video_config = VideoConfig(fps=fps, quality=int(quality))
        logging.debug(f"Created video config: fps={fps}, quality={quality}")
        
        # Create overlay config
        colormap_enum = ColorMap.JET  # Default
        try:
            colormap_enum = ColorMap(colormap.lower())
        except ValueError:
            logging.warning(f"Unknown colormap '{colormap}', using 'jet'")
        
        overlay_config = OverlayConfig(
            alpha=overlay_alpha,
            colormap=colormap_enum
        )
        logging.debug(f"Created overlay config: alpha={overlay_alpha}, colormap={colormap}")
        
        # Create a temporary AttentionAnalyzer and AttentionVisualizer
        # We need to extract the video_id and token from the file path for compatibility
        token_dir = attention_file.parent
        token_name = token_dir.name.replace('token_', '')
        video_dir = token_dir.parent
        prompt_dir = video_dir.parent
        logging.debug(f"Parsed paths: token={token_name}, video_dir={video_dir.name}, prompt_dir={prompt_dir.name}")
        
        # Construct video_id in format expected by AttentionVisualizer
        video_id = f"{prompt_dir.name}_{video_dir.name}"
        logging.debug(f"Constructed video_id: {video_id}")
        
        # Set up temporary analyzer (we won't actually use it for loading since we already loaded the data)
        logging.debug(f"Creating AttentionAnalyzer with path: {prompt_dir.parent}")
        temp_analyzer = AttentionAnalyzer(prompt_dir.parent)
        
        # Set up visualizer with output directory
        visualizer = AttentionVisualizer(
            analyzer=temp_analyzer,
            output_dir=output_path.parent,
            fps=fps,
            overlay_alpha=overlay_alpha
        )
        
        # Get step number from filename
        step_name = attention_file.stem.replace('.npy', '')
        try:
            step_num = int(step_name.replace('step_', ''))
        except ValueError:
            step_num = 0
        
        # Create a mock spatial attention tensor from the loaded data
        import torch
        if isinstance(attention_data, np.ndarray):
            spatial_attention = torch.from_numpy(attention_data).float()
        else:
            spatial_attention = attention_data
        
        # Generate video using existing attention visualizer logic
        # We'll need to create a custom method since the existing one expects to load from analyzer
        output_video_path = visualizer._generate_attention_video_from_data(
            spatial_attention=spatial_attention,
            video_id=video_id,
            token_word=token_name,
            step=step_num,
            video_config=video_config,
            overlay_config=overlay_config,
            output_filename=output_path.name,
            source_video_path=str(latent_video_path) if latent_video_path else None,
            metadata=metadata
        )
        
        result.success = True
        result.output_path = output_video_path
        result.has_overlay = latent_video_path is not None
        
    except Exception as e:
        result.success = False
        result.error_message = str(e)
        logging.error(f"Failed to decode {attention_file}: {e}")
    
    result.decode_time = time.time() - start_time
    return result


def decode_experiment_with_progress(experiment_dir: Path,
                                  output_dir: Optional[Path] = None,
                                  prompt_filter: Optional[str] = None,
                                  video_filter: Optional[str] = None,
                                  token_filter: Optional[str] = None,
                                  step_filter: Optional[str] = None,
                                  overlay_alpha: float = 0.6,
                                  colormap: str = "jet",
                                  fps: int = 15,
                                  quality: float = 8.0,
                                  scale_factor: float = 1.0,
                                  force: bool = False,
                                  thumbnail_frame: float = 0.5) -> Dict:
    """
    Decode attention maps with progress tracking.
    
    Args:
        experiment_dir: Experiment directory
        output_dir: Output directory for videos (default: experiment_dir/attention_videos/)
        prompt_filter: Filter for prompt directories
        video_filter: Filter for video directories
        token_filter: Filter for token directories
        step_filter: Filter for step files
        overlay_alpha: Overlay transparency
        colormap: Color scheme
        fps: Video frame rate
        quality: Video quality
        scale_factor: Resolution scaling
        force: Force regeneration of existing videos
        
    Returns:
        Nested dictionary with results
    """
    attention_maps_dir = experiment_dir / "attention_maps"
    
    if not attention_maps_dir.exists():
        logging.error(f"Attention maps directory not found: {attention_maps_dir}")
        return {}
    
    # Set up output directory
    if output_dir is None:
        output_dir = experiment_dir / "attention_videos"
    
    # Find all token directories
    token_dirs = find_attention_directories(
        attention_maps_dir, 
        prompt_filter=prompt_filter,
        video_filter=video_filter, 
        token_filter=token_filter
    )
    
    if not token_dirs:
        logging.warning("No token directories found after filtering")
        return {}
    
    # Count total steps for progress tracking
    total_steps = 0
    token_step_counts = {}
    
    for token_dir, prompt_id, video_id, token_name in token_dirs:
        step_files = find_attention_steps(token_dir, step_filter)
        token_key = f"{prompt_id}/{video_id}/{token_name}"
        token_step_counts[token_key] = len(step_files)
        total_steps += len(step_files)
    
    logging.info(f"Decoding {len(token_dirs)} token directories with {total_steps} total steps")
    
    all_results = {}
    
    # Create progress bars
    if TQDM_AVAILABLE:
        token_pbar = tqdm(token_dirs, desc="Tokens", unit="token")
        step_pbar = tqdm(total=total_steps, desc="Steps", unit="step", leave=False)
    else:
        token_pbar = token_dirs
        step_pbar = None
    
    try:
        for token_idx, (token_dir, prompt_id, video_id, token_name) in enumerate(token_pbar):
            # Update progress description
            if TQDM_AVAILABLE:
                token_desc = f"{prompt_id}/{video_id}/token_{token_name}"
                token_pbar.set_description(f"Processing {token_desc}")
            else:
                logging.info(f"Processing token directory ({token_idx + 1}/{len(token_dirs)}): {token_dir}")
            
            try:
                # Get steps for this token
                step_files = find_attention_steps(token_dir, step_filter)
                
                # Convert video_id to consistent format (vid001 -> vid_001)
                consistent_video_id = video_id.replace('vid', 'vid_') if not video_id.startswith('vid_') else video_id
                
                # Set up output directory for this token
                token_output_dir = output_dir / prompt_id / consistent_video_id / f"token_{token_name}"
                token_output_dir.mkdir(parents=True, exist_ok=True)
                
                results = {}
                
                for step_file in step_files:
                    step_name = step_file.name.replace('.npy.gz', '')  # e.g., "step_000"
                    output_path = token_output_dir / f"{step_name}.mp4"
                    thumbnail_path = token_output_dir / f"{step_name}.jpg"
                    
                    # Skip if already exists and not forcing
                    if not force and output_path.exists():
                        result = AttentionDecodeResult()
                        result.success = True
                        result.output_path = str(output_path)
                        result.thumbnail_generated = thumbnail_path.exists()
                        result.thumbnail_path = str(thumbnail_path) if result.thumbnail_generated else None
                        results[step_name] = result
                        
                        if TQDM_AVAILABLE:
                            step_pbar.update(1)
                        continue
                    
                    # Update step progress description
                    if TQDM_AVAILABLE:
                        step_pbar.set_description(f"Decoding {step_name}")
                    
                    # Find corresponding latent video
                    latent_video_path = get_corresponding_latent_video(
                        experiment_dir, prompt_id, video_id, step_name
                    )
                    
                    # Decode attention step
                    result = decode_attention_step_to_video(
                        attention_file=step_file,
                        output_path=output_path,
                        latent_video_path=latent_video_path,
                        overlay_alpha=overlay_alpha,
                        colormap=colormap,
                        fps=fps,
                        quality=quality,
                        scale_factor=scale_factor
                    )
                    
                    # Generate thumbnail if video was successful
                    thumbnail_success = False
                    if result.success and output_path.exists():
                        thumbnail_success = generate_thumbnail_for_video(output_path, thumbnail_path, thumbnail_frame)
                    
                    # Store thumbnail results
                    result.thumbnail_generated = thumbnail_success
                    result.thumbnail_path = str(thumbnail_path) if thumbnail_success else None
                    
                    results[step_name] = result
                    
                    # Log progress
                    if result.success:
                        overlay_status = "🎭" if result.has_overlay else "🎨"
                        thumbnail_status = "📸" if thumbnail_success else "⚠️"
                        if not TQDM_AVAILABLE:
                            print(f"✅ {token_desc}/{step_name} in {result.decode_time:.2f}s {overlay_status}{thumbnail_status}")
                    else:
                        if not TQDM_AVAILABLE:
                            print(f"❌ Failed {token_desc}/{step_name}: {result.error_message}")
                        else:
                            logging.error(f"❌ Failed to decode {step_name}: {result.error_message}")
                    
                    # Update step progress
                    if TQDM_AVAILABLE:
                        step_pbar.update(1)
                
                all_results[f"{prompt_id}/{video_id}/{token_name}"] = results
                
            except Exception as e:
                logging.error(f"Failed to process {token_dir}: {e}")
                all_results[f"{prompt_id}/{video_id}/{token_name}"] = {}
                
                # Update progress for skipped steps
                if TQDM_AVAILABLE:
                    token_key = f"{prompt_id}/{video_id}/{token_name}"
                    step_pbar.update(token_step_counts.get(token_key, 0))
    
    finally:
        # Close progress bars
        if TQDM_AVAILABLE:
            token_pbar.close()
            if step_pbar:
                step_pbar.close()
    
    return all_results


def create_attention_decode_summary_report(results: Dict, report_path: Path) -> Dict:
    """Create summary report for attention decoding results."""
    total_steps = 0
    successful_steps = 0
    failed_steps = 0
    total_decode_time = 0.0
    thumbnail_generated = 0
    overlays_created = 0
    
    for token_path, token_results in results.items():
        for step_name, result in token_results.items():
            total_steps += 1
            if result.success:
                successful_steps += 1
                total_decode_time += result.decode_time
                if result.thumbnail_generated:
                    thumbnail_generated += 1
                if result.has_overlay:
                    overlays_created += 1
            else:
                failed_steps += 1
    
    summary = {
        'total_steps': total_steps,
        'successful_steps': successful_steps,
        'failed_steps': failed_steps,
        'success_rate': successful_steps / total_steps if total_steps > 0 else 0.0,
        'avg_decode_time': total_decode_time / successful_steps if successful_steps > 0 else 0.0,
        'thumbnails_generated': thumbnail_generated,
        'overlays_created': overlays_created,
        'timestamp': time.time()
    }
    
    # Save detailed report
    detailed_report = {
        'summary': summary,
        'results': {token_path: {step: {
            'success': result.success,
            'decode_time': result.decode_time,
            'thumbnail_generated': result.thumbnail_generated,
            'has_overlay': result.has_overlay,
            'error_message': result.error_message if not result.success else None
        } for step, result in token_results.items()} for token_path, token_results in results.items()}
    }
    
    with open(report_path, 'w') as f:
        json.dump(detailed_report, f, indent=2)
    
    return summary


def parse_filter_list(filter_str: str) -> list:
    """Parse comma-separated filter string into list."""
    if not filter_str:
        return []
    return [item.strip() for item in filter_str.split(',')]


def main():
    parser = argparse.ArgumentParser(
        description="Decode attention maps to videos with optional latent video overlays",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Path to experiment directory containing attention_maps/ and optionally latents_videos/"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for decoded videos (default: experiment_dir/attention_videos/)"
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
        "--token-filter",
        help="Filter for token directories (e.g., 'car' or 'car,truck')"
    )
    
    parser.add_argument(
        "--step-filter",
        help="Filter for step files (e.g., 'step_000' or 'step_000,step_010,step_019')"
    )
    
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.6,
        help="Overlay alpha/transparency (0=only latent video, 1=only attention) [default: 0.6]"
    )
    
    parser.add_argument(
        "--colormap",
        choices=["viridis", "plasma", "inferno", "magma", "jet", "hot", "cool", "turbo"],
        default="jet",
        help="Color scheme for attention visualization [default: jet]"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Output video FPS [default: 15]"
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
        "--no-overwrite",
        action="store_true",
        help="Skip existing videos instead of overwriting them (default: overwrite existing videos)"
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
    
    parser.add_argument(
        "--thumbnail-frame",
        type=float,
        default=0.5,
        help="Frame position for thumbnail generation (0.0=first frame, 0.5=middle frame, 1.0=last frame)"
    )
    
    args = parser.parse_args()
    
    # Validate thumbnail frame parameter
    if not (0.0 <= args.thumbnail_frame <= 1.0):
        parser.error("--thumbnail-frame must be between 0.0 and 1.0")
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Check if ffmpeg is available for thumbnail generation
    if not check_ffmpeg():
        logging.warning("ffmpeg is not available. Thumbnails will not be generated.")
        logging.warning("To enable thumbnail generation, install ffmpeg:")
        logging.warning("  Ubuntu/Debian: sudo apt install ffmpeg")
        logging.warning("  macOS: brew install ffmpeg")
        logging.warning("  Windows: Download from https://ffmpeg.org/download.html")
    else:
        logging.info("ffmpeg found - thumbnails will be generated alongside videos")
    
    # Validate experiment directory
    if not args.experiment_dir.exists():
        logging.error(f"Experiment directory not found: {args.experiment_dir}")
        return 1
    
    attention_maps_dir = args.experiment_dir / "attention_maps"
    if not attention_maps_dir.exists():
        logging.error(f"Attention maps directory not found: {attention_maps_dir}")
        return 1
    
    # Check for latents_videos directory
    latents_videos_dir = args.experiment_dir / "latents_videos"
    if latents_videos_dir.exists():
        logging.info("Found latents_videos directory - will create overlay videos")
    else:
        logging.info("No latents_videos directory - will create pure attention videos")
    
    try:
        # Find attention directories
        token_dirs = find_attention_directories(
            attention_maps_dir,
            prompt_filter=args.prompt_filter,
            video_filter=args.video_filter,
            token_filter=args.token_filter
        )
        
        logging.info(f"Found {len(token_dirs)} token directories")
        
        if not token_dirs:
            logging.warning("No attention directories match the specified filters")
            return 1
        
        # Count steps that would be processed
        total_steps = 0
        for token_dir, prompt_id, video_id, token_name in token_dirs:
            step_files = find_attention_steps(token_dir, args.step_filter)
            total_steps += len(step_files)
            
            if args.dry_run:
                logging.info(f"Would decode {len(step_files)} steps from {prompt_id}/{video_id}/token_{token_name}")
        
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
            experiment_dir=args.experiment_dir,
            output_dir=args.output_dir,
            prompt_filter=args.prompt_filter,
            video_filter=args.video_filter,
            token_filter=args.token_filter,
            step_filter=args.step_filter,
            overlay_alpha=args.overlay_alpha,
            colormap=args.colormap,
            fps=args.fps,
            quality=args.quality,
            scale_factor=args.scale,
            force=not args.no_overwrite,  # Invert the logic - force by default
            thumbnail_frame=args.thumbnail_frame
        )
        
        total_time = time.time() - start_time
        
        # Generate summary report
        output_dir = args.output_dir or (args.experiment_dir / "attention_videos")
        report_path = output_dir / "attention_decode_summary.json"
        
        summary = create_attention_decode_summary_report(results, report_path)
        
        # Log final results
        logging.info("=" * 60)
        logging.info("ATTENTION DECODING COMPLETE")
        logging.info("=" * 60)
        logging.info(f"Total steps processed: {summary['total_steps']}")
        logging.info(f"Successful: {summary['successful_steps']}")
        logging.info(f"Failed: {summary['failed_steps']}")
        logging.info(f"Success rate: {summary['success_rate']:.1%}")
        logging.info(f"Overlays created: {summary['overlays_created']}")
        
        if summary['thumbnails_generated'] > 0:
            logging.info(f"Thumbnails generated: {summary['thumbnails_generated']}")
        
        logging.info(f"Total time: {total_time:.2f}s")
        logging.info(f"Average time per step: {summary['avg_decode_time']:.2f}s")
        logging.info(f"Output directory: {output_dir}")
        logging.info(f"Summary report: {report_path}")
        
        # Show any failures
        failed_count = 0
        for token_path, token_results in results.items():
            for step_name, result in token_results.items():
                if not result.success:
                    failed_count += 1
                    if failed_count <= 5:  # Show first 5 failures
                        logging.error(f"Failed: {token_path}/{step_name} - {result.error_message}")
        
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


if __name__ == "__main__":
    sys.exit(main())
