"""
Utilities for extracting metadata and information from batch directories.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
from .config_utils import load_generation_config, extract_model_metadata, extract_video_metadata, get_videos_per_variation
from .prompt_utils import load_prompt_template, load_prompt_metadata


def get_batch_name(batch_path: str) -> str:
    """Extract batch name from the path.
    
    Args:
        batch_path: Path to the batch directory
        
    Returns:
        Batch name (directory name)
    """
    return Path(batch_path).name


def get_prompt_groups(batch_path: str) -> List[str]:
    """Get all prompt groups from the videos directory.
    
    Args:
        batch_path: Path to the batch directory
        
    Returns:
        Sorted list of prompt group names (e.g., ['prompt_000', 'prompt_001'])
    """
    videos_dir = Path(batch_path) / "videos"
    if not videos_dir.exists():
        return []
    
    prompt_dirs = [d.name for d in videos_dir.iterdir() if d.is_dir() and d.name.startswith('prompt_')]
    return sorted(prompt_dirs)


def get_video_variations(batch_path: str, prompt_group: str) -> List[str]:
    """Get all video variations for a specific prompt group.
    
    Args:
        batch_path: Path to the batch directory
        prompt_group: Name of the prompt group (e.g., 'prompt_000')
        
    Returns:
        Sorted list of video file names (e.g., ['video_001.jpg', 'video_002.jpg'])
    """
    prompt_dir = Path(batch_path) / "videos" / prompt_group
    if not prompt_dir.exists():
        return []
    
    video_files = [f.name for f in prompt_dir.iterdir() if f.name.endswith('.jpg') and f.name.startswith('video_')]
    return sorted(video_files)


def calculate_seed_for_video(base_seed: int, video_number: int) -> int:
    """Calculate the seed for a specific video variation.
    
    Args:
        base_seed: Base seed from configuration
        video_number: Video number (1-based, e.g., video_001 = 1)
        
    Returns:
        Calculated seed for the video
    """
    return base_seed + video_number - 1


def extract_batch_metadata(batch_path: str) -> Dict[str, Any]:
    """Extract comprehensive metadata from a batch directory.
    
    Args:
        batch_path: Path to the batch directory
        
    Returns:
        Dictionary containing all extracted metadata
    """
    logger = logging.getLogger(__name__)
    
    # Load generation config
    config = load_generation_config(batch_path)
    
    # Extract metadata
    batch_name = get_batch_name(batch_path)
    prompt_template = load_prompt_template(batch_path)
    prompt_groups = get_prompt_groups(batch_path)
    model_metadata = extract_model_metadata(config)
    video_metadata = extract_video_metadata(config)
    videos_per_variation = get_videos_per_variation(config)
    
    # Load prompt metadata
    if prompt_groups:
        prompt_metadata = load_prompt_metadata(batch_path, prompt_groups)
    else:
        prompt_metadata = [], {}
    
    return {
        'batch_name': batch_name,
        'batch_path': batch_path,
        'prompt_template': prompt_template,
        'prompt_groups': prompt_groups,
        'prompt_descriptions': prompt_metadata.get('prompt_descriptions', []),
        'prompt_metadata': prompt_metadata,
        'model_metadata': model_metadata,
        'video_metadata': video_metadata,
        'videos_per_variation': videos_per_variation,
        'config': config
    }
