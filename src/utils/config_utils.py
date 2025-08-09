"""
Utilities for loading and processing batch configuration files.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional


def load_generation_config(batch_path: str) -> Dict[str, Any]:
    """Load the generation configuration from batch directory.
    
    Args:
        batch_path: Path to the batch directory
        
    Returns:
        Dictionary containing the generation configuration
    """
    logger = logging.getLogger(__name__)
    
    config_dir = Path(batch_path) / "configs"
    config_file = config_dir / "generation_config.yaml"
    
    if not config_file.exists():
        logger.warning(f"⚠️ No generation_config.yaml found at {config_file}")
        return {}
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config if config else {}
    except Exception as e:
        logger.error(f"❌ Error loading generation config: {e}")
        return {}


def extract_model_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract core model metadata from generation config.
    
    Args:
        config: Generation configuration dictionary
        
    Returns:
        Dictionary with model_id, steps, cfg_scale, sampler, and seed
    """
    model_settings = config.get('model_settings', {})
    
    return {
        'model_id': model_settings.get('model_id', 'Unknown'),
        'steps': model_settings.get('steps', 'Unknown'),
        'cfg_scale': model_settings.get('cfg_scale', 'Unknown'),
        'sampler': model_settings.get('sampler', 'Unknown'),
        'seed': model_settings.get('seed', 'Unknown')
    }


def extract_video_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract video settings from generation config.
    
    Args:
        config: Generation configuration dictionary
        
    Returns:
        Dictionary with width, height, and frames
    """
    video_settings = config.get('video_settings', {})
    
    return {
        'width': video_settings.get('width', 'Unknown'),
        'height': video_settings.get('height', 'Unknown'),
        'frames': video_settings.get('frames', 'Unknown')
    }


def get_videos_per_variation(config: Dict[str, Any]) -> int:
    """Get the number of videos per variation from config.
    
    Args:
        config: Generation configuration dictionary
        
    Returns:
        Number of videos per variation
    """
    return config.get('videos_per_variation', 8)
