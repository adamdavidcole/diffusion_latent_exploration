import torch
import numpy as np
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_frequency_patterns(
        group_tensors: Dict[str, Dict[str, torch.Tensor]]
) -> Dict[str, Any]:
    """Simplified frequency analysis for trajectories."""
    frequency_analysis = {
        'trajectory_frequency_characteristics': {},
        'temporal_patterns': {}
    }
    
    for group_name in sorted(group_tensors.keys()):
        data = group_tensors[group_name]
        logger.info(f"Analyzing frequency patterns for {group_name}")

        trajectories = data['trajectory_tensor']  # [n_videos, steps, 1, 16, frames, H, W]
        trajectories = trajectories.squeeze(2)   # [n_videos, steps, 16, frames, H, W]
        
        # Simple frequency-domain analysis
        trajectory_fft_magnitude = 0.0
        if hasattr(torch.fft, 'fft'):
            try:
                trajectory_fft_magnitude = float(torch.abs(torch.fft.fft(trajectories.flatten())).mean().item())
            except:
                trajectory_fft_magnitude = 0.0
        
        frequency_analysis['trajectory_frequency_characteristics'][group_name] = {
            'fft_magnitude_mean': trajectory_fft_magnitude,
            'spectral_energy': float(torch.norm(trajectories).item())
        }
        
        frequency_analysis['temporal_patterns'][group_name] = {
            'temporal_smoothness': float(torch.mean(torch.abs(torch.diff(trajectories, dim=1))).item())
        }
    
    return frequency_analysis