import torch
import numpy as np
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_complexity_measures(
        group_tensors: Dict[str, Dict[str, torch.Tensor]]
) -> Dict[str, Any]:
    """Simplified complexity analysis for trajectories."""
    complexity_analysis = {
        'trajectory_complexity': {},
        'evolution_complexity': {}
    }
    
    for group_name in sorted(group_tensors.keys()):
        data = group_tensors[group_name]
        logger.info(f"Analyzing complexity measures for {group_name}")
        
        trajectories = data['trajectory_tensor']  # [n_videos, steps, 1, 16, frames, H, W]
        trajectories = trajectories.squeeze(2)   # [n_videos, steps, 16, frames, H, W]
        
        # Simple complexity measures
        trajectory_std = float(torch.std(trajectories).item())
        trajectory_range = float((torch.max(trajectories) - torch.min(trajectories)).item())
        
        complexity_analysis['trajectory_complexity'][group_name] = {
            'standard_deviation': trajectory_std,
            'value_range': trajectory_range
        }
        
        complexity_analysis['evolution_complexity'][group_name] = {
            'temporal_variation': float(torch.var(trajectories, dim=1).mean().item())
        }
    
    return complexity_analysis