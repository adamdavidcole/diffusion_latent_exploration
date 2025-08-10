import torch
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)    


def analyze_information_content(
        group_tensors: Dict[str, Dict[str, torch.Tensor]]
) -> Dict[str, Any]:
    """Simplified information-theoretic analysis for trajectories."""
    info_analysis = {
        'trajectory_information_content': {},
        'information_evolution': {}
    }
    
    for group_name in sorted(group_tensors.keys()):
        data = group_tensors[group_name]
        logger.info(f"Analyzing information content for {group_name}")

        trajectories = data['trajectory_tensor']  # [n_videos, steps, 1, 16, frames, H, W]
        trajectories = trajectories.squeeze(2)   # [n_videos, steps, 16, frames, H, W]
        
        # Simple information content measures
        trajectory_variance = float(torch.var(trajectories).item())
        
        info_analysis['trajectory_information_content'][group_name] = {
            'variance_measure': trajectory_variance
        }
        
        info_analysis['information_evolution'][group_name] = {
            'complexity_trend': 'simplified_analysis'
        }
    
    return info_analysis