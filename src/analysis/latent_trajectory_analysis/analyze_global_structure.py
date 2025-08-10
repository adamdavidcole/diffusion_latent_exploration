import torch
import numpy as np
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_global_structure(
        group_tensors: Dict[str, Dict[str, torch.Tensor]]
) -> Dict[str, Any]:
    """GPU-accelerated global structure analysis with trajectory focus."""
    global_analysis = {
        'trajectory_global_evolution': {},
        'convergence_patterns': {}
    }
    
    for group_name in sorted(group_tensors.keys()):
        data = group_tensors[group_name]
        logger.info(f"GPU analyzing global structure for {group_name}")
        
        trajectories = data['trajectory_tensor']  # [n_videos, steps, 1, 16, frames, H, W]
        trajectories = trajectories.squeeze(2)   # [n_videos, steps, 16, frames, H, W]
        
        n_videos, n_steps = trajectories.shape[:2]
        
        # Global evolution patterns
        global_evolutions = []
        for video_idx in range(n_videos):
            video_traj = trajectories[video_idx]  # [steps, 16, frames, H, W]
            
            video_evolution = []
            for step in range(n_steps):
                step_data = video_traj[step]  # [16, frames, H, W]
                
                global_variance = torch.var(step_data).item()
                global_magnitude = torch.norm(step_data).item()
                
                video_evolution.append({
                    'step': step,
                    'global_variance': global_variance,
                    'global_magnitude': global_magnitude
                })
            
            global_evolutions.append(video_evolution)
        
        # Store results
        global_analysis['trajectory_global_evolution'][group_name] = {
            'variance_progression': [np.mean([evol[i]['global_variance'] for evol in global_evolutions if i < len(evol)]) for i in range(min(n_steps, 20))],
            'magnitude_progression': [np.mean([evol[i]['global_magnitude'] for evol in global_evolutions if i < len(evol)]) for i in range(min(n_steps, 20))]
        }
        
        global_analysis['convergence_patterns'][group_name] = {
            'overall_diversity_score': float(torch.var(trajectories, dim=0).mean().item())
        }
    
    return global_analysis