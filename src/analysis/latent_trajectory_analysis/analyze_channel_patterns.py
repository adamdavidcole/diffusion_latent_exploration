import torch
import numpy as np
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)    

def analyze_channel_patterns(
        group_tensors: Dict[str, Dict[str, torch.Tensor]]
) -> Dict[str, Any]:
    """GPU-accelerated channel pattern analysis with trajectory focus."""
    channel_analysis = {
        'channel_trajectory_evolution': {},
        'channel_specialization_patterns': {}
    }
    
    for group_name in sorted(group_tensors.keys()):
        data = group_tensors[group_name]
        logger.info(f"GPU analyzing channel patterns for {group_name}")
        
        trajectories = data['trajectory_tensor']  # [n_videos, steps, 1, 16, frames, H, W]
        trajectories = trajectories.squeeze(2)   # [n_videos, steps, 16, frames, H, W]
        
        n_videos, n_steps, n_channels = trajectories.shape[:3]
        
        # Channel evolution analysis
        channel_evolutions = []
        for video_idx in range(min(n_videos, 4)):  # Sample videos
            video_traj = trajectories[video_idx]  # [steps, 16, frames, H, W]
            
            video_channel_evolutions = []
            for channel in range(n_channels):
                channel_traj = video_traj[:, channel]  # [steps, frames, H, W]
                
                channel_magnitudes = []
                for step in range(n_steps):
                    magnitude = torch.norm(channel_traj[step]).item()
                    channel_magnitudes.append(magnitude)
                
                video_channel_evolutions.append(channel_magnitudes)
            
            channel_evolutions.append(video_channel_evolutions)
        
        # Store results
        channel_analysis['channel_trajectory_evolution'][group_name] = {
            'mean_evolution_patterns': np.mean(channel_evolutions, axis=0).tolist() if channel_evolutions else [],
            'evolution_variability': np.std(channel_evolutions, axis=0).tolist() if channel_evolutions else []
        }
        
        channel_analysis['channel_specialization_patterns'][group_name] = {
            'overall_variance': float(torch.var(trajectories).item()),
            'temporal_variance': float(torch.var(trajectories, dim=1).mean().item())
        }
    
    return channel_analysis