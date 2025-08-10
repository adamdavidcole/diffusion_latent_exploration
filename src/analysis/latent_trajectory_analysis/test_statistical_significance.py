import torch
import numpy as np
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_statistical_significance(
        group_tensors: Dict[str, Dict[str, torch.Tensor]], 
) -> Dict[str, Any]:
    """Simplified statistical significance testing for trajectories."""
    significance_analysis = {
        'trajectory_group_differences': {},
        'statistical_summary': {}
    }
    
    # Extract key trajectory statistics
    group_statistics = {}
    
    for group_name in sorted(group_tensors.keys()):
        data = group_tensors[group_name]
        trajectories = data['trajectory_tensor']  # [n_videos, steps, 1, 16, frames, H, W]
        trajectories = trajectories.squeeze(2)   # [n_videos, steps, 16, frames, H, W]
        
        # Compute trajectory statistics
        trajectory_variances = torch.var(trajectories, dim=(1, 2, 3, 4, 5))  # [n_videos]
        trajectory_means = torch.mean(trajectories, dim=(1, 2, 3, 4, 5))     # [n_videos]
        
        group_statistics[group_name] = {
            'variance': trajectory_variances.cpu().numpy(),
            'mean': trajectory_means.cpu().numpy()
        }
    
    # Simple group comparisons
    group_names = list(group_statistics.keys())
    for i, group1 in enumerate(group_names):
        for j, group2 in enumerate(group_names[i+1:], i+1):
            variance_diff = np.mean(group_statistics[group1]['variance']) - np.mean(group_statistics[group2]['variance'])
            mean_diff = np.mean(group_statistics[group1]['mean']) - np.mean(group_statistics[group2]['mean'])
            
            significance_analysis['trajectory_group_differences'][f"{group1}_vs_{group2}"] = {
                'variance_difference': float(variance_diff),
                'mean_difference': float(mean_diff)
            }
    
    significance_analysis['statistical_summary'] = {
        'groups_analyzed': len(group_names),
        'comparisons_made': len(group_names) * (len(group_names) - 1) // 2
    }
    
    return significance_analysis