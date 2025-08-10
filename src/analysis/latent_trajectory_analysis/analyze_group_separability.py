import torch
import numpy as np
from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_group_separability(
        group_tensors: Dict[str, Dict[str, torch.Tensor]]
) -> Dict[str, Any]:
    """Simplified group separability analysis for trajectories."""
    separability_analysis = {
        'trajectory_group_separation': {},
        'inter_group_distances': {}
    }
    
    # Extract trajectory features
    group_centroids = {}
    
    for group_name in sorted(group_tensors.keys()):
        data = group_tensors[group_name]
        trajectories = data['trajectory_tensor']  # [n_videos, steps, 1, 16, frames, H, W]
        trajectories = trajectories.squeeze(2)   # [n_videos, steps, 16, frames, H, W]
        
        # Compute group centroid
        group_centroid = torch.mean(trajectories, dim=(0, 1))  # [16, frames, H, W]
        group_centroids[group_name] = group_centroid
    
    # Compute inter-group distances
    inter_distances = {}
    for group1 in group_centroids:
        for group2 in group_centroids:
            if group1 != group2:
                distance = float(torch.norm(group_centroids[group1] - group_centroids[group2]).item())
                inter_distances[f"{group1}_vs_{group2}"] = distance
    
    separability_analysis['trajectory_group_separation'] = {
        'group_count': len(group_centroids),
        'separability_measure': 'simplified_centroid_analysis'
    }
    
    separability_analysis['inter_group_distances'] = inter_distances
    
    return separability_analysis