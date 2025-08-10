import torch
import numpy as np
from typing import Dict, Any, List
import logging

from .utils.select_baseline_group import select_baseline_group
from .utils.apply_normalization import apply_normalization

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Temporal Analysis Helper Methods
def trajectory_length(flat_trajectories: torch.Tensor) -> torch.Tensor:
    """Calculate trajectory lengths using GPU operations."""
    # flat_trajectories: [n_videos, steps, flattened_latent]
    step_differences = flat_trajectories[:, 1:] - flat_trajectories[:, :-1]
    step_norms = torch.linalg.norm(step_differences, dim=2)
    trajectory_lengths = torch.sum(step_norms, dim=1)
    return trajectory_lengths

def velocity_analysis(flat_trajectories: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Calculate velocity statistics using GPU operations."""
    step_differences = flat_trajectories[:, 1:] - flat_trajectories[:, :-1]
    velocities = torch.linalg.norm(step_differences, dim=2)
    
    return {
        'mean_velocity': torch.mean(velocities, dim=1),
        'velocity_variance': torch.var(velocities, dim=1)
    }

def acceleration_analysis(flat_trajectories: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Calculate acceleration statistics using GPU operations."""
    velocities = flat_trajectories[:, 1:] - flat_trajectories[:, :-1]
    accelerations = torch.linalg.norm(velocities[:, 1:] - velocities[:, :-1], dim=2)
    
    return {
        'mean_acceleration': torch.mean(accelerations, dim=1),
        'acceleration_variance': torch.var(accelerations, dim=1)
    }

def endpoint_distance(flat_trajectories: torch.Tensor) -> torch.Tensor:
    """Calculate endpoint distances using GPU operations."""
    return torch.linalg.norm(flat_trajectories[:, -1] - flat_trajectories[:, 0], dim=1)

def calculate_tortuosity(trajectory_lengths: torch.Tensor, 
                                endpoint_distances: torch.Tensor) -> torch.Tensor:
    """Calculate tortuosity (ratio of path length to straight-line distance)."""
    return trajectory_lengths / (endpoint_distances + 1e-8)

def semantic_convergence_rate(flat_trajectories: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Calculate semantic convergence rate using GPU operations."""
    num_videos, num_steps = flat_trajectories.shape[0], flat_trajectories.shape[1]
    
    # Calculate distances to final state for each trajectory
    final_latents = flat_trajectories[:, -1, :].unsqueeze(1)  # [n_videos, 1, latent_dim]
    distances_to_end = torch.linalg.norm(flat_trajectories - final_latents, dim=2)  # [n_videos, steps]
    
    # Find half-life: step where distance falls below half of initial distance
    half_distance = distances_to_end[:, 0] / 2.0  # [n_videos]
    half_life_mask = distances_to_end <= half_distance.unsqueeze(1)  # [n_videos, steps]
    
    # Find first step where condition is met
    half_life_step = torch.argmax(half_life_mask.int(), dim=1)
    
    # Handle cases where convergence never happens
    not_converged_mask = (half_life_step == 0) & ~half_life_mask[:, 0]
    half_life_step[not_converged_mask] = num_steps
    
    return {
        'half_life_step': half_life_step,
        'distances_to_end': distances_to_end
    }

def cross_group_trajectory_distances(trajectories1: torch.Tensor, 
                                        trajectories2: torch.Tensor) -> torch.Tensor:
    """Calculate cross-group trajectory distances."""
    # Average over steps for each video, then calculate pairwise distances
    traj1_mean = torch.mean(trajectories1, dim=1)  # [n_videos1, latent_dim]
    traj2_mean = torch.mean(trajectories2, dim=1)  # [n_videos2, latent_dim]
    
    # Calculate distances between all pairs
    distances = torch.cdist(traj1_mean, traj2_mean)  # [n_videos1, n_videos2]
    
    # Return minimum distances (closest match for each trajectory in group 1)
    return torch.min(distances, dim=1)[0]

def analyze_temporal_trajectories(
    group_tensors: Dict[str, Dict[str, torch.Tensor]], 
    prompt_groups: List[str],
    device: torch.device,
    norm_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """GPU-accelerated temporal trajectory analysis based on TemporalTrajectoryAnalysis."""
    temporal_analysis = {}
    
    # Set primary baseline for comparison analysis
    baseline_group = sorted(prompt_groups)[0]
    
    # Test both baseline strategies for research comparison
    baseline_group_empty = select_baseline_group(prompt_groups, "empty_prompt")
    baseline_group_class = select_baseline_group(prompt_groups, "first_class_specific")
    
    logger.info(f"Temporal analysis using baseline strategies:")
    logger.info(f"  Primary baseline: {baseline_group}")
    logger.info(f"  Empty prompt baseline: {baseline_group_empty}")
    logger.info(f"  First class-specific baseline: {baseline_group_class}")
    
    for group_name, group_data in group_tensors.items():
        trajectory_tensor = group_data['trajectory_tensor'].to(device)  # [n_videos, steps, ...]
        
        # Flatten trajectory for analysis (keep videos and steps dimensions)
        flat_trajectories = apply_normalization(trajectory_tensor, group_data, norm_cfg)  # [n_videos, steps, D]
        
        # Trajectory Length Analysis
        trajectory_lengths = trajectory_length(flat_trajectories)
        
        # Velocity Analysis
        velocity_results = velocity_analysis(flat_trajectories)
        
        # Acceleration Analysis
        acceleration_results = acceleration_analysis(flat_trajectories)
        
        # Endpoint Distance Analysis
        endpoint_distances = endpoint_distance(flat_trajectories)
        
        # Tortuosity Calculation
        tortuosity = calculate_tortuosity(trajectory_lengths, endpoint_distances)
        
        # Semantic Convergence Analysis
        convergence_results = semantic_convergence_rate(flat_trajectories)
        
        # Store results
        temporal_analysis[group_name] = {
            'trajectory_length': {
                'mean_length': float(torch.mean(trajectory_lengths)),
                'std_length': float(torch.std(trajectory_lengths)),
                'min_length': float(torch.min(trajectory_lengths)),
                'max_length': float(torch.max(trajectory_lengths)),
                'individual_lengths': trajectory_lengths.cpu().numpy().tolist()
            },
            'velocity_analysis': {
                'mean_velocity': velocity_results['mean_velocity'].cpu().numpy().tolist(),
                'velocity_variance': velocity_results['velocity_variance'].cpu().numpy().tolist(),
                'overall_mean_velocity': float(torch.mean(velocity_results['mean_velocity'])),
                'overall_velocity_variance': float(torch.mean(velocity_results['velocity_variance']))
            },
            'acceleration_analysis': {
                'mean_acceleration': acceleration_results['mean_acceleration'].cpu().numpy().tolist(),
                'acceleration_variance': acceleration_results['acceleration_variance'].cpu().numpy().tolist(),
                'overall_mean_acceleration': float(torch.mean(acceleration_results['mean_acceleration'])),
                'overall_acceleration_variance': float(torch.mean(acceleration_results['acceleration_variance']))
            },
            'endpoint_distance': {
                'mean_endpoint_distance': float(torch.mean(endpoint_distances)),
                'std_endpoint_distance': float(torch.std(endpoint_distances)),
                'individual_distances': endpoint_distances.cpu().numpy().tolist()
            },
            'tortuosity': {
                'mean_tortuosity': float(torch.mean(tortuosity)),
                'std_tortuosity': float(torch.std(tortuosity)),
                'individual_tortuosity': tortuosity.cpu().numpy().tolist()
            },
            'semantic_convergence': {
                'half_life_steps': convergence_results['half_life_step'].cpu().numpy().tolist(),
                'mean_half_life': float(torch.mean(convergence_results['half_life_step'].float())),
                'distances_to_end_final': convergence_results['distances_to_end'][:, -1].cpu().numpy().tolist(),
                'convergence_rate': float(torch.mean(1.0 / (convergence_results['half_life_step'].float() + 1.0)))
            }
        }
        
        # Baseline comparison if this is not the baseline group
        if group_name != baseline_group and baseline_group in group_tensors:
            baseline_tensor = group_tensors[baseline_group]['trajectory_tensor'].to(device)
            baseline_flat = baseline_tensor.flatten(start_dim=2)
            
            # Cross-group distance analysis
            cross_distances = cross_group_trajectory_distances(flat_trajectories, baseline_flat)
            temporal_analysis[group_name]['baseline_comparison'] = {
                'mean_distance_to_baseline': float(torch.mean(cross_distances)),
                'std_distance_to_baseline': float(torch.std(cross_distances)),
                'baseline_group': baseline_group
            }
    
    return temporal_analysis