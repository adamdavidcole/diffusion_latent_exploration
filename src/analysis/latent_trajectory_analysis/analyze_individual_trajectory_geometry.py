import torch
import numpy as np
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from scipy.spatial import ConvexHull
    from scipy.spatial.distance import pdist, squareform
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.neighbors import NearestNeighbors
    ADVANCED_GEOMETRY_AVAILABLE = True
except ImportError:
    ADVANCED_GEOMETRY_AVAILABLE = False
    

def analyze_individual_trajectory_geometry(
        group_tensors: Dict[str, Dict[str, torch.Tensor]], 
) -> Dict[str, Any]:
    """
    Compute individual trajectory geometry metrics: speed, volume, circuitousness.
    
    For each trajectory in each group, computes:
    1. Speed: Average step size (Euclidean distance between consecutive points)
    2. Volume: Convex hull volume of the individual trajectory points
    3. Circuitousness: Ratio of path length to straight-line endpoint distance
    """
    trajectory_geometry = {}
    
    for group_name in sorted(group_tensors.keys()):
        try:
            trajectory_tensor = group_tensors[group_name]['trajectory_tensor']  # [videos, steps, ...]
            flat_trajectories = trajectory_tensor.view(trajectory_tensor.shape[0], trajectory_tensor.shape[1], -1)
            
            n_videos, n_steps, latent_dim = flat_trajectories.shape
            
            individual_metrics = {
                'speeds': [],
                'log_bbox_volumes': [],
                'effective_sides': [],
                'circuitousness': [],
                'endpoint_alignment': [],
                'turning_angle': [],
                'path_lengths': [],
                'endpoint_distances': [],
                'step_size_variability': []
            }
            
            for video_idx in range(n_videos):
                trajectory = flat_trajectories[video_idx].cpu().numpy()  # [steps, latent_dim]
                
                # 1. Speed calculation
                step_differences = trajectory[1:] - trajectory[:-1]
                step_sizes = np.linalg.norm(step_differences, axis=1)
                mean_speed = np.mean(step_sizes)
                speed_variability = np.std(step_sizes)
                
                # 2. Path length and endpoint distance
                path_length = np.sum(step_sizes)
                endpoint_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
                
                # 3. Circuitousness (tortuosity)
                circuitousness = path_length / (endpoint_distance + 1e-8)

                # 4. Endpoint alignment and turning angle
                e = trajectory[-1] - trajectory[0]
                e_norm = np.linalg.norm(e) + 1e-8
                v = step_differences
                v_norms = np.linalg.norm(v, axis=1) + 1e-8
                endpoint_alignment = float(np.mean((v @ e) / (v_norms * e_norm)))
                u = v[:-1] / v_norms[:-1, None]
                w = v[1:]  / v_norms[1:,  None]
                cosang = np.clip(np.sum(u*w, axis=1), -1.0, 1.0)
                turning_angle = float(np.sum(np.arccos(cosang)))

                # 5. Log-volume (bbox proxy) and effective side
                mins = np.min(trajectory, axis=0)
                maxs = np.max(trajectory, axis=0)
                ranges = np.clip(maxs - mins, 1e-12, None)
                log_bbox_vol = float(np.sum(np.log(ranges)))
                eff_side = float(np.exp(log_bbox_vol / ranges.size))
                
                # 4. Individual trajectory volume (convex hull)
                try:
                    if ADVANCED_GEOMETRY_AVAILABLE and latent_dim <= 20 and n_steps >= latent_dim + 1:
                        # Direct convex hull computation
                        unique_points = np.unique(trajectory, axis=0)
                        if len(unique_points) >= latent_dim + 1:
                            hull = ConvexHull(unique_points)
                            individual_volume = hull.volume
                        else:
                            # Bounding box volume
                            mins = np.min(trajectory, axis=0)
                            maxs = np.max(trajectory, axis=0)
                            individual_volume = np.prod(maxs - mins + 1e-8)
                    else:
                        # Bounding box approximation for high dimensions
                        mins = np.min(trajectory, axis=0)
                        maxs = np.max(trajectory, axis=0)
                        individual_volume = np.prod(maxs - mins + 1e-8)
                except Exception:
                    # Fallback: bounding box volume
                    mins = np.min(trajectory, axis=0)
                    maxs = np.max(trajectory, axis=0)
                    individual_volume = np.prod(maxs - mins + 1e-8)
                
                # Store metrics
                individual_metrics['speeds'].append(float(mean_speed))
                individual_metrics['log_bbox_volumes'].append(float(log_bbox_vol)); individual_metrics['effective_sides'].append(float(eff_side))
                individual_metrics['circuitousness'].append(float(circuitousness)); individual_metrics['endpoint_alignment'].append(float(endpoint_alignment)); individual_metrics['turning_angle'].append(float(turning_angle))
                individual_metrics['path_lengths'].append(float(path_length))
                individual_metrics['endpoint_distances'].append(float(endpoint_distance))
                individual_metrics['step_size_variability'].append(float(speed_variability))
            
            # Compute summary statistics
            trajectory_geometry[group_name] = {
                'speed_stats': {
                    'mean': float(np.mean(individual_metrics['speeds'])),
                    'std': float(np.std(individual_metrics['speeds'])),
                    'min': float(np.min(individual_metrics['speeds'])),
                    'max': float(np.max(individual_metrics['speeds'])),
                    'median': float(np.median(individual_metrics['speeds'])),
                    'individual_values': individual_metrics['speeds']
                },
                'log_volume_stats': {
                    'mean': float(np.mean(individual_metrics['log_bbox_volumes'])),
                    'std': float(np.std(individual_metrics['log_bbox_volumes'])),
                    'min': float(np.min(individual_metrics['log_bbox_volumes'])),
                    'max': float(np.max(individual_metrics['log_bbox_volumes'])),
                    'median': float(np.median(individual_metrics['log_bbox_volumes'])),
                    'individual_values': individual_metrics['log_bbox_volumes']
                },
                'effective_side_stats': {
                    'mean': float(np.mean(individual_metrics['effective_sides'])),
                    'std': float(np.std(individual_metrics['effective_sides'])),
                    'min': float(np.min(individual_metrics['effective_sides'])),
                    'max': float(np.max(individual_metrics['effective_sides'])),
                    'median': float(np.median(individual_metrics['effective_sides'])),
                    'individual_values': individual_metrics['effective_sides']
                },
                'endpoint_alignment_stats': {
                    'mean': float(np.mean(individual_metrics['endpoint_alignment'])),
                    'std': float(np.std(individual_metrics['endpoint_alignment'])),
                    'min': float(np.min(individual_metrics['endpoint_alignment'])),
                    'max': float(np.max(individual_metrics['endpoint_alignment'])),
                    'median': float(np.median(individual_metrics['endpoint_alignment'])),
                    'individual_values': individual_metrics['endpoint_alignment']
                },
                'turning_angle_stats': {
                    'mean': float(np.mean(individual_metrics['turning_angle'])),
                    'std': float(np.std(individual_metrics['turning_angle'])),
                    'min': float(np.min(individual_metrics['turning_angle'])),
                    'max': float(np.max(individual_metrics['turning_angle'])),
                    'median': float(np.median(individual_metrics['turning_angle'])),
                    'individual_values': individual_metrics['turning_angle']
                },

                'circuitousness_stats': {
                    'mean': float(np.mean(individual_metrics['circuitousness'])),
                    'std': float(np.std(individual_metrics['circuitousness'])),
                    'min': float(np.min(individual_metrics['circuitousness'])),
                    'max': float(np.max(individual_metrics['circuitousness'])),
                    'median': float(np.median(individual_metrics['circuitousness'])),
                    'individual_values': individual_metrics['circuitousness']
                },
                'path_length_stats': {
                    'mean': float(np.mean(individual_metrics['path_lengths'])),
                    'std': float(np.std(individual_metrics['path_lengths'])),
                    'individual_values': individual_metrics['path_lengths']
                },
                'endpoint_distance_stats': {
                    'mean': float(np.mean(individual_metrics['endpoint_distances'])),
                    'std': float(np.std(individual_metrics['endpoint_distances'])),
                    'individual_values': individual_metrics['endpoint_distances']
                },
                'step_variability_stats': {
                    'mean': float(np.mean(individual_metrics['step_size_variability'])),
                    'std': float(np.std(individual_metrics['step_size_variability'])),
                    'individual_values': individual_metrics['step_size_variability']
                },
                'efficiency_metrics': {
                    'mean_efficiency': float(np.mean([1.0/c for c in individual_metrics['circuitousness']])),
                    'ballistic_trajectories_count': int(np.sum([c < 1.5 for c in individual_metrics['circuitousness']])),
                    'meandering_trajectories_count': int(np.sum([c > 3.0 for c in individual_metrics['circuitousness']]))
                },
                'n_trajectories': int(n_videos)
            }
            
        except Exception as e:
            logger.error(f"Error computing individual trajectory geometry for {group_name}: {e}")
            trajectory_geometry[group_name] = {
                'error': str(e),
                'n_trajectories': 0
            }
    
    return trajectory_geometry

