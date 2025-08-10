import torch
import numpy as np
from typing import Dict, Any
import logging

from .utils.corrcoef import corrcoef

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_spatial_patterns(
        group_tensors: Dict[str, Dict[str, torch.Tensor]]
) -> Dict[str, Any]:
    """GPU-accelerated spatial pattern analysis preserving trajectory structure."""
    spatial_analysis = {
        'spatial_variance_maps': {},
        'trajectory_spatial_evolution': {},
        'spatial_progression_patterns': {},
        'video_spatial_diversity': {},
        'edge_density_evolution': {},
        'spatial_coherence_patterns': {}
    }

    for group_name in sorted(group_tensors.keys()):
        data = group_tensors[group_name]
        logger.info(f"GPU analyzing spatial patterns for {group_name}")
        
        trajectories = data['trajectory_tensor']  # [n_videos, steps, 1, 16, frames, H, W]
        trajectories = trajectories.squeeze(2)   # [n_videos, steps, 16, frames, H, W]
        
        n_videos, n_steps = trajectories.shape[:2]
        
        # 1. Trajectory-aware spatial variance evolution
        # Compute spatial variance for each video at each diffusion step
        spatial_vars_per_step = torch.var(trajectories, dim=(-2, -1))  # [n_videos, steps, 16, frames]
        spatial_vars_mean_per_step = torch.mean(spatial_vars_per_step, dim=(2, 3))  # [n_videos, steps]
        
        # Average across videos to get group trajectory pattern
        group_spatial_trajectory = torch.mean(spatial_vars_mean_per_step, dim=0)  # [steps]
        
        # 2. Step-to-step spatial changes within trajectories
        spatial_trajectory_deltas = torch.diff(spatial_vars_mean_per_step, dim=1)  # [n_videos, steps-1]
        
        # 3. Video-level spatial diversity (how much each video varies spatially)
        video_spatial_diversity = torch.std(spatial_vars_mean_per_step, dim=1)  # [n_videos]
        
        # 4. Cross-video spatial consistency at each step
        step_consistency = torch.std(spatial_vars_mean_per_step, dim=0)  # [steps]
        
        # 5. Early vs Late diffusion spatial patterns
        early_steps = spatial_vars_mean_per_step[:, :n_steps//3]  # First third
        late_steps = spatial_vars_mean_per_step[:, -n_steps//3:]  # Last third
        
        early_spatial_mean = torch.mean(early_steps)
        late_spatial_mean = torch.mean(late_steps)
        spatial_evolution_ratio = late_spatial_mean / (early_spatial_mean + 1e-8)
        
        # 6. Edge density evolution in trajectory
        sample_indices = torch.randperm(n_videos)[:min(4, n_videos)]  # Sample videos
        edge_evolutions = []
        
        for video_idx in sample_indices:
            video_traj = trajectories[video_idx]  # [steps, 16, frames, H, W]
            
            # Compute edge density for each step
            step_edge_densities = []
            for step in range(min(n_steps, 20)):  # Sample steps
                step_data = video_traj[step]  # [16, frames, H, W]
                
                # Compute gradients
                grad_x = torch.diff(step_data, dim=-1).abs().mean()
                grad_y = torch.diff(step_data, dim=-2).abs().mean()
                edge_density = (grad_x + grad_y) / 2
                step_edge_densities.append(edge_density.item())
            
            edge_evolutions.append(step_edge_densities)
        
        # 7. Spatial coherence patterns (spatial autocorrelation evolution)
        spatial_coherences = []
        for video_idx in range(min(n_videos, 6)):  # Sample videos
            video_traj = trajectories[video_idx]  # [steps, 16, frames, H, W]
            
            video_coherences = []
            for step in range(0, n_steps, max(1, n_steps//10)):  # Sample steps
                step_data = video_traj[step]  # [16, frames, H, W]
                
                # Spatial autocorrelation for each channel/frame
                coherence_values = []
                for c in range(min(4, step_data.shape[0])):  # Sample channels
                    for f in range(step_data.shape[1]):  # All frames
                        spatial_map = step_data[c, f]  # [H, W]
                        
                        if spatial_map.shape[0] > 4 and spatial_map.shape[1] > 4:
                            # Simple spatial autocorrelation using shifted correlation
                            shifted_h = torch.roll(spatial_map, 1, dims=0)
                            shifted_w = torch.roll(spatial_map, 1, dims=1)

                            corr_h = corrcoef(spatial_map.flatten(), shifted_h.flatten())
                            corr_w = corrcoef(spatial_map.flatten(), shifted_w.flatten())

                            if not (torch.isnan(corr_h) or torch.isnan(corr_w)):
                                coherence_values.append((corr_h + corr_w).item() / 2)
                
                if coherence_values:
                    video_coherences.append(np.mean(coherence_values))
            
            if video_coherences:
                spatial_coherences.append(video_coherences)
        
        # Store trajectory-aware results (optimized data storage)
        spatial_analysis['spatial_variance_maps'][group_name] = {
            'mean': float(torch.mean(spatial_vars_per_step).item()),
            'std': float(torch.std(spatial_vars_per_step).item()),
            'distribution_sample': spatial_vars_per_step.flatten().cpu().numpy().tolist()[:50]  # Reduced from 1000 to 50
        }
        
        spatial_analysis['trajectory_spatial_evolution'][group_name] = {
            'trajectory_pattern': group_spatial_trajectory.cpu().numpy().tolist(),
            'evolution_ratio': float(spatial_evolution_ratio.item()),
            'early_vs_late_significance': float(torch.abs(early_spatial_mean - late_spatial_mean).item()),
            'trajectory_smoothness': float(torch.mean(torch.abs(spatial_trajectory_deltas)).item()),
            'phase_transition_strength': float(torch.std(group_spatial_trajectory).item())
        }
        
        spatial_analysis['spatial_progression_patterns'][group_name] = {
            'step_deltas_mean': torch.mean(spatial_trajectory_deltas, dim=0).cpu().numpy().tolist(),
            'step_deltas_std': torch.std(spatial_trajectory_deltas, dim=0).cpu().numpy().tolist(),
            'progression_consistency': float(torch.mean(step_consistency).item()),
            'progression_variability': float(torch.std(step_consistency).item()),
            'edge_evolution_patterns': edge_evolutions
        }
        
        spatial_analysis['video_spatial_diversity'][group_name] = {
            'inter_video_diversity_mean': float(torch.mean(video_spatial_diversity).item()),
            'inter_video_diversity_std': float(torch.std(video_spatial_diversity).item()),
            'diversity_distribution': video_spatial_diversity.cpu().numpy().tolist()
        }
        
        spatial_analysis['edge_density_evolution'][group_name] = {
            'mean_evolution_pattern': np.mean(edge_evolutions, axis=0).tolist() if edge_evolutions else [],
            'evolution_variability': np.std(edge_evolutions, axis=0).tolist() if edge_evolutions else [],
            'edge_formation_trend': 'increasing' if len(edge_evolutions) > 0 and np.mean([evo[-1] for evo in edge_evolutions]) > np.mean([evo[0] for evo in edge_evolutions]) else 'decreasing'
        }
        
        spatial_analysis['spatial_coherence_patterns'][group_name] = {
            'coherence_evolution': spatial_coherences,
            'mean_coherence_trajectory': np.mean(spatial_coherences, axis=0).tolist() if spatial_coherences else [],
            'coherence_stability': np.std([np.std(coh) for coh in spatial_coherences]) if spatial_coherences else 0
        }

    return spatial_analysis