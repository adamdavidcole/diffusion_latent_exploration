import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from pathlib import Path

from src.analysis.data_structures import LatentTrajectoryAnalysis


def plot_spatial_coherence_patterns(
    results: LatentTrajectoryAnalysis, 
    viz_dir: Path,
    logger: logging.Logger = None
) -> Path:
    """Plot spatial coherence analysis."""
    if logger is None:
        logger = logging.getLogger(__name__)
        
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    coherence_data = results.spatial_patterns['spatial_coherence_patterns']
    prompt_names = sorted(coherence_data.keys())
    colors = sns.color_palette("husl", len(prompt_names))
    
    # Debug: Check actual data structure
    sample_prompt = prompt_names[0]
    sample_data = coherence_data[sample_prompt]
    logger.info(f"Spatial coherence data structure for {sample_prompt}: {list(sample_data.keys())}")
    
    # Plot 1: Aggregate spatial coherence evolution by prompt group (not individual trajectories)
    has_evolution_data = False
    for i, prompt_name in enumerate(prompt_names):
        if 'coherence_evolution' in coherence_data[prompt_name] and coherence_data[prompt_name]['coherence_evolution']:
            evolution_data = coherence_data[prompt_name]['coherence_evolution']
            if evolution_data and len(evolution_data) > 0:
                # If evolution_data contains multiple trajectories, aggregate them
                if isinstance(evolution_data[0], (list, np.ndarray)):
                    # Multiple video trajectories - calculate mean trajectory
                    evolution_arrays = [np.array(traj) for traj in evolution_data if len(traj) > 0]
                    if evolution_arrays:
                        # Ensure all have same length by taking minimum
                        min_length = min(len(arr) for arr in evolution_arrays)
                        trimmed_arrays = [arr[:min_length] for arr in evolution_arrays]
                        mean_evolution = np.mean(trimmed_arrays, axis=0)
                        steps = range(len(mean_evolution))
                        ax1.plot(steps, mean_evolution, 'o-', label=f"{prompt_name} (N={len(evolution_arrays)})", 
                                color=colors[i], alpha=0.8, linewidth=2, markersize=3)
                        has_evolution_data = True
                else:
                    # Single trajectory
                    steps = range(len(evolution_data))
                    ax1.plot(steps, evolution_data, 'o-', label=prompt_name, 
                            color=colors[i], alpha=0.8, linewidth=2, markersize=3)
                    has_evolution_data = True
    
    if has_evolution_data:
        ax1.set_xlabel('Diffusion Step')
        ax1.set_ylabel('Spatial Coherence')
        ax1.set_title('Spatial Coherence Evolution by Prompt\n(Individual Trajectories)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No coherence evolution data available', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Spatial Coherence Evolution (No Data)')
    
    # Plot 2: Mean coherence trajectory evolution over diffusion steps
    trajectory_data_available = any('mean_coherence_trajectory' in coherence_data[prompt] and 
                                   coherence_data[prompt]['mean_coherence_trajectory'] is not None 
                                   for prompt in prompt_names)
    
    if trajectory_data_available:
        has_trajectory_evolution = False
        for i, prompt in enumerate(prompt_names):
            if 'mean_coherence_trajectory' in coherence_data[prompt] and coherence_data[prompt]['mean_coherence_trajectory'] is not None:
                trajectory = coherence_data[prompt]['mean_coherence_trajectory']
                if isinstance(trajectory, (list, np.ndarray)) and len(trajectory) > 0:
                    # Plot the trajectory evolution over diffusion steps
                    trajectory_array = np.array(trajectory)
                    if trajectory_array.ndim > 1:
                        # If multidimensional, take mean across non-time dimensions
                        trajectory_1d = np.mean(trajectory_array.reshape(trajectory_array.shape[0], -1), axis=1)
                    else:
                        trajectory_1d = trajectory_array
                    
                    steps = range(len(trajectory_1d))
                    ax2.plot(steps, trajectory_1d, 'o-', label=prompt, 
                            color=colors[i], alpha=0.8, linewidth=2, markersize=3)
                    has_trajectory_evolution = True
        
        if has_trajectory_evolution:
            ax2.set_xlabel('Diffusion Step')
            ax2.set_ylabel('Mean Coherence Value')
            ax2.set_title('Mean Coherence Trajectory Evolution\n(By Prompt Over Diffusion Steps)')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No valid trajectory evolution data', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Mean Coherence Trajectory (No Data)')
    else:
        ax2.text(0.5, 0.5, 'No mean coherence trajectory data available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Mean Coherence Trajectory (No Data)')
    
    # Plot 3: Coherence stability (using coherence_stability)
    stability_data_available = any('coherence_stability' in coherence_data[prompt] and 
                                  coherence_data[prompt]['coherence_stability'] is not None 
                                  for prompt in prompt_names)
    
    if stability_data_available:
        stabilities = []
        valid_prompts = []
        for prompt in prompt_names:
            if 'coherence_stability' in coherence_data[prompt] and coherence_data[prompt]['coherence_stability'] is not None:
                stabilities.append(coherence_data[prompt]['coherence_stability'])
                valid_prompts.append(prompt)
        
        if stabilities:
            bars3 = ax3.bar(valid_prompts, stabilities, alpha=0.7,
                           color=sns.color_palette("viridis", len(valid_prompts)))
            ax3.set_xlabel('Prompt ID')
            ax3.set_ylabel('Coherence Stability')
            ax3.set_title('Spatial Coherence Stability by Prompt')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, stab in zip(bars3, stabilities):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + max(stabilities) * 0.01,
                        f'{stab:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            ax3.text(0.5, 0.5, 'No valid stability data', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Coherence Stability (No Data)')
    else:
        ax3.text(0.5, 0.5, 'No coherence stability data available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Coherence Stability (No Data)')
    
    # Plot 4: Coherence evolution heatmap
    if has_evolution_data:
        evolution_matrix = []
        valid_prompts = []
        for prompt_name in prompt_names:
            if 'coherence_evolution' in coherence_data[prompt_name] and coherence_data[prompt_name]['coherence_evolution']:
                evolution = coherence_data[prompt_name]['coherence_evolution']
                if evolution and len(evolution) > 0:
                    # Handle multidimensional data by taking mean across extra dimensions
                    evolution_array = np.array(evolution)
                    if evolution_array.ndim > 1:
                        # If 3D like (6, 6, 10), flatten the first two dimensions or take mean
                        if evolution_array.ndim == 3:
                            # Take mean across spatial dimensions (assuming first two are spatial)
                            evolution_1d = np.mean(evolution_array, axis=(0, 1))
                        elif evolution_array.ndim == 2:
                            # Take mean across one dimension
                            evolution_1d = np.mean(evolution_array, axis=0)
                    else:
                        evolution_1d = evolution_array
                    
                    evolution_matrix.append(evolution_1d)
                    valid_prompts.append(prompt_name)
        
        if evolution_matrix:
            try:
                evolution_2d = np.array(evolution_matrix)
                # Ensure we have a 2D array for imshow
                if evolution_2d.ndim == 2:
                    im = ax4.imshow(evolution_2d, cmap='RdYlBu_r', aspect='auto')
                    ax4.set_yticks(range(len(valid_prompts)))
                    ax4.set_yticklabels(valid_prompts)
                    ax4.set_xlabel('Diffusion Step')
                    ax4.set_ylabel('Prompt ID')
                    ax4.set_title('Spatial Coherence Evolution Heatmap\n(By Prompt Over Diffusion Steps)')
                    plt.colorbar(im, ax=ax4, label='Coherence Value')
                else:
                    ax4.text(0.5, 0.5, f'Data shape not suitable for heatmap: {evolution_2d.shape}', 
                            ha='center', va='center', transform=ax4.transAxes)
                    ax4.set_title('Coherence Evolution Heatmap (Incompatible Shape)')
            except Exception as e:
                logger.warning(f"Failed to create coherence heatmap: {e}")
                ax4.text(0.5, 0.5, f'Failed to create heatmap: {str(e)}', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Coherence Evolution Heatmap (Error)')
        else:
            ax4.text(0.5, 0.5, 'No evolution matrix data', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Coherence Evolution Heatmap (No Data)')
    else:
        ax4.text(0.5, 0.5, 'No coherence evolution data available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Coherence Evolution Heatmap (No Data)')

    plt.tight_layout()
    
    output_path = viz_dir / "spatial_coherence_patterns.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path
