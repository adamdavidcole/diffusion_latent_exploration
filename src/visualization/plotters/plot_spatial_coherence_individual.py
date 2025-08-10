import matplotlib.pyplot as plt
import numpy as np
import logging
from pathlib import Path

from src.analysis.data_structures import LatentTrajectoryAnalysis
from src.visualization.visualization_config import VisualizationConfig


def plot_spatial_coherence_individual(
    results: LatentTrajectoryAnalysis, 
    viz_dir: Path,
    viz_config: VisualizationConfig = None
) -> Path:
    """Create separate visualization files for each prompt group showing all individual video trajectories."""
    if viz_config is None:
        viz_config = VisualizationConfig()
        
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🎬 Creating individual spatial coherence trajectory visualizations...")
        
        coherence_data = results.spatial_patterns['spatial_coherence_patterns']
        sorted_group_names = sorted(coherence_data.keys())
        
        # Create subfolder for individual spatial coherence visualizations
        individual_viz_dir = viz_dir / "spatial_coherence_individual"
        individual_viz_dir.mkdir(exist_ok=True)
        
        # Create individual files for each prompt group
        for group_name in sorted_group_names:
            try:
                logger.info(f"📊 Creating individual trajectories for group: {group_name}")
                
                data = coherence_data[group_name]
                coherence_evolution = data.get('coherence_evolution', [])
                
                if not coherence_evolution:
                    logger.warning(f"⚠️ No coherence evolution data for group: {group_name}")
                    continue
                
                # Create figure for this group
                fig = plt.figure(figsize=viz_config.figsize_dashboard)
                fig.suptitle(f'Spatial Coherence Individual Trajectories: {group_name}\n' +
                            f'All video trajectories for this prompt group',
                            fontsize=viz_config.fontsize_title, fontweight=viz_config.fontweight_title)
                
                # Create grid: 2x2 for different views
                gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
                
                # Plot 1: All individual trajectories
                ax1 = fig.add_subplot(gs[0, :])
                
                video_colors = viz_config.get_colors(min(len(coherence_evolution), 12))  # Limit colors
                valid_trajectories = 0
                
                for i, video_coherence in enumerate(coherence_evolution):
                    try:
                        if isinstance(video_coherence, (list, np.ndarray)) and len(video_coherence) > 0:
                            # Handle multidimensional video coherence data
                            coherence_array = np.array(video_coherence)
                            if coherence_array.ndim > 1:
                                # If multidimensional, take mean across spatial dimensions
                                coherence_1d = np.mean(coherence_array.reshape(coherence_array.shape[0], -1), axis=1)
                            else:
                                coherence_1d = coherence_array
                            
                            if len(coherence_1d) > 0:
                                steps = range(len(coherence_1d))
                                color_idx = i % len(video_colors)
                                ax1.plot(steps, coherence_1d, 'o-', 
                                        label=f'Video {i+1}', 
                                        alpha=viz_config.alpha,
                                        color=video_colors[color_idx],
                                        linewidth=viz_config.linewidth,
                                        markersize=viz_config.markersize)
                                valid_trajectories += 1
                                
                                logger.debug(f"✅ Plotted trajectory for video {i+1}: {len(coherence_1d)} steps")
                    except Exception as e:
                        logger.error(f"❌ Error plotting video {i}: {e}")
                
                if valid_trajectories > 0:
                    ax1.set_xlabel('Diffusion Step', fontsize=viz_config.fontsize_labels)
                    ax1.set_ylabel('Spatial Coherence', fontsize=viz_config.fontsize_labels)
                    ax1.set_title(f'All Individual Video Trajectories (N={valid_trajectories})',
                                 fontsize=viz_config.fontsize_title, fontweight=viz_config.fontweight_title)
                    if valid_trajectories <= 12:  # Only show legend if manageable
                        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=viz_config.fontsize_legend)
                    ax1.grid(True, alpha=viz_config.grid_alpha)
                    ax1.tick_params(axis='both', labelsize=viz_config.fontsize_labels)
                else:
                    ax1.text(0.5, 0.5, 'No valid trajectory data available', 
                            ha='center', va='center', transform=ax1.transAxes)
                    ax1.set_title('Individual Video Trajectories (No Data)')
                
                # Plot 2: Statistical summary
                ax2 = fig.add_subplot(gs[1, 0])
                
                if valid_trajectories > 1:
                    try:
                        # Calculate trajectory statistics
                        all_trajectories = []
                        for video_coherence in coherence_evolution:
                            if isinstance(video_coherence, (list, np.ndarray)) and len(video_coherence) > 0:
                                coherence_array = np.array(video_coherence)
                                if coherence_array.ndim > 1:
                                    coherence_1d = np.mean(coherence_array.reshape(coherence_array.shape[0], -1), axis=1)
                                else:
                                    coherence_1d = coherence_array
                                if len(coherence_1d) > 0:
                                    all_trajectories.append(coherence_1d)
                        
                        if len(all_trajectories) > 1:
                            # Ensure same length for statistics
                            min_length = min(len(traj) for traj in all_trajectories)
                            trimmed_trajectories = [traj[:min_length] for traj in all_trajectories]
                            trajectory_matrix = np.array(trimmed_trajectories)
                            
                            # Calculate mean and std
                            mean_trajectory = np.mean(trajectory_matrix, axis=0)
                            std_trajectory = np.std(trajectory_matrix, axis=0)
                            
                            steps = range(len(mean_trajectory))
                            ax2.plot(steps, mean_trajectory, 'o-', color='red', linewidth=3, 
                                    label='Mean', alpha=0.9)
                            ax2.fill_between(steps, 
                                           mean_trajectory - std_trajectory,
                                           mean_trajectory + std_trajectory,
                                           alpha=0.3, color='red', label='±1 Std')
                            
                            ax2.set_xlabel('Diffusion Step', fontsize=viz_config.fontsize_labels)
                            ax2.set_ylabel('Spatial Coherence', fontsize=viz_config.fontsize_labels)
                            ax2.set_title('Mean ± Standard Deviation',
                                         fontsize=viz_config.fontsize_title, fontweight=viz_config.fontweight_title)
                            ax2.legend(fontsize=viz_config.fontsize_legend)
                            ax2.grid(True, alpha=viz_config.grid_alpha)
                            ax2.tick_params(axis='both', labelsize=viz_config.fontsize_labels)
                    except Exception as e:
                        logger.error(f"❌ Error creating statistical summary: {e}")
                        ax2.text(0.5, 0.5, f'Statistical summary failed:\n{str(e)}', 
                                ha='center', va='center', transform=ax2.transAxes)
                        ax2.set_title('Statistical Summary (Error)')
                else:
                    ax2.text(0.5, 0.5, 'Insufficient data for statistics', 
                            ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title('Statistical Summary (Insufficient Data)')
                
                # Plot 3: Trajectory variance over steps
                ax3 = fig.add_subplot(gs[1, 1])
                
                if valid_trajectories > 1 and 'all_trajectories' in locals():
                    try:
                        # Calculate variance at each step
                        step_variances = np.var(trajectory_matrix, axis=0)
                        steps = range(len(step_variances))
                        
                        ax3.plot(steps, step_variances, 'o-', color='purple', 
                                linewidth=viz_config.linewidth, 
                                markersize=viz_config.markersize, alpha=0.8)
                        
                        ax3.set_xlabel('Diffusion Step', fontsize=viz_config.fontsize_labels)
                        ax3.set_ylabel('Trajectory Variance', fontsize=viz_config.fontsize_labels)
                        ax3.set_title('Inter-Video Variance by Step',
                                     fontsize=viz_config.fontsize_title, fontweight=viz_config.fontweight_title)
                        ax3.grid(True, alpha=viz_config.grid_alpha)
                        ax3.tick_params(axis='both', labelsize=viz_config.fontsize_labels)
                        
                        # Add interpretation
                        max_var_step = np.argmax(step_variances)
                        min_var_step = np.argmin(step_variances)
                        ax3.text(0.02, 0.98, f'Max var: Step {max_var_step}\nMin var: Step {min_var_step}', 
                                transform=ax3.transAxes, fontsize=10, fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
                    except Exception as e:
                        logger.error(f"❌ Error creating variance plot: {e}")
                        ax3.text(0.5, 0.5, f'Variance analysis failed:\n{str(e)}', 
                                ha='center', va='center', transform=ax3.transAxes)
                        ax3.set_title('Trajectory Variance (Error)')
                else:
                    ax3.text(0.5, 0.5, 'Insufficient data for variance analysis', 
                            ha='center', va='center', transform=ax3.transAxes)
                    ax3.set_title('Trajectory Variance (Insufficient Data)')
                
                plt.tight_layout()
                
                # Save with sanitized group name
                safe_group_name = "".join(c for c in group_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_group_name = safe_group_name.replace(' ', '_')
                output_path = individual_viz_dir / f"spatial_coherence_individual_{safe_group_name}.{viz_config.save_format}"
                plt.savefig(output_path, dpi=viz_config.dpi, bbox_inches=viz_config.bbox_inches)
                plt.close()
                
                logger.info(f"✅ Individual coherence visualization for '{group_name}' saved to: {output_path}")
                
            except Exception as e:
                logger.error(f"❌ Failed to create individual visualization for group '{group_name}': {e}")
                continue
        
        logger.info(f"✅ Completed individual spatial coherence visualizations for {len(sorted_group_names)} groups")
        
    except Exception as e:
        logger.error(f"❌ Critical error in individual spatial coherence visualization: {e}")
        logger.exception("Full traceback:")
        raise

    # Return the directory path since multiple files are created
    return individual_viz_dir
