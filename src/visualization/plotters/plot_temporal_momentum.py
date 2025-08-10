import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
from src.analysis.data_structures import LatentTrajectoryAnalysis
from src.visualization.visualization_config import VisualizationConfig


def plot_temporal_momentum_analysis(results: LatentTrajectoryAnalysis, viz_dir: Path, labels_map: dict[str, str], viz_config=VisualizationConfig()) -> Path:
    """Plot temporal momentum patterns with improved clarity and individual group views."""
    momentum_data = results.temporal_coherence['temporal_momentum_analysis']
    group_names = sorted(momentum_data.keys())
    colors = viz_config.get_colors(len(group_names))
    
    # Create main overlaid analysis figure
    fig_main = plt.figure(figsize=viz_config.figsize_standard)
    
    # Main plots (2x2 grid)
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    
    # Plot 1: Velocity patterns with confidence intervals (overlaid)
    for i, group_name in enumerate(group_names):
        data = momentum_data[group_name]
        velocity_mean = np.array(data['velocity_mean']).flatten()
        velocity_std = np.array(data['velocity_std']).flatten()
        
        # Ensure arrays have the same length
        min_len = min(len(velocity_mean), len(velocity_std))
        velocity_mean = velocity_mean[:min_len]
        velocity_std = velocity_std[:min_len]
        steps = np.arange(min_len)

        label = labels_map[group_name]
        ax1.plot(steps, velocity_mean, 'o-', label=label,
                color=colors[i], alpha=viz_config.alpha,
                linewidth=viz_config.linewidth, markersize=viz_config.markersize)
        ax1.fill_between(steps, velocity_mean - velocity_std, velocity_mean + velocity_std,
                        alpha=0.15, color=colors[i])
    
    ax1.set_xlabel('Diffusion Step', fontsize=viz_config.fontsize_labels)
    ax1.set_ylabel('Mean Velocity (±1σ)', fontsize=viz_config.fontsize_labels)
    ax1.set_title('Temporal Velocity Evolution - All Groups\n(Denoising Speed with Uncertainty)',
                    fontsize=viz_config.fontsize_title, fontweight=viz_config.fontweight_title)
    ax1.legend(bbox_to_anchor=viz_config.legend_bbox_anchor, loc=viz_config.legend_loc, 
                fontsize=viz_config.fontsize_legend)
    ax1.grid(True, alpha=viz_config.grid_alpha)
    
    # Plot 2: Acceleration patterns with confidence intervals (overlaid)
    for i, group_name in enumerate(group_names):
        data = momentum_data[group_name]
        accel_mean = np.array(data['acceleration_mean']).flatten()
        accel_std = np.array(data['acceleration_std']).flatten()
        
        # Ensure arrays have the same length
        min_len = min(len(accel_mean), len(accel_std))
        accel_mean = accel_mean[:min_len]
        accel_std = accel_std[:min_len]
        steps = np.arange(min_len)

        label = labels_map[group_name]
        ax2.plot(steps, accel_mean, 's-', label=label,
                color=colors[i], alpha=viz_config.alpha,
                linewidth=viz_config.linewidth, markersize=viz_config.markersize)
        ax2.fill_between(steps, accel_mean - accel_std, accel_mean + accel_std,
                        alpha=0.15, color=colors[i])
    
    ax2.set_xlabel('Diffusion Step', fontsize=viz_config.fontsize_labels)
    ax2.set_ylabel('Mean Acceleration (±1σ)', fontsize=viz_config.fontsize_labels)
    ax2.set_title('Temporal Acceleration Evolution - All Groups\n(Denoising Rate Changes)',
                    fontsize=viz_config.fontsize_title, fontweight=viz_config.fontweight_title)
    ax2.legend(bbox_to_anchor=viz_config.legend_bbox_anchor, loc=viz_config.legend_loc,
                fontsize=viz_config.fontsize_legend)
    ax2.grid(True, alpha=viz_config.grid_alpha)
    
    # Plot 3: Direction instability patterns (overlaid)
    for i, group_name in enumerate(group_names):
        data = momentum_data[group_name]
        direction_changes = np.array(data['momentum_direction_changes']).flatten()
        steps = np.arange(len(direction_changes))

        label = labels_map[group_name]
        ax3.plot(steps, direction_changes, '^-', label=label,
                color=colors[i], alpha=viz_config.alpha,
                linewidth=viz_config.linewidth, markersize=viz_config.markersize)
    
    ax3.set_xlabel('Diffusion Step', fontsize=viz_config.fontsize_labels)
    ax3.set_ylabel('Direction Change Count', fontsize=viz_config.fontsize_labels)
    ax3.set_title('Momentum Direction Changes - All Groups\n(Trajectory Instability)',
                    fontsize=viz_config.fontsize_title, fontweight=viz_config.fontweight_title)
    ax3.legend(fontsize=viz_config.fontsize_legend)
    ax3.grid(True, alpha=viz_config.grid_alpha)
    
    # Plot 4: Momentum phase space with error ellipses
    for i, group_name in enumerate(group_names):
        data = momentum_data[group_name]
        avg_velocity = np.mean(data['velocity_mean'])
        avg_acceleration = np.mean(data['acceleration_mean'])
        vel_uncertainty = np.mean(data['velocity_std'])
        accel_uncertainty = np.mean(data['acceleration_std'])
        
        # Plot point
        ax4.scatter(avg_velocity, avg_acceleration, s=120, 
                    color=colors[i], alpha=0.8, edgecolors='black', linewidth=1)
        
        # Plot uncertainty ellipse
        from matplotlib.patches import Ellipse
        ellipse = Ellipse((avg_velocity, avg_acceleration), 
                        2*vel_uncertainty, 2*accel_uncertainty,
                        alpha=0.3, color=colors[i])
        ax4.add_patch(ellipse)
        
        # Label
        ax4.annotate(group_name, (avg_velocity, avg_acceleration), 
                    xytext=(8, 8), textcoords='offset points', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3))
    
    ax4.set_xlabel('Average Velocity')
    ax4.set_ylabel('Average Acceleration')
    ax4.set_title('Momentum Phase Space\n(Velocity vs Acceleration with Uncertainty)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path_main = viz_dir / "temporal_momentum_analysis.png"
    plt.savefig(output_path_main, dpi=300, bbox_inches='tight')
    plt.close(fig_main)
    
    # Create separate figure for individual group velocity plots
    n_groups = len(group_names)
    n_cols = min(3, n_groups)
    n_rows = (n_groups + n_cols - 1) // n_cols
    
    fig_individual = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    fig_individual.suptitle('Individual Group Velocity Evolution', fontsize=14, fontweight='bold')
    
    # Calculate global y-axis range for consistent scaling
    all_velocities = []
    all_stds = []
    for group_name in group_names:
        data = momentum_data[group_name]
        velocity_mean = np.array(data['velocity_mean']).flatten()
        velocity_std = np.array(data['velocity_std']).flatten()
        all_velocities.extend(velocity_mean)
        all_stds.extend(velocity_std)
    
    global_min = min(all_velocities) - max(all_stds)
    global_max = max(all_velocities) + max(all_stds)
    y_margin = (global_max - global_min) * 0.1
    global_ylim = (global_min - y_margin, global_max + y_margin)
    
    for i, group_name in enumerate(group_names):
        ax_ind = plt.subplot(n_rows, n_cols, i + 1)
        
        data = momentum_data[group_name]
        velocity_mean = np.array(data['velocity_mean']).flatten()
        velocity_std = np.array(data['velocity_std']).flatten()
        
        min_len = min(len(velocity_mean), len(velocity_std))
        velocity_mean = velocity_mean[:min_len]
        velocity_std = velocity_std[:min_len]
        steps = np.arange(min_len)
        
        ax_ind.plot(steps, velocity_mean, 'o-', color=colors[i], 
                    alpha=0.9, linewidth=2.5, markersize=4)
        ax_ind.fill_between(steps, velocity_mean - velocity_std, velocity_mean + velocity_std,
                            alpha=0.3, color=colors[i])
        
        ax_ind.set_xlabel('Diffusion Step')
        ax_ind.set_ylabel('Velocity')
        ax_ind.set_title(f'{group_name}\nVelocity Evolution')
        ax_ind.set_ylim(global_ylim)  # Set consistent y-axis range
        ax_ind.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path_individual = viz_dir / "temporal_momentum_individual.png"
    plt.savefig(output_path_individual, dpi=300, bbox_inches='tight')
    plt.close(fig_individual)

    return output_path_main, output_path_individual