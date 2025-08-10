"""
Temporal stability windows plotting functionality.
Extracted from LatentTrajectoryAnalyzer.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.analysis.data_structures import LatentTrajectoryAnalysis
from src.config.visualization_config import VisualizationConfig


def plot_temporal_stability_windows(results: LatentTrajectoryAnalysis, viz_dir: Path, 
                                   viz_config: VisualizationConfig = None, 
                                   labels_map: dict = None, **kwargs) -> Path:
    """Plot temporal stability windows analysis with consistent design system."""
    # Set defaults for optional parameters
    if viz_config is None:
        viz_config = VisualizationConfig()
    
    output_path = viz_dir / "temporal_stability_windows.png"
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    stability_data = results.temporal_stability_windows['window_stability_analysis']
    sorted_group_names = sorted(stability_data.keys())
    
    # Design system settings
    colors = sns.color_palette("husl", len(sorted_group_names))
    alpha = 0.8
    linewidth = 2
    markersize = 3
    fontsize_labels = 8
    fontsize_legend = 9
    
    # Plot 1: Window stability metrics for each group
    for i, group_name in enumerate(sorted_group_names):
        data = stability_data[group_name]
        window_stabilities = data.get('window_stabilities', [])
        if window_stabilities and len(window_stabilities) > 0:
            windows = list(range(len(window_stabilities)))
            ax1.plot(windows, window_stabilities, 'o-', label=group_name, 
                    alpha=alpha, color=colors[i], linewidth=linewidth, markersize=markersize)
    
    ax1.set_xlabel('Window Index', fontsize=fontsize_labels)
    ax1.set_ylabel('Stability Metric', fontsize=fontsize_labels)
    ax1.set_title('Temporal Window Stability Analysis', fontsize=fontsize_legend, fontweight='bold')
    ax1.legend(fontsize=fontsize_legend, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=fontsize_labels)
    
    # Plot 2: Window size comparison
    window_size_analysis = results.temporal_stability_windows['window_size_comparison']
    window_sizes = sorted(window_size_analysis.keys())
    
    # Calculate average stability for each window size
    size_stabilities = []
    for size in window_sizes:
        size_data = window_size_analysis[size]
        avg_stability = np.mean([size_data[group]['average_stability'] 
                               for group in sorted_group_names 
                               if group in size_data])
        size_stabilities.append(avg_stability)
    
    bars = ax2.bar(window_sizes, size_stabilities, alpha=alpha, color=colors[0])
    ax2.set_xlabel('Window Size', fontsize=fontsize_labels)
    ax2.set_ylabel('Average Stability', fontsize=fontsize_labels)
    ax2.set_title('Window Size vs Average Stability', fontsize=fontsize_legend, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=fontsize_labels)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, stability in zip(bars, size_stabilities):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(size_stabilities) * 0.01,
                f'{stability:.3f}', ha='center', va='bottom', fontsize=fontsize_labels)
    
    # Plot 3: Stability variance by group with design system
    stability_variances = []
    for group_name in sorted_group_names:
        data = stability_data[group_name]
        window_stabilities = data.get('window_stabilities', [])
        if window_stabilities:
            variance = np.var(window_stabilities)
            stability_variances.append(variance)
        else:
            stability_variances.append(0)
    
    bars = ax3.bar(sorted_group_names, stability_variances, alpha=alpha, color=colors)
    ax3.set_xlabel('Prompt Group', fontsize=fontsize_labels)
    ax3.set_ylabel('Stability Variance', fontsize=fontsize_labels)
    ax3.set_title('Window Stability Variance by Group', fontsize=fontsize_legend, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45, labelsize=fontsize_labels)
    ax3.tick_params(axis='y', labelsize=fontsize_labels)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, variance in zip(bars, stability_variances):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(stability_variances) * 0.01,
                f'{variance:.3f}', ha='center', va='bottom', fontsize=fontsize_labels)
    
    # Plot 4: Overall stability ranking with design system
    stability_ranking = sorted(zip(sorted_group_names, stability_variances), key=lambda x: x[1])
    ranked_groups, ranked_variances = zip(*stability_ranking)
    
    # Use consistent colors based on original group order
    ranking_colors = [colors[sorted_group_names.index(group)] for group in ranked_groups]
    
    bars = ax4.bar(ranked_groups, ranked_variances, alpha=alpha, color=ranking_colors)
    ax4.set_xlabel('Prompt Group (Stable → Variable)', fontsize=fontsize_labels)
    ax4.set_ylabel('Stability Variance', fontsize=fontsize_labels)
    ax4.set_title('Stability Ranking (Most to Least Stable)', fontsize=fontsize_legend, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45, labelsize=fontsize_labels)
    ax4.tick_params(axis='y', labelsize=fontsize_labels)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels to bars
    for bar, variance in zip(bars, ranked_variances):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(ranked_variances) * 0.01,
                f'{variance:.3f}', ha='center', va='bottom', fontsize=fontsize_labels)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path
