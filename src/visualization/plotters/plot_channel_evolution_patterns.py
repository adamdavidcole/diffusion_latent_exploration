"""
Channel evolution patterns plotting functionality.
Extracted from LatentTrajectoryAnalyzer.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.analysis.data_structures import LatentTrajectoryAnalysis
from src.config.visualization_config import VisualizationConfig


def plot_channel_evolution_patterns(results: LatentTrajectoryAnalysis, viz_dir: Path, 
                                   viz_config: VisualizationConfig = None, 
                                   labels_map: dict = None, **kwargs) -> Path:
    """Plot channel evolution analysis with consistent design system."""
    # Set defaults for optional parameters
    if viz_config is None:
        viz_config = VisualizationConfig()
    
    output_path = viz_dir / "channel_evolution_patterns.png"
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    channel_data = results.channel_analysis['channel_trajectory_evolution']
    sorted_group_names = sorted(channel_data.keys())
    
    # Design system settings
    colors = sns.color_palette("husl", len(sorted_group_names))
    alpha = 0.8
    linewidth = 2
    markersize = 3
    fontsize_labels = 8
    fontsize_legend = 9
    
    # Plot 1: Mean evolution patterns for first few channels with design system
    for i, group_name in enumerate(sorted_group_names):
        data = channel_data[group_name]
        evolution_patterns = data.get('mean_evolution_patterns', [])
        if evolution_patterns and len(evolution_patterns) > 0:
            # Show evolution of first channel
            if len(evolution_patterns[0]) > 0:
                steps = list(range(len(evolution_patterns[0])))
                ax1.plot(steps, evolution_patterns[0], 'o-', label=f'{group_name} Ch0', 
                        alpha=alpha, color=colors[i], linewidth=linewidth, markersize=markersize)
    
    ax1.set_xlabel('Diffusion Step', fontsize=fontsize_labels)
    ax1.set_ylabel('Channel Magnitude', fontsize=fontsize_labels)
    ax1.set_title('Channel 0 Evolution Patterns', fontsize=fontsize_legend, fontweight='bold')
    ax1.legend(fontsize=fontsize_legend, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=fontsize_labels)
    
    # Plot 2: Channel variability with design system
    specialization_data = results.channel_analysis['channel_specialization_patterns']
    overall_variances = [specialization_data[group]['overall_variance'] for group in sorted_group_names]
    bars = ax2.bar(sorted_group_names, overall_variances, alpha=alpha, color=colors)
    ax2.set_xlabel('Prompt Group', fontsize=fontsize_labels)
    ax2.set_ylabel('Overall Channel Variance', fontsize=fontsize_labels)
    ax2.set_title('Channel Specialization Variance', fontsize=fontsize_legend, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45, labelsize=fontsize_labels)
    ax2.tick_params(axis='y', labelsize=fontsize_labels)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, variance in zip(bars, overall_variances):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(overall_variances) * 0.01,
                f'{variance:.3f}', ha='center', va='bottom', fontsize=fontsize_labels)
    
    # Plot 3: Temporal variance with design system
    temporal_variances = [specialization_data[group]['temporal_variance'] for group in sorted_group_names]
    bars = ax3.bar(sorted_group_names, temporal_variances, alpha=alpha, color=colors)
    ax3.set_xlabel('Prompt Group', fontsize=fontsize_labels)
    ax3.set_ylabel('Temporal Channel Variance', fontsize=fontsize_labels)
    ax3.set_title('Channel Temporal Specialization', fontsize=fontsize_legend, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45, labelsize=fontsize_labels)
    ax3.tick_params(axis='y', labelsize=fontsize_labels)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, variance in zip(bars, temporal_variances):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(temporal_variances) * 0.01,
                f'{variance:.3f}', ha='center', va='bottom', fontsize=fontsize_labels)
    
    # Plot 4: Variance ratio with design system
    variance_ratios = [ov/tv if tv > 0 else 0 for ov, tv in zip(overall_variances, temporal_variances)]
    bars = ax4.bar(sorted_group_names, variance_ratios, alpha=alpha, color=colors)
    ax4.set_xlabel('Prompt Group', fontsize=fontsize_labels)
    ax4.set_ylabel('Overall/Temporal Variance Ratio', fontsize=fontsize_labels)
    ax4.set_title('Channel Specialization Ratio', fontsize=fontsize_legend, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45, labelsize=fontsize_labels)
    ax4.tick_params(axis='y', labelsize=fontsize_labels)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, ratio in zip(bars, variance_ratios):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(variance_ratios) * 0.01,
                f'{ratio:.2f}', ha='center', va='bottom', fontsize=fontsize_labels)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path
