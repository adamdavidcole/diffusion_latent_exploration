"""
Temporal stability windows plotting functionality.
Extracted from LatentTrajectoryAnalyzer.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.analysis.data_structures import LatentTrajectoryAnalysis
from src.visualization.visualization_config import VisualizationConfig

def plot_temporal_stability_windows(results: LatentTrajectoryAnalysis, viz_dir: Path, 
                                   viz_config: VisualizationConfig = None, 
                                   labels_map: dict = None, **kwargs) -> Path:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
    stability_data = results.temporal_coherence['temporal_stability_windows']
    sorted_group_names = sorted(stability_data.keys())
    
    # Design system settings
    colors = sns.color_palette("husl", len(sorted_group_names))
    alpha = 0.8
    linewidth = 2
    markersize = 3
    fontsize_labels = 8
    fontsize_legend = 9
    
    # Plot different window sizes
    window_sizes = ['window_3', 'window_5', 'window_7']
    axes = [ax1, ax2, ax3]
    
    for ax, window_size in zip(axes, window_sizes):
        for i, group_name in enumerate(sorted_group_names):
            data = stability_data[group_name].get(window_size, [])
            if data:
                window_starts = [item['window_start'] for item in data]
                mean_stabilities = [item['mean_stability'] for item in data]
                label = labels_map[group_name]
                ax.plot(window_starts, mean_stabilities, 'o-', label=label, 
                        alpha=alpha, color=colors[i], linewidth=linewidth, markersize=markersize)
        
        ax.set_xlabel('Window Start Position', fontsize=fontsize_labels)
        ax.set_ylabel('Mean Stability', fontsize=fontsize_labels)
        ax.set_title(f'Temporal Stability: {window_size.replace("_", " ").title()}', 
                    fontsize=fontsize_legend, fontweight='bold')
        ax.legend(fontsize=fontsize_legend, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=fontsize_labels)
    
    # Plot 4: Stability variance comparison with design system
    stability_variances = []
    for group_name in sorted_group_names:
        group_variance = 0
        count = 0
        for window_size in window_sizes:
            data = stability_data[group_name].get(window_size, [])
            if data:
                variances = [item['stability_variance'] for item in data]
                group_variance += np.mean(variances)
                count += 1
        stability_variances.append(group_variance / max(count, 1))
    
    bars = ax4.bar(sorted_group_names, stability_variances, alpha=alpha, color=colors)
    ax4.set_xlabel('Prompt Group', fontsize=fontsize_labels)
    ax4.set_ylabel('Average Stability Variance', fontsize=fontsize_labels)
    ax4.set_title('Overall Temporal Stability Variance', 
                    fontsize=fontsize_legend, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45, labelsize=fontsize_labels)
    ax4.tick_params(axis='y', labelsize=fontsize_labels)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels to bars
    for bar, variance in zip(bars, stability_variances):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(stability_variances) * 0.01,
                f'{variance:.3f}', ha='center', va='bottom', fontsize=fontsize_labels)
    
    plt.tight_layout()

    output_path = viz_dir / "temporal_stability_windows.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path