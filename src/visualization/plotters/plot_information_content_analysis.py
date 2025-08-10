"""
Information content analysis plotting functionality.
Extracted from LatentTrajectoryAnalyzer.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.analysis.data_structures import LatentTrajectoryAnalysis
from src.config.visualization_config import VisualizationConfig


def plot_information_content_analysis(results: LatentTrajectoryAnalysis, viz_dir: Path, 
                                     viz_config: VisualizationConfig = None, 
                                     labels_map: dict = None, **kwargs) -> Path:
    """Plot information content analysis with consistent design system."""
    # Set defaults for optional parameters
    if viz_config is None:
        viz_config = VisualizationConfig()
    
    output_path = viz_dir / "information_content_analysis.png"
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    info_data = results.information_content['trajectory_information_content']
    sorted_group_names = sorted(info_data.keys())
    
    # Design system settings
    colors = sns.color_palette("husl", len(sorted_group_names))
    alpha = 0.8
    fontsize_labels = 8
    fontsize_legend = 9
    
    # Plot 1: Variance measures with design system
    variance_measures = [info_data[group]['variance_measure'] for group in sorted_group_names]
    bars = ax1.bar(sorted_group_names, variance_measures, alpha=alpha, color=colors)
    ax1.set_xlabel('Prompt Group', fontsize=fontsize_labels)
    ax1.set_ylabel('Information Variance Measure', fontsize=fontsize_labels)
    ax1.set_title('Trajectory Information Content', fontsize=fontsize_legend, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45, labelsize=fontsize_labels)
    ax1.tick_params(axis='y', labelsize=fontsize_labels)
    ax1.grid(True, alpha=0.3)
    
    # Enhanced value labels
    for bar, measure in zip(bars, variance_measures):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + max(variance_measures)*0.01, 
                f'{measure:.3f}', ha='center', va='bottom', fontsize=fontsize_labels, fontweight='bold')
    
    # Plot 2: Information ranking with consistent colors
    info_ranking = sorted(zip(sorted_group_names, variance_measures), key=lambda x: x[1], reverse=True)
    ranked_groups, ranked_measures = zip(*info_ranking)
    
    # Use reversed husl colors to maintain consistency
    ranking_colors = [colors[sorted_group_names.index(group)] for group in ranked_groups]
    
    ax2.barh(range(len(ranked_groups)), ranked_measures, alpha=alpha, color=ranking_colors)
    ax2.set_yticks(range(len(ranked_groups)))
    ax2.set_yticklabels(ranked_groups, fontsize=fontsize_labels)
    ax2.set_xlabel('Information Variance Measure', fontsize=fontsize_labels)
    ax2.set_title('Information Content Ranking (Highest to Lowest)', 
                 fontsize=fontsize_legend, fontweight='bold')
    ax2.tick_params(axis='x', labelsize=fontsize_labels)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels to horizontal bars
    for i, measure in enumerate(ranked_measures):
        ax2.text(measure + max(ranked_measures)*0.01, i, f'{measure:.3f}', 
                va='center', ha='left', fontsize=fontsize_labels, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path
