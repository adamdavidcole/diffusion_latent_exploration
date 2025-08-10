"""
Complexity measures plotting functionality.
Extracted from LatentTrajectoryAnalyzer.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.analysis.latent_trajectory_analyzer import LatentTrajectoryAnalysis
from src.config.visualization_config import VisualizationConfig


def plot_complexity_measures(results: LatentTrajectoryAnalysis, viz_dir: Path, 
                            viz_config: VisualizationConfig = None, 
                            labels_map: dict = None, **kwargs) -> Path:
    """Plot complexity measures analysis."""
    # Set defaults for optional parameters
    if viz_config is None:
        viz_config = VisualizationConfig()
    
    output_path = viz_dir / "complexity_measures.png"
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    complexity_data = results.complexity_measures['trajectory_complexity']
    evolution_data = results.complexity_measures['evolution_complexity']
    sorted_group_names = sorted(complexity_data.keys())
    colors = sns.color_palette("rocket", len(sorted_group_names))
    
    # Plot 1: Standard deviation
    std_values = [complexity_data[group]['standard_deviation'] for group in sorted_group_names]
    bars = ax1.bar(sorted_group_names, std_values, alpha=0.7, color=colors)
    ax1.set_xlabel('Prompt Group')
    ax1.set_ylabel('Standard Deviation')
    ax1.set_title('Trajectory Complexity: Standard Deviation')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Value range
    range_values = [complexity_data[group]['value_range'] for group in sorted_group_names]
    bars = ax2.bar(sorted_group_names, range_values, alpha=0.7, color=colors)
    ax2.set_xlabel('Prompt Group')
    ax2.set_ylabel('Value Range')
    ax2.set_title('Trajectory Complexity: Value Range')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Temporal variation
    temporal_variations = [evolution_data[group]['temporal_variation'] for group in sorted_group_names]
    bars = ax3.bar(sorted_group_names, temporal_variations, alpha=0.7, color=colors)
    ax3.set_xlabel('Prompt Group')
    ax3.set_ylabel('Temporal Variation')
    ax3.set_title('Evolution Complexity: Temporal Variation')
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Complexity correlation matrix
    complexity_matrix = np.array([std_values, range_values, temporal_variations])
    complexity_labels = ['Std Dev', 'Range', 'Temporal Var']
    
    im = ax4.imshow(complexity_matrix, cmap='RdYlBu_r', aspect='auto')
    ax4.set_xticks(range(len(sorted_group_names)))
    ax4.set_yticks(range(len(complexity_labels)))
    ax4.set_xticklabels(sorted_group_names, rotation=45)
    ax4.set_yticklabels(complexity_labels)
    ax4.set_title('Complexity Measures Heatmap')
    plt.colorbar(im, ax=ax4, label='Complexity Value')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path
