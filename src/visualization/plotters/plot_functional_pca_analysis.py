"""
Functional PCA analysis plotting functionality.
Extracted from LatentTrajectoryAnalyzer.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from src.analysis.data_structures import LatentTrajectoryAnalysis
from src.visualization.visualization_config import VisualizationConfig

def plot_functional_pca_analysis(results: LatentTrajectoryAnalysis, viz_dir: Path, viz_config: VisualizationConfig = None, labels_map: dict = None, **kwargs) -> Path:
    """Plot Functional PCA analysis showing trajectory shape decomposition."""
    if viz_config is None:
        viz_config = VisualizationConfig()
    output_path = viz_dir / f"functional_pca_analysis.{viz_config.save_format}"
    try:
        fpca_data = results.functional_pca_analysis
        sorted_group_names = sorted(fpca_data.keys())
        fig, axes = plt.subplots(2, 2, figsize=viz_config.figsize_standard)
        ax1, ax2, ax3, ax4 = axes.flatten()
        colors = viz_config.get_colors(len(sorted_group_names))

        # Plot 1: Mean Trajectories
        for i, group_name in enumerate(sorted_group_names):
            data = fpca_data[group_name]
            if 'error' not in data and data.get('mean_trajectory'):
                mean_traj = np.array(data['mean_trajectory'])
                if mean_traj.ndim == 2:
                    mean_traj_avg = np.mean(mean_traj, axis=1)
                    label = labels_map[group_name] if labels_map and group_name in labels_map else group_name
                    steps = range(len(mean_traj_avg))
                    ax1.plot(steps, mean_traj_avg, 'o-', label=label, color=colors[i], linewidth=viz_config.linewidth, markersize=viz_config.markersize)
        ax1.set_xlabel('Diffusion Step', fontsize=viz_config.fontsize_labels)
        ax1.set_ylabel('Mean Trajectory Value', fontsize=viz_config.fontsize_labels)
        ax1.set_title('Mean Trajectory Functions\n(FPCA Center)', fontsize=viz_config.fontsize_title, fontweight=viz_config.fontweight_title)
        ax1.legend(fontsize=viz_config.fontsize_legend)
        ax1.grid(True, alpha=viz_config.grid_alpha)

        # Plot 2: Explained Variance
        for i, group_name in enumerate(sorted_group_names):
            data = fpca_data[group_name]
            if 'error' not in data and data.get('explained_variance_ratio'):
                var_ratios = data['explained_variance_ratio'][:5]
                components = range(1, len(var_ratios) + 1)
                label = labels_map[group_name] if labels_map and group_name in labels_map else group_name
                ax2.plot(components, var_ratios, 'o-', label=label, color=colors[i], linewidth=viz_config.linewidth, markersize=viz_config.markersize)
        ax2.set_xlabel('Principal Component', fontsize=viz_config.fontsize_labels)
        ax2.set_ylabel('Explained Variance Ratio', fontsize=viz_config.fontsize_labels)
        ax2.set_title('FPCA Variance Decomposition\n(Modes of Variation)', fontsize=viz_config.fontsize_title, fontweight=viz_config.fontweight_title)
        ax2.legend(fontsize=viz_config.fontsize_legend)
        ax2.grid(True, alpha=viz_config.grid_alpha)

        # Plot 3: Effective Components (95% variance)
        effective_components = []
        for group_name in sorted_group_names:
            data = fpca_data[group_name]
            if 'error' not in data:
                effective_components.append(data.get('effective_components_95', 0))
            else:
                effective_components.append(0)
        bars = ax3.bar(sorted_group_names, effective_components, alpha=viz_config.alpha, color=colors)
        ax3.set_xlabel('Prompt Group', fontsize=viz_config.fontsize_labels)
        ax3.set_ylabel('Effective Components (95% var)', fontsize=viz_config.fontsize_labels)
        ax3.set_title('Functional Complexity\nComponents Needed for 95% Variance', fontsize=viz_config.fontsize_title, fontweight=viz_config.fontweight_title)
        ax3.tick_params(axis='x', rotation=45, labelsize=viz_config.fontsize_labels)
        ax3.grid(True, alpha=viz_config.grid_alpha)
        for bar, comp in zip(bars, effective_components):
            height = bar.get_height()
            ax3.annotate(f'{comp}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=viz_config.fontsize_labels)

        # Plot 4: Mode Diversity Index
        diversity_indices = []
        for group_name in sorted_group_names:
            data = fpca_data[group_name]
            if 'error' not in data:
                diversity_indices.append(data.get('mode_diversity_index', 0))
            else:
                diversity_indices.append(0)
        bars = ax4.bar(sorted_group_names, diversity_indices, alpha=viz_config.alpha, color=colors)
        ax4.set_xlabel('Prompt Group', fontsize=viz_config.fontsize_labels)
        ax4.set_ylabel('Mode Diversity Index', fontsize=viz_config.fontsize_labels)
        ax4.set_title('Trajectory Shape Diversity\nHigher = More Varied Shapes', fontsize=viz_config.fontsize_title, fontweight=viz_config.fontweight_title)
        ax4.tick_params(axis='x', rotation=45, labelsize=viz_config.fontsize_labels)
        ax4.grid(True, alpha=viz_config.grid_alpha)

        plt.tight_layout()
        plt.savefig(output_path, dpi=viz_config.dpi, bbox_inches=viz_config.bbox_inches)
        plt.close()
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error creating FPCA visualization: {e}")
        plt.close()
    return output_path
