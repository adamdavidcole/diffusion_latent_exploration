"""
Intrinsic Dimension Analysis plotting functionality.
Extracted from LatentTrajectoryAnalyzer.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.analysis.data_structures import LatentTrajectoryAnalysis
from src.visualization.visualization_config import VisualizationConfig

def plot_intrinsic_dimension_analysis(results: LatentTrajectoryAnalysis, viz_dir: Path, viz_config: VisualizationConfig = None, labels_map: dict = None, **kwargs) -> Path:
    """Plot intrinsic dimension analysis showing manifold complexity."""
    if viz_config is None:
        viz_config = VisualizationConfig()
    output_path = viz_dir / f"intrinsic_dimension_analysis.{viz_config.save_format}"
    try:
        id_data = results.intrinsic_dimension_analysis
        if not id_data:
            fig, ax = plt.subplots(1,1, figsize=viz_config.figsize_standard)
            ax.text(0.5, 0.5, "Intrinsic dimension not computed", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(output_path, dpi=viz_config.dpi, bbox_inches=viz_config.bbox_inches)
            plt.close()
            return output_path
        sorted_group_names = sorted(id_data.keys())
        colors = viz_config.get_colors(len(sorted_group_names))
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=viz_config.figsize_standard)
        # Plot 1: Consensus Intrinsic Dimension
        consensus_dims = []
        for group_name in sorted_group_names:
            data = id_data[group_name]
            if 'error' not in data:
                consensus_dims.append(data.get('consensus_intrinsic_dimension', 0))
            else:
                consensus_dims.append(0)
        bars = ax1.bar(sorted_group_names, consensus_dims, alpha=viz_config.alpha, color=colors)
        ax1.set_xlabel('Prompt Group', fontsize=viz_config.fontsize_labels)
        ax1.set_ylabel('Intrinsic Dimension', fontsize=viz_config.fontsize_labels)
        ax1.set_title('Manifold Complexity\n(Consensus Intrinsic Dimension)', fontsize=viz_config.fontsize_title, fontweight=viz_config.fontweight_title)
        ax1.tick_params(axis='x', rotation=45, labelsize=viz_config.fontsize_labels)
        ax1.grid(True, alpha=viz_config.grid_alpha)
        for bar, dim in zip(bars, consensus_dims):
            height = bar.get_height()
            ax1.annotate(f'{dim:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=viz_config.fontsize_labels)
        # Plot 2: Dimension Reduction Ratio
        reduction_ratios = []
        for group_name in sorted_group_names:
            data = id_data[group_name]
            if 'error' not in data:
                reduction_ratios.append(data.get('dimension_reduction_ratio', 0))
            else:
                reduction_ratios.append(0)
        bars = ax2.bar(sorted_group_names, reduction_ratios, alpha=viz_config.alpha, color=colors)
        ax2.set_xlabel('Prompt Group', fontsize=viz_config.fontsize_labels)
        ax2.set_ylabel('Intrinsic/Ambient Ratio', fontsize=viz_config.fontsize_labels)
        ax2.set_title('Dimension Efficiency\n(Lower = More Efficient Representation)', fontsize=viz_config.fontsize_title, fontweight=viz_config.fontweight_title)
        ax2.tick_params(axis='x', rotation=45, labelsize=viz_config.fontsize_labels)
        ax2.grid(True, alpha=viz_config.grid_alpha)
        # Plot 3: Multiple ID Estimates Comparison
        pca_95_dims, mle_dims, twonn_dims = [], [], []
        for group_name in sorted_group_names:
            data = id_data[group_name]
            if 'error' not in data:
                pca_95_dims.append(data.get('pca_95_percent', 0))
                mle_dims.append(data.get('mle_estimate', 0))
                twonn_dims.append(data.get('twonn_estimate', 0))
            else:
                pca_95_dims.append(0)
                mle_dims.append(0)
                twonn_dims.append(0)
        x = np.arange(len(sorted_group_names))
        width = 0.25
        bars1 = ax3.bar(x - width, pca_95_dims, width, label='PCA (95%)', alpha=viz_config.alpha)
        bars2 = ax3.bar(x, mle_dims, width, label='MLE', alpha=viz_config.alpha)
        bars3 = ax3.bar(x + width, twonn_dims, width, label='TwoNN', alpha=viz_config.alpha)
        ax3.set_xlabel('Prompt Group', fontsize=viz_config.fontsize_labels)
        ax3.set_ylabel('Estimated Dimension', fontsize=viz_config.fontsize_labels)
        ax3.set_title('ID Estimation Methods Comparison\n(Multiple Algorithms)', fontsize=viz_config.fontsize_title, fontweight=viz_config.fontweight_title)
        ax3.set_xticks(x)
        ax3.set_xticklabels(sorted_group_names, rotation=45, fontsize=viz_config.fontsize_labels)
        ax3.legend(fontsize=viz_config.fontsize_legend)
        ax3.grid(True, alpha=viz_config.grid_alpha)
        # Plot 4: Complexity Categories
        complexity_categories = {'low': 0, 'medium': 0, 'high': 0}
        for group_name in sorted_group_names:
            data = id_data[group_name]
            if 'error' not in data:
                complexity = data.get('manifold_complexity', 'low')
                complexity_categories[complexity] += 1
        categories = list(complexity_categories.keys())
        counts = list(complexity_categories.values())
        colors_cat = ['green', 'orange', 'red']
        bars = ax4.bar(categories, counts, alpha=viz_config.alpha, color=colors_cat)
        ax4.set_xlabel('Complexity Category', fontsize=viz_config.fontsize_labels)
        ax4.set_ylabel('Number of Groups', fontsize=viz_config.fontsize_labels)
        ax4.set_title('Manifold Complexity Distribution\n(Low<10, Medium<50, High≥50)', fontsize=viz_config.fontsize_title, fontweight=viz_config.fontweight_title)
        ax4.grid(True, alpha=viz_config.grid_alpha)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax4.annotate(f'{count}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=viz_config.fontsize_labels)
        plt.tight_layout()
        plt.savefig(output_path, dpi=viz_config.dpi, bbox_inches=viz_config.bbox_inches)
        plt.close()
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error creating intrinsic dimension visualization: {e}")
        plt.close()
    return output_path
