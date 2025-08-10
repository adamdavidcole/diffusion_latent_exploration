"""
Statistical significance plotting functionality.
Extracted from LatentTrajectoryAnalyzer.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from src.analysis.data_structures import LatentTrajectoryAnalysis
from src.visualization.visualization_config import VisualizationConfig

def plot_statistical_significance(results: LatentTrajectoryAnalysis, viz_dir: Path, 
                                 viz_config: VisualizationConfig = None, 
                                 labels_map: dict = None, **kwargs) -> Path:
    """Plot statistical significance analysis."""
    # Set defaults for optional parameters
    if viz_config is None:
        viz_config = VisualizationConfig()
    
    output_path = viz_dir / "statistical_significance.png"
    logger = logging.getLogger(__name__)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    significance_data = results.statistical_significance['trajectory_group_differences']
    summary_data = results.statistical_significance['statistical_summary']
    
    # Extract variance and mean differences
    comparisons = list(significance_data.keys())
    variance_diffs = [significance_data[comp]['variance_difference'] for comp in comparisons]
    mean_diffs = [significance_data[comp]['mean_difference'] for comp in comparisons]
    
    # Plot 1: Variance differences
    bars = ax1.bar(range(len(comparisons)), variance_diffs, alpha=0.7)
    ax1.set_xlabel('Group Comparisons')
    ax1.set_ylabel('Variance Difference')
    ax1.set_title('Statistical Significance: Variance Differences')
    ax1.set_xticks(range(len(comparisons)))
    ax1.set_xticklabels([comp.replace('_vs_', ' vs ') for comp in comparisons], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mean differences
    bars = ax2.bar(range(len(comparisons)), mean_diffs, alpha=0.7, color='orange')
    ax2.set_xlabel('Group Comparisons')
    ax2.set_ylabel('Mean Difference')
    ax2.set_title('Statistical Significance: Mean Differences')
    ax2.set_xticks(range(len(comparisons)))
    ax2.set_xticklabels([comp.replace('_vs_', ' vs ') for comp in comparisons], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Significance magnitude
    significance_magnitudes = [abs(vd) + abs(md) for vd, md in zip(variance_diffs, mean_diffs)]
    bars = ax3.bar(range(len(comparisons)), significance_magnitudes, alpha=0.7, color='red')
    ax3.set_xlabel('Group Comparisons')
    ax3.set_ylabel('Combined Difference Magnitude')
    ax3.set_title('Overall Statistical Significance')
    ax3.set_xticks(range(len(comparisons)))
    ax3.set_xticklabels([comp.replace('_vs_', ' vs ') for comp in comparisons], rotation=45, ha='right')
    
    # Plot 4: Summary statistics
    ax4.axis('off')
    summary_text = None
    try: 
        summary_text = f"""
Statistical Analysis Summary:

Groups Analyzed: {summary_data['groups_analyzed']}
Total Comparisons: {summary_data['comparisons_made']}

Variance Differences:
• Maximum: {max(variance_diffs):.6f}
• Minimum: {min(variance_diffs):.6f}
• Range: {max(variance_diffs) - min(variance_diffs):.6f}

Mean Differences:
• Maximum: {max(mean_diffs):.6f}
• Minimum: {min(mean_diffs):.6f}
• Range: {max(mean_diffs) - min(mean_diffs):.6f}

Most Significant Comparison:
{comparisons[significance_magnitudes.index(max(significance_magnitudes))].replace('_vs_', ' vs ')}
(Combined magnitude: {max(significance_magnitudes):.6f})
        """
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        summary_text = f"Error generating summary: {e}"

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path
