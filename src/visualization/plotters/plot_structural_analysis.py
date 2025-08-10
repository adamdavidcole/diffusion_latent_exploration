"""
Structural analysis plotting functionality.
Extracted from LatentTrajectoryAnalyzer.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import traceback

from src.analysis.data_structures import LatentTrajectoryAnalysis
from src.config.visualization_config import VisualizationConfig


def plot_structural_analysis(results: LatentTrajectoryAnalysis, viz_dir: Path, 
                            viz_config: VisualizationConfig = None, 
                            labels_map: dict = None, **kwargs) -> Path:
    """Plot structural analysis visualizations."""
    # Set defaults for optional parameters
    if viz_config is None:
        viz_config = VisualizationConfig()
    
    output_path = viz_dir / "structural_analysis.png"
    logger = logging.getLogger(__name__)
    
    try:
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 18))
        
        structural_data = results.structural_analysis
        sorted_group_names = sorted(structural_data.keys())
        colors = sns.color_palette("plasma", len(sorted_group_names))
        
        # Plot 1: Variance Analysis
        overall_variances = [structural_data[group]['latent_space_variance']['overall_variance'] for group in sorted_group_names]
        video_variances = [structural_data[group]['latent_space_variance']['variance_across_videos'] for group in sorted_group_names]
        step_variances = [structural_data[group]['latent_space_variance']['variance_across_steps'] for group in sorted_group_names]
        
        x = np.arange(len(sorted_group_names))
        width = 0.25
        
        ax1.bar(x - width, overall_variances, width, label='Overall', alpha=0.7)
        ax1.bar(x, video_variances, width, label='Across Videos', alpha=0.7)
        ax1.bar(x + width, step_variances, width, label='Across Steps', alpha=0.7)
        
        ax1.set_ylabel('Variance')
        ax1.set_title('Latent Space Variance Analysis')
        ax1.set_xticks(x)
        ax1.set_xticklabels(sorted_group_names, rotation=45)
        ax1.legend()
        
        # Plot 2: PCA Effective Dimensionality
        effective_dims = [structural_data[group]['pca_analysis']['effective_dimensionality'] for group in sorted_group_names]
        cumulative_var_90 = [structural_data[group]['pca_analysis']['cumulative_variance_90'] for group in sorted_group_names]
        
        bars = ax2.bar(sorted_group_names, effective_dims, alpha=0.7, color=colors)
        ax2.set_ylabel('Effective Dimensionality')
        ax2.set_title('PCA Effective Dimensionality (90% Variance)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add cumulative variance labels
        for bar, dim, cum_var in zip(bars, effective_dims, cumulative_var_90):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{cum_var:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 3: Shannon Entropy
        entropies = [structural_data[group]['shannon_entropy']['entropy_estimate'] for group in sorted_group_names]
        # Handle missing entropy per dimension std - use the new stats structure
        try:
            entropy_stds = [structural_data[group]['shannon_entropy']['entropy_per_dimension_stats']['std'] 
                           for group in sorted_group_names]
        except KeyError:
            # Fallback for old data format
            try:
                entropy_stds = [np.std(structural_data[group]['shannon_entropy']['entropy_per_dimension']) 
                               for group in sorted_group_names]
            except KeyError:
                entropy_stds = [0.0] * len(sorted_group_names)
        
        bars = ax3.bar(sorted_group_names, entropies, yerr=entropy_stds, alpha=0.7, color=colors, capsize=5)
        ax3.set_ylabel('Shannon Entropy Estimate')
        ax3.set_title('Information Content (Shannon Entropy)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: KL Divergence from Baseline
        kl_divergences = []
        baseline_group = None
        for group in sorted_group_names:
            kl_div = structural_data[group]['kl_divergence']['divergence_from_baseline']
            baseline = structural_data[group]['kl_divergence']['baseline_group']
            if baseline is not None:
                kl_divergences.append(kl_div)
                if baseline_group is None:
                    baseline_group = baseline
            else:
                kl_divergences.append(0.0)  # Baseline group itself
        
        bars = ax4.bar(sorted_group_names, kl_divergences, alpha=0.7, color=colors)
        ax4.set_ylabel('KL Divergence from Baseline')
        ax4.set_title(f'Structural Divergence from {baseline_group or "Baseline"}')
        ax4.tick_params(axis='x', rotation=45)
        
        # Plot 5: Structural Complexity Measures
        rank_estimates = [structural_data[group]['structural_complexity']['rank_estimate'] for group in sorted_group_names]
        condition_numbers = [structural_data[group]['structural_complexity']['condition_number'] for group in sorted_group_names]
        
        # Use log scale for condition numbers if they're very large
        log_condition_numbers = [np.log10(max(cn, 1e-10)) for cn in condition_numbers]
        
        ax5_twin = ax5.twinx()
        bars1 = ax5.bar([x - width/2 for x in range(len(sorted_group_names))], rank_estimates, 
                       width, alpha=0.7, color='blue', label='Rank Estimate')
        bars2 = ax5_twin.bar([x + width/2 for x in range(len(sorted_group_names))], log_condition_numbers, 
                            width, alpha=0.7, color='red', label='Log10(Condition Number)')
        
        ax5.set_ylabel('Rank Estimate', color='blue')
        ax5_twin.set_ylabel('Log10(Condition Number)', color='red')
        ax5.set_title('Structural Complexity Measures')
        ax5.set_xticks(range(len(sorted_group_names)))
        ax5.set_xticklabels(sorted_group_names, rotation=45)
        
        # Plot 6: Spectral Analysis
        spectral_entropies = [structural_data[group]['structural_complexity']['spectral_entropy'] for group in sorted_group_names]
        trace_norms = [structural_data[group]['structural_complexity']['trace_norm'] for group in sorted_group_names]
        
        ax6.scatter(spectral_entropies, trace_norms, s=100, alpha=0.7, c=range(len(sorted_group_names)), cmap='viridis')
        for i, group in enumerate(sorted_group_names):
            ax6.annotate(group, (spectral_entropies[i], trace_norms[i]), xytext=(5, 5), textcoords='offset points')
        
        ax6.set_xlabel('Spectral Entropy')
        ax6.set_ylabel('Trace Norm')
        ax6.set_title('Spectral Analysis: Entropy vs Trace Norm')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error(f"Failed to create structural analysis visualization: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        # Create a simple fallback visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Structural Analysis Visualization Failed\nError: {str(e)}", 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Structural Analysis - Error')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return output_path
