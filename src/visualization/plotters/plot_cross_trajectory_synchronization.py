import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.analysis.data_structures import LatentTrajectoryAnalysis

def plot_cross_trajectory_synchronization(
        results: LatentTrajectoryAnalysis, 
        viz_dir: Path
    ) -> Path:
    """Plot cross-trajectory synchronization analysis."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    sync_data = results.temporal_coherence['cross_trajectory_synchronization']
    
    # Extract data with alphabetical ordering
    group_names = sorted(sync_data.keys())
    mean_correlations = [sync_data[group]['mean_correlation'] for group in group_names]
    correlation_stds = [sync_data[group]['correlation_std'] for group in group_names]
    high_sync_ratios = [sync_data[group]['high_sync_ratio'] for group in group_names]
    
    colors = sns.color_palette("husl", len(group_names))
    
    # Plot 1: Mean correlation by group
    bars1 = ax1.bar(group_names, mean_correlations, alpha=0.7, color=colors)
    ax1.set_ylabel('Mean Cross-Trajectory Correlation')
    ax1.set_title('Cross-Trajectory Synchronization Strength')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, corr in zip(bars1, mean_correlations):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(mean_correlations) * 0.01,
                f'{corr:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Correlation variability
    ax2.errorbar(group_names, mean_correlations, yerr=correlation_stds, 
                fmt='o', capsize=5, capthick=2, linewidth=2, markersize=4, alpha=0.8)
    ax2.set_ylabel('Correlation ± Std Dev')
    ax2.set_title('Synchronization Consistency')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: High synchronization ratio
    bars3 = ax3.bar(group_names, high_sync_ratios, alpha=0.7, 
                    color=sns.color_palette("plasma", len(group_names)))
    ax3.set_ylabel('High Sync Ratio (>0.7 correlation)')
    ax3.set_title('Percentage of Highly Synchronized Videos')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add percentage labels
    for bar, ratio in zip(bars3, high_sync_ratios):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(high_sync_ratios) * 0.01,
                f'{ratio:.1%}', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Synchronization ranking
    sync_ranking = sorted(zip(group_names, mean_correlations), key=lambda x: x[1], reverse=True)
    ranked_groups, ranked_corrs = zip(*sync_ranking)
    
    ax4.barh(range(len(ranked_groups)), ranked_corrs, alpha=0.7,
            color=sns.color_palette("coolwarm", len(ranked_groups)))
    ax4.set_yticks(range(len(ranked_groups)))
    ax4.set_yticklabels(ranked_groups)
    ax4.set_xlabel('Mean Cross-Trajectory Correlation')
    ax4.set_title('Synchronization Ranking (Best to Worst)')
    
    plt.tight_layout()

    output_path = viz_dir / "cross_trajectory_synchronization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path