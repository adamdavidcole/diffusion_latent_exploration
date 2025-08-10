import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

from src.analysis.data_structures import LatentTrajectoryAnalysis


def plot_edge_formation_trends_dashboard(
    results: LatentTrajectoryAnalysis, 
    viz_dir: Path, 
    labels_map: dict[str, str]
) -> Path:
    """Plot edge formation trends dashboard (extracted from spatial progression patterns)."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Get both spatial progression and edge density data
    spatial_data = results.spatial_patterns['spatial_progression_patterns']
    edge_data = results.spatial_patterns['edge_density_evolution']
    sorted_group_names = sorted(spatial_data.keys())
    colors = sns.color_palette("plasma", len(sorted_group_names))
    
    # Plot 1: Edge evolution patterns from spatial progression data
    ax1.set_title('Edge Formation Trends by Group\n(From Spatial Progression Analysis)')
    for i, group_name in enumerate(sorted_group_names):
        data = spatial_data[group_name]
        edge_patterns = data.get('edge_evolution_patterns', [])
        if edge_patterns:
            mean_pattern = np.mean(edge_patterns, axis=0)
            steps = list(range(len(mean_pattern)))
            label = labels_map[group_name]
            ax1.plot(steps, mean_pattern, 'o-', label=label, 
                    alpha=0.8, color=colors[i], linewidth=2)
    
    ax1.set_xlabel('Diffusion Step')
    ax1.set_ylabel('Edge Density')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mean evolution patterns from edge density analysis
    ax2.set_title('Edge Density Evolution Patterns\n(From Edge Density Analysis)')
    for i, group_name in enumerate(sorted_group_names):
        if group_name in edge_data:
            data = edge_data[group_name]
            evolution_pattern = data.get('mean_evolution_pattern', [])
            if evolution_pattern:
                steps = list(range(len(evolution_pattern)))
                label = labels_map[group_name]
                ax2.plot(steps, evolution_pattern, 's-', label=label, 
                        alpha=0.8, color=colors[i], linewidth=2)
    
    ax2.set_xlabel('Diffusion Step')
    ax2.set_ylabel('Mean Edge Density')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Edge formation trend distribution
    trend_counts = {'increasing': 0, 'decreasing': 0, 'stable': 0}
    for group_name in sorted_group_names:
        if group_name in edge_data:
            data = edge_data[group_name]
            trend = data.get('formation_trend', 'stable')
            if trend in trend_counts:
                trend_counts[trend] += 1
    
    if sum(trend_counts.values()) > 0:
        ax3.pie(trend_counts.values(), labels=trend_counts.keys(), autopct='%1.1f%%',
               colors=sns.color_palette("Set2", len(trend_counts)))
        ax3.set_title('Edge Formation Trend Distribution\n(Across All Groups)')
    else:
        ax3.text(0.5, 0.5, 'No edge trend data available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Edge Formation Trends (No Data)')
    
    # Plot 4: Edge density summary statistics
    mean_densities = []
    group_labels = []
    for group_name in sorted_group_names:
        if group_name in edge_data:
            data = edge_data[group_name]
            evolution_pattern = data.get('mean_evolution_pattern', [])
            if evolution_pattern:
                mean_densities.append(np.mean(evolution_pattern))
                group_labels.append(group_name)
    
    if mean_densities:
        bars = ax4.bar(group_labels, mean_densities, alpha=0.7, color=colors[:len(group_labels)])
        ax4.set_xlabel('Prompt Group')
        ax4.set_ylabel('Average Edge Density')
        ax4.set_title('Average Edge Density by Group')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, density in zip(bars, mean_densities):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(mean_densities) * 0.01,
                    f'{density:.3f}', ha='center', va='bottom', fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'No edge density data available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Average Edge Density (No Data)')
    
    plt.tight_layout()
    
    output_path = viz_dir / "edge_formation_trends_dashboard.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path
