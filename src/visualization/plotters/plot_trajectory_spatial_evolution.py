import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.analysis.data_structures import LatentTrajectoryAnalysis

def plot_trajectory_spatial_evolution(results: LatentTrajectoryAnalysis, viz_dir: Path, labels_map: dict[str, str]) -> Path:
    """Plot the U-shaped trajectory spatial evolution pattern."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract trajectory patterns with alphabetical ordering
    spatial_data = results.spatial_patterns['trajectory_spatial_evolution']
    
    # Sort group names alphabetically for consistent ordering
    sorted_group_names = sorted(spatial_data.keys())
    
    # Plot 1: Individual trajectory patterns
    colors = sns.color_palette("husl", len(sorted_group_names))
    for i, group_name in enumerate(sorted_group_names):
        data = spatial_data[group_name]
        trajectory_pattern = data['trajectory_pattern']
        steps = list(range(len(trajectory_pattern)))
        label = labels_map[group_name]
        ax1.plot(steps, trajectory_pattern, 'o-', label=label, alpha=0.8, linewidth=2, 
                markersize=3, color=colors[i])
    
    ax1.set_xlabel('Diffusion Step')
    ax1.set_ylabel('Spatial Variance')
    ax1.set_title('Trajectory Spatial Evolution Patterns\n(Universal U-Shaped Denoising Pattern)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Evolution ratio comparison
    evolution_ratios = [spatial_data[group]['evolution_ratio'] for group in sorted_group_names]
    
    bars = ax2.bar(sorted_group_names, evolution_ratios, alpha=0.7, color=colors)
    ax2.set_xlabel('Prompt Group')
    ax2.set_ylabel('Late/Early Spatial Variance Ratio')
    ax2.set_title('Spatial Evolution Ratio by Prompt\n(Recovery Strength in Late Diffusion)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, ratio in zip(bars, evolution_ratios):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(evolution_ratios) * 0.01,
                f'{ratio:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()

    output_path = viz_dir / "trajectory_spatial_evolution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path