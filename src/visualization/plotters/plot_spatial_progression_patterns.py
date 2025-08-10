import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.analysis.data_structures import LatentTrajectoryAnalysis


def plot_spatial_progression_patterns(
    results: LatentTrajectoryAnalysis, 
    viz_dir: Path, 
    labels_map: dict[str, str]
) -> Path:
    """Plot spatial progression pattern analysis."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    spatial_data = results.spatial_patterns['spatial_progression_patterns']
    group_names = sorted(spatial_data.keys())
    colors = sns.color_palette("husl", len(group_names))
    
    # Plot 1: Progression consistency
    consistency_values = [spatial_data[group]['progression_consistency'] for group in group_names]
    bars1 = ax1.bar(group_names, consistency_values, alpha=0.7, color=colors)
    ax1.set_xlabel('Prompt Group')
    ax1.set_ylabel('Progression Consistency')
    ax1.set_title('Spatial Progression Consistency')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Progression variability
    variability_values = [spatial_data[group]['progression_variability'] for group in group_names]
    bars2 = ax2.bar(group_names, variability_values, alpha=0.7, 
                   color=sns.color_palette("viridis", len(group_names)))
    ax2.set_xlabel('Prompt Group')
    ax2.set_ylabel('Progression Variability')
    ax2.set_title('Spatial Progression Variability')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Step deltas evolution over time
    for i, group_name in enumerate(group_names):
        step_deltas = spatial_data[group_name]['step_deltas_mean']
        steps = range(len(step_deltas))
        label = labels_map[group_name]
        ax3.plot(steps, step_deltas, 'o-', label=label, 
                color=colors[i], alpha=0.8, linewidth=2)
    
    ax3.set_xlabel('Diffusion Step')
    ax3.set_ylabel('Step Delta Mean')
    ax3.set_title('Spatial Step Delta Evolution')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Step delta standard deviation patterns
    for i, group_name in enumerate(group_names):
        step_deltas_std = spatial_data[group_name]['step_deltas_std']
        steps = range(len(step_deltas_std))
        label = labels_map[group_name]
        ax4.plot(steps, step_deltas_std, '^-', label=label, 
                color=colors[i], alpha=0.8, linewidth=2, markersize=4)
    
    ax4.set_xlabel('Diffusion Step')
    ax4.set_ylabel('Step Delta Std Dev')
    ax4.set_title('Spatial Step Delta Variability')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = viz_dir / "spatial_progression_patterns.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path
