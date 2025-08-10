import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

from src.analysis.data_structures import LatentTrajectoryAnalysis


def plot_group_separability(results: LatentTrajectoryAnalysis, viz_dir: Path) -> Path:
    """Plot group separability analysis with consistent design system."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    separability_data = results.group_separability['inter_group_distances']
    
    # Design system settings
    alpha = 0.8
    fontsize_labels = 8
    fontsize_legend = 9
    
    # Create distance matrix
    group_names = set()
    for key in separability_data.keys():
        group1, group2 = key.split('_vs_')
        group_names.add(group1)
        group_names.add(group2)
    
    group_names = sorted(list(group_names))
    n_groups = len(group_names)
    distance_matrix = np.zeros((n_groups, n_groups))
    colors = sns.color_palette("husl", n_groups)
    
    for i, group1 in enumerate(group_names):
        for j, group2 in enumerate(group_names):
            if i != j:
                key1 = f"{group1}_vs_{group2}"
                key2 = f"{group2}_vs_{group1}"
                if key1 in separability_data:
                    distance_matrix[i, j] = separability_data[key1]
                elif key2 in separability_data:
                    distance_matrix[i, j] = separability_data[key2]
    
    # Plot 1: Distance matrix heatmap with enhanced styling
    im1 = ax1.imshow(distance_matrix, cmap='RdYlBu_r', alpha=0.9)
    ax1.set_xticks(range(n_groups))
    ax1.set_yticks(range(n_groups))
    ax1.set_xticklabels(group_names, rotation=45, fontsize=fontsize_labels)
    ax1.set_yticklabels(group_names, fontsize=fontsize_labels)
    ax1.set_title('Inter-Group Distance Matrix\n(Trajectory Separability)', 
                 fontsize=fontsize_legend, fontweight='bold')
    
    # Enhanced colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Distance', fontsize=fontsize_labels, fontweight='bold')
    cbar1.ax.tick_params(labelsize=fontsize_labels)
    
    # Plot 2: Average distances with design system colors
    avg_distances = np.mean(distance_matrix, axis=1)
    bars = ax2.bar(group_names, avg_distances, alpha=alpha, color=colors)
    ax2.set_xlabel('Prompt Group', fontsize=fontsize_labels)
    ax2.set_ylabel('Average Distance to Other Groups', fontsize=fontsize_labels)
    ax2.set_title('Group Isolation Index\n(Higher = More Unique)', 
                 fontsize=fontsize_legend, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45, labelsize=fontsize_labels)
    ax2.tick_params(axis='y', labelsize=fontsize_labels)
    ax2.grid(True, alpha=0.3)
    
    # Enhanced value labels
    for bar, dist in zip(bars, avg_distances):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + max(avg_distances) * 0.01, 
                f'{dist:.2f}', ha='center', va='bottom', fontsize=fontsize_labels,
                fontweight='bold')
    
    plt.tight_layout()
    
    output_path = viz_dir / "group_separability.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path
