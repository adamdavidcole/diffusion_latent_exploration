import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

from src.analysis.data_structures import LatentTrajectoryAnalysis


def plot_individual_video_coherence_dashboard(
    results: LatentTrajectoryAnalysis, 
    viz_dir: Path
) -> Path:
    """Plot individual video coherence trajectories by prompt group (extracted from spatial coherence)."""
    coherence_data = results.spatial_patterns['spatial_coherence_patterns']
    sorted_group_names = sorted(coherence_data.keys())
    
    # Determine the grid layout based on the number of groups
    n_groups = len(sorted_group_names)
    n_cols = min(3, n_groups)
    n_rows = (n_groups + n_cols - 1) // n_cols
    
    # Create figure with appropriate size
    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
    fig.suptitle('Individual Video Coherence Trajectories by Prompt Group', 
                fontsize=16, fontweight='bold')
    
    # Color palette for videos within each group
    video_colors = sns.color_palette("tab10", 10)  # Support up to 10 videos per group
    
    for idx, group_name in enumerate(sorted_group_names):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        data = coherence_data[group_name]
        coherence_evolution = data.get('coherence_evolution', [])
        
        videos_plotted = 0
        if coherence_evolution:
            # Plot individual video trajectories
            max_videos_to_show = min(8, len(coherence_evolution))  # Show up to 8 videos per group
            
            for i, video_coherence in enumerate(coherence_evolution[:max_videos_to_show]):
                if isinstance(video_coherence, (list, np.ndarray)) and len(video_coherence) > 0:
                    # Handle multidimensional video coherence data
                    coherence_array = np.array(video_coherence)
                    if coherence_array.ndim > 1:
                        # If multidimensional, take mean across spatial dimensions
                        coherence_1d = np.mean(coherence_array.reshape(coherence_array.shape[0], -1), axis=1)
                    else:
                        coherence_1d = coherence_array
                    
                    steps = list(range(len(coherence_1d)))
                    ax.plot(steps, coherence_1d, alpha=0.7, linewidth=1.5,
                           color=video_colors[i % len(video_colors)], 
                           label=f'Video {i+1}')
                    videos_plotted += 1
            
            # Overlay mean trajectory if available
            mean_trajectory = data.get('mean_coherence_trajectory', [])
            if mean_trajectory:
                mean_array = np.array(mean_trajectory)
                if mean_array.ndim > 1:
                    mean_1d = np.mean(mean_array.reshape(mean_array.shape[0], -1), axis=1)
                else:
                    mean_1d = mean_array
                
                steps = list(range(len(mean_1d)))
                ax.plot(steps, mean_1d, 'k-', linewidth=3, alpha=0.8, 
                       label='Group Mean')
        
        if videos_plotted > 0:
            ax.set_xlabel('Diffusion Step')
            ax.set_ylabel('Spatial Coherence')
            ax.set_title(f'{group_name}\n({videos_plotted} Individual Videos)')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No individual video\ncoherence data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{group_name}\n(No Data)')
    
    plt.tight_layout()
    
    output_path = viz_dir / "individual_video_coherence_dashboard.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path
