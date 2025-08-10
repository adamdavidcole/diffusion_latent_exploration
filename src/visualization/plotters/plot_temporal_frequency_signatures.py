import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.analysis.data_structures import LatentTrajectoryAnalysis
from src.visualization.visualization_config import VisualizationConfig


def plot_temporal_frequency_signatures(
    results: LatentTrajectoryAnalysis, 
    viz_dir: Path, 
    viz_config: VisualizationConfig = None
) -> Path:
    """Plot temporal frequency analysis with consistent design system."""
    if viz_config is None:
        viz_config = VisualizationConfig()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=viz_config.figsize_standard)
    
    freq_data = results.temporal_coherence['temporal_frequency_signatures']
    
    # Get design system settings from config
    group_names = sorted(freq_data.keys())  # Alphabetical ordering
    colors = viz_config.get_colors(len(group_names))
    
    dominant_freqs = []
    dominant_powers = []
    
    for group_name in group_names:
        data = freq_data[group_name]
        if data['dominant_frequencies']:
            # Ensure we get scalar values - handle both arrays and scalars
            freq_val = data['dominant_frequencies'][0]
            power_val = data['dominant_powers'][0]
            
            # If they're arrays, take the mean
            if isinstance(freq_val, (list, tuple, np.ndarray)):
                freq_val = np.mean(freq_val)
            if isinstance(power_val, (list, tuple, np.ndarray)):
                power_val = np.mean(power_val)
                
            dominant_freqs.append(float(freq_val))
            dominant_powers.append(float(power_val))
        else:
            dominant_freqs.append(0.0)
            dominant_powers.append(0.0)
    
    # Plot 1: Dominant frequencies - unified design
    bars1 = ax1.bar(group_names, dominant_freqs, alpha=viz_config.alpha, color=colors)
    ax1.set_xlabel('Prompt Group', fontsize=viz_config.fontsize_labels)
    ax1.set_ylabel('Dominant Frequency', fontsize=viz_config.fontsize_labels)
    ax1.set_title('Primary Temporal Frequency by Group', 
                 fontsize=viz_config.fontsize_title, fontweight=viz_config.fontweight_title)
    ax1.tick_params(axis='x', rotation=45, labelsize=viz_config.fontsize_labels)
    ax1.tick_params(axis='y', labelsize=viz_config.fontsize_labels)
    ax1.grid(True, alpha=viz_config.grid_alpha)
    
    # Add value labels
    for bar, freq in zip(bars1, dominant_freqs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(dominant_freqs) * 0.01,
                f'{freq:.3f}', ha='center', va='bottom', fontsize=viz_config.fontsize_labels)
    
    # Plot 2: Spectral power - unified design
    bars2 = ax2.bar(group_names, dominant_powers, alpha=viz_config.alpha, color=colors)
    ax2.set_xlabel('Prompt Group', fontsize=viz_config.fontsize_labels)
    ax2.set_ylabel('Spectral Power', fontsize=viz_config.fontsize_labels)
    ax2.set_title('Dominant Frequency Power', 
                 fontsize=viz_config.fontsize_title, fontweight=viz_config.fontweight_title)
    ax2.tick_params(axis='x', rotation=45, labelsize=viz_config.fontsize_labels)
    ax2.tick_params(axis='y', labelsize=viz_config.fontsize_labels)
    ax2.grid(True, alpha=viz_config.grid_alpha)
    
    # Add value labels
    for bar, power in zip(bars2, dominant_powers):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(dominant_powers) * 0.01,
                f'{power:.3f}', ha='center', va='bottom', fontsize=viz_config.fontsize_labels)
    
    # Plot 3: Spectral centroid - unified design
    centroids = []
    for data in freq_data.values():
        centroid = data['spectral_centroid']
        if isinstance(centroid, (list, tuple, np.ndarray)):
            centroid = np.mean(centroid)
        centroids.append(float(centroid))
        
    bars3 = ax3.bar(group_names, centroids, alpha=viz_config.alpha, color=colors)
    ax3.set_xlabel('Prompt Group', fontsize=viz_config.fontsize_labels)
    ax3.set_ylabel('Spectral Centroid', fontsize=viz_config.fontsize_labels)
    ax3.set_title('Frequency Distribution Center', 
                 fontsize=viz_config.fontsize_title, fontweight=viz_config.fontweight_title)
    ax3.tick_params(axis='x', rotation=45, labelsize=viz_config.fontsize_labels)
    ax3.tick_params(axis='y', labelsize=viz_config.fontsize_labels)
    ax3.grid(True, alpha=viz_config.grid_alpha)
    
    # Add value labels
    for bar, centroid in zip(bars3, centroids):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(centroids) * 0.01,
                f'{centroid:.3f}', ha='center', va='bottom', fontsize=viz_config.fontsize_labels)
    
    # Plot 4: Spectral entropy - unified design
    entropies = []
    for data in freq_data.values():
        entropy = data['spectral_entropy']
        if isinstance(entropy, (list, tuple, np.ndarray)):
            entropy = np.mean(entropy)
        entropies.append(float(entropy))
        
    bars4 = ax4.bar(group_names, entropies, alpha=viz_config.alpha, color=colors)
    ax4.set_xlabel('Prompt Group', fontsize=viz_config.fontsize_labels)
    ax4.set_ylabel('Spectral Entropy', fontsize=viz_config.fontsize_labels)
    ax4.set_title('Temporal Frequency Diversity', 
                 fontsize=viz_config.fontsize_title, fontweight=viz_config.fontweight_title)
    ax4.tick_params(axis='x', rotation=45, labelsize=viz_config.fontsize_labels)
    ax4.tick_params(axis='y', labelsize=viz_config.fontsize_labels)
    ax4.grid(True, alpha=viz_config.grid_alpha)
    
    # Add value labels
    for bar, entropy in zip(bars4, entropies):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(entropies) * 0.01,
                f'{entropy:.3f}', ha='center', va='bottom', fontsize=viz_config.fontsize_labels)
    
    plt.tight_layout()
    
    output_path = viz_dir / f"temporal_frequency_signatures.{viz_config.save_format}"
    plt.savefig(output_path, dpi=viz_config.dpi, bbox_inches=viz_config.bbox_inches)
    plt.close()

    return output_path
