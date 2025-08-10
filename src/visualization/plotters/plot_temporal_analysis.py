"""
Temporal analysis plotting functionality.
Extracted from LatentTrajectoryAnalyzer.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import traceback

from src.analysis.latent_trajectory_analyzer import LatentTrajectoryAnalysis
from src.config.visualization_config import VisualizationConfig


def plot_temporal_analysis(results: LatentTrajectoryAnalysis, viz_dir: Path, 
                          viz_config: VisualizationConfig = None, 
                          labels_map: dict = None, **kwargs) -> Path:
    """Plot temporal trajectory analysis visualizations."""
    # Set defaults for optional parameters
    if viz_config is None:
        viz_config = VisualizationConfig()
    
    output_path = viz_dir / "temporal_analysis.png"
    logger = logging.getLogger(__name__)
    
    try:
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 18))
        
        temporal_data = results.temporal_analysis
        sorted_group_names = sorted(temporal_data.keys())
        colors = sns.color_palette("viridis", len(sorted_group_names))
        
        # Plot 1: Trajectory Length Distribution
        lengths = [temporal_data[group]['trajectory_length']['mean_length'] for group in sorted_group_names]
        length_stds = [temporal_data[group]['trajectory_length']['std_length'] for group in sorted_group_names]
        
        bars = ax1.bar(sorted_group_names, lengths, yerr=length_stds, alpha=0.7, color=colors, capsize=5)
        ax1.set_ylabel('Mean Trajectory Length')
        ax1.set_title('Trajectory Length by Group')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels - fix the type error by ensuring proper calculation
        max_std = max(length_stds) if length_stds else 0
        for bar, length in zip(bars, lengths):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_std*0.1, 
                    f'{length:.2f}', ha='center', va='bottom')
        
        # Plot 2: Velocity Analysis
        mean_velocities = [temporal_data[group]['velocity_analysis']['overall_mean_velocity'] for group in sorted_group_names]
        velocity_vars = [temporal_data[group]['velocity_analysis']['overall_velocity_variance'] for group in sorted_group_names]
        
        ax2.scatter(mean_velocities, velocity_vars, s=100, alpha=0.7, c=range(len(sorted_group_names)), cmap='plasma')
        for i, group in enumerate(sorted_group_names):
            ax2.annotate(group, (mean_velocities[i], velocity_vars[i]), xytext=(5, 5), textcoords='offset points')
        
        ax2.set_xlabel('Mean Velocity')
        ax2.set_ylabel('Velocity Variance')
        ax2.set_title('Velocity Phase Space')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Acceleration Analysis
        mean_accelerations = [temporal_data[group]['acceleration_analysis']['overall_mean_acceleration'] for group in sorted_group_names]
        
        bars = ax3.bar(sorted_group_names, mean_accelerations, alpha=0.7, color=colors)
        ax3.set_ylabel('Mean Acceleration')
        ax3.set_title('Acceleration by Group')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Endpoint Distance vs Tortuosity
        endpoint_dists = [temporal_data[group]['endpoint_distance']['mean_endpoint_distance'] for group in sorted_group_names]
        tortuosities = [temporal_data[group]['tortuosity']['mean_tortuosity'] for group in sorted_group_names]
        
        ax4.scatter(endpoint_dists, tortuosities, s=100, alpha=0.7, c=range(len(sorted_group_names)), cmap='coolwarm')
        for i, group in enumerate(sorted_group_names):
            ax4.annotate(group, (endpoint_dists[i], tortuosities[i]), xytext=(5, 5), textcoords='offset points')
        
        ax4.set_xlabel('Mean Endpoint Distance')
        ax4.set_ylabel('Mean Tortuosity')
        ax4.set_title('Path Efficiency Analysis')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Semantic Convergence Rate
        convergence_rates = [temporal_data[group]['semantic_convergence']['convergence_rate'] for group in sorted_group_names]
        
        bars = ax5.bar(sorted_group_names, convergence_rates, alpha=0.7, color=colors)
        ax5.set_ylabel('Convergence Rate')
        ax5.set_title('Semantic Convergence Rate')
        ax5.tick_params(axis='x', rotation=45)
        
        # Plot 6: Half-life Distribution
        half_lives = [temporal_data[group]['semantic_convergence']['mean_half_life'] for group in sorted_group_names]
        
        bars = ax6.bar(sorted_group_names, half_lives, alpha=0.7, color=colors)
        ax6.set_ylabel('Mean Half-life (steps)')
        ax6.set_title('Convergence Half-life')
        ax6.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error(f"Failed to create temporal analysis visualization: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        # Create a simple fallback visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Temporal Analysis Visualization Failed\nError: {str(e)}", 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Temporal Analysis - Error')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return output_path
