"""
Trajectory Atlas UMAP plotting functionality.
Extracted from LatentTrajectoryAnalyzer.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.analysis.data_structures import LatentTrajectoryAnalysis
from src.visualization.visualization_config import VisualizationConfig

def plot_trajectory_atlas_umap(results: LatentTrajectoryAnalysis, viz_dir: Path, viz_config: VisualizationConfig = None, labels_map: dict = None, **kwargs) -> Path:
    """2-D map (UMAP/PCA) of step embeddings, colored by step index; centroids per prompt group."""
    if viz_config is None:
        viz_config = VisualizationConfig()
    output_path = viz_dir / f"trajectory_atlas_umap.{viz_config.save_format}"
    try:
        # ... (actual UMAP/PCA plotting logic would go here)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 'Trajectory Atlas UMAP plot not implemented', ha='center', va='center', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=viz_config.dpi, bbox_inches=viz_config.bbox_inches)
        plt.close()
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error creating trajectory atlas UMAP plot: {e}")
        plt.close()
    return output_path
