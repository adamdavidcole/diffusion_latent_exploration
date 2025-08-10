"""
Log Volume Delta Panel plotting functionality.
Extracted from LatentTrajectoryAnalyzer.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.analysis.data_structures import LatentTrajectoryAnalysis
from src.visualization.visualization_config import VisualizationConfig

def plot_log_volume_delta_panel(results: LatentTrajectoryAnalysis, viz_dir: Path, viz_config: VisualizationConfig = None, labels_map: dict = None, title_suffix: str = "", **kwargs) -> Path:
    """One clean bar panel: mean individual log-volume Δ% vs baseline group."""
    if viz_config is None:
        viz_config = VisualizationConfig()
    output_path = viz_dir / f"log_volume_delta_vs_baseline.{viz_config.save_format}"
    try:
        geom = results.individual_trajectory_geometry
        groups = sorted(geom.keys())
        means = np.array([float(geom[g]['log_volume_stats']['mean']) if 'error' not in geom[g] else np.nan for g in groups])
        if means.size == 0:
            return output_path
        base = means[0]
        delta = 100.0 * (means - base) / (base + 1e-12)
        plt.figure(figsize=(8, 4.5))
        plt.bar(groups, delta, color=plt.get_cmap('tab10')(0))
        plt.title(f"Per-trajectory Log BBox Volume Δ% vs baseline{(' — ' + title_suffix) if title_suffix else ''}")
        plt.ylabel("% change"); plt.xticks(rotation=45); plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=viz_config.dpi, bbox_inches=viz_config.bbox_inches)
        plt.close()
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error creating log volume delta panel: {e}")
        plt.close()
    return output_path
