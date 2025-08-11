"""
Log Volume Delta Panel plotting functionality.
Extracted from LatentTrajectoryAnalyzer.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict
from src.analysis.data_structures import LatentTrajectoryAnalysis
from src.visualization.visualization_config import VisualizationConfig

# src/visualization/plotters/plot_log_volume_delta_panel.py

def plot_log_volume_delta_panel(
    results: LatentTrajectoryAnalysis,
    viz_dir: Path,
    viz_config: Optional[VisualizationConfig] = None,
    labels_map: Optional[Dict[str, str]] = None,
) -> Path:
    viz_config = viz_config or VisualizationConfig()
    viz_config.apply_style_settings()

    geom = results.individual_trajectory_geometry
    groups = results.analysis_metadata['prompt_groups']
    labels = [labels_map.get(g, g) if labels_map else g for g in groups]

    means = np.array([float(geom[g]['log_volume_stats']['mean']) if g in geom and 'error' not in geom[g] else np.nan for g in groups])
    if means.size == 0 or not np.isfinite(means).any():
        return None

    base = means[0]
    delta = 100.0 * (means - base) / (base + 1e-12)

    cmap = plt.get_cmap(viz_config.name_cmap or 'tab10')
    color = cmap(0)

    plt.figure(figsize=(10, 5))
    plt.bar(labels, delta, color=color)
    plt.title("Per-trajectory Log BBox Volume Δ% vs baseline")
    plt.ylabel("% change"); plt.xticks(rotation=45); plt.grid(True, axis='y', alpha=0.3)

    out = viz_dir / f"log_volume_delta_vs_baseline.{viz_config.save_format}"
    plt.tight_layout(); plt.savefig(out, dpi=viz_config.dpi, bbox_inches=viz_config.bbox_inches); plt.close()
    return out
