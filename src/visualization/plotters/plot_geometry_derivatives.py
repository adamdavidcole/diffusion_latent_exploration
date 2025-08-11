# src/visualization/plotters/plot_geometry_derivatives.py
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import matplotlib.pyplot as plt

from src.visualization.visualization_config import VisualizationConfig
from src.analysis.data_structures import LatentTrajectoryAnalysis

def plot_geometry_derivatives(
    results: LatentTrajectoryAnalysis,
    viz_dir: Path,
    viz_config: Optional[VisualizationConfig] = None,
    labels_map: Optional[Dict[str, str]] = None,
) -> Path:
    viz_config = viz_config or VisualizationConfig()
    viz_config.apply_style_settings()

    gd = getattr(results, 'geometry_derivatives', {})
    if not gd: return None

    groups = results.analysis_metadata['prompt_groups']
    labels = [labels_map.get(g, g) if labels_map else g for g in groups]
    cmap = plt.get_cmap(viz_config.name_cmap or 'tab10')
    cols = [cmap(i % 10) for i in range(len(groups))]

    curv = np.array([gd.get(g, {}).get('curvature_peak_mean', np.nan) for g in groups], float)
    cstp = np.array([gd.get(g, {}).get('curvature_peak_step_mean', np.nan) for g in groups], float)
    jerk = np.array([gd.get(g, {}).get('jerk_peak_mean', np.nan) for g in groups], float)
    jstp = np.array([gd.get(g, {}).get('jerk_peak_step_mean', np.nan) for g in groups], float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(labels, curv, color=cols); ax1.set_title("Curvature peak (mean)"); ax1.tick_params(axis='x', rotation=45)
    for i, s in enumerate(cstp):
        if np.isfinite(s): ax1.text(i, curv[i], f"@{s:.1f}", ha='center', va='bottom', fontsize=8)
    ax1.grid(True, axis='y', alpha=0.3)

    ax2.bar(labels, jerk, color=cols); ax2.set_title("Jerk peak (mean)"); ax2.tick_params(axis='x', rotation=45)
    for i, s in enumerate(jstp):
        if np.isfinite(s): ax2.text(i, jerk[i], f"@{s:.1f}", ha='center', va='bottom', fontsize=8)
    ax2.grid(True, axis='y', alpha=0.3)

    out = viz_dir / f"geometry_derivatives.{viz_config.save_format}"
    plt.tight_layout(); plt.savefig(out, dpi=viz_config.dpi, bbox_inches=viz_config.bbox_inches); plt.close()
    return out
