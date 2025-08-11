# src/visualization/plotters/plot_convex_hull_analysis.py
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import matplotlib.pyplot as plt

from src.visualization.visualization_config import VisualizationConfig
from src.analysis.data_structures import LatentTrajectoryAnalysis

def plot_convex_hull_analysis(
    results: LatentTrajectoryAnalysis,
    viz_dir: Path,
    viz_config: Optional[VisualizationConfig] = None,
    labels_map: Optional[Dict[str, str]] = None,
) -> Path:
    viz_config = viz_config or VisualizationConfig()
    viz_config.apply_style_settings()
    data = results.convex_hull_analysis
    groups = results.analysis_metadata['prompt_groups']
    if not groups: return None
    labels = [labels_map.get(g, g) if labels_map else g for g in groups]

    logvol = np.array([data.get(g, {}).get('log_bbox_volume', np.nan) for g in groups], float)

    # derive effective side from SAME basis
    traj_shape = results.analysis_metadata.get('trajectory_shape', [1,16,16,60,106])
    flat_dim = int(np.prod(traj_shape))
    eff = np.exp(logvol / max(1, flat_dim))

    def pct_delta(arr):
        base = arr[0]
        return 100.0 * (arr - base) / (base + 1e-12)

    dv, de = pct_delta(logvol), pct_delta(eff)
    cmap = plt.get_cmap(viz_config.name_cmap or 'tab10')
    c1, c2 = cmap(1), cmap(2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.bar(labels, dv, color=c1); ax1.set_title("Log BBox Volume Δ% vs baseline")
    ax1.set_ylabel("% change"); ax1.tick_params(axis='x', rotation=45); ax1.grid(True, axis='y', alpha=0.3)
    ax2.bar(labels, de, color=c2); ax2.set_title("Effective Side Δ% vs baseline")
    ax2.set_ylabel("% change"); ax2.tick_params(axis='x', rotation=45); ax2.grid(True, axis='y', alpha=0.3)
    out = viz_dir / f"convex_hull_proxies_delta.{viz_config.save_format}"
    plt.tight_layout(); plt.savefig(out, dpi=viz_config.dpi, bbox_inches=viz_config.bbox_inches); plt.close()
    return out
