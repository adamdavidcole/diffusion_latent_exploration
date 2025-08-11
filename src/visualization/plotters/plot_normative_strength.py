# src/visualization/plotters/plot_normative_strength.py
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import matplotlib.pyplot as plt

from src.visualization.visualization_config import VisualizationConfig
from src.analysis.data_structures import LatentTrajectoryAnalysis

def plot_normative_strength(
    results: LatentTrajectoryAnalysis,
    viz_dir: Path,
    viz_config: Optional[VisualizationConfig] = None,
    labels_map: Optional[Dict[str, str]] = None,
) -> Path:
    viz_config = viz_config or VisualizationConfig()
    viz_config.apply_style_settings()
    ns = getattr(results, 'normative_strength', {})
    if not ns: return None

    groups = results.analysis_metadata['prompt_groups']
    labels = [labels_map.get(g, g) if labels_map else g for g in groups]
    cmap = plt.get_cmap(viz_config.name_cmap or 'tab10')
    cols = [cmap(i % 10) for i in range(len(groups))]

    def get(key):
        vals = []
        for g in groups:
            v = ns.get(g, {}).get(key, np.nan)
            vals.append(np.nan if (v is None or (isinstance(v,float) and np.isnan(v))) else float(v))
        return np.array(vals, float)

    z_w   = get('z_early_width')
    z_x   = get('z_exit_distance')
    z_la  = get('z_late_area')
    # composite (simple example; use your definition if different)
    comp  = np.nan_to_num(z_w, 0.0) - np.nan_to_num(z_x, 0.0) + np.nan_to_num(z_la, 0.0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, y, title in zip(axes.flatten(),
                            [z_w, z_x, z_la, comp],
                            ['Z Early Width', 'Z Exit Distance (neg better corridor)', 'Z Late Area', 'Composite Dominance Index']):
        ax.bar(labels, y, color=cols); ax.tick_params(axis='x', rotation=45)
        ax.set_title(title); ax.grid(True, axis='y', alpha=0.3)
    out = viz_dir / f"normative_strength.{viz_config.save_format}"
    plt.tight_layout(); plt.savefig(out, dpi=viz_config.dpi, bbox_inches=viz_config.bbox_inches); plt.close()
    return out
