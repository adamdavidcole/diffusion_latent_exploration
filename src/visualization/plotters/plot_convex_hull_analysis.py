"""
Convex hull volume analysis plotting functionality.
Extracted from LatentTrajectoryAnalyzer._plot_convex_hull_analysis.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
from src.analysis.data_structures import LatentTrajectoryAnalysis
from src.visualization.visualization_config import VisualizationConfig

def plot_convex_hull_analysis(
    results: LatentTrajectoryAnalysis,
    viz_dir: Path,
    viz_config: Optional[VisualizationConfig] = None,
    labels_map: Optional[Dict[str, str]] = None,
    **kwargs
) -> Path:
    """Convex hull proxies: plot Δ% vs baseline (first group), with optional CIs if present."""
    if viz_config is None:
        viz_config = VisualizationConfig()

    data = results.convex_hull_analysis
    groups = sorted(data.keys())
    if not groups:
        return None

    logvol = np.array([data[g].get('log_bbox_volume', np.nan) for g in groups], dtype=float)
    eff    = np.array([data[g].get('effective_side',   np.nan) for g in groups], dtype=float)

    def pct_delta(arr):
        base = arr[0]
        return 100.0 * (arr - base) / (base + 1e-12)

    y1 = pct_delta(logvol)
    y2 = pct_delta(eff)

    def maybe_ci(key):
        lows, highs = [], []
        have_ci = True
        for g in groups:
            boot = data[g].get(key)
            if isinstance(boot, (list, tuple)) and len(boot) > 8:
                b = np.asarray(boot, dtype=float)
                lo, hi = np.percentile(b, [16, 84])
                lows.append(lo); highs.append(hi)
            else:
                have_ci = False
                break
        if not have_ci:
            return None, None
        return np.array(lows), np.array(highs)

    lv_lo, lv_hi = maybe_ci('bootstrap_log_bbox_volume')
    es_lo, es_hi = maybe_ci('bootstrap_effective_side')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Log volume Δ%
    ax1.bar(groups, y1, alpha=0.85)
    if lv_lo is not None:
        ax1.errorbar(groups, pct_delta(lv_lo*0 + logvol),
                    yerr=[pct_delta(logvol) - pct_delta(lv_lo),
                        pct_delta(lv_hi) - pct_delta(logvol)],
                    fmt='none', ecolor='k', capsize=3, linewidth=1)
    ax1.set_title('Log-BBox Volume Δ% vs baseline')
    ax1.set_ylabel('% change'); ax1.tick_params(axis='x', rotation=45); ax1.grid(True, alpha=0.3)

    # Effective side Δ%
    ax2.bar(groups, y2, alpha=0.85)
    if es_lo is not None:
        ax2.errorbar(groups, pct_delta(es_lo*0 + eff),
                    yerr=[pct_delta(eff) - pct_delta(es_lo),
                        pct_delta(es_hi) - pct_delta(eff)],
                    fmt='none', ecolor='k', capsize=3, linewidth=1)
    ax2.set_title('Effective Side Δ% vs baseline')
    ax2.set_ylabel('% change'); ax2.tick_params(axis='x', rotation=45); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = viz_dir / f'convex_hull_proxies_delta.{viz_config.save_format}'
    plt.savefig(output_path, dpi=viz_config.dpi, bbox_inches=viz_config.bbox_inches)
    plt.close()
    return output_path
