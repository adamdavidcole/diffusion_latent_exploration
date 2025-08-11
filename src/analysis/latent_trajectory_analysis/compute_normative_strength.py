# src/analysis/latent_trajectory_analysis/compute_normative_strength.py
import numpy as np
from typing import Dict, Any

def compute_normative_strength(results: Dict[str, Any]) -> Dict[str, Any]:
    """Build z-scores for early width, exit distance, late area with NaN-safe stats."""
    cm = results.get('corridor_metrics', {})
    groups = sorted(cm.get('width_by_step', {}).keys())
    if not groups: return {}

    # early width = mean width over first third; late area = sum width over last third
    widths = {g: np.asarray(cm['width_by_step'][g], float) for g in groups}
    T = min(len(w) for w in widths.values()) if widths else 0
    a, b = 0, max(1, T//3); c = max(1, 2*T//3)

    early = np.array([np.nanmean(w[a:b]) for w in widths.values()], float)
    lateA = np.array([np.nansum(np.maximum(w[c:], 0.0)) for w in widths.values()], float)  # area proxy
    exitD = np.array([float(np.nansum(np.asarray(cm['branch_divergence'][g], float))) for g in groups], float)

    def zscore(x):
        x = np.asarray(x, float)
        m = np.nanmean(x); s = np.nanstd(x) if np.nanstd(x)>1e-12 else 1.0
        z = (x - m)/s
        z[~np.isfinite(z)] = 0.0
        return z

    z_early = zscore(early)
    z_exit  = zscore(exitD)
    z_lateA = zscore(lateA)

    out = {}
    for i,g in enumerate(groups):
        # simple composite: dominance = z_early - z_exit + z_lateA
        out[g] = {
            'z_early_width': float(z_early[i]),
            'z_exit_distance': float(z_exit[i]),
            'z_late_area': float(z_lateA[i]),
            'dominance_index': float(z_early[i] - z_exit[i] + z_lateA[i]),
        }
    return out
