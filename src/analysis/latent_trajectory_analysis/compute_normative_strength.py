import numpy as np
import logging

from src.analysis.data_structures import LatentTrajectoryAnalysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_normative_strength(
    results: LatentTrajectoryAnalysis
):
    """
    Prototype dominance index combining:
    + early corridor width (steps 0..k)
    - exit distance from baseline (sum over steps)
    - late-recovery area (from spatial variance curve; larger area = more late effort)
    Returns z-scored composite per group.
    """
    out = {}
    # need corridor metrics & spatial curves
    corridor = getattr(results, 'corridor_metrics', None)
    if not corridor: 
        logger.warning("No corridor metrics found.")
        return out
    groups = sorted(corridor['width_by_step'].keys())
    k = 3  # early window (t=0..3) for width

    # late recovery area from spatial curve
    def late_area(g):
        sp = results['spatial_patterns']['trajectory_spatial_evolution'][g]
        curve = None
        for kname in ('spatial_variance_curve','spatial_variance_by_step','variance_curve'):
            if kname in sp: curve = np.array(sp[kname], dtype=float)
        if curve is None or curve.size < 4: return np.nan
        t = np.arange(curve.size)
        # area after the minimum (late recovery)
        t0 = int(np.argmin(curve))
        return float(np.trapz(curve[t0:], t[t0:]))

    w_early = np.array([np.nanmean(corridor['width_by_step'][g][:k+1]) for g in groups], dtype=float)
    exit_d  = np.array([corridor['exit_distance'][g] for g in groups], dtype=float)
    late    = np.array([late_area(g) for g in groups], dtype=float)

    def z(a): 
        m, s = np.nanmean(a), np.nanstd(a)
        return (a - m) / (s + 1e-12)

    # dominance: more early width, less distance to exit, less late recovery
    score = +z(w_early) - z(exit_d) - z(late)
    for i,g in enumerate(groups):
        out[g] = {'dominance_index': float(score[i]),
                'z_early_width': float(z(w_early)[i]),
                'z_exit_distance': float(z(exit_d)[i]),
                'z_late_area': float(z(late)[i])}
    return out