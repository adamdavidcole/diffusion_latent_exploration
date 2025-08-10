import numpy as np
import logging

from src.analysis.data_structures import LatentTrajectoryAnalysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_volume_deltas(
    results: LatentTrajectoryAnalysis
):
    """Adds group-level and (if possible) paired per-video Δ% vs baseline for individual log-volumes."""
    geom = results['individual_trajectory_geometry']
    groups = sorted(geom.keys())
    means = np.array([float(geom[g]['log_volume_stats']['mean']) if 'error' not in geom[g] else np.nan for g in groups])
    base = means[0] if means.size else np.nan
    group_delta = 100.0 * (means - base) / (base + 1e-12)

    res = {
        'groups': groups,
        'group_means': means.tolist(),
        'group_delta_percent': group_delta.tolist()
    }
    return res