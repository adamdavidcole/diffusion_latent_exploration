from typing import Dict, Any
import numpy as np

def log_volume_deltas(results: Dict[str, Any]) -> Dict[str, Any]:
    geom = results.get('individual_trajectory_geometry', {})
    groups = sorted(geom.keys())
    means = np.array([float(geom[g]['log_volume_stats']['mean']) if 'error' not in geom[g] else np.nan for g in groups], float)
    if means.size == 0:
        return {}
    base = means[0]
    delta = 100.0 * (means - base) / (base + 1e-12)
    return {'groups': groups, 'group_means': means.tolist(), 'group_delta_percent': delta.tolist()}
