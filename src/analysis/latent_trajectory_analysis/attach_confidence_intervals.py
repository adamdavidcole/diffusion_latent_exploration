import numpy as np
import logging
from typing import Dict, Any

from .utils.bootstrap_ci import bootstrap_ci
from src.analysis.data_structures import LatentTrajectoryAnalysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def attach_confidence_intervals(results: Dict[str, Any]) -> Dict[str, Any]:
    """Bootstrap CIs for bars + log-volume."""
    CIs: Dict[str, Any] = {}
    ta = results['temporal_analysis']
    geom = results.get('individual_trajectory_geometry', {})

    for g in ta.keys():
        CIs[g] = {}
        L = np.asarray(ta[g]['trajectory_length']['individual_lengths'], float)
        V = np.asarray(ta[g]['velocity_analysis'].get('mean_velocity_by_video',
                          ta[g]['velocity_analysis'].get('mean_velocity', [])), float)
        A = np.asarray(ta[g]['acceleration_analysis'].get('mean_acceleration_individual',
                          ta[g]['acceleration_analysis'].get('mean_acceleration', [])), float)
        CIs[g]['length']       = bootstrap_ci(L)
        CIs[g]['velocity']     = bootstrap_ci(V)
        CIs[g]['acceleration'] = bootstrap_ci(A)

        if g in geom and 'error' not in geom[g]:
            circ = np.asarray(geom[g]['circuitousness_stats']['individual_values'], float) - 1.0
            turn = np.asarray(geom[g]['turning_angle_stats']['individual_values'], float)
            ali  = np.asarray(geom[g]['endpoint_alignment_stats']['individual_values'], float)
            lv   = np.asarray(geom[g]['log_volume_stats']['individual_values'], float)
            CIs[g]['circuitousness_minus1'] = bootstrap_ci(circ)
            CIs[g]['turning_angle']         = bootstrap_ci(turn)
            CIs[g]['endpoint_alignment']    = bootstrap_ci(ali)
            CIs[g]['log_volume']            = bootstrap_ci(lv)

    return CIs
