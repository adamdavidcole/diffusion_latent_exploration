import numpy as np
import logging

from .utils.bootstrap_ci import bootstrap_ci
from src.analysis.data_structures import LatentTrajectoryAnalysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def attach_confidence_intervals(results: LatentTrajectoryAnalysis):
    """
    Adds bootstrap CIs for per-group bar metrics:
    length, velocity, acceleration, circuitousness−1, turning, alignment, late/early
    """
    CIs = {}

    temporal_analysis = results['temporal_analysis']
    geom = results['individual_trajectory_geometry']

    # temporal_analysis = getattr(results, 'temporal_analysis', {})
    groups = sorted(temporal_analysis.keys())

    for g in groups:
        CIs[g] = {}
        # per-video arrays
        L = np.array(temporal_analysis[g]['trajectory_length']['individual_lengths'], dtype=float)
        V = np.array(temporal_analysis[g]['velocity_analysis'].get('mean_velocity_by_video',
                temporal_analysis[g]['velocity_analysis'].get('mean_velocity', [])), dtype=float)
        A = np.array(temporal_analysis[g]['acceleration_analysis'].get('mean_acceleration_individual',
                temporal_analysis[g]['acceleration_analysis'].get('mean_acceleration', [])), dtype=float)
        CIs[g]['length']       = bootstrap_ci(L)
        CIs[g]['velocity']     = bootstrap_ci(V)
        CIs[g]['acceleration'] = bootstrap_ci(A)

        if g in geom and 'error' not in geom[g]:
            circ = np.array(geom[g]['circuitousness_stats']['individual_values'], dtype=float) - 1.0
            turn = np.array(geom[g]['turning_angle_stats']['individual_values'], dtype=float)
            ali  = np.array(geom[g]['endpoint_alignment_stats']['individual_values'], dtype=float)
            CIs[g]['circuitousness_minus1'] = bootstrap_ci(circ)
            CIs[g]['turning_angle']         = bootstrap_ci(turn)
            CIs[g]['endpoint_alignment']    = bootstrap_ci(ali)
        # late/early is group-level; skip CI unless you store per-video curves
    return CIs