
import torch
from typing import Dict, Any
import logging

from .utils.apply_normalization import apply_normalization

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_geometry_derivatives(
    group_tensors: Dict[str, Dict[str, torch.Tensor]],
    norm_cfg: Dict[str, Any]
):
    """
    Derivatives along trajectories (Full norm):
    curvature_t   = ||Δv_t|| / (||v_t|| + eps),  v_t = x_{t+1} - x_t
    jerk_t        = ||Δa_t||,                    a_t = v_{t+1} - v_t
    Returns per-group summaries: mean peak curvature, mean peak jerk, and their step indices (averaged).
    """
    import numpy as np
    out = {}
    for g, pack in group_tensors.items():
        X = apply_normalization(pack['trajectory_tensor'], pack, norm_cfg).cpu().numpy()  # [N,T,D]
        N, T, D = X.shape
        peaks_c, peaks_j, steps_c, steps_j = [], [], [], []
        for n in range(N):
            traj = X[n]                           # [T,D]
            v = np.diff(traj, axis=0)             # [T-1, D]
            a = np.diff(v, axis=0)                # [T-2, D]
            curv = np.linalg.norm(np.diff(v, axis=0), axis=1) / (np.linalg.norm(v[1:], axis=1) + 1e-12)  # [T-2]
            jerk = np.linalg.norm(np.diff(a, axis=0), axis=1)  # [T-3]
            if curv.size:
                k_idx = int(np.argmax(curv)); peaks_c.append(float(np.max(curv))); steps_c.append(k_idx+1)
            if jerk.size:
                j_idx = int(np.argmax(jerk)); peaks_j.append(float(np.max(jerk))); steps_j.append(j_idx+2)
        out[g] = {
            'curvature_peak_mean': float(np.mean(peaks_c)) if peaks_c else np.nan,
            'curvature_peak_step_mean': float(np.mean(steps_c)) if steps_c else np.nan,
            'jerk_peak_mean': float(np.mean(peaks_j)) if peaks_j else np.nan,
            'jerk_peak_step_mean': float(np.mean(steps_j)) if steps_j else np.nan,
        }
    return out
