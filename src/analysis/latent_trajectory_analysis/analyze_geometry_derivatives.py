import torch
from typing import Dict, Any
from .utils.apply_normalization import apply_normalization

@torch.no_grad()
def analyze_geometry_derivatives(
    group_tensors: Dict[str, Dict[str, torch.Tensor]],
    norm_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Curvature & jerk on Full norm (GPU):
      curvature_t = ||Δv|| / (||v||+eps), v = x_{t+1}-x_t
      jerk_t      = ||Δa||,               a = v_{t+1}-v_t
    Returns per-group mean peak values and mean peak steps.
    """
    out: Dict[str, Any] = {}
    for g, pack in group_tensors.items():
        X = apply_normalization(pack['trajectory_tensor'], pack, norm_cfg)  # [N,T,D]
        v = X[:, 1:, :] - X[:, :-1, :]              # [N,T-1,D]
        dv = v[:, 2:, :] - v[:, 1:-1, :]            # [N,T-3,D]
        a  = v[:, 1:, :] - v[:, :-1, :]             # [N,T-2,D]
        j  = a[:, 1:, :] - a[:, :-1, :]             # [N,T-3,D]

        eps = 1e-12
        curv = dv.norm(dim=2) / (v[:, 2:, :].norm(dim=2) + eps)  # [N,T-3]
        jerk = j.norm(dim=2)                                     # [N,T-3]

        if curv.numel():
            kmax, kidx = curv.max(dim=1)
            out[g] = {
                'curvature_peak_mean': float(kmax.mean().item()),
                'curvature_peak_step_mean': float((kidx + 2).float().mean().item()),
            }
        else:
            out[g] = {'curvature_peak_mean': float('nan'), 'curvature_peak_step_mean': float('nan')}

        if jerk.numel():
            jmax, jidx = jerk.max(dim=1)
            out[g].update({
                'jerk_peak_mean': float(jmax.mean().item()),
                'jerk_peak_step_mean': float((jidx + 2).float().mean().item())
            })
        else:
            out[g].update({'jerk_peak_mean': float('nan'), 'jerk_peak_step_mean': float('nan')})
    return out
