import torch
from typing import Dict, Any
from .utils.apply_normalization import apply_normalization

@torch.no_grad()
def analyze_corridor_metrics(
    group_tensors: Dict[str, Dict[str, torch.Tensor]],
    norm_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Fast, compact corridor metrics (Full norm):
      width_by_step[g]: [T]   — L2 of per-dim std across seeds (corridor width)
      branch_divergence[g]:[T]— L2 centroid distance to baseline centroid
      exit_distance[g]: float — sum_t branch_divergence[g][t]
    NOTE: We deliberately DO NOT store centroid vectors (T×D) to avoid huge JSON.
    """
    device = next(iter(group_tensors.values()))['trajectory_tensor'].device
    groups = sorted(group_tensors.keys())
    out: Dict[str, Any] = {'width_by_step': {}, 'branch_divergence': {}, 'exit_distance': {}}
    if not groups: return out

    # Full-norm & flatten once per group
    flat: Dict[str, torch.Tensor] = {}
    for g in groups:
        X = apply_normalization(group_tensors[g]['trajectory_tensor'], group_tensors[g], norm_cfg)  # [N,T,D]
        flat[g] = X.to(device, non_blocking=True)

    T = flat[groups[0]].shape[1]
    base = groups[0]

    # baseline centroid path (T,D) kept in GPU only; never serialized
    base_centroid = flat[base].mean(dim=0)  # [T,D]

    for g in groups:
        X = flat[g]                                # [N,T,D]
        mu = X.mean(dim=0)                         # [T,D]
        std = (X - mu.unsqueeze(0)).pow(2).mean(dim=0).sqrt()  # [T,D]
        width = std.norm(dim=1)                    # [T]
        branch = (mu - base_centroid).norm(dim=1)  # [T]
        out['width_by_step'][g] = width.detach().cpu().tolist()
        out['branch_divergence'][g] = branch.detach().cpu().tolist()
        out['exit_distance'][g] = float(branch.sum().item())

    return out
