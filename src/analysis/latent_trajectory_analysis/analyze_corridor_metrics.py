import torch
import numpy as np
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .utils.apply_normalization import apply_normalization

def analyze_corridor_metrics(
    group_tensors: Dict[str, Dict[str, torch.Tensor]],
    norm_cfg: Dict[str, Any]
):
    """
    Corridor metrics computed on Full normalization:
    - width_by_step[g][t]     : mean cross-seed std at step t (corridor width)
    - centroid_path[g][t,:]   : mean embedding at step t
    - exit_distance[g]        : L2 distance between g's centroid path and baseline centroid path (cum. over steps)
    - branch_divergence[g][t] : distance between g's centroid and baseline's centroid at step t
    """
    import numpy as np
    metrics = {'width_by_step': {}, 'centroid_path': {}, 'branch_divergence': {}, 'exit_distance': {}}
    groups = sorted(group_tensors.keys())
    if not groups: return metrics

    # compute flattened Full-norm embeddings per group
    flat_by_group = {}
    for g in groups:
        tens = group_tensors[g]['trajectory_tensor']  # [N, T, ...]
        flat = apply_normalization(tens, group_tensors[g], norm_cfg=norm_cfg)  # [N, T, D]
        flat_by_group[g] = flat.cpu().numpy()

    T = flat_by_group[groups[0]].shape[1]
    base = groups[0]

    for g in groups:
        X = flat_by_group[g]  # [N,T,D]
        # width = mean std across seeds per step (norm of std vector)
        stds = X.std(axis=0)               # [T, D]
        width = np.linalg.norm(stds, axis=1)  # [T]
        metrics['width_by_step'][g] = width.tolist()
        centroid = X.mean(axis=0)          # [T, D]
        metrics['centroid_path'][g] = centroid

    # baseline centroid
    base_centroid = metrics['centroid_path'][base]  # [T, D]

    for g in groups:
        C = metrics['centroid_path'][g]
        branch = np.linalg.norm(C - base_centroid, axis=1)  # [T]
        metrics['branch_divergence'][g] = branch.tolist()
        metrics['exit_distance'][g] = float(np.sum(branch))

    # convert centroids to lists for JSON
    metrics['centroid_path'] = {g: v.tolist() for g, v in metrics['centroid_path'].items()}
    return metrics
