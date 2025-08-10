import torch
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)    

def apply_normalization(
        trajectory_tensor: torch.Tensor, 
        group_data: Dict[str, Any],
        norm_cfg: Dict[str, Any]
) -> torch.Tensor:
    """Return [N,T,D] normalized according to norm_cfg."""
    traj = trajectory_tensor[:, :, 0] if trajectory_tensor.shape[2] == 1 else trajectory_tensor
    if norm_cfg.get("per_channel_standardize", False):
        mean_c = traj.mean(dim=(0,1,3,4,5), keepdim=True)
        std_c  = traj.std(dim=(0,1,3,4,5), keepdim=True) + 1e-6
        traj = (traj - mean_c) / std_c
    flat = traj.flatten(start_dim=2)
    if norm_cfg.get("per_step_whiten", False):
        mean = flat.mean(dim=2, keepdim=True)
        std  = flat.std(dim=2, keepdim=True) + 1e-6
        flat = (flat - mean) / std
    if norm_cfg.get("snr_normalize", False):
        sigmas = None
        try:
            meta_list = group_data.get('trajectory_metadata', [])
            if meta_list and 'step_metadata' in meta_list[0]:
                steps_meta = meta_list[0]['step_metadata']
                if isinstance(steps_meta, list) and steps_meta:
                    for key in ('sigma','flow_sigma','snr','sigma_t'):
                        if key in steps_meta[0]:
                            import torch as _torch
                            sigmas = _torch.tensor([m.get(key, 1.0) for m in steps_meta],
                                                    device=flat.device, dtype=flat.dtype)
                            break
        except Exception:
            sigmas = None
        if sigmas is None:
            step_std = flat.permute(1,0,2).reshape(flat.shape[1], -1).std(dim=1) + 1e-6
            inv = 1.0 / step_std
        else:
            inv = 1.0 / (sigmas + 1e-6)
            if inv.numel() != flat.shape[1]:
                inv = torch.nn.functional.interpolate(inv.view(1,1,-1), size=flat.shape[1],
                                                        mode='linear', align_corners=False).view(-1)
        flat = flat * inv.view(1,-1,1)
    return flat