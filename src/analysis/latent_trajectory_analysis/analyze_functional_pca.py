import torch
import numpy as np
from typing import Dict, Any
import logging
import time
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_functional_pca(
    group_tensors: Dict[str, Dict[str, torch.Tensor]],
    device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    fpca_cfg: Dict[str, Any] = {}
) -> Dict[str, Any]:
    """
    GPU-friendly FPCA with fp16-safe SVD:
    1) Time subsampling (stride)
    2) Optional GPU Gaussian random projection to K dims (float32)
    3) Center across videos per time-feature slice (if enabled)
    4) PCA via compact SVD on [N, T_used*K], with autocast disabled for SVD
    """
    cfg = fpca_cfg
    K_target = int(cfg.get('feature_dim', 128))
    time_stride = int(cfg.get('time_stride', 2))
    max_components = int(cfg.get('max_components', 8))
    center = bool(cfg.get('center', True))
    use_rp = bool(cfg.get('use_random_projection', True))
    seed = int(cfg.get('random_seed', 42))

    out: Dict[str, Any] = {}
    for group_name in sorted(group_tensors.keys()):
        t0 = time.time()
        try:
            traj = group_tensors[group_name]['trajectory_tensor']  # [N, T, ...]
            N, T, F_full = traj.view(traj.shape[0], traj.shape[1], -1).shape
            X = traj.view(N, T, -1)  # [N, T, F_full]

            # Time subsampling
            t_idx = torch.arange(0, T, time_stride, device=X.device)
            X = X.index_select(1, t_idx)  # [N, T_used, F_full]
            T_used = X.shape[1]

            # Random projection to K dims (use float32 to avoid fp16 kernel paths)
            F = X.shape[-1]
            K = min(K_target, F)
            if use_rp and K < F:
                g = torch.Generator(device=X.device)
                g.manual_seed(seed)
                R = torch.randn(F, K, device=X.device, dtype=torch.float32, generator=g) / math.sqrt(F)
                X2 = (X.reshape(N * T_used, F).to(torch.float32) @ R).reshape(N, T_used, K)
            else:
                X2 = X[..., :K] if F >= K else X  # upcast happens below before SVD

            # Center across videos per time-feature (optional)
            if center and N > 1:
                mean_tf = X2.mean(dim=0, keepdim=True)  # [1, T_used, K]
                X2 = X2 - mean_tf

            # Flatten time-feature for PCA across videos and ensure float32
            X_flat = X2.reshape(N, T_used * X2.shape[-1]).to(torch.float32)  # [N, T_used*K]
            if center:
                X_flat = X_flat - X_flat.mean(dim=0, keepdim=True)

            # SVD-based PCA: explicitly disable autocast to avoid fp16 on CUDA
            try:
                if device.startswith('cuda'):
                    with torch.amp.autocast('cuda', enabled=False):
                        U, S, Vh = torch.linalg.svd(X_flat, full_matrices=False)
                else:
                    U, S, Vh = torch.linalg.svd(X_flat, full_matrices=False)
            except Exception as svd_err:
                logger.warning(f"[FPCA] {group_name}: CUDA SVD failed ({svd_err}); retrying on CPU in float32")
                U, S, Vh = torch.linalg.svd(X_flat.cpu(), full_matrices=False)
                U = U.to(X_flat.device); S = S.to(X_flat.device); Vh = Vh.to(X_flat.device)

            # Explained variance
            S2 = (S ** 2)
            total_var = float(S2.sum().item()) if S2.numel() > 0 else 0.0

            r = min(max_components, U.shape[1])
            evr_t = (S2[:r] / (S2.sum() + 1e-12)).detach().float().cpu().numpy()
            evr = evr_t.tolist()
            cum_evr = np.cumsum(evr_t).tolist()

            # Principal functions: reshape first r right-singular vectors to [r, T_used, K]
            Vh_r = Vh[:r, :]  # [r, T_used*K]
            principal_functions = (
                Vh_r.reshape(r, T_used, X2.shape[-1])
                    .detach().float().cpu().numpy().tolist()
            )

            # Scores per video: U[:, :r] * S[:r]
            scores = (U[:, :r] * S[:r]).detach().float().cpu().numpy().tolist()

            # Variance profiles
            temporal_var_profile = X2.var(dim=(0, 2)).detach().float().cpu().numpy().tolist()  # [T_used]
            across_video_var = X2.var(dim=0).mean(dim=-1).detach().float().cpu().numpy().tolist()  # [T_used]

            out[group_name] = {
                'principal_functions': principal_functions,          # [r, T_used, K]
                'explained_variance_ratio': evr,                     # [r]
                'cumulative_variance_ratio': cum_evr,                # [r]
                'scores': scores,                                    # [N, r]
                'temporal_variance_profile': temporal_var_profile,   # [T_used]
                'across_video_variance': across_video_var,           # [T_used]
                'metadata': {
                    'videos': int(N),
                    'time_used': int(T_used),
                    'feature_dim_used': int(X2.shape[-1]),
                    'time_stride': int(time_stride),
                    'max_components': int(max_components),
                    'dimensionality_reduced': bool(use_rp and K < F),
                    'original_feature_dim': int(F_full),
                    'total_variance': total_var
                }
            }

            logger.info(
                f"[FPCA] {group_name}: N={N} T_used={T_used} K={X2.shape[-1]} "
                f"r={r} EVR1={(evr[0] if evr else 0.0):.3f} "
                f"t={(time.time()-t0)*1000:.0f}ms"
            )

        except Exception as e:
            logger.error(f"[FPCA] error for {group_name}: {e}")
            out[group_name] = {
                'principal_functions': [],
                'explained_variance_ratio': [],
                'cumulative_variance_ratio': [],
                'scores': [],
                'temporal_variance_profile': [],
                'across_video_variance': [],
                'metadata': {'error': str(e)}
            }

    return out
