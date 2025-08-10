import torch
import numpy as np
from typing import Dict, Any, Tuple
import logging
import time
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_convex_hull_metrics_safe(
    group_tensors: Dict[str, Dict[str, torch.Tensor]],
    hull_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Performance-safe convex hull analysis that avoids Qhull entirely.
    Uses proxy metrics:
    - log_bbox_volume: sum(log(range_d)) for numerical stability
    - hull_volume: exp(clipped log volume)
    - effective_side: exp(log_bbox_volume / D)   (geometric mean side length)
    - hull_surface_area: sum of per-dim ranges (area proxy)
    - ellipsoid_log_det: log-det of covariance via SVD (sum log eigenvalues)
    - pairwise diameter and mean pairwise distance (sampled)
    Aggressively samples points and features for speed and robustness.
    """
    cfg = hull_cfg
    sample_points = int(cfg.get('sample_points', 2000))
    sample_features = int(cfg.get('sample_features', 8192))

    logger.info(
        f"[Hull] entering safe hull method; groups={len(group_tensors)} "
        f"SP={sample_points} SF={sample_features}"
    )
    start_ts = time.time()

    def _pairwise_stats(points_np: np.ndarray) -> Tuple[float, float]:
        m = min(points_np.shape[0], 500)
        if m < 2:
            return 0.0, 0.0
        X = points_np[:m]
        diffs = X[:, None, :] - X[None, :, :]
        d = np.sqrt(np.sum(diffs * diffs, axis=-1))
        iu = np.triu_indices(m, 1)
        dv = d[iu]
        if dv.size == 0:
            return 0.0, 0.0
        return float(dv.max()), float(dv.mean())

    results: Dict[str, Any] = {}

    for group_name in sorted(group_tensors.keys()):
        grp_t0 = time.time()
        try:
            traj = group_tensors[group_name]['trajectory_tensor']  # [videos, steps, ...]
            flat = traj.view(traj.shape[0], traj.shape[1], -1)     # [N, T, F]
            n_videos, n_steps, latent_dim_full = flat.shape

            # Feature sampling
            dim_reduced = False
            latent_dim_used = latent_dim_full
            if latent_dim_full > sample_features:
                idx = torch.randperm(latent_dim_full, device=flat.device)[:sample_features]
                flat = flat[..., idx]
                latent_dim_used = flat.shape[-1]
                dim_reduced = True

            # Point sampling
            points = flat.reshape(-1, latent_dim_used)  # [N*T, F_used]
            total_points = points.shape[0]
            if total_points > sample_points:
                ridx = torch.randperm(total_points, device=points.device)[:sample_points]
                points = points.index_select(0, ridx)

            # To numpy
            pts_np = points.float().cpu().numpy()

            # Bounding box ranges
            mins = np.min(pts_np, axis=0)
            maxs = np.max(pts_np, axis=0)
            ranges = np.maximum(maxs - mins, 1e-12)

            # Stable volume proxies
            log_bbox_vol = float(np.sum(np.log(ranges)))
            bbox_vol = float(np.exp(np.clip(log_bbox_vol, -700.0, 700.0)))
            area_proxy = float(np.sum(ranges))

            # Effective side length (geometric mean of ranges)
            if latent_dim_used > 0:
                eff_side = float(math.exp(np.clip(log_bbox_vol / latent_dim_used, -100.0, 100.0)))
            else:
                eff_side = 0.0

            # Ellipsoidal log-det via SVD of centered data
            X = pts_np - pts_np.mean(axis=0, keepdims=True)
            if X.shape[0] > 1 and X.shape[1] > 0:
                try:
                    svals = np.linalg.svd(X, full_matrices=False, compute_uv=False)
                    eig = (svals ** 2) / max(1, X.shape[0] - 1)
                    ellipsoid_log_det = float(np.sum(np.log(eig + 1e-12)))
                except Exception:
                    ellipsoid_log_det = float('nan')
            else:
                ellipsoid_log_det = float('nan')

            # Pairwise stats
            diameter, mean_dist = _pairwise_stats(pts_np)

            # Density-like metrics
            volume_per_point = float(bbox_vol / max(1, n_videos * n_steps))
            density_metric = float((n_videos * n_steps) / (bbox_vol + 1e-10))

            results[group_name] = {
                'hull_volume': bbox_vol,
                'log_bbox_volume': log_bbox_vol,
                'effective_side': eff_side,
                'hull_surface_area': area_proxy,
                'n_hull_vertices': 0,  # not computed in proxy mode
                'point_cloud_diameter': diameter,
                'mean_pairwise_distance': mean_dist,
                'total_trajectory_points': int(n_videos * n_steps),
                'latent_dim_used': int(latent_dim_used),
                'dimensionality_reduced': bool(dim_reduced),
                'volume_per_point': volume_per_point,
                'density_metric': density_metric,
                'ellipsoid_log_det': ellipsoid_log_det,
                'method': 'proxy_safe'
            }

            logger.info(
                f"[Hull] {group_name}: pts={pts_np.shape[0]}/{n_videos*n_steps} "
                f"dim={latent_dim_used} logVol={log_bbox_vol:.3f} "
                f"effSide={eff_side:.3e} area≈{area_proxy:.3e} "
                f"diam≈{diameter:.3e} mean_d≈{mean_dist:.3e} "
                f"t={(time.time()-grp_t0)*1000:.0f}ms"
            )

        except Exception as e:
            logger.error(f"[Hull] error for {group_name}: {e}")
            results[group_name] = {
                'hull_volume': 0.0,
                'log_bbox_volume': 0.0,
                'effective_side': 0.0,
                'hull_surface_area': 0.0,
                'n_hull_vertices': 0,
                'point_cloud_diameter': 0.0,
                'mean_pairwise_distance': 0.0,
                'total_trajectory_points': 0,
                'latent_dim_used': 0,
                'dimensionality_reduced': False,
                'volume_per_point': 0.0,
                'density_metric': 0.0,
                'ellipsoid_log_det': float('nan'),
                'method': 'proxy_safe',
                'error': str(e)
            }

    logger.info(f"[Hull] safe hull analysis completed in {(time.time()-start_ts)*1000:.0f}ms")
    return results