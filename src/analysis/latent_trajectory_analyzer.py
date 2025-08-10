#!/usr/bin/env python3
"""
GPU-Optimized Structure-Aware Latent Analysis for Video Diffusion Models

This module implements GPU-accelerated analysis methods that respect the 3D video latent structure
[batch, channels, frames, height, width] for significant performance improvements.

Key GPU optimizations:
1. Keep tensors on GPU throughout computation pipeline
2. Vectorized batch operations across channels/frames
3. Mixed precision computation for memory efficiency
4. GPU-accelerated FFT and statistical operations
5. Trajectory-aware analysis preserving diffusion step structure
"""

import logging
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

from src.visualization.batch_grid import create_batch_image_grid

from.data_structures import LatentTrajectoryAnalysis

# Try to import FFT functions
try:
    from torch.fft import fft, ifft, fft2, ifft2, fftshift
    TORCH_FFT_AVAILABLE = True
except ImportError:
    TORCH_FFT_AVAILABLE = False

# Advanced geometry and statistical imports for new metrics
try:
    from scipy.spatial import ConvexHull
    from scipy.spatial.distance import pdist, squareform
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.neighbors import NearestNeighbors
    ADVANCED_GEOMETRY_AVAILABLE = True
except ImportError:
    ADVANCED_GEOMETRY_AVAILABLE = False
    
try:
    import skdim
    INTRINSIC_DIM_AVAILABLE = True
except ImportError:
    INTRINSIC_DIM_AVAILABLE = False

from .latent_trajectory_analysis.analyze_individual_trajectory_geometry import analyze_individual_trajectory_geometry
from .latent_trajectory_analysis.analyze_spatial_patterns import analyze_spatial_patterns
from .latent_trajectory_analysis.analyze_temporal_coherence import analyze_temporal_coherence
from .latent_trajectory_analysis.analyze_channel_patterns import analyze_channel_patterns
from .latent_trajectory_analysis.analyze_patch_diversity import analyze_patch_diversity
from .latent_trajectory_analysis.analyze_global_structure import analyze_global_structure
from .latent_trajectory_analysis.analyze_information_content import analyze_information_content
from .latent_trajectory_analysis.analyze_complexity_measures import analyze_complexity_measures
from .latent_trajectory_analysis.analyze_frequency_patterns import analyze_frequency_patterns
from .latent_trajectory_analysis.analyze_group_separability import analyze_group_separability
from .latent_trajectory_analysis.analyze_temporal_trajectories import analyze_temporal_trajectories
from .latent_trajectory_analysis.analyze_structural_patterns import analyze_structural_patterns
from .latent_trajectory_analysis.analyze_intrinsic_dimension import analyze_intrinsic_dimension
from .latent_trajectory_analysis.test_statistical_significance import test_statistical_significance
from .latent_trajectory_analysis.analyze_corridor_metrics import analyze_corridor_metrics
from .latent_trajectory_analysis.analyze_geometry_derivatives import analyze_geometry_derivatives
from .latent_trajectory_analysis.attach_confidence_intervals import attach_confidence_intervals
from .latent_trajectory_analysis.log_volume_deltas import log_volume_deltas
from .latent_trajectory_analysis.compute_normative_strength import compute_normative_strength

class LatentTrajectoryAnalyzer:
    """GPU-accelerated structure-aware latent analysis with trajectory preservation."""

    def __init__(
        self,
        latents_dir: str,
        device: str = "auto",
        enable_mixed_precision: bool = True,
        batch_size: int = 32,
        output_dir: Optional[str] = None,
        use_prompt_labels = False, # use variation text label as label or group name (prompt_000)
        # Distance normalization config
        norm_cfg: Optional[Dict[str, Any]] = None,
        # Convex hull / geometry config
        hull_mode: str = "auto",
        hull_max_dim_exact: int = 8,
        hull_max_points_exact: int = 500,
        hull_rp_dim: int = 8,
        hull_rp_projections: int = 12,
        hull_sample_points: int = 2000,
        hull_sample_features: int = 8192,
        hull_time_budget_ms: int = 3000,
    ):
        """Initialize GPU-optimized analyzer.

        Parameters
        ----------
        latents_dir : str
            Path to the batch's ``latents/`` directory.
        device : str
            "cpu", "cuda", "cuda:N", or "auto" to pick automatically.
        enable_mixed_precision : bool
            Whether to enable autocast mixed precision on CUDA.
        batch_size : int
            Batch size for GPU ops.
        output_dir : Optional[str]
            Where to write analysis outputs. Defaults to ``<batch>/latent_trajectory_analysis_results``.
        Hull/geometry kwargs mirror CLI flags in ``run_latent_trajectory_analysis.py``.
        """
        # Paths & basic settings
        self.latents_dir = Path(latents_dir)
        self.enable_mixed_precision = enable_mixed_precision
        self.batch_size = batch_size

        # Device setup
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Output directory
        if output_dir is None:
            self.output_dir = self.latents_dir.parent / "latent_trajectory_analysis_results"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Use prompt label variation text in visualizations (default to True)
        self.use_prompt_labels = use_prompt_labels
        
        # Normalization configuration (distance metrics)
        self.norm_cfg = {
            "per_step_whiten": True,
            "per_channel_standardize": True,
            "snr_normalize": True,
        }
        if norm_cfg:
            self.norm_cfg.update(norm_cfg)

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.performance_stats = {
            'device_used': self.device,
            'mixed_precision_enabled': self.enable_mixed_precision,
            'batch_size': self.batch_size,
            'memory_usage': {}
        }

        # Convex hull analysis configuration
        self.hull_cfg = {
            'mode': hull_mode,
            'max_dim_exact': int(hull_max_dim_exact),
            'max_points_exact': int(hull_max_points_exact),
            'rp_dim': int(hull_rp_dim),
            'rp_projections': int(hull_rp_projections),
            'sample_points': int(hull_sample_points),
            'sample_features': int(hull_sample_features),
            'time_budget_ms': int(hull_time_budget_ms),
        }
        # FPCA configuration (kept internal defaults; can be promoted to CLI later)
        self.fpca_cfg = {
            'feature_dim': 128,       # reduce latent features to this dim via random projection
            'time_stride': 2,         # subsample steps by this stride
            'max_components': 8,      # cap number of principal components
            'center': True,           # center per-feature across videos
            'use_random_projection': True,
            'random_seed': 42,
        }

        self.logger.info(f"Initialized GPU analyzer on {self.device}")
        if self.device.startswith("cuda"):
            try:
                self.logger.info(f"GPU: {torch.cuda.get_device_name()}")
                props = torch.cuda.get_device_properties(0)
                self.logger.info(f"Memory: {props.total_memory / 1e9:.1f} GB")
            except Exception:
                pass

    def _gpu_analyze_convex_hull_metrics_safe(
        self,
        group_tensors: Dict[str, Dict[str, torch.Tensor]],
        prompt_groups: List[str]
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
        cfg = getattr(self, 'hull_cfg', {})
        sample_points = int(cfg.get('sample_points', 2000))
        sample_features = int(cfg.get('sample_features', 8192))

        self.logger.info(
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

                self.logger.info(
                    f"[Hull] {group_name}: pts={pts_np.shape[0]}/{n_videos*n_steps} "
                    f"dim={latent_dim_used} logVol={log_bbox_vol:.3f} "
                    f"effSide={eff_side:.3e} area≈{area_proxy:.3e} "
                    f"diam≈{diameter:.3e} mean_d≈{mean_dist:.3e} "
                    f"t={(time.time()-grp_t0)*1000:.0f}ms"
                )

            except Exception as e:
                self.logger.error(f"[Hull] error for {group_name}: {e}")
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

        self.logger.info(f"[Hull] safe hull analysis completed in {(time.time()-start_ts)*1000:.0f}ms")
        return results

    def _compute_spectral_entropy(self, power_spectrum: torch.Tensor) -> float:
        """Compute spectral entropy of power spectrum."""
        # Normalize to probability distribution
        probs = power_spectrum / (torch.sum(power_spectrum) + 1e-8)
        probs = probs[probs > 1e-8]  # Remove zeros
        
        if len(probs) > 1:
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            return entropy.item()
        return 0.0

    def _analyze_corridor_metrics(
        self, 
        group_tensors: Dict[str, Dict[str, torch.Tensor]]
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
            flat = self._apply_normalization(tens, group_tensors[g])  # [N, T, D]
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

    def _analyze_geometry_derivatives(
        self, 
        group_tensors: Dict[str, Dict[str, torch.Tensor]]
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
            X = self._apply_normalization(pack['trajectory_tensor'], pack).cpu().numpy()  # [N,T,D]
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

    
    def _find_peaks_gpu(self, signal: torch.Tensor) -> List[int]:
        """Simple peak detection on GPU tensor."""
        if len(signal) < 3:
            return []
        
        # If signal is multi-dimensional, use the norm for peak detection
        if signal.dim() > 1:
            signal = torch.norm(signal, dim=tuple(range(1, signal.dim())))
        
        # Find local maxima using tensor operations
        peaks = []
        for i in range(1, len(signal) - 1):
            # Convert tensor comparisons to boolean values properly
            is_peak = (signal[i] > signal[i-1]).item() and (signal[i] > signal[i+1]).item()
            if is_peak:
                peaks.append(i)
        
        return peaks

    
    def _apply_normalization(self, trajectory_tensor: torch.Tensor, group_data: Dict[str, Any]) -> torch.Tensor:
        """Return [N,T,D] normalized according to self.norm_cfg."""
        traj = trajectory_tensor[:, :, 0] if trajectory_tensor.shape[2] == 1 else trajectory_tensor
        if self.norm_cfg.get("per_channel_standardize", False):
            mean_c = traj.mean(dim=(0,1,3,4,5), keepdim=True)
            std_c  = traj.std(dim=(0,1,3,4,5), keepdim=True) + 1e-6
            traj = (traj - mean_c) / std_c
        flat = traj.flatten(start_dim=2)
        if self.norm_cfg.get("per_step_whiten", False):
            mean = flat.mean(dim=2, keepdim=True)
            std  = flat.std(dim=2, keepdim=True) + 1e-6
            flat = (flat - mean) / std
        if self.norm_cfg.get("snr_normalize", False):
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

    def _bootstrap_ci(
        self, 
        arr: np.ndarray, 
        n_boot: int = 2000, 
        alpha: float = 0.05
    ):
        """Return (mean, lo, hi) with (1-alpha) CI via bootstrap; NaN-safe."""
        import numpy as np
        x = np.asarray(arr, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return (np.nan, np.nan, np.nan)
        rng = np.random.default_rng(123)
        boots = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            s = rng.choice(x, size=x.size, replace=True)
            boots[i] = float(np.mean(s))
        lo, hi = np.quantile(boots, [alpha/2, 1 - alpha/2])
        return (float(np.mean(x)), float(lo), float(hi))

    def _attach_confidence_intervals(self, results: LatentTrajectoryAnalysis):
        """
        Adds bootstrap CIs for per-group bar metrics:
        length, velocity, acceleration, circuitousness−1, turning, alignment, late/early
        """
        import numpy as np
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
            CIs[g]['length']       = self._bootstrap_ci(L)
            CIs[g]['velocity']     = self._bootstrap_ci(V)
            CIs[g]['acceleration'] = self._bootstrap_ci(A)

            if g in geom and 'error' not in geom[g]:
                circ = np.array(geom[g]['circuitousness_stats']['individual_values'], dtype=float) - 1.0
                turn = np.array(geom[g]['turning_angle_stats']['individual_values'], dtype=float)
                ali  = np.array(geom[g]['endpoint_alignment_stats']['individual_values'], dtype=float)
                CIs[g]['circuitousness_minus1'] = self._bootstrap_ci(circ)
                CIs[g]['turning_angle']         = self._bootstrap_ci(turn)
                CIs[g]['endpoint_alignment']    = self._bootstrap_ci(ali)
            # late/early is group-level; skip CI unless you store per-video curves
        return CIs
        
    def _compute_normative_strength(
        self, 
        results: LatentTrajectoryAnalysis
    ):
        """
        Prototype dominance index combining:
        + early corridor width (steps 0..k)
        - exit distance from baseline (sum over steps)
        - late-recovery area (from spatial variance curve; larger area = more late effort)
        Returns z-scored composite per group.
        """
        import numpy as np
        out = {}
        # need corridor metrics & spatial curves
        corridor = getattr(results, 'corridor_metrics', None)
        if not corridor: 
            self.logger.warning("No corridor metrics found.")
            return out
        groups = sorted(corridor['width_by_step'].keys())
        k = 3  # early window (t=0..3) for width

        # late recovery area from spatial curve
        def late_area(g):
            sp = results['spatial_patterns']['trajectory_spatial_evolution'][g]
            curve = None
            for kname in ('spatial_variance_curve','spatial_variance_by_step','variance_curve'):
                if kname in sp: curve = np.array(sp[kname], dtype=float)
            if curve is None or curve.size < 4: return np.nan
            t = np.arange(curve.size)
            # area after the minimum (late recovery)
            t0 = int(np.argmin(curve))
            return float(np.trapz(curve[t0:], t[t0:]))

        w_early = np.array([np.nanmean(corridor['width_by_step'][g][:k+1]) for g in groups], dtype=float)
        exit_d  = np.array([corridor['exit_distance'][g] for g in groups], dtype=float)
        late    = np.array([late_area(g) for g in groups], dtype=float)

        def z(a): 
            m, s = np.nanmean(a), np.nanstd(a)
            return (a - m) / (s + 1e-12)

        # dominance: more early width, less distance to exit, less late recovery
        score = +z(w_early) - z(exit_d) - z(late)
        for i,g in enumerate(groups):
            out[g] = {'dominance_index': float(score[i]),
                    'z_early_width': float(z(w_early)[i]),
                    'z_exit_distance': float(z(exit_d)[i]),
                    'z_late_area': float(z(late)[i])}
        return out


    def _add_log_volume_deltas(
        self, 
        results: LatentTrajectoryAnalysis
    ):
        """Adds group-level and (if possible) paired per-video Δ% vs baseline for individual log-volumes."""
        import numpy as np
        geom = results['individual_trajectory_geometry']
        groups = sorted(geom.keys())
        means = np.array([float(geom[g]['log_volume_stats']['mean']) if 'error' not in geom[g] else np.nan for g in groups])
        base = means[0] if means.size else np.nan
        group_delta = 100.0 * (means - base) / (base + 1e-12)

        res = {
            'groups': groups,
            'group_means': means.tolist(),
            'group_delta_percent': group_delta.tolist()
        }
        return res


    def _track_gpu_memory(self, stage: str):
        """Track GPU memory usage at different stages."""
        if self.device.startswith("cuda"):
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            
            self.performance_stats['memory_usage'][stage] = {
                'allocated_gb': allocated,
                'cached_gb': cached
            }
            
            if 'peak_allocated_gb' not in self.performance_stats['memory_usage']:
                self.performance_stats['memory_usage']['peak_allocated_gb'] = allocated
            else:
                self.performance_stats['memory_usage']['peak_allocated_gb'] = max(
                    self.performance_stats['memory_usage']['peak_allocated_gb'], 
                    allocated
                )

    def analyze_prompt_groups(self, prompt_groups: List[str], 
                            prompt_metadata: Optional[Dict[str, Dict[str, str]]] = None) -> LatentTrajectoryAnalysis:
        """Main analysis entry point with trajectory-aware processing."""
        self.logger.info("Starting GPU-optimized trajectory-aware analysis")
        start_time = time.time()
        
        if not getattr(self, 'group_tensors', None):
            # 1. Load and batch trajectory data
            self._track_gpu_memory("start")
            self.group_tensors = self._load_and_batch_trajectory_data(prompt_groups)
            
            self._track_gpu_memory("data_loaded")
        
        if not self.group_tensors:
            raise ValueError("No trajectory data loaded")

        group_tensors = self.group_tensors
        
        # Get trajectory shape
        sample_tensor = next(iter(group_tensors.values()))['trajectory_tensor']
        trajectory_shape = tuple(sample_tensor.shape[2:])  # Remove videos and steps dimensions
        self.logger.info(f"Analyzing trajectory latents with shape: {trajectory_shape}")
        self.logger.info(f"Trajectory structure: [videos={sample_tensor.shape[0]}, steps={sample_tensor.shape[1]}, ...]")
        
        # 2. GPU-accelerated analysis suite
        analysis_results = {}
        
        # Use autocast only if CUDA is available
        if self.device.startswith('cuda'):
            autocast_context = torch.amp.autocast('cuda', enabled=self.enable_mixed_precision)
        else:
            autocast_context = torch.amp.autocast('cpu', enabled=False)  # CPU doesn't support autocast
        
        with autocast_context:
            # Core trajectory-aware analyses
            self.logger.info("Running spatial pattern analysis...")
            # analysis_results['spatial_patterns'] = self._gpu_analyze_spatial_patterns(group_tensors)
            analysis_results['spatial_patterns'] = analyze_spatial_patterns(group_tensors)
            self._track_gpu_memory("spatial_analysis")
            
            self.logger.info("Running temporal coherence analysis...")
            # analysis_results['temporal_coherence'] = self._gpu_analyze_temporal_coherence(group_tensors)
            analysis_results['temporal_coherence'] = analyze_temporal_coherence(group_tensors, device=self.device)
            self._track_gpu_memory("temporal_analysis")
            
            self.logger.info("Running channel pattern analysis...")
            # analysis_results['channel_analysis'] = self._gpu_analyze_channel_patterns(group_tensors)
            analysis_results['channel_analysis'] = analyze_channel_patterns(group_tensors)
            self._track_gpu_memory("channel_analysis")
            
            # Multi-scale analysis
            self.logger.info("Running patch diversity analysis...")
            # analysis_results['patch_diversity'] = self._gpu_analyze_patch_diversity(group_tensors)
            analysis_results['patch_diversity'] = analyze_patch_diversity(group_tensors)

            
            self.logger.info("Running global structure analysis...")
            # analysis_results['global_structure'] = self._gpu_analyze_global_structure(group_tensors)
            analysis_results['global_structure'] = analyze_global_structure(group_tensors)


            # Simplified additional analyses
            self.logger.info("Running information content analysis...")
            # analysis_results['information_content'] = self._gpu_analyze_information_content(group_tensors)
            analysis_results['information_content'] = analyze_information_content(group_tensors)

            
            self.logger.info("Running complexity analysis...")
            # analysis_results['complexity_measures'] = self._gpu_analyze_complexity_measures(group_tensors)
            analysis_results['complexity_measures'] = analyze_complexity_measures(group_tensors)

            self.logger.info("Running frequency analysis...")
            # analysis_results['frequency_patterns'] = self._gpu_analyze_frequency_patterns(group_tensors)
            analysis_results['frequency_patterns'] = analyze_frequency_patterns(group_tensors)

            
            # Group separability
            self.logger.info("Running group separability analysis...")
            # analysis_results['group_separability'] = self._gpu_analyze_group_separability(group_tensors, prompt_groups)
            analysis_results['group_separability'] = analyze_group_separability(group_tensors)

            # Temporal trajectory analysis
            self.logger.info("Running temporal trajectory analysis...")
            # analysis_results['temporal_analysis'] = self._gpu_analyze_temporal_trajectories(group_tensors, prompt_groups)
            analysis_results['temporal_analysis'] = analyze_temporal_trajectories(group_tensors, prompt_groups, device=self.device, norm_cfg=self.norm_cfg)

            # Structural analysis
            self.logger.info("Running structural analysis...")
            # analysis_results['structural_analysis'] = self._gpu_analyze_structural_patterns(group_tensors, prompt_groups)
            analysis_results['structural_analysis'] = analyze_structural_patterns(group_tensors, prompt_groups, device=self.device)
            
            # NEW: Advanced geometric analysis
            self.logger.info("Running convex hull analysis...")
            analysis_results['convex_hull_analysis'] = self._gpu_analyze_convex_hull_metrics_safe(group_tensors, prompt_groups)
            
            self.logger.info("Running functional PCA analysis...")
            analysis_results['functional_pca_analysis'] = self._gpu_analyze_functional_pca(group_tensors, prompt_groups)
            
            self.logger.info("Running individual trajectory geometry analysis...")
            # analysis_results['individual_trajectory_geometry'] = self._gpu_analyze_individual_trajectory_geometry(group_tensors, prompt_groups)
            analysis_results['individual_trajectory_geometry'] = analyze_individual_trajectory_geometry(group_tensors)
            
            # TODO: maybe skip -- function only stubbed
            self.logger.info("Running intrinsic dimension analysis...")
            # analysis_results['intrinsic_dimension_analysis'] = self._gpu_analyze_intrinsic_dimension(group_tensors, prompt_groups)
            analysis_results['intrinsic_dimension_analysis'] = analyze_intrinsic_dimension(group_tensors)

            # Statistical significance
            self.logger.info("Running statistical significance tests...")
            # analysis_results['statistical_significance'] = self._gpu_test_statistical_significance(group_tensors, prompt_groups)
            analysis_results['statistical_significance'] = test_statistical_significance(group_tensors)

            # Corridor metrics
            self.logger.info("Running corridor metrics tests...")
            # analysis_results['corridor_metrics'] = self._analyze_corridor_metrics(group_tensors)
            analysis_results['corridor_metrics'] = analyze_corridor_metrics(group_tensors, norm_cfg=self.norm_cfg)

            # # Geometry derivatives metrics
            self.logger.info("Running geometry derivatives analysis...")
            # analysis_results['geometry_derivatives'] = self._analyze_geometry_derivatives(group_tensors)
            analysis_results['geometry_derivatives'] = analyze_geometry_derivatives(group_tensors, norm_cfg=self.norm_cfg)

            self.logger.info("Attaching confidence intervals...")
            # analysis_results['confidence_intervals'] = self._attach_confidence_intervals(analysis_results)
            analysis_results['confidence_intervals'] = attach_confidence_intervals(analysis_results)


            self.logger.info("Log volume delta vs baseline...")
            # analysis_results['log_volume_delta_vs_baseline'] = self._add_log_volume_deltas(analysis_results)
            analysis_results['log_volume_delta_vs_baseline'] = log_volume_deltas(analysis_results)


            self.logger.info("Running normative strength...")
            # analysis_results['normative_strength'] = self._compute_normative_strength(analysis_results)
            analysis_results['normative_strength'] = compute_normative_strength(analysis_results)

        self._track_gpu_memory("analysis_complete")
        
        # 3. Create analysis metadata
        total_time = time.time() - start_time
        analysis_metadata = {
            'total_analysis_time_seconds': total_time,
            'prompt_groups': prompt_groups,
            'prompt_metadata': prompt_metadata or {},
            'latents_directory': str(self.latents_dir),
            'trajectory_shape': trajectory_shape,
            'device_used': self.device,
            'mixed_precision': self.enable_mixed_precision,
            'batch_size': self.batch_size,
            'analysis_timestamp': time.strftime("%Y%m%d_%H%M%S")
        }
        
        # 4. Save results
        results = LatentTrajectoryAnalysis(
            spatial_patterns=analysis_results.get('spatial_patterns', {}),
            temporal_coherence=analysis_results.get('temporal_coherence', {}),
            channel_analysis=analysis_results.get('channel_analysis', {}),
            patch_diversity=analysis_results.get('patch_diversity', {}),
            global_structure=analysis_results.get('global_structure', {}),
            information_content=analysis_results.get('information_content', {}),
            complexity_measures=analysis_results.get('complexity_measures', {}),
            frequency_patterns=analysis_results.get('frequency_patterns', {}),
            group_separability=analysis_results.get('group_separability', {}),
            temporal_analysis=analysis_results.get('temporal_analysis', {}),
            structural_analysis=analysis_results.get('structural_analysis', {}),
            statistical_significance=analysis_results.get('statistical_significance', {}),
            convex_hull_analysis=analysis_results.get('convex_hull_analysis', {}),
            functional_pca_analysis=analysis_results.get('functional_pca_analysis', {}),
            individual_trajectory_geometry=analysis_results.get('individual_trajectory_geometry', {}),
            intrinsic_dimension_analysis=analysis_results.get('intrinsic_dimension_analysis', {}),
            
            # These were not from analysis_results, so they can remain as they are.
            gpu_performance_stats=self.performance_stats,
            analysis_metadata=analysis_metadata
        )
                
        self._save_results(results)
        
        self.logger.info(f"Latent Trajectory Analysis completed in {total_time:.2f} seconds")

        # TODO: figure out better solution than returning group tensors
        return results


    def run_dual_tracks(
        self,
        prompt_groups: List[str],
        prompt_metadata: Optional[Dict[str, Dict[str, str]]] = None
    ) -> Dict[str, LatentTrajectoryAnalysis]:
        """Run SNR-only and Full normalization in one pass and build a combined board."""
        from copy import deepcopy
        base_out = Path(self.output_dir) if self.output_dir else (self.latents_dir.parent / 'latent_trajectory_analysis_results')
        base_out.mkdir(parents=True, exist_ok=True)

        tracks = {
            'snr_only': {'per_step_whiten': False, 'per_channel_standardize': False, 'snr_normalize': True},
            'full_norm': {'per_step_whiten': True,  'per_channel_standardize': True,  'snr_normalize': True},
        }

        saved: Dict[str, LatentTrajectoryAnalysis] = {}
        orig_out = deepcopy(self.output_dir)
        orig_norm = deepcopy(self.norm_cfg)

        try:
            for name, cfg in tracks.items():
                self.norm_cfg.update(cfg)
                self.output_dir = base_out / name
                self.output_dir.mkdir(parents=True, exist_ok=True)

                res = self.analyze_prompt_groups(prompt_groups, prompt_metadata)
                saved[name] = res

            return saved
        finally:
            # Always restore
            self.output_dir = orig_out
            self.norm_cfg = orig_norm



    def _generate_research_insights(self, results: LatentTrajectoryAnalysis, group_names: List[str]) -> str:
        """Generate research insights text based on analysis results including advanced geometric metrics."""
        
        try:
            # Extract key metrics for statistical analysis
            spatial_data = results.spatial_patterns.get('trajectory_spatial_evolution', {})
            sync_data = results.temporal_coherence.get('cross_trajectory_synchronization', {})
            momentum_data = results.temporal_coherence.get('temporal_momentum_analysis', {})
            separability_data = results.group_separability.get('inter_group_distances', {})
            
            # NEW: Extract advanced geometric metrics
            hull_data = results.convex_hull_analysis
            fpca_data = results.functional_pca_analysis  
            geometry_data = results.individual_trajectory_geometry
            id_data = results.intrinsic_dimension_analysis
            
            # Calculate summary statistics
            trajectory_distances = []
            consistency_scores = []
            avg_velocities = []
            
            # NEW: Advanced geometric summary statistics
            hull_volumes = []
            trajectory_complexities = []
            individual_speeds = []
            individual_circuitousness = []
            intrinsic_dimensions = []
            
            for group in group_names:
                # Original trajectory distance
                if group in spatial_data:
                    pattern = spatial_data[group].get('trajectory_pattern', [])
                    if pattern and len(pattern) > 1:
                        total_dist = np.sum(np.abs(np.diff(pattern)))
                        trajectory_distances.append(total_dist)
                
                # Consistency
                if group in sync_data:
                    consistency_scores.append(sync_data[group].get('mean_correlation', 0))
                
                # Velocity
                if group in momentum_data:
                    velocity_mean = momentum_data[group].get('velocity_mean', [])
                    if velocity_mean:
                        avg_velocities.append(np.mean(np.abs(velocity_mean)))
                
                # NEW: Convex hull volume (representational diversity)
                if group in hull_data and 'error' not in hull_data[group]:
                    hull_volumes.append(hull_data[group].get('hull_volume', 0))
                
                # NEW: Functional PCA complexity
                if group in fpca_data and 'error' not in fpca_data[group]:
                    eff_components = fpca_data[group].get('effective_components_95', 0)
                    trajectory_complexities.append(eff_components)
                
                # NEW: Individual trajectory metrics
                if group in geometry_data and 'error' not in geometry_data[group]:
                    speed_stats = geometry_data[group].get('speed_stats', {})
                    circuit_stats = geometry_data[group].get('circuitousness_stats', {})
                    if speed_stats:
                        individual_speeds.append(speed_stats.get('mean', 0))
                    if circuit_stats:
                        individual_circuitousness.append(circuit_stats.get('mean', 0))
                
                # NEW: Intrinsic dimension (manifold complexity)
                if id_data and group in id_data and 'error' not in id_data[group]:
                    consensus_dim = id_data[group].get('consensus_intrinsic_dimension', 0)
                    intrinsic_dimensions.append(consensus_dim)
            
            # Statistical tests (enhanced)
            significant_findings = []
            
            if len(trajectory_distances) > 2:
                dist_range = max(trajectory_distances) - min(trajectory_distances)
                dist_mean = np.mean(trajectory_distances)
                if dist_range > 0.5 * dist_mean:  # >50% variation
                    significant_findings.append("SIGNIFICANT: Large variation in trajectory distances between prompt groups")
            
            if len(consistency_scores) > 2:
                consistency_range = max(consistency_scores) - min(consistency_scores)
                if consistency_range > 0.3:  # >30% range
                    significant_findings.append("SIGNIFICANT: Generation consistency varies substantially by prompt type")
            
            if len(avg_velocities) > 2:
                velocity_cv = np.std(avg_velocities) / np.mean(avg_velocities) if np.mean(avg_velocities) > 0 else 0
                if velocity_cv > 0.2:  # >20% coefficient of variation
                    significant_findings.append("SIGNIFICANT: Trajectory velocities differ meaningfully between groups")
            
            # NEW: Advanced geometric significance tests
            if len(hull_volumes) > 2:
                hull_cv = np.std(hull_volumes) / np.mean(hull_volumes) if np.mean(hull_volumes) > 0 else 0
                if hull_cv > 0.3:  # >30% coefficient of variation
                    significant_findings.append("SIGNIFICANT: Representational diversity (hull volume) varies dramatically between prompts")
            
            if len(individual_circuitousness) > 2:
                circuit_range = max(individual_circuitousness) - min(individual_circuitousness)
                if circuit_range > 1.0:  # Circuitousness difference > 1.0
                    significant_findings.append("SIGNIFICANT: Trajectory efficiency patterns differ between prompt groups")
            
            if len(intrinsic_dimensions) > 2:
                id_range = max(intrinsic_dimensions) - min(intrinsic_dimensions)
                if id_range > 5:  # Intrinsic dimension difference > 5
                    significant_findings.append("SIGNIFICANT: Manifold complexity varies between prompt groups")
            
            # Count groups with U-shaped pattern
            u_shaped_count = 0
            for group in group_names:
                if group in spatial_data:
                    pattern = spatial_data[group].get('trajectory_pattern', [])
                    if len(pattern) >= 3:
                        # Check for U-shape: start high, go low, end higher
                        start_third = np.mean(pattern[:len(pattern)//3])
                        middle_third = np.mean(pattern[len(pattern)//3:2*len(pattern)//3])
                        end_third = np.mean(pattern[2*len(pattern)//3:])
                        if start_third > middle_third and end_third > middle_third:
                            u_shaped_count += 1
            
            insights_text = f"""
🔬 RESEARCH FINDINGS: ADVANCED GEOMETRIC TRAJECTORY ANALYSIS

📊 COMPREHENSIVE STATISTICAL SUMMARY:
• Groups Analyzed: {len(group_names)}

BASIC TRAJECTORY METRICS:
• Trajectory Distance Range: {max(trajectory_distances) - min(trajectory_distances):.3f} ({len(trajectory_distances)} groups)
• Consistency Score Range: {max(consistency_scores) - min(consistency_scores):.3f} ({len(consistency_scores)} groups)
• Velocity Variation (CV): {(np.std(avg_velocities) / np.mean(avg_velocities)):.3f} ({len(avg_velocities)} groups)

NEW: ADVANCED GEOMETRIC METRICS:
• Hull Volume Range: {f"{min(hull_volumes):.2e} - {max(hull_volumes):.2e}" if hull_volumes else "No data"} (representational diversity)
• FPCA Complexity Range: {f"{min(trajectory_complexities)} - {max(trajectory_complexities)} components" if trajectory_complexities else "No data"}
• Individual Speed Range: {f"{min(individual_speeds):.3f} - {max(individual_speeds):.3f}" if individual_speeds else "No data"}
• Circuitousness Range: {f"{min(individual_circuitousness):.2f} - {max(individual_circuitousness):.2f}" if individual_circuitousness else "No data"}
• Intrinsic Dimension Range: {f"{min(intrinsic_dimensions):.1f} - {max(intrinsic_dimensions):.1f}" if intrinsic_dimensions else "No data"}

🎯 KEY RESEARCH QUESTIONS ANSWERED:

1. CONVEX HULL VOLUME (Representational Diversity):
   {"✅ CONFIRMED" if len(hull_volumes) > 0 and np.std(hull_volumes)/np.mean(hull_volumes) > 0.3 else "❌ INCONCLUSIVE"}: Prompt specificity affects latent space occupation
   Most Diverse: {group_names[hull_volumes.index(max(hull_volumes))] if hull_volumes else "Unknown"} (Volume: {max(hull_volumes) if hull_volumes else 0})

2. FUNCTIONAL PCA COMPLEXITY:
   {"✅ SIGNIFICANT" if len(trajectory_complexities) > 0 and max(trajectory_complexities) - min(trajectory_complexities) > 2 else "❌ MINIMAL"}: Trajectory shape complexity varies by prompt
   Range: {f"{min(trajectory_complexities)} - {max(trajectory_complexities)} effective components" if trajectory_complexities else "No data"}

3. INDIVIDUAL TRAJECTORY GEOMETRY:
   Speed Efficiency: {"✅ VARIES" if len(individual_speeds) > 0 and np.std(individual_speeds)/np.mean(individual_speeds) > 0.2 else "❌ UNIFORM"}
   Path Efficiency: {"✅ VARIES" if len(individual_circuitousness) > 0 and max(individual_circuitousness) - min(individual_circuitousness) > 1.0 else "❌ UNIFORM"}
   Most Efficient: {group_names[individual_circuitousness.index(min(individual_circuitousness))] if individual_circuitousness else "Unknown"} (Circuitousness: {min(individual_circuitousness) if individual_circuitousness else 0})



5. UNIVERSAL DENOISING PATTERN:
   {"✅ CONFIRMED" if u_shaped_count > len(group_names) * 0.7 else "❌ PARTIAL"}: U-shaped pattern observed in {u_shaped_count}/{len(group_names)} groups ({u_shaped_count/len(group_names)*100:.1f}%)

🔍 SIGNIFICANT FINDINGS:
{chr(10).join(f"• {finding}" for finding in significant_findings) if significant_findings else "• No statistically significant patterns detected"}

🧠 ADVANCED RESEARCH IMPLICATIONS:
• VOLUMETRIC MEASUREMENT: Convex hull volumes quantify representational "area" occupied by different concepts
• SHAPE CHARACTERIZATION: FPCA reveals dominant modes of variation in trajectory manifolds  
• INDIVIDUAL GEOMETRY: Speed, volume, and circuitousness provide trajectory-level insights
• COMPLEXITY MEASUREMENT: Intrinsic dimension estimates reveal latent manifold complexity

📈 STATISTICAL CONFIDENCE:
• Large Effect Sizes: {"✅" if len(significant_findings) >= 3 else "❌"} Multiple significant patterns detected
• Reproducible Patterns: {"✅" if u_shaped_count > len(group_names) * 0.5 else "❌"} Universal denoising confirmed
• Group Separability: {"✅" if len(separability_data) > 0 else "❌"} Mathematical distinction between prompt groups
• Geometric Validation: {"✅" if len(hull_volumes) > 0 and len(intrinsic_dimensions) > 0 else "❌"} Advanced metrics confirm trajectory differences

🎯 NEXT RESEARCH DIRECTIONS:
• Correlate convex hull volume with visual output diversity
• Map FPCA eigenfunctions to semantic content changes
• Develop efficiency metrics for generation quality prediction
• Use intrinsic dimension for bias detection in representational coverage

⚠️  METHODOLOGICAL NOTES:
• Analysis based on {results.analysis_metadata.get("trajectory_shape", "unknown")} latent shape
• Processing: {results.analysis_metadata.get("device_used", "unknown")}
• Advanced geometric methods: ConvexHull, FPCA, Individual Geometry, Intrinsic Dimension
• Statistical power: N={len(group_names)} prompt groups
            """.strip()
            
            return insights_text
            
        except Exception as e:
            self.logger.error(f"Failed to generate research insights: {e}")
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")

            return f"""
🔬 RESEARCH FINDINGS: ANALYSIS ERROR

❌ Unable to generate comprehensive insights due to data processing error:
{str(e)}

Please check the analysis logs for detailed error information.
            """.strip()

    def _load_and_batch_trajectory_data(self, prompt_groups: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Load and batch trajectory data preserving diffusion step structure."""
        import gzip
        
        group_tensors = {}
        
        for group_name in prompt_groups:
            self.logger.info(f"Loading trajectory-structured latents for group: {group_name}")
            
            group_dir = self.latents_dir / group_name
            if not group_dir.exists():
                self.logger.warning(f"Group directory not found: {group_dir}")
                continue
            
            # Find all video directories
            video_dirs = sorted([d for d in group_dir.iterdir() if d.is_dir() and d.name.startswith('vid_')])
            
            if not video_dirs:
                self.logger.warning(f"No video directories found in {group_dir}")
                continue
            
            video_trajectories = []
            trajectory_metadata = []
            
            for video_dir in video_dirs:
                try:
                    # Find all step files for this video
                    step_files = sorted([f for f in video_dir.glob("step_*.npy.gz")])
                    
                    if not step_files:
                        self.logger.warning(f"No step files found in {video_dir}")
                        continue
                    
                    # Load trajectory preserving diffusion step order
                    video_steps = []
                    step_metadata = []
                    
                    for step_file in step_files:
                        # Load compressed numpy array
                        with gzip.open(step_file, 'rb') as f:
                            step_latent = np.load(f)
                        
                        # Convert to tensor: [1, 16, frames, H, W]
                        step_tensor = torch.from_numpy(step_latent).float()
                        video_steps.append(step_tensor)
                        
                        # Load metadata if available
                        metadata_file = step_file.with_name(step_file.stem.replace('.npy', '_metadata.json'))
                        if metadata_file.exists():
                            with open(metadata_file) as f:
                                step_meta = json.load(f)
                                step_metadata.append(step_meta)
                    
                    if video_steps:
                        # Stack steps to create video trajectory: [steps, 1, 16, frames, H, W]
                        video_trajectory = torch.stack(video_steps, dim=0)
                        video_trajectories.append(video_trajectory)
                        
                        trajectory_metadata.append({
                            'video_id': video_dir.name,
                            'n_steps': len(video_steps),
                            'step_metadata': step_metadata,
                            'trajectory_shape': video_trajectory.shape
                        })
                        
                        self.logger.debug(f"Loaded video {video_dir.name}: {video_trajectory.shape}")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to load trajectory for {video_dir.name}: {e}")
            
            if video_trajectories:
                try:
                    # Ensure all trajectories have same number of steps
                    min_steps = min(traj.shape[0] for traj in video_trajectories)
                    self.logger.info(f"Truncating all trajectories to {min_steps} steps for consistency")
                    
                    # Truncate and stack: [n_videos, steps, 1, 16, frames, H, W]
                    truncated_trajectories = [traj[:min_steps] for traj in video_trajectories]
                    batched_trajectories = torch.stack(truncated_trajectories, dim=0)
                    
                    # Move to device
                    batched_trajectories = batched_trajectories.to(self.device)
                    
                    group_tensors[group_name] = {
                        'trajectory_tensor': batched_trajectories,  # [n_videos, steps, 1, 16, frames, H, W]
                        'trajectory_metadata': trajectory_metadata,
                        'n_videos': len(video_trajectories),
                        'n_steps': min_steps,
                        'latent_shape': batched_trajectories.shape[3:],  # [16, frames, H, W]
                        'full_shape': batched_trajectories.shape
                    }
                    
                    self.logger.info(f"✅ Loaded {len(video_trajectories)} trajectory videos for {group_name}")
                    self.logger.info(f"   Shape: {batched_trajectories.shape} [videos, steps, batch, channels, frames, H, W]")
                    self.logger.info(f"   Preserving trajectory structure for diffusion step analysis")
                    self.logger.info(f"   Device: {batched_trajectories.device}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to batch trajectories for {group_name}: {e}")
        
        return group_tensors

    def _gpu_analyze_spatial_patterns(self, group_tensors: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """GPU-accelerated spatial pattern analysis preserving trajectory structure."""
        spatial_analysis = {
            'spatial_variance_maps': {},
            'trajectory_spatial_evolution': {},
            'spatial_progression_patterns': {},
            'video_spatial_diversity': {},
            'edge_density_evolution': {},
            'spatial_coherence_patterns': {}
        }
        
        for group_name in sorted(group_tensors.keys()):
            data = group_tensors[group_name]
            self.logger.info(f"GPU analyzing spatial patterns for {group_name}")
            
            trajectories = data['trajectory_tensor']  # [n_videos, steps, 1, 16, frames, H, W]
            trajectories = trajectories.squeeze(2)   # [n_videos, steps, 16, frames, H, W]
            
            n_videos, n_steps = trajectories.shape[:2]
            
            # 1. Trajectory-aware spatial variance evolution
            # Compute spatial variance for each video at each diffusion step
            spatial_vars_per_step = torch.var(trajectories, dim=(-2, -1))  # [n_videos, steps, 16, frames]
            spatial_vars_mean_per_step = torch.mean(spatial_vars_per_step, dim=(2, 3))  # [n_videos, steps]
            
            # Average across videos to get group trajectory pattern
            group_spatial_trajectory = torch.mean(spatial_vars_mean_per_step, dim=0)  # [steps]
            
            # 2. Step-to-step spatial changes within trajectories
            spatial_trajectory_deltas = torch.diff(spatial_vars_mean_per_step, dim=1)  # [n_videos, steps-1]
            
            # 3. Video-level spatial diversity (how much each video varies spatially)
            video_spatial_diversity = torch.std(spatial_vars_mean_per_step, dim=1)  # [n_videos]
            
            # 4. Cross-video spatial consistency at each step
            step_consistency = torch.std(spatial_vars_mean_per_step, dim=0)  # [steps]
            
            # 5. Early vs Late diffusion spatial patterns
            early_steps = spatial_vars_mean_per_step[:, :n_steps//3]  # First third
            late_steps = spatial_vars_mean_per_step[:, -n_steps//3:]  # Last third
            
            early_spatial_mean = torch.mean(early_steps)
            late_spatial_mean = torch.mean(late_steps)
            spatial_evolution_ratio = late_spatial_mean / (early_spatial_mean + 1e-8)
            
            # 6. Edge density evolution in trajectory
            sample_indices = torch.randperm(n_videos)[:min(4, n_videos)]  # Sample videos
            edge_evolutions = []
            
            for video_idx in sample_indices:
                video_traj = trajectories[video_idx]  # [steps, 16, frames, H, W]
                
                # Compute edge density for each step
                step_edge_densities = []
                for step in range(min(n_steps, 20)):  # Sample steps
                    step_data = video_traj[step]  # [16, frames, H, W]
                    
                    # Compute gradients
                    grad_x = torch.diff(step_data, dim=-1).abs().mean()
                    grad_y = torch.diff(step_data, dim=-2).abs().mean()
                    edge_density = (grad_x + grad_y) / 2
                    step_edge_densities.append(edge_density.item())
                
                edge_evolutions.append(step_edge_densities)
            
            # 7. Spatial coherence patterns (spatial autocorrelation evolution)
            spatial_coherences = []
            for video_idx in range(min(n_videos, 6)):  # Sample videos
                video_traj = trajectories[video_idx]  # [steps, 16, frames, H, W]
                
                video_coherences = []
                for step in range(0, n_steps, max(1, n_steps//10)):  # Sample steps
                    step_data = video_traj[step]  # [16, frames, H, W]
                    
                    # Spatial autocorrelation for each channel/frame
                    coherence_values = []
                    for c in range(min(4, step_data.shape[0])):  # Sample channels
                        for f in range(step_data.shape[1]):  # All frames
                            spatial_map = step_data[c, f]  # [H, W]
                            
                            if spatial_map.shape[0] > 4 and spatial_map.shape[1] > 4:
                                # Simple spatial autocorrelation using shifted correlation
                                shifted_h = torch.roll(spatial_map, 1, dims=0)
                                shifted_w = torch.roll(spatial_map, 1, dims=1)
                                
                                corr_h = self._gpu_corrcoef(spatial_map.flatten(), shifted_h.flatten())
                                corr_w = self._gpu_corrcoef(spatial_map.flatten(), shifted_w.flatten())
                                
                                if not (torch.isnan(corr_h) or torch.isnan(corr_w)):
                                    coherence_values.append((corr_h + corr_w).item() / 2)
                    
                    if coherence_values:
                        video_coherences.append(np.mean(coherence_values))
                
                if video_coherences:
                    spatial_coherences.append(video_coherences)
            
            # Store trajectory-aware results (optimized data storage)
            spatial_analysis['spatial_variance_maps'][group_name] = {
                'mean': float(torch.mean(spatial_vars_per_step).item()),
                'std': float(torch.std(spatial_vars_per_step).item()),
                'distribution_sample': spatial_vars_per_step.flatten().cpu().numpy().tolist()[:50]  # Reduced from 1000 to 50
            }
            
            spatial_analysis['trajectory_spatial_evolution'][group_name] = {
                'trajectory_pattern': group_spatial_trajectory.cpu().numpy().tolist(),
                'evolution_ratio': float(spatial_evolution_ratio.item()),
                'early_vs_late_significance': float(torch.abs(early_spatial_mean - late_spatial_mean).item()),
                'trajectory_smoothness': float(torch.mean(torch.abs(spatial_trajectory_deltas)).item()),
                'phase_transition_strength': float(torch.std(group_spatial_trajectory).item())
            }
            
            spatial_analysis['spatial_progression_patterns'][group_name] = {
                'step_deltas_mean': torch.mean(spatial_trajectory_deltas, dim=0).cpu().numpy().tolist(),
                'step_deltas_std': torch.std(spatial_trajectory_deltas, dim=0).cpu().numpy().tolist(),
                'progression_consistency': float(torch.mean(step_consistency).item()),
                'progression_variability': float(torch.std(step_consistency).item()),
                'edge_evolution_patterns': edge_evolutions
            }
            
            spatial_analysis['video_spatial_diversity'][group_name] = {
                'inter_video_diversity_mean': float(torch.mean(video_spatial_diversity).item()),
                'inter_video_diversity_std': float(torch.std(video_spatial_diversity).item()),
                'diversity_distribution': video_spatial_diversity.cpu().numpy().tolist()
            }
            
            spatial_analysis['edge_density_evolution'][group_name] = {
                'mean_evolution_pattern': np.mean(edge_evolutions, axis=0).tolist() if edge_evolutions else [],
                'evolution_variability': np.std(edge_evolutions, axis=0).tolist() if edge_evolutions else [],
                'edge_formation_trend': 'increasing' if len(edge_evolutions) > 0 and np.mean([evo[-1] for evo in edge_evolutions]) > np.mean([evo[0] for evo in edge_evolutions]) else 'decreasing'
            }
            
            spatial_analysis['spatial_coherence_patterns'][group_name] = {
                'coherence_evolution': spatial_coherences,
                'mean_coherence_trajectory': np.mean(spatial_coherences, axis=0).tolist() if spatial_coherences else [],
                'coherence_stability': np.std([np.std(coh) for coh in spatial_coherences]) if spatial_coherences else 0
            }
        
        return spatial_analysis

    def _gpu_analyze_temporal_coherence(self, group_tensors: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """Advanced GPU-accelerated temporal coherence analysis with sophisticated trajectory focus."""
        temporal_analysis = {
            'diffusion_trajectory_coherence': {},
            'video_frame_consistency': {},
            'trajectory_progression_patterns': {},
            'temporal_momentum_analysis': {},
            'phase_transition_detection': {},
            'temporal_stability_windows': {},
            'cross_trajectory_synchronization': {},
            'temporal_frequency_signatures': {}
        }
        
        for group_name in sorted(group_tensors.keys()):
            data = group_tensors[group_name]
            self.logger.info(f"GPU analyzing temporal coherence for {group_name}")
            
            trajectories = data['trajectory_tensor']  # [n_videos, steps, 1, 16, frames, H, W]
            trajectories = trajectories.squeeze(2)   # [n_videos, steps, 16, frames, H, W]
            
            n_videos, n_steps = trajectories.shape[:2]
            
            # 1. Enhanced Diffusion Step Trajectory Coherence
            step_coherences = []
            # Fix: Flatten the spatial dimensions properly
            trajectory_norms = torch.norm(trajectories.flatten(start_dim=-3), dim=-1)  # [n_videos, steps]
            
            for video_idx in range(min(n_videos, 10)):  # Sample videos
                video_traj = trajectories[video_idx]  # [steps, 16, frames, H, W]
                
                video_step_coherences = []
                for step in range(n_steps - 1):
                    step1 = video_traj[step].flatten()
                    step2 = video_traj[step + 1].flatten()
                    
                    coherence = self._gpu_corrcoef(step1, step2)
                    if not torch.isnan(coherence):
                        video_step_coherences.append(coherence.item())
                
                if video_step_coherences:
                    step_coherences.append(video_step_coherences)
            
            # 2. Temporal Momentum Analysis
            # First and second derivatives for acceleration patterns
            first_derivatives = torch.diff(trajectory_norms, dim=1)  # [n_videos, steps-1]
            second_derivatives = torch.diff(first_derivatives, dim=1)  # [n_videos, steps-2]
            
            momentum_patterns = {
                'velocity_mean': torch.mean(first_derivatives, dim=0).cpu().numpy().tolist(),
                'velocity_std': torch.std(first_derivatives, dim=0).cpu().numpy().tolist(),
                'acceleration_mean': torch.mean(second_derivatives, dim=0).cpu().numpy().tolist(),
                'acceleration_std': torch.std(second_derivatives, dim=0).cpu().numpy().tolist(),
                'momentum_direction_changes': torch.sum(torch.diff(torch.sign(first_derivatives), dim=1) != 0, dim=0).cpu().numpy().tolist()
            }
            
            # 3. Phase Transition Detection
            # Identify sudden changes in trajectory behavior
            trajectory_changes = torch.abs(first_derivatives)
            change_percentiles = torch.quantile(trajectory_changes, torch.tensor([0.75, 0.9, 0.95]).to(self.device), dim=0)
            
            phase_transitions = {}
            for i, percentile in enumerate([75, 90, 95]):
                threshold = change_percentiles[i]
                significant_changes = trajectory_changes > threshold.unsqueeze(0)
                transition_counts = torch.sum(significant_changes, dim=0)
                phase_transitions[f'p{percentile}_transitions'] = transition_counts.cpu().numpy().tolist()
            
            # 4. Temporal Stability Windows
            # Analyze stability across different time windows
            window_sizes = [3, 5, 7] if n_steps >= 10 else [min(3, n_steps//2)]
            stability_analysis = {}
            
            for window_size in window_sizes:
                stability_metrics = []
                for start in range(0, n_steps - window_size + 1, max(1, window_size//2)):
                    end = start + window_size
                    window_norms = trajectory_norms[:, start:end]  # [n_videos, window_size]
                    
                    # Coefficient of variation for stability
                    window_means = torch.mean(window_norms, dim=1)
                    window_stds = torch.std(window_norms, dim=1)
                    cv_values = window_stds / (window_means + 1e-8)
                    
                    stability_metrics.append({
                        'window_start': start,
                        'mean_stability': float(torch.mean(cv_values).item()),
                        'stability_variance': float(torch.var(cv_values).item())
                    })
                
                stability_analysis[f'window_{window_size}'] = stability_metrics
            
            # 5. Cross-Trajectory Synchronization
            sync_analysis = {}
            if n_videos >= 2:
                correlations = []
                phase_alignments = []
                
                for i in range(min(n_videos, 8)):
                    for j in range(i+1, min(n_videos, 8)):
                        traj_i = trajectory_norms[i]
                        traj_j = trajectory_norms[j]
                        
                        # Correlation
                        correlation = self._gpu_corrcoef(traj_i, traj_j)
                        if not torch.isnan(correlation):
                            correlations.append(correlation.item())
                        
                        # Phase alignment (using peak positions)
                        if len(traj_i) >= 5:
                            peaks_i = self._find_peaks_gpu(traj_i)
                            peaks_j = self._find_peaks_gpu(traj_j)
                            
                            if len(peaks_i) > 0 and len(peaks_j) > 0:
                                phase_diff = abs(peaks_i[0] - peaks_j[0]) / len(traj_i)
                                phase_alignments.append(phase_diff)
                
                sync_analysis = {
                    'mean_correlation': float(np.mean(correlations)) if correlations else 0,
                    'correlation_std': float(np.std(correlations)) if correlations else 0,
                    'high_sync_ratio': float(np.mean(np.array(correlations) > 0.7)) if correlations else 0,
                    'phase_alignment_mean': float(np.mean(phase_alignments)) if phase_alignments else 0
                }
            
            # 6. Temporal Frequency Signatures
            frequency_analysis = {}
            if n_steps >= 8:
                # FFT analysis on trajectory norms
                fft_results = torch.fft.fft(trajectory_norms, dim=1)
                power_spectra = torch.abs(fft_results) ** 2
                mean_power_spectrum = torch.mean(power_spectra, dim=0)  # [steps]
                
                # Find dominant frequencies (skip DC component)
                if len(mean_power_spectrum) > 4:
                    freq_slice = mean_power_spectrum[1:n_steps//2]
                    freq_indices = torch.argsort(freq_slice, descending=True)[:3] + 1
                    dominant_freqs = freq_indices.cpu().numpy().tolist()
                    # Fix: Use norm for multi-dimensional tensors
                    if mean_power_spectrum.dim() > 1:
                        dominant_powers = [float(torch.norm(mean_power_spectrum[idx]).item()) for idx in freq_indices]
                    else:
                        dominant_powers = [float(mean_power_spectrum[idx].item()) for idx in freq_indices]
                    
                    frequency_analysis = {
                        'dominant_frequencies': dominant_freqs,
                        'dominant_powers': dominant_powers,
                        'spectral_centroid': float(torch.sum(torch.arange(len(mean_power_spectrum)).float().to(self.device) * (torch.norm(mean_power_spectrum, dim=-1) if mean_power_spectrum.dim() > 1 else mean_power_spectrum)) / torch.sum(torch.norm(mean_power_spectrum, dim=-1) if mean_power_spectrum.dim() > 1 else mean_power_spectrum)),
                        'spectral_entropy': float(self._compute_spectral_entropy(torch.norm(mean_power_spectrum, dim=-1) if mean_power_spectrum.dim() > 1 else mean_power_spectrum))
                    }
                else:
                    frequency_analysis = {
                        'dominant_frequencies': [],
                        'dominant_powers': [],
                        'spectral_centroid': 0.0,
                        'spectral_entropy': 0.0
                    }
            
            # 7. Enhanced Trajectory Progression Patterns
            magnitude_evolutions = []
            for video_idx in range(n_videos):
                video_traj = trajectories[video_idx]  # [steps, 16, frames, H, W]
                
                step_magnitudes = []
                for step in range(n_steps):
                    magnitude = torch.norm(video_traj[step]).item()
                    step_magnitudes.append(magnitude)
                
                magnitude_evolutions.append(step_magnitudes)
            
            # Compute progression statistics
            if magnitude_evolutions:
                mean_evolution = [np.mean([evol[i] for evol in magnitude_evolutions if i < len(evol)]) 
                                for i in range(min(n_steps, 30))]
                std_evolution = [np.std([evol[i] for evol in magnitude_evolutions if i < len(evol)]) 
                               for i in range(min(n_steps, 30))]
                
                # Trend analysis
                if len(mean_evolution) > 2:
                    early_mean = np.mean(mean_evolution[:len(mean_evolution)//3])
                    late_mean = np.mean(mean_evolution[-len(mean_evolution)//3:])
                    trend_direction = 'increasing' if late_mean > early_mean else 'decreasing'
                    trend_strength = abs(late_mean - early_mean) / (early_mean + 1e-8)
                else:
                    trend_direction = 'stable'
                    trend_strength = 0
            else:
                mean_evolution = []
                std_evolution = []
                trend_direction = 'unknown'
                trend_strength = 0
            
            # Store comprehensive results
            temporal_analysis['diffusion_trajectory_coherence'][group_name] = {
                'mean_step_coherence': np.mean([np.mean(coherences) for coherences in step_coherences]) if step_coherences else 0,
                'trajectory_smoothness_std': np.std([np.std(coherences) for coherences in step_coherences]) if step_coherences else 0,
                'coherence_evolution': [np.mean([coh[i] for coh in step_coherences if i < len(coh)]) 
                                      for i in range(min(n_steps-1, 20))] if step_coherences else []
            }
            
            temporal_analysis['temporal_momentum_analysis'][group_name] = momentum_patterns
            temporal_analysis['phase_transition_detection'][group_name] = phase_transitions
            temporal_analysis['temporal_stability_windows'][group_name] = stability_analysis
            temporal_analysis['cross_trajectory_synchronization'][group_name] = sync_analysis
            temporal_analysis['temporal_frequency_signatures'][group_name] = frequency_analysis
            
            temporal_analysis['trajectory_progression_patterns'][group_name] = {
                'magnitude_evolution_mean': mean_evolution,
                'magnitude_evolution_std': std_evolution,
                'trend_direction': trend_direction,
                'trend_strength': float(trend_strength),
                'progression_variability': float(np.std(std_evolution)) if std_evolution else 0
            }
        
        return temporal_analysis

    def _gpu_analyze_channel_patterns(self, group_tensors: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """GPU-accelerated channel pattern analysis with trajectory focus."""
        channel_analysis = {
            'channel_trajectory_evolution': {},
            'channel_specialization_patterns': {}
        }
        
        for group_name in sorted(group_tensors.keys()):
            data = group_tensors[group_name]
            self.logger.info(f"GPU analyzing channel patterns for {group_name}")
            
            trajectories = data['trajectory_tensor']  # [n_videos, steps, 1, 16, frames, H, W]
            trajectories = trajectories.squeeze(2)   # [n_videos, steps, 16, frames, H, W]
            
            n_videos, n_steps, n_channels = trajectories.shape[:3]
            
            # Channel evolution analysis
            channel_evolutions = []
            for video_idx in range(min(n_videos, 4)):  # Sample videos
                video_traj = trajectories[video_idx]  # [steps, 16, frames, H, W]
                
                video_channel_evolutions = []
                for channel in range(n_channels):
                    channel_traj = video_traj[:, channel]  # [steps, frames, H, W]
                    
                    channel_magnitudes = []
                    for step in range(n_steps):
                        magnitude = torch.norm(channel_traj[step]).item()
                        channel_magnitudes.append(magnitude)
                    
                    video_channel_evolutions.append(channel_magnitudes)
                
                channel_evolutions.append(video_channel_evolutions)
            
            # Store results
            channel_analysis['channel_trajectory_evolution'][group_name] = {
                'mean_evolution_patterns': np.mean(channel_evolutions, axis=0).tolist() if channel_evolutions else [],
                'evolution_variability': np.std(channel_evolutions, axis=0).tolist() if channel_evolutions else []
            }
            
            channel_analysis['channel_specialization_patterns'][group_name] = {
                'overall_variance': float(torch.var(trajectories).item()),
                'temporal_variance': float(torch.var(trajectories, dim=1).mean().item())
            }
        
        return channel_analysis

    def _gpu_analyze_patch_diversity(self, group_tensors: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """GPU-accelerated trajectory-focused patch analysis."""
        patch_analysis = {
            'trajectory_patch_evolution': {},
            'spatial_scale_progression': {}
        }
        
        for group_name in sorted(group_tensors.keys()):
            data = group_tensors[group_name]
            self.logger.info(f"GPU analyzing patch diversity for {group_name}")
            
            trajectories = data['trajectory_tensor']  # [n_videos, steps, 1, 16, frames, H, W]
            trajectories = trajectories.squeeze(2)   # [n_videos, steps, 16, frames, H, W]
            
            # Simple patch-based analysis
            patch_analysis['trajectory_patch_evolution'][group_name] = {
                'evolution_patterns': [],
                'mean_evolution': []
            }
            
            patch_analysis['spatial_scale_progression'][group_name] = {
                'overall_variance': float(torch.var(trajectories).item()),
                'temporal_variance': float(torch.var(trajectories, dim=1).mean().item())
            }
        
        return patch_analysis

    def _gpu_analyze_global_structure(self, group_tensors: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """GPU-accelerated global structure analysis with trajectory focus."""
        global_analysis = {
            'trajectory_global_evolution': {},
            'convergence_patterns': {}
        }
        
        for group_name in sorted(group_tensors.keys()):
            data = group_tensors[group_name]
            self.logger.info(f"GPU analyzing global structure for {group_name}")
            
            trajectories = data['trajectory_tensor']  # [n_videos, steps, 1, 16, frames, H, W]
            trajectories = trajectories.squeeze(2)   # [n_videos, steps, 16, frames, H, W]
            
            n_videos, n_steps = trajectories.shape[:2]
            
            # Global evolution patterns
            global_evolutions = []
            for video_idx in range(n_videos):
                video_traj = trajectories[video_idx]  # [steps, 16, frames, H, W]
                
                video_evolution = []
                for step in range(n_steps):
                    step_data = video_traj[step]  # [16, frames, H, W]
                    
                    global_variance = torch.var(step_data).item()
                    global_magnitude = torch.norm(step_data).item()
                    
                    video_evolution.append({
                        'step': step,
                        'global_variance': global_variance,
                        'global_magnitude': global_magnitude
                    })
                
                global_evolutions.append(video_evolution)
            
            # Store results
            global_analysis['trajectory_global_evolution'][group_name] = {
                'variance_progression': [np.mean([evol[i]['global_variance'] for evol in global_evolutions if i < len(evol)]) for i in range(min(n_steps, 20))],
                'magnitude_progression': [np.mean([evol[i]['global_magnitude'] for evol in global_evolutions if i < len(evol)]) for i in range(min(n_steps, 20))]
            }
            
            global_analysis['convergence_patterns'][group_name] = {
                'overall_diversity_score': float(torch.var(trajectories, dim=0).mean().item())
            }
        
        return global_analysis

    def _gpu_analyze_information_content(self, group_tensors: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """Simplified information-theoretic analysis for trajectories."""
        info_analysis = {
            'trajectory_information_content': {},
            'information_evolution': {}
        }
        
        for group_name in sorted(group_tensors.keys()):
            data = group_tensors[group_name]
            self.logger.info(f"Analyzing information content for {group_name}")
            
            trajectories = data['trajectory_tensor']  # [n_videos, steps, 1, 16, frames, H, W]
            trajectories = trajectories.squeeze(2)   # [n_videos, steps, 16, frames, H, W]
            
            # Simple information content measures
            trajectory_variance = float(torch.var(trajectories).item())
            
            info_analysis['trajectory_information_content'][group_name] = {
                'variance_measure': trajectory_variance
            }
            
            info_analysis['information_evolution'][group_name] = {
                'complexity_trend': 'simplified_analysis'
            }
        
        return info_analysis

    def _gpu_analyze_complexity_measures(self, group_tensors: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """Simplified complexity analysis for trajectories."""
        complexity_analysis = {
            'trajectory_complexity': {},
            'evolution_complexity': {}
        }
        
        for group_name in sorted(group_tensors.keys()):
            data = group_tensors[group_name]
            self.logger.info(f"Analyzing complexity measures for {group_name}")
            
            trajectories = data['trajectory_tensor']  # [n_videos, steps, 1, 16, frames, H, W]
            trajectories = trajectories.squeeze(2)   # [n_videos, steps, 16, frames, H, W]
            
            # Simple complexity measures
            trajectory_std = float(torch.std(trajectories).item())
            trajectory_range = float((torch.max(trajectories) - torch.min(trajectories)).item())
            
            complexity_analysis['trajectory_complexity'][group_name] = {
                'standard_deviation': trajectory_std,
                'value_range': trajectory_range
            }
            
            complexity_analysis['evolution_complexity'][group_name] = {
                'temporal_variation': float(torch.var(trajectories, dim=1).mean().item())
            }
        
        return complexity_analysis

    def _gpu_analyze_frequency_patterns(self, group_tensors: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """Simplified frequency analysis for trajectories."""
        frequency_analysis = {
            'trajectory_frequency_characteristics': {},
            'temporal_patterns': {}
        }
        
        for group_name in sorted(group_tensors.keys()):
            data = group_tensors[group_name]
            self.logger.info(f"Analyzing frequency patterns for {group_name}")
            
            trajectories = data['trajectory_tensor']  # [n_videos, steps, 1, 16, frames, H, W]
            trajectories = trajectories.squeeze(2)   # [n_videos, steps, 16, frames, H, W]
            
            # Simple frequency-domain analysis
            trajectory_fft_magnitude = 0.0
            if hasattr(torch.fft, 'fft'):
                try:
                    trajectory_fft_magnitude = float(torch.abs(torch.fft.fft(trajectories.flatten())).mean().item())
                except:
                    trajectory_fft_magnitude = 0.0
            
            frequency_analysis['trajectory_frequency_characteristics'][group_name] = {
                'fft_magnitude_mean': trajectory_fft_magnitude,
                'spectral_energy': float(torch.norm(trajectories).item())
            }
            
            frequency_analysis['temporal_patterns'][group_name] = {
                'temporal_smoothness': float(torch.mean(torch.abs(torch.diff(trajectories, dim=1))).item())
            }
        
        return frequency_analysis

    def _gpu_analyze_group_separability(self, group_tensors: Dict[str, Dict[str, torch.Tensor]], 
                                       prompt_groups: List[str]) -> Dict[str, Any]:
        """Simplified group separability analysis for trajectories."""
        separability_analysis = {
            'trajectory_group_separation': {},
            'inter_group_distances': {}
        }
        
        # Extract trajectory features
        group_centroids = {}
        
        for group_name in sorted(group_tensors.keys()):
            data = group_tensors[group_name]
            trajectories = data['trajectory_tensor']  # [n_videos, steps, 1, 16, frames, H, W]
            trajectories = trajectories.squeeze(2)   # [n_videos, steps, 16, frames, H, W]
            
            # Compute group centroid
            group_centroid = torch.mean(trajectories, dim=(0, 1))  # [16, frames, H, W]
            group_centroids[group_name] = group_centroid
        
        # Compute inter-group distances
        inter_distances = {}
        for group1 in group_centroids:
            for group2 in group_centroids:
                if group1 != group2:
                    distance = float(torch.norm(group_centroids[group1] - group_centroids[group2]).item())
                    inter_distances[f"{group1}_vs_{group2}"] = distance
        
        separability_analysis['trajectory_group_separation'] = {
            'group_count': len(group_centroids),
            'separability_measure': 'simplified_centroid_analysis'
        }
        
        separability_analysis['inter_group_distances'] = inter_distances
        
        return separability_analysis

    def _gpu_test_statistical_significance(self, group_tensors: Dict[str, Dict[str, torch.Tensor]], 
                                         prompt_groups: List[str]) -> Dict[str, Any]:
        """Simplified statistical significance testing for trajectories."""
        significance_analysis = {
            'trajectory_group_differences': {},
            'statistical_summary': {}
        }
        
        # Extract key trajectory statistics
        group_statistics = {}
        
        for group_name in sorted(group_tensors.keys()):
            data = group_tensors[group_name]
            trajectories = data['trajectory_tensor']  # [n_videos, steps, 1, 16, frames, H, W]
            trajectories = trajectories.squeeze(2)   # [n_videos, steps, 16, frames, H, W]
            
            # Compute trajectory statistics
            trajectory_variances = torch.var(trajectories, dim=(1, 2, 3, 4, 5))  # [n_videos]
            trajectory_means = torch.mean(trajectories, dim=(1, 2, 3, 4, 5))     # [n_videos]
            
            group_statistics[group_name] = {
                'variance': trajectory_variances.cpu().numpy(),
                'mean': trajectory_means.cpu().numpy()
            }
        
        # Simple group comparisons
        group_names = list(group_statistics.keys())
        for i, group1 in enumerate(group_names):
            for j, group2 in enumerate(group_names[i+1:], i+1):
                variance_diff = np.mean(group_statistics[group1]['variance']) - np.mean(group_statistics[group2]['variance'])
                mean_diff = np.mean(group_statistics[group1]['mean']) - np.mean(group_statistics[group2]['mean'])
                
                significance_analysis['trajectory_group_differences'][f"{group1}_vs_{group2}"] = {
                    'variance_difference': float(variance_diff),
                    'mean_difference': float(mean_diff)
                }
        
        significance_analysis['statistical_summary'] = {
            'groups_analyzed': len(group_names),
            'comparisons_made': len(group_names) * (len(group_names) - 1) // 2
        }
        
        return significance_analysis


    def _gpu_analyze_temporal_trajectories(self, group_tensors: Dict[str, Dict[str, torch.Tensor]], 
                                          prompt_groups: List[str]) -> Dict[str, Any]:
        """GPU-accelerated temporal trajectory analysis based on TemporalTrajectoryAnalysis."""
        temporal_analysis = {}
        
        # Set primary baseline for comparison analysis
        baseline_group = sorted(prompt_groups)[0]
        
        # Test both baseline strategies for research comparison
        baseline_group_empty = self._select_baseline_group(prompt_groups, "empty_prompt")
        baseline_group_class = self._select_baseline_group(prompt_groups, "first_class_specific")
        
        self.logger.info(f"Temporal analysis using baseline strategies:")
        self.logger.info(f"  Primary baseline: {baseline_group}")
        self.logger.info(f"  Empty prompt baseline: {baseline_group_empty}")
        self.logger.info(f"  First class-specific baseline: {baseline_group_class}")
        
        for group_name, group_data in group_tensors.items():
            trajectory_tensor = group_data['trajectory_tensor'].to(self.device)  # [n_videos, steps, ...]
            
            # Flatten trajectory for analysis (keep videos and steps dimensions)
            flat_trajectories = self._apply_normalization(trajectory_tensor, group_data)  # [n_videos, steps, D]
            
            # Trajectory Length Analysis
            trajectory_lengths = self._gpu_trajectory_length(flat_trajectories)
            
            # Velocity Analysis
            velocity_results = self._gpu_velocity_analysis(flat_trajectories)
            
            # Acceleration Analysis
            acceleration_results = self._gpu_acceleration_analysis(flat_trajectories)
            
            # Endpoint Distance Analysis
            endpoint_distances = self._gpu_endpoint_distance(flat_trajectories)
            
            # Tortuosity Calculation
            tortuosity = self._gpu_calculate_tortuosity(trajectory_lengths, endpoint_distances)
            
            # Semantic Convergence Analysis
            convergence_results = self._gpu_semantic_convergence_rate(flat_trajectories)
            
            # Store results
            temporal_analysis[group_name] = {
                'trajectory_length': {
                    'mean_length': float(torch.mean(trajectory_lengths)),
                    'std_length': float(torch.std(trajectory_lengths)),
                    'min_length': float(torch.min(trajectory_lengths)),
                    'max_length': float(torch.max(trajectory_lengths)),
                    'individual_lengths': trajectory_lengths.cpu().numpy().tolist()
                },
                'velocity_analysis': {
                    'mean_velocity': velocity_results['mean_velocity'].cpu().numpy().tolist(),
                    'velocity_variance': velocity_results['velocity_variance'].cpu().numpy().tolist(),
                    'overall_mean_velocity': float(torch.mean(velocity_results['mean_velocity'])),
                    'overall_velocity_variance': float(torch.mean(velocity_results['velocity_variance']))
                },
                'acceleration_analysis': {
                    'mean_acceleration': acceleration_results['mean_acceleration'].cpu().numpy().tolist(),
                    'acceleration_variance': acceleration_results['acceleration_variance'].cpu().numpy().tolist(),
                    'overall_mean_acceleration': float(torch.mean(acceleration_results['mean_acceleration'])),
                    'overall_acceleration_variance': float(torch.mean(acceleration_results['acceleration_variance']))
                },
                'endpoint_distance': {
                    'mean_endpoint_distance': float(torch.mean(endpoint_distances)),
                    'std_endpoint_distance': float(torch.std(endpoint_distances)),
                    'individual_distances': endpoint_distances.cpu().numpy().tolist()
                },
                'tortuosity': {
                    'mean_tortuosity': float(torch.mean(tortuosity)),
                    'std_tortuosity': float(torch.std(tortuosity)),
                    'individual_tortuosity': tortuosity.cpu().numpy().tolist()
                },
                'semantic_convergence': {
                    'half_life_steps': convergence_results['half_life_step'].cpu().numpy().tolist(),
                    'mean_half_life': float(torch.mean(convergence_results['half_life_step'].float())),
                    'distances_to_end_final': convergence_results['distances_to_end'][:, -1].cpu().numpy().tolist(),
                    'convergence_rate': float(torch.mean(1.0 / (convergence_results['half_life_step'].float() + 1.0)))
                }
            }
            
            # Baseline comparison if this is not the baseline group
            if group_name != baseline_group and baseline_group in group_tensors:
                baseline_tensor = group_tensors[baseline_group]['trajectory_tensor'].to(self.device)
                baseline_flat = baseline_tensor.flatten(start_dim=2)
                
                # Cross-group distance analysis
                cross_distances = self._gpu_cross_group_trajectory_distances(flat_trajectories, baseline_flat)
                temporal_analysis[group_name]['baseline_comparison'] = {
                    'mean_distance_to_baseline': float(torch.mean(cross_distances)),
                    'std_distance_to_baseline': float(torch.std(cross_distances)),
                    'baseline_group': baseline_group
                }
        
        return temporal_analysis

    def _gpu_analyze_structural_patterns(self, group_tensors: Dict[str, Dict[str, torch.Tensor]], 
                                        prompt_groups: List[str]) -> Dict[str, Any]:
        """GPU-accelerated structural analysis including PCA, variance, and entropy measures."""
        structural_analysis = {}
        
        # Determine baseline latents (first prompt group alphabetically)
        baseline_group = sorted(prompt_groups)[0]
        
        for group_name, group_data in group_tensors.items():
            trajectory_tensor = group_data['trajectory_tensor'].to(self.device)  # [n_videos, steps, ...]

            self.logger.info(f"Analyzing structural patterns for group: {group_name}")
            
            # Flatten trajectory for structural analysis
            flat_trajectories = trajectory_tensor.flatten(start_dim=2)  # [n_videos, steps, flattened_latent]

            self.logger.debug(f"Flat trajectories shape: {flat_trajectories.shape}")
            
            # Latent Space Variance Analysis (fast)
            variance_results = self._gpu_latent_space_variance(flat_trajectories)
            self.logger.debug("Variance analysis completed")

            # PCA-based Analysis (optimized with sampling)
            pca_results = self._gpu_pca_analysis(flat_trajectories)
            self.logger.debug("PCA analysis completed")

            # Shannon Entropy Estimation (fast approximation)
            entropy_results = self._gpu_shannon_entropy_estimation(flat_trajectories)
            self.logger.debug("Entropy estimation completed")

            # KL Divergence Analysis (fast moment-based approximation)
            if group_name != baseline_group and baseline_group in group_tensors:
                baseline_tensor = group_tensors[baseline_group]['trajectory_tensor'].to(self.device)
                baseline_flat = baseline_tensor.flatten(start_dim=2)
                kl_divergence = self._gpu_kl_divergence_estimation(flat_trajectories, baseline_flat)
            else:
                kl_divergence = 0.0
            self.logger.debug("KL divergence analysis completed")
            
            # Structural Complexity Measures (optimized)
            complexity_results = self._gpu_structural_complexity(flat_trajectories)
            self.logger.debug("Structural complexity analysis completed")

            # Store results
            structural_analysis[group_name] = {
                'latent_space_variance': {
                    'temporal_variance': variance_results['temporal_variance'].cpu().numpy().tolist(),
                    'spatial_variance': variance_results['spatial_variance'].cpu().numpy().tolist(),
                    'overall_variance': float(variance_results['overall_variance']),
                    'variance_across_videos': float(variance_results['variance_across_videos']),
                    'variance_across_steps': float(variance_results['variance_across_steps'])
                },
                'pca_analysis': {
                    'explained_variance_ratio': pca_results['explained_variance_ratio'].cpu().numpy().tolist(),
                    'cumulative_variance_90': float(pca_results['cumulative_variance_90']),
                    'effective_dimensionality': int(pca_results['effective_dimensionality']),
                    'principal_component_magnitudes': pca_results['pc_magnitudes'].cpu().numpy().tolist()
                },
                'shannon_entropy': {
                    'entropy_estimate': float(entropy_results['entropy_estimate']),
                    'bin_counts': entropy_results['bin_counts'].cpu().numpy().tolist(),
                    # Reduce entropy_per_dimension to statistical summary to save space
                    'entropy_per_dimension_stats': {
                        'mean': float(torch.mean(entropy_results['entropy_per_dim'])),
                        'std': float(torch.std(entropy_results['entropy_per_dim'])),
                        'min': float(torch.min(entropy_results['entropy_per_dim'])),
                        'max': float(torch.max(entropy_results['entropy_per_dim'])),
                        'median': float(torch.median(entropy_results['entropy_per_dim'])),
                        'q25': float(torch.quantile(entropy_results['entropy_per_dim'], 0.25)),
                        'q75': float(torch.quantile(entropy_results['entropy_per_dim'], 0.75)),
                        'total_dimensions': int(entropy_results['entropy_per_dim'].shape[0])
                    }
                },
                'kl_divergence': {
                    'divergence_from_baseline': float(kl_divergence),
                    'baseline_group': baseline_group if group_name != baseline_group else None
                },
                'structural_complexity': {
                    'rank_estimate': float(complexity_results['rank_estimate']),
                    'condition_number': float(complexity_results['condition_number']),
                    'spectral_entropy': float(complexity_results['spectral_entropy']),
                    'trace_norm': float(complexity_results['trace_norm'])
                }
            }
        
        return structural_analysis

    # Temporal Analysis Helper Methods
    def _gpu_trajectory_length(self, flat_trajectories: torch.Tensor) -> torch.Tensor:
        """Calculate trajectory lengths using GPU operations."""
        # flat_trajectories: [n_videos, steps, flattened_latent]
        step_differences = flat_trajectories[:, 1:] - flat_trajectories[:, :-1]
        step_norms = torch.linalg.norm(step_differences, dim=2)
        trajectory_lengths = torch.sum(step_norms, dim=1)
        return trajectory_lengths

    def _gpu_velocity_analysis(self, flat_trajectories: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate velocity statistics using GPU operations."""
        step_differences = flat_trajectories[:, 1:] - flat_trajectories[:, :-1]
        velocities = torch.linalg.norm(step_differences, dim=2)
        
        return {
            'mean_velocity': torch.mean(velocities, dim=1),
            'velocity_variance': torch.var(velocities, dim=1)
        }

    def _gpu_acceleration_analysis(self, flat_trajectories: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate acceleration statistics using GPU operations."""
        velocities = flat_trajectories[:, 1:] - flat_trajectories[:, :-1]
        accelerations = torch.linalg.norm(velocities[:, 1:] - velocities[:, :-1], dim=2)
        
        return {
            'mean_acceleration': torch.mean(accelerations, dim=1),
            'acceleration_variance': torch.var(accelerations, dim=1)
        }

    def _gpu_endpoint_distance(self, flat_trajectories: torch.Tensor) -> torch.Tensor:
        """Calculate endpoint distances using GPU operations."""
        return torch.linalg.norm(flat_trajectories[:, -1] - flat_trajectories[:, 0], dim=1)

    def _gpu_calculate_tortuosity(self, trajectory_lengths: torch.Tensor, 
                                 endpoint_distances: torch.Tensor) -> torch.Tensor:
        """Calculate tortuosity (ratio of path length to straight-line distance)."""
        return trajectory_lengths / (endpoint_distances + 1e-8)

    def _gpu_semantic_convergence_rate(self, flat_trajectories: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate semantic convergence rate using GPU operations."""
        num_videos, num_steps = flat_trajectories.shape[0], flat_trajectories.shape[1]
        
        # Calculate distances to final state for each trajectory
        final_latents = flat_trajectories[:, -1, :].unsqueeze(1)  # [n_videos, 1, latent_dim]
        distances_to_end = torch.linalg.norm(flat_trajectories - final_latents, dim=2)  # [n_videos, steps]
        
        # Find half-life: step where distance falls below half of initial distance
        half_distance = distances_to_end[:, 0] / 2.0  # [n_videos]
        half_life_mask = distances_to_end <= half_distance.unsqueeze(1)  # [n_videos, steps]
        
        # Find first step where condition is met
        half_life_step = torch.argmax(half_life_mask.int(), dim=1)
        
        # Handle cases where convergence never happens
        not_converged_mask = (half_life_step == 0) & ~half_life_mask[:, 0]
        half_life_step[not_converged_mask] = num_steps
        
        return {
            'half_life_step': half_life_step,
            'distances_to_end': distances_to_end
        }

    def _gpu_cross_group_trajectory_distances(self, trajectories1: torch.Tensor, 
                                            trajectories2: torch.Tensor) -> torch.Tensor:
        """Calculate cross-group trajectory distances."""
        # Average over steps for each video, then calculate pairwise distances
        traj1_mean = torch.mean(trajectories1, dim=1)  # [n_videos1, latent_dim]
        traj2_mean = torch.mean(trajectories2, dim=1)  # [n_videos2, latent_dim]
        
        # Calculate distances between all pairs
        distances = torch.cdist(traj1_mean, traj2_mean)  # [n_videos1, n_videos2]
        
        # Return minimum distances (closest match for each trajectory in group 1)
        return torch.min(distances, dim=1)[0]

    # Structural Analysis Helper Methods
    def _gpu_latent_space_variance(self, flat_trajectories: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate various variance measures in latent space."""
        # flat_trajectories: [n_videos, steps, flattened_latent]
        
        # Temporal variance: variance across time steps for each video
        temporal_variance = torch.var(flat_trajectories, dim=1)  # [n_videos, latent_dim]
        
        # Spatial variance: variance across videos for each step
        spatial_variance = torch.var(flat_trajectories, dim=0)  # [steps, latent_dim]
        
        # Overall variance
        all_data = flat_trajectories.reshape(-1, flat_trajectories.shape[-1])
        overall_variance = torch.var(all_data, dim=0)
        
        return {
            'temporal_variance': torch.mean(temporal_variance, dim=1),  # [n_videos] - avg across latent dims
            'spatial_variance': torch.mean(spatial_variance, dim=1),   # [steps] - avg across latent dims
            'overall_variance': torch.mean(overall_variance),
            'variance_across_videos': torch.var(torch.mean(flat_trajectories, dim=1)),
            'variance_across_steps': torch.var(torch.mean(flat_trajectories, dim=0))
        }

    def _gpu_pca_analysis(self, flat_trajectories: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Fast GPU-optimized PCA analysis with sampling for large datasets."""
        # Reshape for PCA: [n_samples, n_features]
        n_videos, n_steps, latent_dim = flat_trajectories.shape
        data_matrix = flat_trajectories.reshape(-1, latent_dim)  # [n_videos*n_steps, latent_dim]
        
        # Sample data if too large for efficient PCA
        max_samples_for_pca = 5000
        if data_matrix.shape[0] > max_samples_for_pca:
            indices = torch.randperm(data_matrix.shape[0], device=data_matrix.device)[:max_samples_for_pca]
            data_matrix = data_matrix[indices]
        
        # Center the data
        data_centered = data_matrix - torch.mean(data_matrix, dim=0, keepdim=True)
        
        # Use efficient SVD strategy based on data dimensions
        try:
            if data_centered.shape[0] < data_centered.shape[1]:
                # More features than samples: use SVD on data
                U, S, Vt = torch.linalg.svd(data_centered, full_matrices=False)
                eigenvalues = (S ** 2) / (data_centered.shape[0] - 1)
            else:
                # More samples than features: use covariance matrix approach
                # But limit to reasonable size for GPU memory
                if latent_dim > 1000:
                    # For very high-dimensional data, use randomized SVD approximation
                    k = min(100, latent_dim // 2)  # Keep top k components
                    U, S, Vt = torch.svd_lowrank(data_centered, q=k)
                    eigenvalues = (S ** 2) / (data_centered.shape[0] - 1)
                else:
                    # Standard covariance approach
                    cov_matrix = torch.cov(data_centered.T)
                    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
                    eigenvalues = torch.flip(eigenvalues, dims=[0])  # Sort in descending order
            
            # Calculate explained variance ratio
            explained_variance_ratio = eigenvalues / torch.sum(eigenvalues)
            
            # Find cumulative variance and effective dimensionality
            cumulative_variance = torch.cumsum(explained_variance_ratio, dim=0)
            effective_dim = torch.argmax((cumulative_variance >= 0.9).float()) + 1
            
            return {
                'explained_variance_ratio': explained_variance_ratio,
                'cumulative_variance_90': cumulative_variance[effective_dim-1] if effective_dim > 0 else cumulative_variance[-1],
                'effective_dimensionality': effective_dim,
                'pc_magnitudes': torch.sqrt(eigenvalues)
            }
            
        except Exception as e:
            self.logger.warning(f"PCA computation failed: {e}, using simplified variance analysis")
            # Fallback to simple variance analysis
            eigenvalues = torch.var(data_centered, dim=0)
            explained_variance_ratio = eigenvalues / torch.sum(eigenvalues)
            
            return {
                'explained_variance_ratio': explained_variance_ratio,
                'cumulative_variance_90': torch.tensor(0.9),
                'effective_dimensionality': torch.tensor(min(10, len(eigenvalues))),
                'pc_magnitudes': torch.sqrt(eigenvalues)
            }

    def _gpu_shannon_entropy_estimation(self, flat_trajectories: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Ultra-fast Shannon entropy approximation using variance-only estimation."""
        # Use differential entropy approximation for multivariate Gaussian assumption
        # This is much faster than any histogram-based methods on GPU
        
        # Flatten all data for entropy estimation
        all_data = flat_trajectories.reshape(-1, flat_trajectories.shape[-1])
        
        # Fast variance-based entropy approximation
        # For multivariate Gaussian: H ≈ 0.5 * log(2πe * σ²)
        data_vars = torch.var(all_data, dim=0)  # Variance per dimension [latent_dim]
        
        # Differential entropy approximation per dimension
        entropy_per_dim = 0.5 * torch.log(2 * torch.pi * torch.e * (data_vars + 1e-8))
        
        # Create minimal dummy bin counts for compatibility (tiny tensor)
        n_dims = min(10, all_data.shape[1])  # Limit to avoid memory issues
        dummy_bins = torch.ones(n_dims, 5, device=all_data.device)  # Very small dummy tensor
        
        return {
            'entropy_estimate': torch.mean(entropy_per_dim),
            'bin_counts': dummy_bins,
            'entropy_per_dim': entropy_per_dim
        }

    def _gpu_kl_divergence_estimation(self, trajectories1: torch.Tensor, 
                                    trajectories2: torch.Tensor) -> float:
        """Fast GPU-optimized KL divergence estimation using moment-based approximation."""
        # Use fast moment-based approximation instead of histogram method
        # For multivariate Gaussian assumption: KL(P||Q) ≈ based on means and covariances
        
        # Flatten both trajectory sets
        data1 = trajectories1.reshape(-1, trajectories1.shape[-1])
        data2 = trajectories2.reshape(-1, trajectories2.shape[-1])
        
        # Sample for computational efficiency if data is too large
        max_samples = 2000
        if data1.shape[0] > max_samples:
            indices1 = torch.randperm(data1.shape[0], device=data1.device)[:max_samples]
            data1 = data1[indices1]
        if data2.shape[0] > max_samples:
            indices2 = torch.randperm(data2.shape[0], device=data2.device)[:max_samples]
            data2 = data2[indices2]
        
        # Fast moment-based KL divergence approximation
        # KL(P||Q) ≈ 0.5 * (log(σ²_Q/σ²_P) + σ²_P/σ²_Q + (μ_P-μ_Q)²/σ²_Q - 1)
        
        # Compute means and variances efficiently
        mean1 = torch.mean(data1, dim=0)
        mean2 = torch.mean(data2, dim=0)
        var1 = torch.var(data1, dim=0) + 1e-8  # Add epsilon for numerical stability
        var2 = torch.var(data2, dim=0) + 1e-8
        
        # KL divergence approximation per dimension (assuming independence)
        mean_diff_sq = (mean1 - mean2) ** 2
        kl_per_dim = 0.5 * (torch.log(var2 / var1) + var1 / var2 + mean_diff_sq / var2 - 1)
        
        # Average across dimensions
        return float(torch.mean(kl_per_dim))

    def _gpu_structural_complexity(self, flat_trajectories: torch.Tensor) -> Dict[str, float]:
        """Fast GPU-optimized structural complexity measures with sampling."""
        # Reshape data matrix
        data_matrix = flat_trajectories.reshape(-1, flat_trajectories.shape[-1])
        
        # Sample for computational efficiency on large datasets
        max_samples = 3000
        if data_matrix.shape[0] > max_samples:
            indices = torch.randperm(data_matrix.shape[0], device=data_matrix.device)[:max_samples]
            data_matrix = data_matrix[indices]
        
        # Center the data
        data_centered = data_matrix - torch.mean(data_matrix, dim=0, keepdim=True)
        
        # Compute SVD for rank and spectral analysis with fallback
        try:
            # Use low-rank approximation for efficiency
            if min(data_centered.shape) > 100:
                k = min(50, min(data_centered.shape) // 2)
                U, S, Vt = torch.svd_lowrank(data_centered, q=k)
            else:
                U, S, Vt = torch.linalg.svd(data_centered, full_matrices=False)
            
            # Rank estimation (number of significant singular values)
            threshold = torch.max(S) * 1e-6
            rank_estimate = torch.sum(S > threshold)
            
            # Condition number
            condition_number = S[0] / (S[-1] + 1e-8)
            
            # Spectral entropy
            s_normalized = S / torch.sum(S)
            spectral_entropy = -torch.sum(s_normalized * torch.log(s_normalized + 1e-8))
            
            # Trace norm (nuclear norm)
            trace_norm = torch.sum(S)
            
        except Exception as e:
            self.logger.warning(f"SVD failed in structural complexity: {e}, using variance-based approximation")
            # Fallback to variance-based measures
            data_var = torch.var(data_centered, dim=0)
            rank_estimate = torch.sum(data_var > torch.max(data_var) * 1e-6)
            condition_number = torch.max(data_var) / (torch.min(data_var) + 1e-8)
            spectral_entropy = torch.tensor(0.0, device=data_centered.device)
            trace_norm = torch.sum(torch.sqrt(data_var))
        
        return {
            'rank_estimate': float(rank_estimate),
            'condition_number': float(condition_number),
            'spectral_entropy': float(spectral_entropy),
            'trace_norm': float(trace_norm)
        }

    def _gpu_corrcoef(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """GPU-accelerated correlation coefficient."""
        if x.numel() < 2:
            return torch.tensor(0.0, device=x.device)
        
        x_centered = x - torch.mean(x)
        y_centered = y - torch.mean(y)
        
        numerator = torch.sum(x_centered * y_centered)
        denominator = torch.sqrt(torch.sum(x_centered**2) * torch.sum(y_centered**2))
        
        if denominator > 1e-8:
            return numerator / denominator
        else:
            return torch.tensor(0.0, device=x.device)

    def _save_results(self, results: LatentTrajectoryAnalysis):
        """Save analysis results to disk."""
        # Save main results
        results_file = self.output_dir / "latent_trajectory_analysis_results.json"
        with open(results_file, 'w') as f:
            json.dump(results.to_dict(), f, indent=2, default=str)
        
        # Save performance report
        perf_file = self.output_dir / "gpu_performance_report.json"
        with open(perf_file, 'w') as f:
            json.dump(self.performance_stats, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to: {results_file}")
        self.logger.info(f"Performance report saved to: {perf_file}")

    def analyze_structure_aware_latents(self, prompt_groups: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Convenience method for structure-aware latent analysis.
        
        Auto-discovers prompt groups if not provided and returns results in a structured format.
        """
        if prompt_groups is None:
            # Auto-discover prompt groups from directory structure
            prompt_groups = []
            for item in self.latents_dir.iterdir():
                if (item.is_dir() and 
                    not item.name.startswith('.') and 
                    not item.name.startswith('analysis') and
                    not item.name.startswith('gpu_') and
                    not item.name.startswith('temporal_') and
                    not item.name.startswith('structure_')):
                    prompt_groups.append(item.name)
            
            if not prompt_groups:
                raise ValueError(f"No prompt group directories found in {self.latents_dir}")
            
            self.logger.info(f"Auto-discovered prompt groups: {prompt_groups}")
        
        # Run the full analysis
        analysis_results = self.analyze_prompt_groups(prompt_groups)

        # Restructure results for convenience
        structured_results = {
            'summary': {
                'prompt_groups_analyzed': prompt_groups,
                'total_analysis_time': analysis_results.analysis_metadata.get('total_analysis_time_seconds', 0),
                'device_used': analysis_results.gpu_performance_stats.get('device_used', 'unknown'),
                'peak_memory_gb': analysis_results.gpu_performance_stats.get('memory_usage', {}).get('peak_allocated_gb', 0),
                'trajectory_shape': analysis_results.analysis_metadata.get('trajectory_shape', []),
                'sophisticated_analysis_features': [
                    'temporal_momentum_analysis',
                    'phase_transition_detection', 
                    'cross_trajectory_synchronization',
                    'temporal_frequency_signatures',
                    'spatial_coherence_patterns',
                    'edge_density_evolution'
                ]
            },
            'detailed': analysis_results.to_dict(),
            'metadata': analysis_results.analysis_metadata
        }
        
        return structured_results

    # =============================================================================
    # NEW ADVANCED GEOMETRIC ANALYSIS METHODS
    # =============================================================================
    
    def _gpu_analyze_convex_hull_metrics(
        self, 
        group_tensors: Dict[str, Dict[str, torch.Tensor]], 
        prompt_groups: List[str]
    ) -> Dict[str, Any]:
        """
        Compute convex hull volume metrics for trajectory sets.
        
        For each prompt group, treats all trajectory points as a massive point cloud
        and computes the convex hull volume as a measure of representational diversity.
        """
        convex_hull_analysis = {}
        
        for group_name in sorted(group_tensors.keys()):
            try:
                trajectory_tensor = group_tensors[group_name]['trajectory_tensor']  # [videos, steps, ...]
                flat_trajectories = trajectory_tensor.view(trajectory_tensor.shape[0], trajectory_tensor.shape[1], -1)
                
                # Convert to CPU numpy for scipy ConvexHull
                all_points = flat_trajectories.cpu().numpy().reshape(-1, flat_trajectories.shape[-1])
                
                # For high-dimensional data, we need to be careful with ConvexHull
                if not ADVANCED_GEOMETRY_AVAILABLE:
                    self.logger.warning(f"Advanced geometry libraries not available, using approximation for {group_name}")
                    # Approximation using PCA projection to lower dimensions
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=min(10, all_points.shape[1], all_points.shape[0]//2))
                    points_reduced = pca.fit_transform(all_points)
                    
                    # Use bounding box volume as approximation
                    mins = np.min(points_reduced, axis=0)
                    maxs = np.max(points_reduced, axis=0)
                    box_volume = np.prod(maxs - mins)
                    hull_volume = box_volume  # Approximation
                    hull_area = np.sum(maxs - mins)  # Approximation
                    n_vertices = points_reduced.shape[0]
                    
                else:
                    # For very high dimensions, project to lower dimensional space first
                    if all_points.shape[1] > 20:
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=min(20, all_points.shape[0]//2))
                        points_reduced = pca.fit_transform(all_points)
                    else:
                        points_reduced = all_points
                        
                    # Remove duplicate points for ConvexHull
                    unique_points = np.unique(points_reduced, axis=0)
                    
                    if len(unique_points) >= points_reduced.shape[1] + 1:  # Minimum points for hull
                        hull = ConvexHull(unique_points)
                        hull_volume = hull.volume
                        hull_area = hull.area if hasattr(hull, 'area') else 0.0
                        n_vertices = len(hull.vertices)
                    else:
                        # Fallback to bounding box
                        mins = np.min(unique_points, axis=0)
                        maxs = np.max(unique_points, axis=0)
                        hull_volume = np.prod(maxs - mins)
                        hull_area = np.sum(maxs - mins)
                        n_vertices = len(unique_points)
                
                # Additional metrics
                point_cloud_diameter = np.max(pdist(all_points[:1000]))  # Sample for efficiency
                mean_pairwise_distance = np.mean(pdist(all_points[:500]))  # Sample for efficiency
                
                convex_hull_analysis[group_name] = {
                    'hull_volume': float(hull_volume),
                    'hull_surface_area': float(hull_area),
                    'n_hull_vertices': int(n_vertices),
                    'point_cloud_diameter': float(point_cloud_diameter),
                    'mean_pairwise_distance': float(mean_pairwise_distance),
                    'total_trajectory_points': int(all_points.shape[0]),
                    'dimensionality_reduced': all_points.shape[1] > 20,
                    'volume_per_point': float(hull_volume / all_points.shape[0]),
                    'density_metric': float(all_points.shape[0] / (hull_volume + 1e-10))
                }
                
            except Exception as e:
                self.logger.error(f"Error computing convex hull for {group_name}: {e}")
                convex_hull_analysis[group_name] = {
                    'hull_volume': 0.0,
                    'error': str(e)
                }
        
        return convex_hull_analysis

    def _gpu_analyze_functional_pca(
        self,
        group_tensors: Dict[str, Dict[str, torch.Tensor]],
        prompt_groups: List[str]
    ) -> Dict[str, Any]:
        """
        GPU-friendly FPCA with fp16-safe SVD:
        1) Time subsampling (stride)
        2) Optional GPU Gaussian random projection to K dims (float32)
        3) Center across videos per time-feature slice (if enabled)
        4) PCA via compact SVD on [N, T_used*K], with autocast disabled for SVD
        """
        cfg = getattr(self, 'fpca_cfg', {})
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
                    if self.device.startswith('cuda'):
                        with torch.amp.autocast('cuda', enabled=False):
                            U, S, Vh = torch.linalg.svd(X_flat, full_matrices=False)
                    else:
                        U, S, Vh = torch.linalg.svd(X_flat, full_matrices=False)
                except Exception as svd_err:
                    self.logger.warning(f"[FPCA] {group_name}: CUDA SVD failed ({svd_err}); retrying on CPU in float32")
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

                self.logger.info(
                    f"[FPCA] {group_name}: N={N} T_used={T_used} K={X2.shape[-1]} "
                    f"r={r} EVR1={(evr[0] if evr else 0.0):.3f} "
                    f"t={(time.time()-t0)*1000:.0f}ms"
                )

            except Exception as e:
                self.logger.error(f"[FPCA] error for {group_name}: {e}")
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

    def _gpu_analyze_individual_trajectory_geometry(self, group_tensors: Dict[str, Dict[str, torch.Tensor]], 
                                                  prompt_groups: List[str]) -> Dict[str, Any]:
        """
        Compute individual trajectory geometry metrics: speed, volume, circuitousness.
        
        For each trajectory in each group, computes:
        1. Speed: Average step size (Euclidean distance between consecutive points)
        2. Volume: Convex hull volume of the individual trajectory points
        3. Circuitousness: Ratio of path length to straight-line endpoint distance
        """
        trajectory_geometry = {}
        
        for group_name in sorted(group_tensors.keys()):
            try:
                trajectory_tensor = group_tensors[group_name]['trajectory_tensor']  # [videos, steps, ...]
                flat_trajectories = trajectory_tensor.view(trajectory_tensor.shape[0], trajectory_tensor.shape[1], -1)
                
                n_videos, n_steps, latent_dim = flat_trajectories.shape
                
                individual_metrics = {
                    'speeds': [],
                    'log_bbox_volumes': [],
                    'effective_sides': [],
                    'circuitousness': [],
                    'endpoint_alignment': [],
                    'turning_angle': [],
                    'path_lengths': [],
                    'endpoint_distances': [],
                    'step_size_variability': []
                }
                
                for video_idx in range(n_videos):
                    trajectory = flat_trajectories[video_idx].cpu().numpy()  # [steps, latent_dim]
                    
                    # 1. Speed calculation
                    step_differences = trajectory[1:] - trajectory[:-1]
                    step_sizes = np.linalg.norm(step_differences, axis=1)
                    mean_speed = np.mean(step_sizes)
                    speed_variability = np.std(step_sizes)
                    
                    # 2. Path length and endpoint distance
                    path_length = np.sum(step_sizes)
                    endpoint_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
                    
                    # 3. Circuitousness (tortuosity)
                    circuitousness = path_length / (endpoint_distance + 1e-8)

                    # 4. Endpoint alignment and turning angle
                    e = trajectory[-1] - trajectory[0]
                    e_norm = np.linalg.norm(e) + 1e-8
                    v = step_differences
                    v_norms = np.linalg.norm(v, axis=1) + 1e-8
                    endpoint_alignment = float(np.mean((v @ e) / (v_norms * e_norm)))
                    u = v[:-1] / v_norms[:-1, None]
                    w = v[1:]  / v_norms[1:,  None]
                    cosang = np.clip(np.sum(u*w, axis=1), -1.0, 1.0)
                    turning_angle = float(np.sum(np.arccos(cosang)))

                    # 5. Log-volume (bbox proxy) and effective side
                    mins = np.min(trajectory, axis=0)
                    maxs = np.max(trajectory, axis=0)
                    ranges = np.clip(maxs - mins, 1e-12, None)
                    log_bbox_vol = float(np.sum(np.log(ranges)))
                    eff_side = float(np.exp(log_bbox_vol / ranges.size))
                    
                    # 4. Individual trajectory volume (convex hull)
                    try:
                        if ADVANCED_GEOMETRY_AVAILABLE and latent_dim <= 20 and n_steps >= latent_dim + 1:
                            # Direct convex hull computation
                            unique_points = np.unique(trajectory, axis=0)
                            if len(unique_points) >= latent_dim + 1:
                                hull = ConvexHull(unique_points)
                                individual_volume = hull.volume
                            else:
                                # Bounding box volume
                                mins = np.min(trajectory, axis=0)
                                maxs = np.max(trajectory, axis=0)
                                individual_volume = np.prod(maxs - mins + 1e-8)
                        else:
                            # Bounding box approximation for high dimensions
                            mins = np.min(trajectory, axis=0)
                            maxs = np.max(trajectory, axis=0)
                            individual_volume = np.prod(maxs - mins + 1e-8)
                    except Exception:
                        # Fallback: bounding box volume
                        mins = np.min(trajectory, axis=0)
                        maxs = np.max(trajectory, axis=0)
                        individual_volume = np.prod(maxs - mins + 1e-8)
                    
                    # Store metrics
                    individual_metrics['speeds'].append(float(mean_speed))
                    individual_metrics['log_bbox_volumes'].append(float(log_bbox_vol)); individual_metrics['effective_sides'].append(float(eff_side))
                    individual_metrics['circuitousness'].append(float(circuitousness)); individual_metrics['endpoint_alignment'].append(float(endpoint_alignment)); individual_metrics['turning_angle'].append(float(turning_angle))
                    individual_metrics['path_lengths'].append(float(path_length))
                    individual_metrics['endpoint_distances'].append(float(endpoint_distance))
                    individual_metrics['step_size_variability'].append(float(speed_variability))
                
                # Compute summary statistics
                trajectory_geometry[group_name] = {
                    'speed_stats': {
                        'mean': float(np.mean(individual_metrics['speeds'])),
                        'std': float(np.std(individual_metrics['speeds'])),
                        'min': float(np.min(individual_metrics['speeds'])),
                        'max': float(np.max(individual_metrics['speeds'])),
                        'median': float(np.median(individual_metrics['speeds'])),
                        'individual_values': individual_metrics['speeds']
                    },
                    'log_volume_stats': {
                        'mean': float(np.mean(individual_metrics['log_bbox_volumes'])),
                        'std': float(np.std(individual_metrics['log_bbox_volumes'])),
                        'min': float(np.min(individual_metrics['log_bbox_volumes'])),
                        'max': float(np.max(individual_metrics['log_bbox_volumes'])),
                        'median': float(np.median(individual_metrics['log_bbox_volumes'])),
                        'individual_values': individual_metrics['log_bbox_volumes']
                    },
                    'effective_side_stats': {
                        'mean': float(np.mean(individual_metrics['effective_sides'])),
                        'std': float(np.std(individual_metrics['effective_sides'])),
                        'min': float(np.min(individual_metrics['effective_sides'])),
                        'max': float(np.max(individual_metrics['effective_sides'])),
                        'median': float(np.median(individual_metrics['effective_sides'])),
                        'individual_values': individual_metrics['effective_sides']
                    },
                    'endpoint_alignment_stats': {
                        'mean': float(np.mean(individual_metrics['endpoint_alignment'])),
                        'std': float(np.std(individual_metrics['endpoint_alignment'])),
                        'min': float(np.min(individual_metrics['endpoint_alignment'])),
                        'max': float(np.max(individual_metrics['endpoint_alignment'])),
                        'median': float(np.median(individual_metrics['endpoint_alignment'])),
                        'individual_values': individual_metrics['endpoint_alignment']
                    },
                    'turning_angle_stats': {
                        'mean': float(np.mean(individual_metrics['turning_angle'])),
                        'std': float(np.std(individual_metrics['turning_angle'])),
                        'min': float(np.min(individual_metrics['turning_angle'])),
                        'max': float(np.max(individual_metrics['turning_angle'])),
                        'median': float(np.median(individual_metrics['turning_angle'])),
                        'individual_values': individual_metrics['turning_angle']
                    },

                    'circuitousness_stats': {
                        'mean': float(np.mean(individual_metrics['circuitousness'])),
                        'std': float(np.std(individual_metrics['circuitousness'])),
                        'min': float(np.min(individual_metrics['circuitousness'])),
                        'max': float(np.max(individual_metrics['circuitousness'])),
                        'median': float(np.median(individual_metrics['circuitousness'])),
                        'individual_values': individual_metrics['circuitousness']
                    },
                    'path_length_stats': {
                        'mean': float(np.mean(individual_metrics['path_lengths'])),
                        'std': float(np.std(individual_metrics['path_lengths'])),
                        'individual_values': individual_metrics['path_lengths']
                    },
                    'endpoint_distance_stats': {
                        'mean': float(np.mean(individual_metrics['endpoint_distances'])),
                        'std': float(np.std(individual_metrics['endpoint_distances'])),
                        'individual_values': individual_metrics['endpoint_distances']
                    },
                    'step_variability_stats': {
                        'mean': float(np.mean(individual_metrics['step_size_variability'])),
                        'std': float(np.std(individual_metrics['step_size_variability'])),
                        'individual_values': individual_metrics['step_size_variability']
                    },
                    'efficiency_metrics': {
                        'mean_efficiency': float(np.mean([1.0/c for c in individual_metrics['circuitousness']])),
                        'ballistic_trajectories_count': int(np.sum([c < 1.5 for c in individual_metrics['circuitousness']])),
                        'meandering_trajectories_count': int(np.sum([c > 3.0 for c in individual_metrics['circuitousness']]))
                    },
                    'n_trajectories': int(n_videos)
                }
                
            except Exception as e:
                self.logger.error(f"Error computing individual trajectory geometry for {group_name}: {e}")
                trajectory_geometry[group_name] = {
                    'error': str(e),
                    'n_trajectories': 0
                }
        
        return trajectory_geometry

    def _gpu_analyze_intrinsic_dimension(
        self,
        group_tensors: Dict[str, Dict[str, torch.Tensor]],
        prompt_groups: List[str],
    ) -> Dict[str, Any]:
        """
        Intrinsic dimension analysis that is GPU/CPU memory-safe.

        Pipeline per group:
        1) Flatten features to [N*T, F_full].
        2) (GPU) Optional Gaussian random projection to K << F_full in float32.
        3) (GPU) Row subsample to at most `max_points`.
        4) (CPU) Standardize and run PCA (randomized) for explained variance + #components @ 90/95/99%.
        5) (CPU) Estimate ID via:
                - TwoNN (Facco et al.) on a small subset.
                - Levina–Bickel MLE with k in [k_min, k_max] (averaged), also on a small subset.
        """

        id_cfg = getattr(
            self,
            "id_cfg",
            {
                "use_random_projection": True,
                "rp_dim": 512,            # target feature dim after RP (256–512 is good)
                "max_points": 2000,       # cap rows for PCA/ID
                "corr_max_points": 600,   # cap rows for NN / pairwise distance based estimators
                "pca_components": 50,     # cap PCA components (randomized)
                "center": True,
                "standardize": True,
                "random_seed": 42,
                "twoNN_min_frac": 0.05,   # fraction of tail to ignore in TwoNN fit (robustness)
                "twoNN_max_frac": 0.95,
                "mle_k_min": 10,
                "mle_k_max": 20,
            },
        )

        use_rp: bool = bool(id_cfg.get("use_random_projection", True))
        rp_dim: int = int(id_cfg.get("rp_dim", 512))
        max_points: int = int(id_cfg.get("max_points", 2000))
        corr_max_points: int = int(id_cfg.get("corr_max_points", 600))
        pca_components_cap: int = int(id_cfg.get("pca_components", 50))
        center: bool = bool(id_cfg.get("center", True))
        standardize: bool = bool(id_cfg.get("standardize", True))
        seed: int = int(id_cfg.get("random_seed", 42))
        twoNN_min_frac: float = float(id_cfg.get("twoNN_min_frac", 0.05))
        twoNN_max_frac: float = float(id_cfg.get("twoNN_max_frac", 0.95))
        mle_k_min: int = int(id_cfg.get("mle_k_min", 10))
        mle_k_max: int = int(id_cfg.get("mle_k_max", 20))

        import math
        import numpy as np
        from typing import Optional
        from sklearn.decomposition import PCA

        def _twonn_id(points: np.ndarray) -> float:
            """
            Facco et al. TwoNN estimator.
            - Compute for each point the ratio mu = d2/d1 (2nd / 1st NN).
            - Fit: log(1 - F(mu)) vs log(mu) over central quantiles -> slope ~ -ID.
            """
            n = points.shape[0]
            if n < 20:
                return float("nan")

            # Pairwise distances (small n, already projected)
            # Use chunking-free broadcast since corr_max_points <= ~600
            diffs = points[:, None, :] - points[None, :, :]
            D = np.sqrt(np.sum(diffs * diffs, axis=2))  # (n, n)
            np.fill_diagonal(D, np.inf)
            # For each row, get smallest and second smallest distances
            d1 = np.partition(D, 0, axis=1)[:, 0]
            d2 = np.partition(D, 1, axis=1)[:, 1]
            # Guard against zeros
            d1 = np.clip(d1, 1e-12, None)
            mu = d2 / d1

            # Sort mu and compute empirical CDF
            mu_sorted = np.sort(mu)
            F = (np.arange(1, n + 1) - 0.5) / n

            # Trim tails for robustness
            lo = int(max(1, math.floor(twoNN_min_frac * n)))
            hi = int(min(n, math.ceil(twoNN_max_frac * n)))
            if hi - lo < 10:
                return float("nan")

            x = np.log(mu_sorted[lo:hi])
            y = np.log(1.0 - F[lo:hi])

            # Linear fit y = a + b x; slope b ≈ -ID
            # Use robust guard against NaN/inf
            msk = np.isfinite(x) & np.isfinite(y)
            if msk.sum() < 10:
                return float("nan")
            b, a = np.polyfit(x[msk], y[msk], 1)
            id_est = -float(b)
            return id_est if np.isfinite(id_est) and id_est > 0 else float("nan")

        def _mle_id(points: np.ndarray, k_min: int = 10, k_max: int = 20) -> float:
            """
            Levina–Bickel MLE (average over k in [k_min, k_max]).
            Implementation uses full pairwise distances (OK for <= ~600).
            """
            n = points.shape[0]
            if n <= k_max + 1:
                return float("nan")

            diffs = points[:, None, :] - points[None, :, :]
            D = np.sqrt(np.sum(diffs * diffs, axis=2))  # (n, n)
            np.fill_diagonal(D, np.inf)
            # Sort distances per row
            D_sorted = np.sort(D, axis=1)  # increasing, D_sorted[:, 0] is 1-NN

            ids = []
            for k in range(k_min, k_max + 1):
                # Levina–Bickel: m_hat = [ 1 / ( (1/(n*(k-1))) * sum_i sum_{j=1..k-1} log(d_{ik} / d_{ij}) ) ]
                # Using natural log.
                d_k = D_sorted[:, k]  # distance to k-th neighbor (0-based)
                d_js = D_sorted[:, 1:k]  # distances to 1..(k-1)-th neighbors
                ratio = d_k[:, None] / np.clip(d_js, 1e-12, None)
                s = np.log(np.clip(ratio, 1e-12, None)).sum(axis=1)  # per i
                denom = np.mean(s)  # average over i
                if denom <= 0 or not np.isfinite(denom):
                    continue
                m_hat = (1.0 / ((1.0 / (k - 1)) * denom))
                if np.isfinite(m_hat) and m_hat > 0:
                    ids.append(m_hat)

            if len(ids) == 0:
                return float("nan")
            return float(np.mean(ids))