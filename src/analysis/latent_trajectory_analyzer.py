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


from .load_and_batch_trajectory_data import load_and_batch_trajectory_data

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
from .latent_trajectory_analysis.analyze_convex_hull_metrics_safe import analyze_convex_hull_metrics_safe
from .latent_trajectory_analysis.analyze_functional_pca import analyze_functional_pca

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

    def analyze_prompt_groups(self, prompt_groups: List[str], 
                            prompt_metadata: Optional[Dict[str, Dict[str, str]]] = None) -> LatentTrajectoryAnalysis:
        """Main analysis entry point with trajectory-aware processing."""
        self.logger.info("Starting GPU-optimized trajectory-aware analysis")
        start_time = time.time()
        
        if not getattr(self, 'group_tensors', None):
            # 1. Load and batch trajectory data
            self._track_gpu_memory("start")
            self.group_tensors = load_and_batch_trajectory_data(
                latents_dir=self.latents_dir,
                prompt_groups=prompt_groups,
                device=self.device
            )

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
            analysis_results['spatial_patterns'] = analyze_spatial_patterns(group_tensors)
            self._track_gpu_memory("spatial_analysis")
            
            self.logger.info("Running temporal coherence analysis...")
            analysis_results['temporal_coherence'] = analyze_temporal_coherence(group_tensors, device=self.device)
            self._track_gpu_memory("temporal_analysis")
            
            self.logger.info("Running channel pattern analysis...")
            analysis_results['channel_analysis'] = analyze_channel_patterns(group_tensors)
            self._track_gpu_memory("channel_analysis")
            
            # Multi-scale analysis
            self.logger.info("Running patch diversity analysis...")
            analysis_results['patch_diversity'] = analyze_patch_diversity(group_tensors)

            
            self.logger.info("Running global structure analysis...")
            analysis_results['global_structure'] = analyze_global_structure(group_tensors)


            # Simplified additional analyses
            self.logger.info("Running information content analysis...")
            analysis_results['information_content'] = analyze_information_content(group_tensors)

            
            self.logger.info("Running complexity analysis...")
            analysis_results['complexity_measures'] = analyze_complexity_measures(group_tensors)

            self.logger.info("Running frequency analysis...")
            analysis_results['frequency_patterns'] = analyze_frequency_patterns(group_tensors)

            
            # Group separability
            self.logger.info("Running group separability analysis...")
            analysis_results['group_separability'] = analyze_group_separability(group_tensors)

            # Temporal trajectory analysis
            self.logger.info("Running temporal trajectory analysis...")
            analysis_results['temporal_analysis'] = analyze_temporal_trajectories(group_tensors, prompt_groups, device=self.device, norm_cfg=self.norm_cfg)

            # Structural analysis
            self.logger.info("Running structural analysis...")
            analysis_results['structural_analysis'] = analyze_structural_patterns(group_tensors, prompt_groups, device=self.device)
            
            # NEW: Advanced geometric analysis
            self.logger.info("Running convex hull analysis...")
            analysis_results['convex_hull_analysis'] = analyze_convex_hull_metrics_safe(group_tensors, hull_cfg=self.hull_cfg)
            
            self.logger.info("Running functional PCA analysis...")
            analysis_results['functional_pca_analysis'] = analyze_functional_pca(group_tensors, prompt_groups)

            self.logger.info("Running individual trajectory geometry analysis...")
            analysis_results['individual_trajectory_geometry'] = analyze_individual_trajectory_geometry(group_tensors)
            
            # TODO: maybe skip -- function only stubbed
            self.logger.info("Running intrinsic dimension analysis...")
            analysis_results['intrinsic_dimension_analysis'] = analyze_intrinsic_dimension(group_tensors)

            # Statistical significance
            self.logger.info("Running statistical significance tests...")
            analysis_results['statistical_significance'] = test_statistical_significance(group_tensors)

            # Corridor metrics
            self.logger.info("Running corridor metrics tests...")
            analysis_results['corridor_metrics'] = analyze_corridor_metrics(group_tensors, norm_cfg=self.norm_cfg)

            # # Geometry derivatives metrics
            self.logger.info("Running geometry derivatives analysis...")
            analysis_results['geometry_derivatives'] = analyze_geometry_derivatives(group_tensors, norm_cfg=self.norm_cfg)

            self.logger.info("Attaching confidence intervals...")
            analysis_results['confidence_intervals'] = attach_confidence_intervals(analysis_results)


            self.logger.info("Log volume delta vs baseline...")
            analysis_results['log_volume_delta_vs_baseline'] = log_volume_deltas(analysis_results)


            self.logger.info("Running normative strength...")
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