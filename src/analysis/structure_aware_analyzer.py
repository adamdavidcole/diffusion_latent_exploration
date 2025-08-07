#!/usr/bin/env python3
"""
Structure-Aware Latent Analysis for Video Diffusion Models

This module implements analysis methods that respect the 3D video latent structure
[batch, channels, frames, height, width] rather than flattening to 1D vectors.

Key innovations:
1. Spatial pattern analysis (preserves height/width structure)
2. Temporal coherence analysis (preserves frame structure) 
3. Channel-specific analysis (different channels may encode different features)
4. Multi-scale analysis (patch-based and global patterns)
5. Information-theoretic measures
6. Spectral analysis (frequency domain patterns)
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats, spatial
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging
from dataclasses import dataclass
import json
from collections import defaultdict
from datetime import datetime

try:
    from scipy.fft import fft2, fftshift
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import entropy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class StructureAwareAnalysis:
    """Container for structure-aware analysis results."""
    
    # Spatial analysis
    spatial_patterns: Dict[str, Any]
    temporal_coherence: Dict[str, Any]
    channel_analysis: Dict[str, Any]
    
    # Multi-scale analysis
    patch_diversity: Dict[str, Any]
    global_structure: Dict[str, Any]
    
    # Information theory
    information_content: Dict[str, Any]
    complexity_measures: Dict[str, Any]
    
    # Spectral analysis
    frequency_patterns: Dict[str, Any]
    
    # Validation
    group_separability: Dict[str, Any]
    statistical_significance: Dict[str, Any]
    
    # Metadata
    analysis_timestamp: str
    latent_shape: Tuple[int, ...]
    groups_analyzed: List[str]


class StructureAwareLatentAnalyzer:
    """
    Advanced analyzer that respects 3D video latent structure.
    
    Addresses fundamental issues with previous flattening-based approaches
    by analyzing spatial patterns, temporal coherence, and channel-specific
    information separately and in combination.
    """
    
    def __init__(self, latents_dir: str, output_dir: Optional[str] = None):
        """Initialize structure-aware analyzer."""
        self.latents_dir = Path(latents_dir)
        self.output_dir = Path(output_dir) if output_dir else self.latents_dir / "structure_aware_analysis"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized StructureAwareLatentAnalyzer for: {latents_dir}")
        
        # Analysis parameters
        self.patch_sizes = [4, 8, 16]  # Multi-scale spatial analysis
        self.temporal_windows = [4, 8, 16]  # Multi-scale temporal analysis
        
        # Load base analyzer
        try:
            from .latent_trajectory_analysis import LatentTrajectoryAnalyzer
            self.base_analyzer = LatentTrajectoryAnalyzer(latents_dir)
        except ImportError:
            self.logger.error("Could not import base LatentTrajectoryAnalyzer")
            raise
    
    def analyze_prompt_groups(self, prompt_groups: List[str], 
                            prompt_descriptions: List[str]) -> StructureAwareAnalysis:
        """
        Comprehensive structure-aware analysis of prompt groups.
        
        This is the main analysis function that implements all methodological
        approaches while respecting 3D video latent structure.
        """
        self.logger.info(f"Starting structure-aware analysis of {len(prompt_groups)} groups")
        
        # 1. Load latent data with structure preservation
        group_latents = self._load_structured_latent_data(prompt_groups)
        
        if not group_latents:
            raise ValueError("No latent data loaded")
        
        # Get latent shape from first sample
        sample_latent = next(iter(group_latents.values()))[0]['latents'][0]
        latent_shape = tuple(sample_latent.shape)
        self.logger.info(f"Analyzing latents with shape: {latent_shape}")
        
        # 2. Multi-dimensional analysis suite
        spatial_patterns = self._analyze_spatial_patterns(group_latents)
        temporal_coherence = self._analyze_temporal_coherence(group_latents)
        channel_analysis = self._analyze_channel_patterns(group_latents)
        
        # 3. Multi-scale analysis
        patch_diversity = self._analyze_patch_diversity(group_latents)
        global_structure = self._analyze_global_structure(group_latents)
        
        # 4. Information-theoretic analysis
        information_content = self._analyze_information_content(group_latents)
        complexity_measures = self._analyze_complexity_measures(group_latents)
        
        # 5. Spectral analysis
        frequency_patterns = self._analyze_frequency_patterns(group_latents)
        
        # 6. Group separability analysis
        group_separability = self._analyze_group_separability(group_latents, prompt_groups)
        
        # 7. Statistical significance testing
        statistical_significance = self._test_statistical_significance(group_latents, prompt_groups)
        
        # 8. Package results
        results = StructureAwareAnalysis(
            spatial_patterns=spatial_patterns,
            temporal_coherence=temporal_coherence,
            channel_analysis=channel_analysis,
            patch_diversity=patch_diversity,
            global_structure=global_structure,
            information_content=information_content,
            complexity_measures=complexity_measures,
            frequency_patterns=frequency_patterns,
            group_separability=group_separability,
            statistical_significance=statistical_significance,
            analysis_timestamp=str(datetime.now()),
            latent_shape=latent_shape,
            groups_analyzed=prompt_groups
        )
        
        # 9. Save results and create visualizations
        self._save_analysis_results(results)
        self._create_comprehensive_visualizations(results, group_latents)
        
        return results
    
    def _load_structured_latent_data(self, prompt_groups: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Load latent data preserving 3D structure."""
        group_latents = {}
        
        for group_name in prompt_groups:
            self.logger.info(f"Loading structured latents for group: {group_name}")
            
            video_ids = self.base_analyzer.discover_videos_in_prompt(group_name)
            group_data = []
            
            for video_id in video_ids:
                try:
                    latents, metadata = self.base_analyzer.load_video_trajectory(video_id)
                    
                    if len(latents) > 0:
                        # Verify expected shape
                        sample_shape = latents[0].shape
                        if len(sample_shape) != 5:
                            self.logger.warning(f"Unexpected latent shape {sample_shape} for {video_id}")
                            continue
                        
                        trajectory_data = {
                            'video_id': video_id,
                            'latents': latents,  # List of [1, 16, frames, H, W] tensors
                            'metadata': metadata,
                            'timesteps': [meta.timestep for meta in metadata],
                            'shape': sample_shape,
                            'sequence_length': len(latents)
                        }
                        group_data.append(trajectory_data)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to load {video_id}: {e}")
            
            group_latents[group_name] = group_data
            self.logger.info(f"Loaded {len(group_data)} videos for {group_name}")
        
        return group_latents
    
    def _analyze_spatial_patterns(self, group_latents: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Analyze spatial patterns in latent representations.
        
        Measures how prompt specificity affects spatial organization
        within each frame of the latent tensor.
        """
        spatial_analysis = {
            'spatial_variance_maps': {},
            'spatial_autocorrelation': {},
            'edge_density': {},
            'spatial_clustering': {}
        }
        
        for group_name, videos in group_latents.items():
            self.logger.info(f"Analyzing spatial patterns for {group_name}")
            
            group_spatial_vars = []
            group_autocorrs = []
            group_edge_densities = []
            
            for video_data in videos:
                latents = video_data['latents']
                
                # Analyze each timestep
                video_spatial_vars = []
                video_autocorrs = []
                video_edge_densities = []
                
                for latent in latents:
                    # latent shape: [1, 16, frames, H, W]
                    latent_np = latent.squeeze(0).numpy()  # [16, frames, H, W]
                    
                    # For each channel and frame
                    timestep_spatial_vars = []
                    timestep_autocorrs = []
                    timestep_edge_densities = []
                    
                    for channel in range(latent_np.shape[0]):
                        for frame in range(latent_np.shape[1]):
                            spatial_map = latent_np[channel, frame]  # [H, W]
                            
                            # Spatial variance (how varied the spatial pattern is)
                            spatial_var = np.var(spatial_map)
                            timestep_spatial_vars.append(spatial_var)
                            
                            # Spatial autocorrelation (how spatially coherent)
                            if spatial_map.shape[0] > 1 and spatial_map.shape[1] > 1:
                                # Moran's I-like measure
                                flat_map = spatial_map.flatten()
                                # Simple autocorrelation: correlation with shifted version
                                shifted_h = np.roll(spatial_map, 1, axis=0).flatten()
                                shifted_w = np.roll(spatial_map, 1, axis=1).flatten()
                                autocorr_h = np.corrcoef(flat_map, shifted_h)[0, 1] if len(flat_map) > 1 else 0
                                autocorr_w = np.corrcoef(flat_map, shifted_w)[0, 1] if len(flat_map) > 1 else 0
                                autocorr = (autocorr_h + autocorr_w) / 2
                                if not np.isnan(autocorr):
                                    timestep_autocorrs.append(autocorr)
                            
                            # Edge density (high-frequency content)
                            if CV2_AVAILABLE and spatial_map.shape[0] > 3 and spatial_map.shape[1] > 3:
                                # Convert to uint8 for edge detection
                                normalized = ((spatial_map - spatial_map.min()) / 
                                            (spatial_map.max() - spatial_map.min() + 1e-8) * 255).astype(np.uint8)
                                edges = cv2.Canny(normalized, 50, 150)
                                edge_density = np.sum(edges > 0) / edges.size
                                timestep_edge_densities.append(edge_density)
                    
                    video_spatial_vars.append(np.mean(timestep_spatial_vars) if timestep_spatial_vars else 0)
                    video_autocorrs.append(np.mean(timestep_autocorrs) if timestep_autocorrs else 0)
                    video_edge_densities.append(np.mean(timestep_edge_densities) if timestep_edge_densities else 0)
                
                group_spatial_vars.extend(video_spatial_vars)
                group_autocorrs.extend(video_autocorrs)
                group_edge_densities.extend(video_edge_densities)
            
            # Aggregate group statistics
            spatial_analysis['spatial_variance_maps'][group_name] = {
                'mean': np.mean(group_spatial_vars) if group_spatial_vars else 0,
                'std': np.std(group_spatial_vars) if group_spatial_vars else 0,
                'distribution': group_spatial_vars
            }
            
            spatial_analysis['spatial_autocorrelation'][group_name] = {
                'mean': np.mean(group_autocorrs) if group_autocorrs else 0,
                'std': np.std(group_autocorrs) if group_autocorrs else 0,
                'distribution': group_autocorrs
            }
            
            spatial_analysis['edge_density'][group_name] = {
                'mean': np.mean(group_edge_densities) if group_edge_densities else 0,
                'std': np.std(group_edge_densities) if group_edge_densities else 0,
                'distribution': group_edge_densities
            }
        
        return spatial_analysis
    
    def _analyze_temporal_coherence(self, group_latents: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Analyze temporal coherence across video frames.
        
        Measures how prompt specificity affects temporal consistency
        within the video latent representation.
        """
        temporal_analysis = {
            'frame_correlation': {},
            'temporal_variance': {},
            'motion_patterns': {},
            'temporal_autocorrelation': {}
        }
        
        for group_name, videos in group_latents.items():
            self.logger.info(f"Analyzing temporal coherence for {group_name}")
            
            group_frame_corrs = []
            group_temporal_vars = []
            group_motion_patterns = []
            group_temporal_autocorrs = []
            
            for video_data in videos:
                latents = video_data['latents']
                
                video_frame_corrs = []
                video_temporal_vars = []
                video_motion_patterns = []
                video_temporal_autocorrs = []
                
                for latent in latents:
                    latent_np = latent.squeeze(0).numpy()  # [16, frames, H, W]
                    
                    # For each channel
                    for channel in range(latent_np.shape[0]):
                        channel_data = latent_np[channel]  # [frames, H, W]
                        
                        if channel_data.shape[0] > 1:  # Need at least 2 frames
                            # Frame-to-frame correlation
                            frame_corrs = []
                            for f in range(channel_data.shape[0] - 1):
                                frame1 = channel_data[f].flatten()
                                frame2 = channel_data[f + 1].flatten()
                                if len(frame1) > 1:
                                    corr = np.corrcoef(frame1, frame2)[0, 1]
                                    if not np.isnan(corr):
                                        frame_corrs.append(corr)
                            
                            if frame_corrs:
                                video_frame_corrs.append(np.mean(frame_corrs))
                            
                            # Temporal variance (how much change over time)
                            temporal_var = np.var(channel_data, axis=0)  # Variance across frames
                            video_temporal_vars.append(np.mean(temporal_var))
                            
                            # Motion patterns (frame differences)
                            frame_diffs = []
                            for f in range(channel_data.shape[0] - 1):
                                diff = np.mean(np.abs(channel_data[f + 1] - channel_data[f]))
                                frame_diffs.append(diff)
                            
                            if frame_diffs:
                                video_motion_patterns.append(np.mean(frame_diffs))
                            
                            # Temporal autocorrelation (periodic patterns)
                            if channel_data.shape[0] > 3:
                                # Average across spatial dimensions
                                temporal_signal = np.mean(channel_data, axis=(1, 2))
                                # Autocorrelation with lag-1
                                if len(temporal_signal) > 1:
                                    autocorr = np.corrcoef(temporal_signal[:-1], temporal_signal[1:])[0, 1]
                                    if not np.isnan(autocorr):
                                        video_temporal_autocorrs.append(autocorr)
                
                group_frame_corrs.extend(video_frame_corrs)
                group_temporal_vars.extend(video_temporal_vars)
                group_motion_patterns.extend(video_motion_patterns)
                group_temporal_autocorrs.extend(video_temporal_autocorrs)
            
            # Aggregate group statistics
            temporal_analysis['frame_correlation'][group_name] = {
                'mean': np.mean(group_frame_corrs) if group_frame_corrs else 0,
                'std': np.std(group_frame_corrs) if group_frame_corrs else 0,
                'distribution': group_frame_corrs
            }
            
            temporal_analysis['temporal_variance'][group_name] = {
                'mean': np.mean(group_temporal_vars) if group_temporal_vars else 0,
                'std': np.std(group_temporal_vars) if group_temporal_vars else 0,
                'distribution': group_temporal_vars
            }
            
            temporal_analysis['motion_patterns'][group_name] = {
                'mean': np.mean(group_motion_patterns) if group_motion_patterns else 0,
                'std': np.std(group_motion_patterns) if group_motion_patterns else 0,
                'distribution': group_motion_patterns
            }
            
            temporal_analysis['temporal_autocorrelation'][group_name] = {
                'mean': np.mean(group_temporal_autocorrs) if group_temporal_autocorrs else 0,
                'std': np.std(group_temporal_autocorrs) if group_temporal_autocorrs else 0,
                'distribution': group_temporal_autocorrs
            }
        
        return temporal_analysis
    
    def _analyze_channel_patterns(self, group_latents: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Analyze channel-specific patterns.
        
        Different channels may encode different types of information
        (e.g., some channels for color, others for texture, etc.)
        """
        channel_analysis = {
            'channel_variance': {},
            'channel_correlation': {},
            'channel_dominance': {},
            'cross_channel_interaction': {}
        }
        
        for group_name, videos in group_latents.items():
            self.logger.info(f"Analyzing channel patterns for {group_name}")
            
            group_channel_vars = []
            group_channel_corrs = []
            group_channel_dominance = []
            group_cross_channel = []
            
            for video_data in videos:
                latents = video_data['latents']
                
                for latent in latents:
                    latent_np = latent.squeeze(0).numpy()  # [16, frames, H, W]
                    
                    # Channel variance (how much each channel varies)
                    channel_vars = []
                    for channel in range(latent_np.shape[0]):
                        channel_data = latent_np[channel]  # [frames, H, W]
                        channel_var = np.var(channel_data)
                        channel_vars.append(channel_var)
                    
                    group_channel_vars.append(channel_vars)
                    
                    # Channel correlation (how similar channels are)
                    if latent_np.shape[0] > 1:
                        channel_corrs = []
                        for c1 in range(latent_np.shape[0]):
                            for c2 in range(c1 + 1, latent_np.shape[0]):
                                flat1 = latent_np[c1].flatten()
                                flat2 = latent_np[c2].flatten()
                                if len(flat1) > 1:
                                    corr = np.corrcoef(flat1, flat2)[0, 1]
                                    if not np.isnan(corr):
                                        channel_corrs.append(corr)
                        
                        if channel_corrs:
                            group_channel_corrs.append(np.mean(channel_corrs))
                    
                    # Channel dominance (which channels have most energy)
                    channel_energies = [np.sum(np.abs(latent_np[c])) for c in range(latent_np.shape[0])]
                    total_energy = np.sum(channel_energies)
                    if total_energy > 0:
                        channel_dominance = [e / total_energy for e in channel_energies]
                        group_channel_dominance.append(channel_dominance)
                    
                    # Cross-channel interaction (how channels interact spatially)
                    if latent_np.shape[0] > 1:
                        cross_channel_interactions = []
                        for frame in range(latent_np.shape[1]):
                            # Compute spatial correlation between channels for this frame
                            frame_interactions = []
                            for c1 in range(latent_np.shape[0]):
                                for c2 in range(c1 + 1, latent_np.shape[0]):
                                    map1 = latent_np[c1, frame].flatten()
                                    map2 = latent_np[c2, frame].flatten()
                                    if len(map1) > 1:
                                        interaction = np.corrcoef(map1, map2)[0, 1]
                                        if not np.isnan(interaction):
                                            frame_interactions.append(interaction)
                            
                            if frame_interactions:
                                cross_channel_interactions.append(np.mean(frame_interactions))
                        
                        if cross_channel_interactions:
                            group_cross_channel.append(np.mean(cross_channel_interactions))
            
            # Aggregate statistics
            if group_channel_vars:
                # Average channel variances across all samples
                channel_vars_array = np.array(group_channel_vars)
                channel_analysis['channel_variance'][group_name] = {
                    'per_channel_mean': np.mean(channel_vars_array, axis=0).tolist(),
                    'per_channel_std': np.std(channel_vars_array, axis=0).tolist(),
                    'total_variance': np.mean(np.sum(channel_vars_array, axis=1)),
                    'variance_distribution': np.std(np.sum(channel_vars_array, axis=1))
                }
            
            channel_analysis['channel_correlation'][group_name] = {
                'mean': np.mean(group_channel_corrs) if group_channel_corrs else 0,
                'std': np.std(group_channel_corrs) if group_channel_corrs else 0,
                'distribution': group_channel_corrs
            }
            
            if group_channel_dominance:
                dominance_array = np.array(group_channel_dominance)
                channel_analysis['channel_dominance'][group_name] = {
                    'mean_dominance': np.mean(dominance_array, axis=0).tolist(),
                    'dominance_entropy': [entropy(dom) for dom in dominance_array if np.sum(dom) > 0],
                    'max_channel_dominance': np.mean(np.max(dominance_array, axis=1))
                }
            
            channel_analysis['cross_channel_interaction'][group_name] = {
                'mean': np.mean(group_cross_channel) if group_cross_channel else 0,
                'std': np.std(group_cross_channel) if group_cross_channel else 0,
                'distribution': group_cross_channel
            }
        
        return channel_analysis
    
    def _analyze_patch_diversity(self, group_latents: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Multi-scale patch diversity analysis.
        
        Analyzes local patterns at different spatial scales to understand
        how prompt specificity affects local vs global organization.
        """
        patch_analysis = {
            'patch_variance': {},
            'patch_distinctiveness': {},
            'multi_scale_patterns': {}
        }
        
        for group_name, videos in group_latents.items():
            self.logger.info(f"Analyzing patch diversity for {group_name}")
            
            scale_results = {}
            
            for patch_size in self.patch_sizes:
                group_patch_vars = []
                group_patch_distinctiveness = []
                
                for video_data in videos:
                    latents = video_data['latents']
                    
                    for latent in latents:
                        latent_np = latent.squeeze(0).numpy()  # [16, frames, H, W]
                        
                        # For each channel and frame
                        for channel in range(latent_np.shape[0]):
                            for frame in range(latent_np.shape[1]):
                                spatial_map = latent_np[channel, frame]  # [H, W]
                                
                                # Extract patches
                                patches = self._extract_patches(spatial_map, patch_size)
                                
                                if len(patches) > 1:
                                    # Patch variance (diversity within patches)
                                    patch_vars = [np.var(patch) for patch in patches]
                                    group_patch_vars.extend(patch_vars)
                                    
                                    # Patch distinctiveness (how different patches are from each other)
                                    patch_distances = []
                                    for i in range(len(patches)):
                                        for j in range(i + 1, len(patches)):
                                            dist = np.linalg.norm(patches[i] - patches[j])
                                            patch_distances.append(dist)
                                    
                                    if patch_distances:
                                        group_patch_distinctiveness.append(np.mean(patch_distances))
                
                scale_results[f'patch_size_{patch_size}'] = {
                    'patch_variance': {
                        'mean': np.mean(group_patch_vars) if group_patch_vars else 0,
                        'std': np.std(group_patch_vars) if group_patch_vars else 0,
                        'distribution': group_patch_vars
                    },
                    'patch_distinctiveness': {
                        'mean': np.mean(group_patch_distinctiveness) if group_patch_distinctiveness else 0,
                        'std': np.std(group_patch_distinctiveness) if group_patch_distinctiveness else 0,
                        'distribution': group_patch_distinctiveness
                    }
                }
            
            patch_analysis['multi_scale_patterns'][group_name] = scale_results
        
        return patch_analysis
    
    def _extract_patches(self, spatial_map: np.ndarray, patch_size: int) -> List[np.ndarray]:
        """Extract non-overlapping patches from spatial map."""
        patches = []
        H, W = spatial_map.shape
        
        for h in range(0, H - patch_size + 1, patch_size):
            for w in range(0, W - patch_size + 1, patch_size):
                patch = spatial_map[h:h+patch_size, w:w+patch_size]
                if patch.shape == (patch_size, patch_size):
                    patches.append(patch.flatten())
        
        return patches
    
    def _analyze_global_structure(self, group_latents: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Analyze global structural properties.
        
        Examines overall organization patterns across the entire latent tensor.
        """
        global_analysis = {
            'global_variance': {},
            'structural_entropy': {},
            'symmetry_measures': {},
            'global_clustering': {}
        }
        
        for group_name, videos in group_latents.items():
            self.logger.info(f"Analyzing global structure for {group_name}")
            
            group_global_vars = []
            group_entropies = []
            group_symmetries = []
            
            for video_data in videos:
                latents = video_data['latents']
                
                for latent in latents:
                    latent_np = latent.squeeze(0).numpy()  # [16, frames, H, W]
                    
                    # Global variance (overall spread)
                    global_var = np.var(latent_np)
                    group_global_vars.append(global_var)
                    
                    # Structural entropy (how organized vs random)
                    # Compute entropy of spatial variance distribution
                    spatial_vars = []
                    for channel in range(latent_np.shape[0]):
                        for frame in range(latent_np.shape[1]):
                            spatial_vars.append(np.var(latent_np[channel, frame]))
                    
                    if spatial_vars:
                        # Discretize for entropy calculation
                        hist, _ = np.histogram(spatial_vars, bins=10, density=True)
                        hist = hist + 1e-10  # Avoid log(0)
                        entropy_val = entropy(hist)
                        group_entropies.append(entropy_val)
                    
                    # Symmetry measures (spatial organization)
                    symmetries = []
                    for channel in range(latent_np.shape[0]):
                        for frame in range(latent_np.shape[1]):
                            spatial_map = latent_np[channel, frame]
                            
                            # Horizontal symmetry
                            left_half = spatial_map[:, :spatial_map.shape[1]//2]
                            right_half = spatial_map[:, spatial_map.shape[1]//2:]
                            right_half_flipped = np.fliplr(right_half)
                            
                            if left_half.shape == right_half_flipped.shape and left_half.size > 0:
                                h_symmetry = np.corrcoef(left_half.flatten(), 
                                                       right_half_flipped.flatten())[0, 1]
                                if not np.isnan(h_symmetry):
                                    symmetries.append(h_symmetry)
                            
                            # Vertical symmetry
                            top_half = spatial_map[:spatial_map.shape[0]//2, :]
                            bottom_half = spatial_map[spatial_map.shape[0]//2:, :]
                            bottom_half_flipped = np.flipud(bottom_half)
                            
                            if top_half.shape == bottom_half_flipped.shape and top_half.size > 0:
                                v_symmetry = np.corrcoef(top_half.flatten(), 
                                                       bottom_half_flipped.flatten())[0, 1]
                                if not np.isnan(v_symmetry):
                                    symmetries.append(v_symmetry)
                    
                    if symmetries:
                        group_symmetries.append(np.mean(symmetries))
            
            global_analysis['global_variance'][group_name] = {
                'mean': np.mean(group_global_vars) if group_global_vars else 0,
                'std': np.std(group_global_vars) if group_global_vars else 0,
                'distribution': group_global_vars
            }
            
            global_analysis['structural_entropy'][group_name] = {
                'mean': np.mean(group_entropies) if group_entropies else 0,
                'std': np.std(group_entropies) if group_entropies else 0,
                'distribution': group_entropies
            }
            
            global_analysis['symmetry_measures'][group_name] = {
                'mean': np.mean(group_symmetries) if group_symmetries else 0,
                'std': np.std(group_symmetries) if group_symmetries else 0,
                'distribution': group_symmetries
            }
        
        return global_analysis
    
    def _analyze_information_content(self, group_latents: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Information-theoretic analysis of latent content.
        
        Measures actual information content rather than simple variance.
        """
        info_analysis = {
            'mutual_information': {},
            'conditional_entropy': {},
            'information_density': {},
            'complexity_measures': {}
        }
        
        for group_name, videos in group_latents.items():
            self.logger.info(f"Analyzing information content for {group_name}")
            
            group_mutual_info = []
            group_cond_entropy = []
            group_info_density = []
            
            for video_data in videos:
                latents = video_data['latents']
                
                for latent in latents:
                    latent_np = latent.squeeze(0).numpy()  # [16, frames, H, W]
                    
                    # Mutual information between channels
                    if latent_np.shape[0] > 1:
                        mutual_infos = []
                        for c1 in range(latent_np.shape[0]):
                            for c2 in range(c1 + 1, latent_np.shape[0]):
                                # Flatten and discretize for MI calculation
                                data1 = latent_np[c1].flatten()
                                data2 = latent_np[c2].flatten()
                                
                                if len(data1) > 10:  # Need enough data points
                                    mi = self._compute_mutual_information(data1, data2)
                                    if not np.isnan(mi):
                                        mutual_infos.append(mi)
                        
                        if mutual_infos:
                            group_mutual_info.append(np.mean(mutual_infos))
                    
                    # Conditional entropy (how predictable is each channel given others)
                    if latent_np.shape[0] > 1 and latent_np.shape[1] > 1:
                        # For temporal conditional entropy
                        temporal_entropies = []
                        for channel in range(latent_np.shape[0]):
                            channel_data = latent_np[channel]  # [frames, H, W]
                            
                            # Entropy of current frame given previous frame
                            for frame in range(1, channel_data.shape[0]):
                                curr_frame = channel_data[frame].flatten()
                                prev_frame = channel_data[frame-1].flatten()
                                
                                if len(curr_frame) > 10:
                                    cond_entropy = self._compute_conditional_entropy(curr_frame, prev_frame)
                                    if not np.isnan(cond_entropy):
                                        temporal_entropies.append(cond_entropy)
                        
                        if temporal_entropies:
                            group_cond_entropy.append(np.mean(temporal_entropies))
                    
                    # Information density (bits per spatial location)
                    total_entropy = 0
                    total_locations = 0
                    
                    for channel in range(latent_np.shape[0]):
                        for frame in range(latent_np.shape[1]):
                            spatial_data = latent_np[channel, frame].flatten()
                            if len(spatial_data) > 1:
                                # Discretize and compute entropy
                                hist, _ = np.histogram(spatial_data, bins=min(50, len(spatial_data)//10), density=True)
                                hist = hist + 1e-10
                                spatial_entropy = entropy(hist)
                                total_entropy += spatial_entropy
                                total_locations += 1
                    
                    if total_locations > 0:
                        info_density = total_entropy / total_locations
                        group_info_density.append(info_density)
            
            info_analysis['mutual_information'][group_name] = {
                'mean': np.mean(group_mutual_info) if group_mutual_info else 0,
                'std': np.std(group_mutual_info) if group_mutual_info else 0,
                'distribution': group_mutual_info
            }
            
            info_analysis['conditional_entropy'][group_name] = {
                'mean': np.mean(group_cond_entropy) if group_cond_entropy else 0,
                'std': np.std(group_cond_entropy) if group_cond_entropy else 0,
                'distribution': group_cond_entropy
            }
            
            info_analysis['information_density'][group_name] = {
                'mean': np.mean(group_info_density) if group_info_density else 0,
                'std': np.std(group_info_density) if group_info_density else 0,
                'distribution': group_info_density
            }
        
        return info_analysis
    
    def _compute_mutual_information(self, x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
        """Compute mutual information between two continuous variables."""
        try:
            # Discretize the continuous variables
            x_bins = np.linspace(x.min(), x.max(), bins)
            y_bins = np.linspace(y.min(), y.max(), bins)
            
            x_digitized = np.digitize(x, x_bins)
            y_digitized = np.digitize(y, y_bins)
            
            # Compute joint and marginal distributions
            joint_hist, _, _ = np.histogram2d(x_digitized, y_digitized, bins=[bins, bins])
            joint_hist = joint_hist + 1e-10  # Avoid log(0)
            joint_prob = joint_hist / np.sum(joint_hist)
            
            marginal_x = np.sum(joint_prob, axis=1)
            marginal_y = np.sum(joint_prob, axis=0)
            
            # Compute mutual information
            mi = 0
            for i in range(len(marginal_x)):
                for j in range(len(marginal_y)):
                    if joint_prob[i, j] > 0:
                        mi += joint_prob[i, j] * np.log2(joint_prob[i, j] / (marginal_x[i] * marginal_y[j]))
            
            return mi
        except:
            return np.nan
    
    def _compute_conditional_entropy(self, y: np.ndarray, x: np.ndarray, bins: int = 20) -> float:
        """Compute conditional entropy H(Y|X)."""
        try:
            # Discretize
            x_bins = np.linspace(x.min(), x.max(), bins)
            y_bins = np.linspace(y.min(), y.max(), bins)
            
            x_digitized = np.digitize(x, x_bins)
            y_digitized = np.digitize(y, y_bins)
            
            # Compute conditional entropy
            cond_entropy = 0
            for x_val in range(1, bins + 1):
                x_mask = x_digitized == x_val
                if np.sum(x_mask) > 0:
                    y_given_x = y_digitized[x_mask]
                    hist_y_given_x, _ = np.histogram(y_given_x, bins=bins, density=True)
                    hist_y_given_x = hist_y_given_x + 1e-10
                    
                    entropy_y_given_x = entropy(hist_y_given_x)
                    prob_x = np.sum(x_mask) / len(x)
                    cond_entropy += prob_x * entropy_y_given_x
            
            return cond_entropy
        except:
            return np.nan
    
    def _analyze_complexity_measures(self, group_latents: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Advanced complexity measures for latent representations.
        """
        complexity_analysis = {
            'lempel_ziv_complexity': {},
            'fractal_dimension': {},
            'effective_dimensionality': {},
            'compressibility': {}
        }
        
        for group_name, videos in group_latents.items():
            self.logger.info(f"Analyzing complexity measures for {group_name}")
            
            group_lz_complexity = []
            group_fractal_dim = []
            group_eff_dim = []
            group_compressibility = []
            
            for video_data in videos:
                latents = video_data['latents']
                
                for latent in latents:
                    latent_np = latent.squeeze(0).numpy()  # [16, frames, H, W]
                    
                    # Effective dimensionality (PCA-based)
                    flattened = latent_np.reshape(latent_np.shape[0], -1)  # [16, frames*H*W]
                    if flattened.shape[1] > flattened.shape[0]:
                        pca = PCA()
                        pca.fit(flattened.T)  # Transpose so features are channels
                        explained_var = pca.explained_variance_ratio_
                        
                        # Effective dimensionality: number of components needed for 95% variance
                        cumsum_var = np.cumsum(explained_var)
                        eff_dim = np.argmax(cumsum_var >= 0.95) + 1
                        group_eff_dim.append(eff_dim)
                    
                    # Compressibility (using simple compression ratio)
                    try:
                        import zlib
                        # Convert to bytes for compression
                        quantized = (latent_np * 1000).astype(np.int16)  # Quantize for compression
                        data_bytes = quantized.tobytes()
                        compressed = zlib.compress(data_bytes)
                        compression_ratio = len(compressed) / len(data_bytes)
                        group_compressibility.append(compression_ratio)
                    except:
                        pass
                    
                    # Fractal dimension estimate (box counting for 2D slices)
                    fractal_dims = []
                    for channel in range(min(4, latent_np.shape[0])):  # Sample channels
                        for frame in range(min(4, latent_np.shape[1])):  # Sample frames
                            spatial_map = latent_np[channel, frame]
                            fractal_dim = self._estimate_fractal_dimension(spatial_map)
                            if not np.isnan(fractal_dim):
                                fractal_dims.append(fractal_dim)
                    
                    if fractal_dims:
                        group_fractal_dim.append(np.mean(fractal_dims))
            
            complexity_analysis['effective_dimensionality'][group_name] = {
                'mean': np.mean(group_eff_dim) if group_eff_dim else 0,
                'std': np.std(group_eff_dim) if group_eff_dim else 0,
                'distribution': group_eff_dim
            }
            
            complexity_analysis['compressibility'][group_name] = {
                'mean': np.mean(group_compressibility) if group_compressibility else 0,
                'std': np.std(group_compressibility) if group_compressibility else 0,
                'distribution': group_compressibility
            }
            
            complexity_analysis['fractal_dimension'][group_name] = {
                'mean': np.mean(group_fractal_dim) if group_fractal_dim else 0,
                'std': np.std(group_fractal_dim) if group_fractal_dim else 0,
                'distribution': group_fractal_dim
            }
        
        return complexity_analysis
    
    def _estimate_fractal_dimension(self, image: np.ndarray, max_box_size: int = 32) -> float:
        """Estimate fractal dimension using box counting method."""
        try:
            # Threshold the image
            threshold = np.mean(image)
            binary_image = (image > threshold).astype(int)
            
            sizes = []
            counts = []
            
            # Different box sizes
            for box_size in range(1, min(max_box_size, min(image.shape) // 2)):
                count = 0
                for i in range(0, image.shape[0], box_size):
                    for j in range(0, image.shape[1], box_size):
                        box = binary_image[i:i+box_size, j:j+box_size]
                        if box.size > 0 and np.sum(box) > 0:
                            count += 1
                
                if count > 0:
                    sizes.append(box_size)
                    counts.append(count)
            
            if len(sizes) > 2:
                # Fit power law: count ~ size^(-dimension)
                log_sizes = np.log(sizes)
                log_counts = np.log(counts)
                slope, _ = np.polyfit(log_sizes, log_counts, 1)
                fractal_dimension = -slope
                return fractal_dimension
            
            return np.nan
        except:
            return np.nan
    
    def _analyze_frequency_patterns(self, group_latents: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Spectral analysis of latent representations.
        
        Analyzes frequency domain patterns to understand spatial/temporal frequencies.
        """
        frequency_analysis = {
            'spatial_frequency_spectrum': {},
            'temporal_frequency_spectrum': {},
            'dominant_frequencies': {},
            'frequency_entropy': {}
        }
        
        if not SCIPY_AVAILABLE:
            self.logger.warning("Scipy not available, skipping frequency analysis")
            return frequency_analysis
        
        for group_name, videos in group_latents.items():
            self.logger.info(f"Analyzing frequency patterns for {group_name}")
            
            group_spatial_spectra = []
            group_temporal_spectra = []
            group_dominant_freqs = []
            group_freq_entropies = []
            
            for video_data in videos:
                latents = video_data['latents']
                
                for latent in latents:
                    latent_np = latent.squeeze(0).numpy()  # [16, frames, H, W]
                    
                    # Spatial frequency analysis
                    spatial_spectra = []
                    for channel in range(latent_np.shape[0]):
                        for frame in range(latent_np.shape[1]):
                            spatial_map = latent_np[channel, frame]
                            
                            if spatial_map.shape[0] > 4 and spatial_map.shape[1] > 4:
                                # 2D FFT for spatial frequencies
                                fft_2d = fft2(spatial_map)
                                power_spectrum = np.abs(fftshift(fft_2d))**2
                                
                                # Average power spectrum
                                spatial_spectra.append(power_spectrum.flatten())
                    
                    if spatial_spectra:
                        # Average across all spatial maps
                        avg_spatial_spectrum = np.mean(spatial_spectra, axis=0)
                        group_spatial_spectra.append(avg_spatial_spectrum)
                        
                        # Frequency entropy
                        norm_spectrum = avg_spatial_spectrum / (np.sum(avg_spatial_spectrum) + 1e-10)
                        freq_entropy = entropy(norm_spectrum + 1e-10)
                        group_freq_entropies.append(freq_entropy)
                    
                    # Temporal frequency analysis
                    if latent_np.shape[1] > 4:  # Need enough frames
                        temporal_spectra = []
                        for channel in range(latent_np.shape[0]):
                            # Average across spatial dimensions for temporal signal
                            temporal_signal = np.mean(latent_np[channel], axis=(1, 2))
                            
                            # 1D FFT for temporal frequencies
                            fft_1d = np.fft.fft(temporal_signal)
                            power_spectrum = np.abs(fft_1d)**2
                            temporal_spectra.append(power_spectrum)
                        
                        if temporal_spectra:
                            avg_temporal_spectrum = np.mean(temporal_spectra, axis=0)
                            group_temporal_spectra.append(avg_temporal_spectrum)
                            
                            # Dominant frequency
                            dominant_freq_idx = np.argmax(avg_temporal_spectrum[1:]) + 1  # Skip DC
                            group_dominant_freqs.append(dominant_freq_idx)
            
            # Aggregate results
            if group_spatial_spectra:
                frequency_analysis['spatial_frequency_spectrum'][group_name] = {
                    'mean_spectrum': np.mean(group_spatial_spectra, axis=0).tolist(),
                    'spectrum_variance': np.var(group_spatial_spectra, axis=0).tolist(),
                    'peak_frequency_consistency': np.std([np.argmax(spectrum) for spectrum in group_spatial_spectra])
                }
            
            if group_temporal_spectra:
                frequency_analysis['temporal_frequency_spectrum'][group_name] = {
                    'mean_spectrum': np.mean(group_temporal_spectra, axis=0).tolist(),
                    'spectrum_variance': np.var(group_temporal_spectra, axis=0).tolist(),
                    'peak_frequency_consistency': np.std([np.argmax(spectrum) for spectrum in group_temporal_spectra])
                }
            
            frequency_analysis['dominant_frequencies'][group_name] = {
                'mean': np.mean(group_dominant_freqs) if group_dominant_freqs else 0,
                'std': np.std(group_dominant_freqs) if group_dominant_freqs else 0,
                'distribution': group_dominant_freqs
            }
            
            frequency_analysis['frequency_entropy'][group_name] = {
                'mean': np.mean(group_freq_entropies) if group_freq_entropies else 0,
                'std': np.std(group_freq_entropies) if group_freq_entropies else 0,
                'distribution': group_freq_entropies
            }
        
        return frequency_analysis
    
    def _analyze_group_separability(self, group_latents: Dict[str, List[Dict[str, Any]]], 
                                  prompt_groups: List[str]) -> Dict[str, Any]:
        """
        Analyze how well different prompt groups separate in various feature spaces.
        """
        separability_analysis = {
            'feature_space_separation': {},
            'classification_accuracy': {},
            'distance_based_separation': {},
            'manifold_separation': {}
        }
        
        # Collect features for all groups
        all_features = []
        all_labels = []
        
        feature_extractors = {
            'spatial_variance': lambda latent: [np.var(latent[0, c, f]) for c in range(latent.shape[1]) for f in range(latent.shape[2])],
            'temporal_variance': lambda latent: [np.var(latent[0, c]) for c in range(latent.shape[1])],
            'channel_energy': lambda latent: [np.sum(np.abs(latent[0, c])) for c in range(latent.shape[1])],
            'global_statistics': lambda latent: [np.mean(latent), np.std(latent), np.max(latent), np.min(latent)]
        }
        
        for group_name, videos in group_latents.items():
            for video_data in videos:
                latents = video_data['latents']
                
                for latent in latents:
                    # Extract multiple types of features
                    combined_features = []
                    
                    for feature_name, extractor in feature_extractors.items():
                        try:
                            features = extractor(latent)
                            combined_features.extend(features)
                        except:
                            continue
                    
                    if combined_features:
                        all_features.append(combined_features)
                        all_labels.append(group_name)
        
        if len(all_features) > 0 and len(set(all_labels)) > 1:
            # Convert to arrays
            features_array = np.array(all_features)
            
            # Handle different feature lengths by padding/truncating
            max_len = max(len(f) for f in all_features)
            features_padded = []
            for f in all_features:
                if len(f) < max_len:
                    padded = f + [0] * (max_len - len(f))
                else:
                    padded = f[:max_len]
                features_padded.append(padded)
            
            features_array = np.array(features_padded)
            
            # PCA for dimensionality reduction
            if features_array.shape[1] > 50:
                pca = PCA(n_components=50)
                features_pca = pca.fit_transform(features_array)
            else:
                features_pca = features_array
            
            # Distance-based separation
            unique_labels = list(set(all_labels))
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            numeric_labels = [label_to_idx[label] for label in all_labels]
            
            # Compute inter-group vs intra-group distances
            distances = pdist(features_pca)
            distance_matrix = squareform(distances)
            
            intra_group_distances = []
            inter_group_distances = []
            
            for i in range(len(all_labels)):
                for j in range(i + 1, len(all_labels)):
                    if all_labels[i] == all_labels[j]:
                        intra_group_distances.append(distance_matrix[i, j])
                    else:
                        inter_group_distances.append(distance_matrix[i, j])
            
            separability_analysis['distance_based_separation'] = {
                'intra_group_distance': {
                    'mean': np.mean(intra_group_distances) if intra_group_distances else 0,
                    'std': np.std(intra_group_distances) if intra_group_distances else 0
                },
                'inter_group_distance': {
                    'mean': np.mean(inter_group_distances) if inter_group_distances else 0,
                    'std': np.std(inter_group_distances) if inter_group_distances else 0
                },
                'separation_ratio': (np.mean(inter_group_distances) / 
                                   (np.mean(intra_group_distances) + 1e-10)) if intra_group_distances and inter_group_distances else 0
            }
            
            # Classification accuracy (simple test)
            if len(unique_labels) > 1 and features_pca.shape[0] > len(unique_labels) * 2:
                try:
                    from sklearn.ensemble import RandomForestClassifier
                    from sklearn.model_selection import cross_val_score
                    
                    rf = RandomForestClassifier(n_estimators=50, random_state=42)
                    scores = cross_val_score(rf, features_pca, numeric_labels, cv=min(5, len(unique_labels)))
                    
                    separability_analysis['classification_accuracy'] = {
                        'mean_accuracy': np.mean(scores),
                        'std_accuracy': np.std(scores),
                        'all_scores': scores.tolist()
                    }
                except:
                    pass
        
        return separability_analysis
    
    def _test_statistical_significance(self, group_latents: Dict[str, List[Dict[str, Any]]], 
                                     prompt_groups: List[str]) -> Dict[str, Any]:
        """
        Test statistical significance of differences between groups.
        """
        significance_analysis = {
            'group_comparison_tests': {},
            'effect_sizes': {},
            'multiple_testing_correction': {}
        }
        
        # Extract summary statistics for each group
        group_statistics = {}
        
        for group_name, videos in group_latents.items():
            group_vars = []
            group_means = []
            group_energies = []
            
            for video_data in videos:
                latents = video_data['latents']
                
                for latent in latents:
                    latent_np = latent.squeeze(0).numpy()
                    
                    group_vars.append(np.var(latent_np))
                    group_means.append(np.mean(latent_np))
                    group_energies.append(np.sum(np.abs(latent_np)))
            
            group_statistics[group_name] = {
                'variance': group_vars,
                'mean': group_means,
                'energy': group_energies
            }
        
        # Pairwise statistical tests
        group_names = list(group_statistics.keys())
        test_results = {}
        
        for metric in ['variance', 'mean', 'energy']:
            metric_tests = {}
            
            for i, group1 in enumerate(group_names):
                for j, group2 in enumerate(group_names[i+1:], i+1):
                    data1 = group_statistics[group1][metric]
                    data2 = group_statistics[group2][metric]
                    
                    if len(data1) > 1 and len(data2) > 1:
                        # Welch's t-test (unequal variances)
                        try:
                            t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                            
                            # Effect size (Cohen's d)
                            pooled_std = np.sqrt((np.var(data1) + np.var(data2)) / 2)
                            effect_size = (np.mean(data1) - np.mean(data2)) / (pooled_std + 1e-10)
                            
                            metric_tests[f"{group1}_vs_{group2}"] = {
                                't_statistic': float(t_stat) if not np.isnan(t_stat) else 0,
                                'p_value': float(p_value) if not np.isnan(p_value) else 1,
                                'effect_size': float(effect_size) if not np.isnan(effect_size) else 0,
                                'significant': p_value < 0.05 if not np.isnan(p_value) else False
                            }
                        except:
                            continue
            
            test_results[metric] = metric_tests
        
        significance_analysis['group_comparison_tests'] = test_results
        
        # Multiple testing correction
        all_p_values = []
        test_names = []
        
        for metric, tests in test_results.items():
            for test_name, results in tests.items():
                all_p_values.append(results['p_value'])
                test_names.append(f"{metric}_{test_name}")
        
        if all_p_values:
            # Bonferroni correction
            corrected_p_values = [min(p * len(all_p_values), 1.0) for p in all_p_values]
            
            significance_analysis['multiple_testing_correction'] = {
                'original_p_values': all_p_values,
                'bonferroni_corrected': corrected_p_values,
                'test_names': test_names,
                'significant_after_correction': [p < 0.05 for p in corrected_p_values]
            }
        
        return significance_analysis
    
    def _save_analysis_results(self, results: StructureAwareAnalysis) -> None:
        """Save comprehensive analysis results."""
        results_path = self.output_dir / "structure_aware_analysis_results.json"
        
        # Convert results to serializable format
        results_dict = {
            'analysis_metadata': {
                'timestamp': results.analysis_timestamp,
                'latent_shape': results.latent_shape,
                'groups_analyzed': results.groups_analyzed,
                'methodology': 'Structure-aware analysis preserving 3D video latent organization'
            },
            'spatial_patterns': results.spatial_patterns,
            'temporal_coherence': results.temporal_coherence,
            'channel_analysis': results.channel_analysis,
            'patch_diversity': results.patch_diversity,
            'global_structure': results.global_structure,
            'information_content': results.information_content,
            'complexity_measures': results.complexity_measures,
            'frequency_patterns': results.frequency_patterns,
            'group_separability': results.group_separability,
            'statistical_significance': results.statistical_significance
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=lambda x: float(x) if isinstance(x, np.number) else str(x))
        
        self.logger.info(f"Structure-aware analysis results saved to: {results_path}")
    
    def _create_comprehensive_visualizations(self, results: StructureAwareAnalysis, 
                                           group_latents: Dict[str, List[Dict[str, Any]]]) -> None:
        """Create comprehensive visualizations for structure-aware analysis."""
        self.logger.info("Creating structure-aware analysis visualizations...")
        
        # Set up plotting
        plt.style.use('default')
        sns.set_palette("husl")
        
        try:
            self._create_spatial_pattern_plots(results)
            self._create_temporal_coherence_plots(results)
            self._create_information_content_plots(results)
            self._create_separability_plots(results)
            self._create_summary_dashboard(results)
        except Exception as e:
            self.logger.error(f"Visualization creation failed: {e}")
    
    def _create_spatial_pattern_plots(self, results: StructureAwareAnalysis) -> None:
        """Create spatial pattern visualizations."""
        # Implementation details for visualization methods would go here
        pass
    
    def _create_temporal_coherence_plots(self, results: StructureAwareAnalysis) -> None:
        """Create temporal coherence visualizations."""
        pass
    
    def _create_information_content_plots(self, results: StructureAwareAnalysis) -> None:
        """Create information content visualizations."""
        pass
    
    def _create_separability_plots(self, results: StructureAwareAnalysis) -> None:
        """Create group separability visualizations."""
        pass
    
    def _create_summary_dashboard(self, results: StructureAwareAnalysis) -> None:
        """Create comprehensive summary dashboard."""
        pass
