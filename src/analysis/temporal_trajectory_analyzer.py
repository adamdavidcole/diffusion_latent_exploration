#!/usr/bin/env python3
"""
Temporal-Aware Latent Trajectory Analysis

This module implements a research-grade analysis framework that addresses the critical
limitation of simple averaging in trajectory analysis. It provides multi-dimensional
investigation of latent space traversal patterns with temporal awareness.

Key innovations:
1. Temporal dynamics analysis (trajectory evolution over diffusion timesteps)
2. Multi-resolution spatial analysis (PCA space + raw dimensions)
3. Comprehensive correlation metrics (Pearson, Spearman, DTW)
4. Phase-specific behavior analysis (early/mid/late diffusion phases)
5. Validation against ground truth specificity gradients
"""

import numpy as np
import torch
import pandas as pd
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
    from scipy import stats
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, dendrogram
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from dtaidistance import dtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False


@dataclass
class TemporalTrajectoryAnalysis:
    """Container for comprehensive temporal trajectory analysis results."""
    
    # Multi-level analysis results
    scalar_metrics: Dict[str, Any]
    temporal_dynamics: Dict[str, Any]
    spatial_analysis: Dict[str, Any]
    correlation_analysis: Dict[str, Any]
    phase_analysis: Dict[str, Any]
    
    # Validation results
    specificity_validation: Dict[str, Any]
    methodological_comparison: Dict[str, Any]
    
    # Visualization paths
    visualization_paths: List[str]
    
    # Metadata
    analysis_timestamp: str
    groups_analyzed: List[str]
    total_trajectories: int


class TemporalTrajectoryAnalyzer:
    """
    Research-grade analyzer for latent trajectory temporal dynamics.
    
    This analyzer addresses the fundamental limitation that simple averaging
    destroys the most important information about trajectory patterns.
    """
    
    def __init__(self, latents_dir: str, output_dir: Optional[str] = None):
        """Initialize the temporal trajectory analyzer."""
        self.latents_dir = Path(latents_dir)
        self.output_dir = Path(output_dir) if output_dir else self.latents_dir / "temporal_analysis"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized TemporalTrajectoryAnalyzer for: {latents_dir}")
        
        # Analysis parameters
        self.phase_boundaries = {
            'early': (1000, 750),   # High noise removal
            'middle': (750, 250),   # Structure formation  
            'late': (250, 0)        # Detail refinement
        }
        
        # Import the base analyzer for data loading
        try:
            from .latent_trajectory_analysis import LatentTrajectoryAnalyzer
            self.base_analyzer = LatentTrajectoryAnalyzer(latents_dir)
        except ImportError:
            self.logger.error("Could not import base LatentTrajectoryAnalyzer")
            raise
    
    def analyze_temporal_specificity_gradient(self, 
                                            prompts_specificity_order: List[str],
                                            prompt_descriptions: List[str]) -> TemporalTrajectoryAnalysis:
        """
        Analyze a gradient of prompt specificity using temporal-aware methods.
        
        This is the main research function that implements all methodological
        approaches to reveal how prompt specificity affects latent traversal.
        
        Args:
            prompts_specificity_order: List of prompt dirs in order of increasing specificity
            prompt_descriptions: Human-readable descriptions of each prompt
            
        Returns:
            Comprehensive analysis results with temporal awareness
        """
        self.logger.info(f"Analyzing specificity gradient: {len(prompts_specificity_order)} levels")
        
        # 1. Load all trajectory data with temporal preservation
        group_trajectories = self._load_temporal_trajectory_data(prompts_specificity_order)
        
        # 2. Multi-level analysis
        scalar_metrics = self._analyze_scalar_metrics_with_caveats(group_trajectories)
        temporal_dynamics = self._analyze_temporal_dynamics(group_trajectories)
        spatial_analysis = self._analyze_spatial_distributions(group_trajectories)
        correlation_analysis = self._analyze_trajectory_correlations(group_trajectories)
        phase_analysis = self._analyze_phase_specific_behavior(group_trajectories)
        
        # 3. Validation against specificity gradient
        specificity_validation = self._validate_specificity_gradient(
            group_trajectories, prompts_specificity_order, prompt_descriptions
        )
        
        # 4. Methodological comparison
        methodological_comparison = self._compare_analysis_methods(group_trajectories)
        
        # 5. Create comprehensive visualizations
        viz_paths = self._create_temporal_visualizations(
            group_trajectories, temporal_dynamics, spatial_analysis, 
            correlation_analysis, phase_analysis, specificity_validation
        )
        
        # 6. Package results
        results = TemporalTrajectoryAnalysis(
            scalar_metrics=scalar_metrics,
            temporal_dynamics=temporal_dynamics,
            spatial_analysis=spatial_analysis,
            correlation_analysis=correlation_analysis,
            phase_analysis=phase_analysis,
            specificity_validation=specificity_validation,
            methodological_comparison=methodological_comparison,
            visualization_paths=viz_paths,
            analysis_timestamp=str(pd.Timestamp.now()),
            groups_analyzed=prompts_specificity_order,
            total_trajectories=sum(len(trajs) for trajs in group_trajectories.values())
        )
        
        # 7. Save comprehensive results
        self._save_temporal_analysis_results(results)
        
        return results
    
    def _load_temporal_trajectory_data(self, prompt_groups: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load trajectory data preserving temporal information.
        
        This is critical - we need the full trajectory sequences, not just summaries.
        """
        group_trajectories = {}
        
        for group_name in prompt_groups:
            self.logger.info(f"Loading trajectories for group: {group_name}")
            
            # Discover videos in this prompt group
            video_ids = self.base_analyzer.discover_videos_in_prompt(group_name)
            group_trajs = []
            
            for video_id in video_ids:
                try:
                    # Load full trajectory with metadata
                    latents, metadata = self.base_analyzer.load_video_trajectory(video_id)
                    
                    if len(latents) > 0:
                        trajectory_data = {
                            'video_id': video_id,
                            'latents': latents,  # Full sequence
                            'metadata': metadata,  # Timestep info
                            'timesteps': [meta.timestep for meta in metadata],
                            'shape': latents[0].shape,
                            'sequence_length': len(latents)
                        }
                        group_trajs.append(trajectory_data)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to load trajectory for {video_id}: {e}")
            
            group_trajectories[group_name] = group_trajs
            self.logger.info(f"Loaded {len(group_trajs)} trajectories for {group_name}")
        
        return group_trajectories
    
    def _analyze_scalar_metrics_with_caveats(self, group_trajectories: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Compute traditional scalar metrics but with explicit warnings about limitations.
        """
        scalar_analysis = {
            'methodology_caveat': (
                "WARNING: Scalar metrics average across trajectories and timesteps, "
                "potentially masking critical temporal patterns and individual trajectory diversity. "
                "Use these only as baseline comparisons alongside temporal analysis."
            ),
            'metrics': {}
        }
        
        # Compute scalar summaries for each group
        for group_name, trajectories in group_trajectories.items():
            group_scalars = []
            
            for traj_data in trajectories:
                latents = traj_data['latents']
                
                # Compute scalar metrics (with temporal information lost)
                scalar_metrics = self._compute_scalar_trajectory_metrics(latents)
                group_scalars.append(scalar_metrics)
            
            # Aggregate across group (further information loss)
            if group_scalars:
                scalar_analysis['metrics'][group_name] = {
                    'n_trajectories': len(group_scalars),
                    'metrics': self._aggregate_scalar_metrics(group_scalars)
                }
        
        return scalar_analysis
    
    def _compute_scalar_trajectory_metrics(self, latents: List[torch.Tensor]) -> Dict[str, float]:
        """
        Compute scalar metrics for a single trajectory.
        WARNING: This method destroys temporal information and should only be used 
        for compatibility. Use temporal analysis methods instead.
        """
        if not latents:
            return {}
        
        # Convert to numpy for analysis
        latent_arrays = [latent.numpy() if torch.is_tensor(latent) else latent for latent in latents]
        
        # Compute basic statistics (temporal information lost)
        flattened_latents = [arr.flatten() for arr in latent_arrays]
        
        # Trajectory length
        total_displacement = 0.0
        for i in range(1, len(latent_arrays)):
            displacement = np.linalg.norm(latent_arrays[i] - latent_arrays[i-1])
            total_displacement += displacement
        
        # Variance measures
        all_values = np.concatenate(flattened_latents)
        
        return {
            'total_displacement': float(total_displacement),
            'mean_step_displacement': float(total_displacement / max(1, len(latent_arrays) - 1)),
            'variance': float(np.var(all_values)),
            'std': float(np.std(all_values)),
            'num_steps': len(latent_arrays)
        }
    
    def _aggregate_scalar_metrics(self, group_scalars: List[Dict]) -> Dict[str, float]:
        """
        Aggregate scalar metrics across trajectories in a group.
        WARNING: This further destroys trajectory-specific information.
        """
        if not group_scalars:
            return {}
        
        # Get all metric keys
        all_keys = set()
        for metrics in group_scalars:
            all_keys.update(metrics.keys())
        
        aggregated = {}
        for key in all_keys:
            values = [metrics.get(key, 0) for metrics in group_scalars if key in metrics]
            if values:
                aggregated[f'{key}_mean'] = float(np.mean(values))
                aggregated[f'{key}_std'] = float(np.std(values))
                aggregated[f'{key}_min'] = float(np.min(values))
                aggregated[f'{key}_max'] = float(np.max(values))
        
        return aggregated
    
    def _analyze_temporal_dynamics(self, group_trajectories: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Analyze how trajectory properties evolve over diffusion timesteps.
        
        This is the KEY innovation that preserves temporal information.
        """
        temporal_analysis = {}
        
        for group_name, trajectories in group_trajectories.items():
            self.logger.info(f"Computing temporal dynamics for {group_name}")
            
            # Extract temporal sequences for each trajectory
            group_temporal_data = []
            
            for traj_data in trajectories:
                latents = traj_data['latents']
                timesteps = traj_data['timesteps']
                
                # Compute time-series of trajectory properties
                temporal_properties = self._compute_temporal_properties(latents, timesteps)
                group_temporal_data.append(temporal_properties)
            
            # Analyze temporal patterns across group
            temporal_analysis[group_name] = self._analyze_group_temporal_patterns(group_temporal_data)
        
        # Cross-group temporal comparison
        temporal_analysis['cross_group_analysis'] = self._compare_temporal_patterns_across_groups(temporal_analysis)
        
        return temporal_analysis
    
    def _compute_temporal_properties(self, latents: List[np.ndarray], timesteps: List[int]) -> Dict[str, List[float]]:
        """Compute time-series of properties for a single trajectory."""
        
        # Initialize time series
        properties = {
            'timesteps': timesteps,
            'variance_sequence': [],
            'mean_magnitude_sequence': [],
            'velocity_sequence': [],
            'acceleration_sequence': [],
            'pca_variance_explained': [],
            'distance_from_start': []
        }
        
        # Compute at each timestep
        for i, latent in enumerate(latents):
            # Convert to numpy if needed
            if torch.is_tensor(latent):
                flat_latent = latent.numpy().flatten()
            else:
                flat_latent = latent.flatten()
            
            # Basic statistics
            properties['variance_sequence'].append(np.var(flat_latent))
            properties['mean_magnitude_sequence'].append(np.mean(np.abs(flat_latent)))
            
            # Distance from start
            if i == 0:
                start_latent = flat_latent
                properties['distance_from_start'].append(0.0)
            else:
                dist = np.linalg.norm(flat_latent - start_latent)
                properties['distance_from_start'].append(dist)
            
            # Velocity (movement between timesteps)
            if i > 0:
                if torch.is_tensor(latents[i-1]):
                    prev_latent = latents[i-1].numpy().flatten()
                else:
                    prev_latent = latents[i-1].flatten()
                velocity = np.linalg.norm(flat_latent - prev_latent)
                properties['velocity_sequence'].append(velocity)
            
            # Acceleration (change in velocity)
            if i > 1 and len(properties['velocity_sequence']) >= 2:
                current_vel = properties['velocity_sequence'][-1]
                prev_vel = properties['velocity_sequence'][-2]
                acceleration = abs(current_vel - prev_vel)
                properties['acceleration_sequence'].append(acceleration)
        
        return properties
    
    def _analyze_group_temporal_patterns(self, group_temporal_data: List[Dict[str, List[float]]]) -> Dict[str, Any]:
        """Analyze temporal patterns across all trajectories in a group."""
        
        if not group_temporal_data:
            return {}
        
        # Synchronize timesteps (all should be same, but check)
        common_timesteps = group_temporal_data[0]['timesteps']
        n_timesteps = len(common_timesteps)
        
        # Aggregate temporal properties
        aggregated = {
            'timesteps': common_timesteps,
            'n_trajectories': len(group_temporal_data),
            'temporal_consistency': {},
            'phase_behavior': {}
        }
        
        # For each property, compute statistics across trajectories at each timestep
        properties = ['variance_sequence', 'velocity_sequence', 'distance_from_start']
        
        for prop in properties:
            # Extract sequences for all trajectories
            all_sequences = []
            for traj_data in group_temporal_data:
                if prop in traj_data and len(traj_data[prop]) > 0:
                    all_sequences.append(traj_data[prop])
            
            if all_sequences:
                # Compute mean and std at each timestep
                max_len = max(len(seq) for seq in all_sequences)
                
                timestep_means = []
                timestep_stds = []
                timestep_correlations = []
                
                for t in range(max_len):
                    values_at_t = []
                    for seq in all_sequences:
                        if t < len(seq):
                            values_at_t.append(seq[t])
                    
                    if len(values_at_t) > 1:
                        timestep_means.append(np.mean(values_at_t))
                        timestep_stds.append(np.std(values_at_t))
                        
                        # Consistency metric: coefficient of variation
                        cv = np.std(values_at_t) / np.mean(values_at_t) if np.mean(values_at_t) != 0 else float('inf')
                        timestep_correlations.append(1.0 / (1.0 + cv))  # Higher = more consistent
                
                aggregated['temporal_consistency'][prop] = {
                    'mean_sequence': timestep_means,
                    'std_sequence': timestep_stds,
                    'consistency_sequence': timestep_correlations,
                    'overall_consistency': np.mean(timestep_correlations) if timestep_correlations else 0.0
                }
        
        # Phase-specific analysis
        for phase_name, (start_t, end_t) in self.phase_boundaries.items():
            phase_indices = [i for i, t in enumerate(common_timesteps) if end_t <= t <= start_t]
            
            if phase_indices:
                phase_analysis = {}
                for prop in properties:
                    if prop in aggregated['temporal_consistency']:
                        phase_values = [aggregated['temporal_consistency'][prop]['consistency_sequence'][i] 
                                      for i in phase_indices if i < len(aggregated['temporal_consistency'][prop]['consistency_sequence'])]
                        if phase_values:
                            phase_analysis[prop] = {
                                'mean_consistency': np.mean(phase_values),
                                'consistency_trend': np.polyfit(range(len(phase_values)), phase_values, 1)[0] if len(phase_values) > 1 else 0.0
                            }
                
                aggregated['phase_behavior'][phase_name] = phase_analysis
        
        return aggregated
    
    def _compare_temporal_patterns_across_groups(self, temporal_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compare temporal patterns between groups."""
        
        cross_group = {
            'consistency_ranking': {},
            'phase_differences': {},
            'temporal_divergence_points': {}
        }
        
        # Rank groups by overall temporal consistency
        consistency_scores = {}
        for group_name, group_data in temporal_analysis.items():
            if group_name != 'cross_group_analysis' and 'temporal_consistency' in group_data:
                scores = []
                for prop, prop_data in group_data['temporal_consistency'].items():
                    if 'overall_consistency' in prop_data:
                        scores.append(prop_data['overall_consistency'])
                
                if scores:
                    consistency_scores[group_name] = np.mean(scores)
        
        # Sort by consistency (higher = more similar trajectories within group)
        sorted_groups = sorted(consistency_scores.items(), key=lambda x: x[1], reverse=True)
        cross_group['consistency_ranking'] = {
            'ranking': sorted_groups,
            'interpretation': (
                "Higher consistency = trajectories within group follow similar patterns. "
                "Specific prompts should show higher consistency than random prompts."
            )
        }
        
        return cross_group
    
    def _analyze_spatial_distributions(self, group_trajectories: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze spatial distribution patterns in multiple coordinate systems."""
        
        spatial_analysis = {
            'raw_latent_space': {},
            'pca_space': {},
            'trajectory_clustering': {}
        }
        
        # Collect all latent states for global PCA
        all_latents = []
        group_labels = []
        trajectory_labels = []
        
        for group_name, trajectories in group_trajectories.items():
            for i, traj_data in enumerate(trajectories):
                for latent in traj_data['latents']:
                    all_latents.append(latent.flatten())
                    group_labels.append(group_name)
                    trajectory_labels.append(f"{group_name}_traj_{i}")
        
        if not all_latents:
            return spatial_analysis
        
        # Convert to array
        all_latents_array = np.array(all_latents)
        
        # Global PCA analysis
        if SCIPY_AVAILABLE:
            # Multiple PCA dimensions for exploration
            pca_dims = [2, 3, 10, 50]
            
            for n_dims in pca_dims:
                if n_dims <= all_latents_array.shape[1]:
                    pca = PCA(n_components=n_dims)
                    pca_coords = pca.fit_transform(all_latents_array)
                    
                    # Analyze group separation in PCA space
                    group_separation = self._analyze_group_separation_pca(
                        pca_coords, group_labels, n_dims
                    )
                    
                    spatial_analysis['pca_space'][f'{n_dims}d'] = {
                        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
                        'group_separation': group_separation
                    }
        
        # Trajectory clustering analysis
        spatial_analysis['trajectory_clustering'] = self._analyze_trajectory_clustering(
            group_trajectories
        )
        
        return spatial_analysis
    
    def _analyze_trajectory_clustering(self, group_trajectories: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Analyze trajectory clustering in latent space.
        NOTE: This is spatial analysis that may lose some temporal information.
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        clustering_analysis = {}
        
        # Collect all trajectories with group labels
        all_trajectories = []
        all_labels = []
        trajectory_endpoints = []
        
        for group_name, trajectories in group_trajectories.items():
            for traj_data in trajectories:
                latents = traj_data['latents']
                
                # Use trajectory endpoint for clustering
                endpoint = latents[-1]
                if torch.is_tensor(endpoint):
                    endpoint = endpoint.numpy()
                trajectory_endpoints.append(endpoint.flatten())
                all_labels.append(group_name)
        
        if not trajectory_endpoints:
            return {'error': 'No trajectories to cluster'}
        
        # Convert to array
        endpoint_array = np.array(trajectory_endpoints)
        
        # Try different numbers of clusters
        cluster_scores = {}
        for n_clusters in range(2, min(10, len(group_trajectories) + 1)):
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(endpoint_array)
                
                # Silhouette score (higher is better)
                sil_score = silhouette_score(endpoint_array, cluster_labels)
                cluster_scores[n_clusters] = sil_score
            except:
                continue
        
        clustering_analysis['silhouette_scores'] = cluster_scores
        clustering_analysis['best_n_clusters'] = max(cluster_scores, key=cluster_scores.get) if cluster_scores else 2
        
        return clustering_analysis
    
    def _analyze_group_separation_pca(self, pca_coords: np.ndarray, group_labels: List[str], n_dims: int) -> Dict[str, Any]:
        """Analyze how well groups separate in PCA space."""
        
        unique_groups = list(set(group_labels))
        n_groups = len(unique_groups)
        
        if n_groups < 2:
            return {}
        
        # Compute group centroids
        centroids = {}
        for group in unique_groups:
            group_mask = [label == group for label in group_labels]
            group_coords = pca_coords[group_mask]
            centroids[group] = np.mean(group_coords, axis=0)
        
        # Inter-group distances
        inter_group_distances = {}
        for i, group1 in enumerate(unique_groups):
            for j, group2 in enumerate(unique_groups[i+1:], i+1):
                dist = np.linalg.norm(centroids[group1] - centroids[group2])
                inter_group_distances[f"{group1}_vs_{group2}"] = dist
        
        # Intra-group dispersions
        intra_group_dispersions = {}
        for group in unique_groups:
            group_mask = [label == group for label in group_labels]
            group_coords = pca_coords[group_mask]
            
            if len(group_coords) > 1:
                # Average distance from centroid
                centroid = centroids[group]
                distances = [np.linalg.norm(coord - centroid) for coord in group_coords]
                intra_group_dispersions[group] = {
                    'mean_distance_from_centroid': np.mean(distances),
                    'std_distance_from_centroid': np.std(distances)
                }
        
        # Silhouette analysis (if scipy available)
        silhouette_scores = {}
        if SCIPY_AVAILABLE and len(unique_groups) > 1:
            try:
                # Convert group labels to numeric
                group_numeric = [unique_groups.index(label) for label in group_labels]
                silhouette_avg = silhouette_score(pca_coords, group_numeric)
                silhouette_scores['average'] = silhouette_avg
                
                # Per-group silhouette scores
                from sklearn.metrics import silhouette_samples
                sample_scores = silhouette_samples(pca_coords, group_numeric)
                
                for i, group in enumerate(unique_groups):
                    group_mask = [label == group for label in group_labels]
                    group_silhouette_scores = sample_scores[group_mask]
                    silhouette_scores[group] = {
                        'mean': np.mean(group_silhouette_scores),
                        'std': np.std(group_silhouette_scores)
                    }
                    
            except Exception as e:
                self.logger.warning(f"Silhouette analysis failed: {e}")
        
        return {
            'centroids': {group: centroid.tolist() for group, centroid in centroids.items()},
            'inter_group_distances': inter_group_distances,
            'intra_group_dispersions': intra_group_dispersions,
            'silhouette_analysis': silhouette_scores,
            'separation_quality': {
                'avg_inter_group_distance': np.mean(list(inter_group_distances.values())),
                'avg_intra_group_dispersion': np.mean([d['mean_distance_from_centroid'] 
                                                     for d in intra_group_dispersions.values() if d])
            }
        }
    
    def _analyze_trajectory_correlations(self, group_trajectories: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Comprehensive trajectory correlation analysis using multiple methods.
        
        This implements Pearson, Spearman, and DTW correlations to explore
        different aspects of trajectory similarity.
        """
        correlation_analysis = {
            'methodology_note': (
                "Multiple correlation methods reveal different aspects: "
                "Pearson (linear relationships), Spearman (monotonic relationships), "
                "DTW (temporal pattern similarity allowing for time warping)"
            ),
            'intra_group_correlations': {},
            'inter_group_correlations': {},
            'correlation_method_comparison': {}
        }
        
        # For each group, compute intra-group correlations
        for group_name, trajectories in group_trajectories.items():
            if len(trajectories) < 2:
                continue
                
            self.logger.info(f"Computing correlations for {group_name} ({len(trajectories)} trajectories)")
            
            # Extract comparable trajectory features
            trajectory_features = []
            for traj_data in trajectories:
                features = self._extract_trajectory_features_for_correlation(traj_data)
                if features is not None:
                    trajectory_features.append(features)
            
            if len(trajectory_features) >= 2:
                intra_correlations = self._compute_trajectory_correlations(
                    trajectory_features, methods=['pearson', 'spearman']
                )
                correlation_analysis['intra_group_correlations'][group_name] = intra_correlations
        
        # Cross-group correlation analysis
        all_group_features = {}
        for group_name, trajectories in group_trajectories.items():
            group_features = []
            for traj_data in trajectories:
                features = self._extract_trajectory_features_for_correlation(traj_data)
                if features is not None:
                    group_features.append(features)
            if group_features:
                all_group_features[group_name] = group_features
        
        # Compare groups pairwise
        group_names = list(all_group_features.keys())
        for i, group1 in enumerate(group_names):
            for j, group2 in enumerate(group_names[i+1:], i+1):
                inter_correlations = self._compute_inter_group_correlations(
                    all_group_features[group1], all_group_features[group2]
                )
                correlation_analysis['inter_group_correlations'][f"{group1}_vs_{group2}"] = inter_correlations
        
        # DTW analysis (if available)
        if DTW_AVAILABLE:
            correlation_analysis['dtw_analysis'] = self._compute_dtw_correlations(group_trajectories)
        
        return correlation_analysis
    
    def _compute_dtw_correlations(self, group_trajectories: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Compute Dynamic Time Warping correlations for temporal pattern analysis."""
        dtw_analysis = {
            'methodology_note': (
                "DTW (Dynamic Time Warping) measures similarity between temporal patterns "
                "allowing for time shifts and warping. This reveals trajectory shape similarity "
                "beyond simple correlation."
            ),
            'intra_group_dtw': {},
            'inter_group_dtw': {}
        }
        
        try:
            # For each group, compute intra-group DTW similarities
            for group_name, trajectories in group_trajectories.items():
                if len(trajectories) < 2:
                    continue
                
                # Extract temporal sequences for DTW
                temporal_sequences = []
                for traj_data in trajectories:
                    # Use distance_from_start as temporal sequence
                    temp_props = self._compute_temporal_properties(
                        traj_data['latents'], traj_data['timesteps']
                    )
                    if 'distance_from_start' in temp_props:
                        seq = temp_props['distance_from_start']
                        if len(seq) > 0:
                            temporal_sequences.append(seq)
                
                if len(temporal_sequences) >= 2:
                    # Compute pairwise DTW distances
                    dtw_distances = []
                    for i in range(len(temporal_sequences)):
                        for j in range(i+1, len(temporal_sequences)):
                            try:
                                distance = dtw.distance(temporal_sequences[i], temporal_sequences[j])
                                dtw_distances.append(distance)
                            except Exception as e:
                                self.logger.warning(f"DTW computation failed: {e}")
                    
                    if dtw_distances:
                        dtw_analysis['intra_group_dtw'][group_name] = {
                            'mean_dtw_distance': np.mean(dtw_distances),
                            'std_dtw_distance': np.std(dtw_distances),
                            'min_dtw_distance': np.min(dtw_distances),
                            'max_dtw_distance': np.max(dtw_distances),
                            'n_pairs': len(dtw_distances)
                        }
            
            # Cross-group DTW analysis would go here but is computationally expensive
            dtw_analysis['inter_group_dtw'] = {
                'note': 'Inter-group DTW analysis not computed due to computational cost'
            }
            
        except Exception as e:
            self.logger.warning(f"DTW analysis failed: {e}")
            dtw_analysis['error'] = str(e)
        
        return dtw_analysis
    
    def _extract_trajectory_features_for_correlation(self, traj_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract features from trajectory for correlation analysis."""
        try:
            latents = traj_data['latents']
            timesteps = traj_data['timesteps']
            
            # Multiple feature extraction strategies
            features = []
            
            # 1. Temporal property sequences
            temp_props = self._compute_temporal_properties(latents, timesteps)
            for prop in ['variance_sequence', 'velocity_sequence', 'distance_from_start']:
                if prop in temp_props:
                    # Normalize sequences to same length and scale
                    seq = np.array(temp_props[prop])
                    if len(seq) > 0:
                        # Z-score normalization
                        seq_norm = (seq - np.mean(seq)) / (np.std(seq) + 1e-8)
                        features.extend(seq_norm.tolist())
            
            # 2. PCA trajectory in reduced space
            flattened_latents = [latent.flatten() for latent in latents]
            if len(flattened_latents) > 1:
                latent_array = np.array(flattened_latents)
                
                # Simple PCA to extract trajectory shape
                from sklearn.decomposition import PCA
                pca = PCA(n_components=min(5, latent_array.shape[1], latent_array.shape[0]))
                pca_traj = pca.fit_transform(latent_array)
                
                # Add PCA trajectory features
                for dim in range(pca_traj.shape[1]):
                    dim_seq = pca_traj[:, dim]
                    # Normalize
                    dim_norm = (dim_seq - np.mean(dim_seq)) / (np.std(dim_seq) + 1e-8)
                    features.extend(dim_norm.tolist())
            
            return np.array(features) if features else None
            
        except Exception as e:
            self.logger.warning(f"Feature extraction failed for {traj_data.get('video_id', 'unknown')}: {e}")
            return None
    
    def _compute_trajectory_correlations(self, trajectory_features: List[np.ndarray], 
                                       methods: List[str]) -> Dict[str, Any]:
        """Compute correlations between trajectory features using specified methods."""
        
        correlations = {}
        n_trajectories = len(trajectory_features)
        
        for method in methods:
            method_correlations = []
            
            # Pairwise correlations
            for i in range(n_trajectories):
                for j in range(i+1, n_trajectories):
                    feat1, feat2 = trajectory_features[i], trajectory_features[j]
                    
                    # Ensure same length
                    min_len = min(len(feat1), len(feat2))
                    feat1_trim = feat1[:min_len]
                    feat2_trim = feat2[:min_len]
                    
                    if min_len > 1:
                        try:
                            if method == 'pearson':
                                corr, p_val = stats.pearsonr(feat1_trim, feat2_trim)
                            elif method == 'spearman':
                                corr, p_val = stats.spearmanr(feat1_trim, feat2_trim)
                            else:
                                continue
                            
                            if not np.isnan(corr):
                                method_correlations.append(corr)
                                
                        except Exception as e:
                            self.logger.warning(f"Correlation computation failed: {e}")
            
            if method_correlations:
                correlations[method] = {
                    'pairwise_correlations': method_correlations,
                    'mean_correlation': np.mean(method_correlations),
                    'std_correlation': np.std(method_correlations),
                    'min_correlation': np.min(method_correlations),
                    'max_correlation': np.max(method_correlations),
                    'n_pairs': len(method_correlations)
                }
        
        return correlations
    
    def _compute_inter_group_correlations(self, group1_features: List[np.ndarray], 
                                        group2_features: List[np.ndarray]) -> Dict[str, Any]:
        """Compute correlations between trajectories from different groups."""
        
        inter_correlations = []
        
        for feat1 in group1_features:
            for feat2 in group2_features:
                # Ensure same length
                min_len = min(len(feat1), len(feat2))
                feat1_trim = feat1[:min_len]
                feat2_trim = feat2[:min_len]
                
                if min_len > 1:
                    try:
                        corr, _ = stats.pearsonr(feat1_trim, feat2_trim)
                        if not np.isnan(corr):
                            inter_correlations.append(corr)
                    except Exception:
                        continue
        
        if inter_correlations:
            return {
                'mean_inter_correlation': np.mean(inter_correlations),
                'std_inter_correlation': np.std(inter_correlations),
                'n_pairs': len(inter_correlations),
                'all_correlations': inter_correlations
            }
        else:
            return {'mean_inter_correlation': 0.0, 'n_pairs': 0}
    
    def _analyze_phase_specific_behavior(self, group_trajectories: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze behavior in different phases of the diffusion process."""
        
        phase_analysis = {}
        
        for phase_name, (start_t, end_t) in self.phase_boundaries.items():
            self.logger.info(f"Analyzing {phase_name} phase ({start_t} → {end_t})")
            
            phase_data = {}
            
            for group_name, trajectories in group_trajectories.items():
                group_phase_data = []
                
                for traj_data in trajectories:
                    timesteps = traj_data['timesteps']
                    latents = traj_data['latents']
                    
                    # Extract latents in this phase
                    phase_indices = [i for i, t in enumerate(timesteps) if end_t <= t <= start_t]
                    
                    if phase_indices:
                        phase_latents = [latents[i] for i in phase_indices]
                        phase_timesteps = [timesteps[i] for i in phase_indices]
                        
                        # Compute phase-specific metrics
                        phase_metrics = self._compute_phase_metrics(phase_latents, phase_timesteps)
                        group_phase_data.append(phase_metrics)
                
                if group_phase_data:
                    # Aggregate phase metrics for group
                    phase_data[group_name] = self._aggregate_phase_metrics(group_phase_data)
            
            phase_analysis[phase_name] = phase_data
        
        # Cross-phase analysis
        phase_analysis['cross_phase_comparison'] = self._compare_phases(phase_analysis)
        
        return phase_analysis
    
    def _compute_phase_metrics(self, phase_latents: List[np.ndarray], phase_timesteps: List[int]) -> Dict[str, float]:
        """Compute metrics specific to a phase of diffusion."""
        
        if len(phase_latents) < 2:
            return {}
        
        metrics = {}
        
        # Movement characteristics in this phase
        distances = []
        for i in range(1, len(phase_latents)):
            dist = np.linalg.norm(phase_latents[i].flatten() - phase_latents[i-1].flatten())
            distances.append(dist)
        
        if distances:
            metrics['mean_movement'] = np.mean(distances)
            metrics['movement_consistency'] = 1.0 / (1.0 + np.std(distances) / np.mean(distances))
        
        # Variance evolution in phase
        variances = []
        for latent in phase_latents:
            if torch.is_tensor(latent):
                flat_latent = latent.numpy().flatten()
            else:
                flat_latent = latent.flatten()
            variances.append(np.var(flat_latent))
        
        if len(variances) > 1:
            variance_trend = np.polyfit(range(len(variances)), variances, 1)[0]
            metrics['variance_trend'] = variance_trend
        
        # Total change in phase
        start_latent = phase_latents[0].flatten()
        end_latent = phase_latents[-1].flatten()
        metrics['total_phase_change'] = np.linalg.norm(end_latent - start_latent)
        
        return metrics
    
    def _aggregate_phase_metrics(self, group_phase_data: List[Dict[str, float]]) -> Dict[str, Any]:
        """Aggregate phase metrics across trajectories in a group."""
        
        if not group_phase_data:
            return {}
        
        aggregated = {'n_trajectories': len(group_phase_data)}
        
        # Get all metric names
        all_metrics = set()
        for phase_metrics in group_phase_data:
            all_metrics.update(phase_metrics.keys())
        
        # Aggregate each metric
        for metric in all_metrics:
            values = [pm[metric] for pm in group_phase_data if metric in pm]
            if values:
                aggregated[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return aggregated
    
    def _compare_phases(self, phase_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compare behavior across different phases."""
        
        phase_comparison = {}
        
        # For each group, compare metrics across phases
        phase_names = [name for name in phase_analysis.keys() if name != 'cross_phase_comparison']
        
        for group_name in set().union(*[phase_data.keys() for phase_data in 
                                       [phase_analysis[phase] for phase in phase_names]]):
            
            group_phase_comparison = {}
            
            # Extract metrics for this group across phases
            for metric in ['mean_movement', 'movement_consistency', 'total_phase_change']:
                metric_across_phases = {}
                
                for phase_name in phase_names:
                    if (group_name in phase_analysis[phase_name] and 
                        metric in phase_analysis[phase_name][group_name]):
                        metric_value = phase_analysis[phase_name][group_name][metric]['mean']
                        metric_across_phases[phase_name] = metric_value
                
                if len(metric_across_phases) > 1:
                    group_phase_comparison[metric] = metric_across_phases
            
            if group_phase_comparison:
                phase_comparison[group_name] = group_phase_comparison
        
        return phase_comparison
    
    def _compare_analysis_methods(self, group_trajectories: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Compare temporal-aware analysis vs simple averaging to demonstrate methodology differences.
        This validates that temporal methods capture patterns that averaging destroys.
        """
        comparison_results = {
            'temporal_vs_averaging': {},
            'methodology_validation': {}
        }
        
        for group_name, trajectories in group_trajectories.items():
            if not trajectories:
                continue
            
            # Method 1: Simple averaging (destroys temporal information)
            all_latents = []
            for traj_data in trajectories:
                latents = traj_data['latents']
                for latent in latents:
                    if torch.is_tensor(latent):
                        all_latents.append(latent.numpy().flatten())
                    else:
                        all_latents.append(latent.flatten())
            
            if all_latents:
                averaged_representation = np.mean(all_latents, axis=0)
                averaged_variance = np.var(averaged_representation)
                
                # Method 2: Temporal-aware analysis (preserves time-series information)
                temporal_properties = []
                for traj_data in trajectories:
                    latents = traj_data['latents']
                    timesteps = traj_data['timesteps']
                    props = self._compute_temporal_properties(latents, timesteps)
                    temporal_properties.append(props)
                
                # Compare information content
                if temporal_properties:
                    # Temporal variance sequences (preserves time information)
                    variance_sequences = [props['variance_sequence'] for props in temporal_properties]
                    temporal_variance_diversity = np.std([np.std(seq) for seq in variance_sequences])
                    
                    # Distance evolution patterns
                    distance_sequences = [props['distance_from_start'] for props in temporal_properties]
                    distance_pattern_diversity = np.std([np.std(seq) for seq in distance_sequences])
                    
                    comparison_results['temporal_vs_averaging'][group_name] = {
                        'averaged_method': {
                            'variance': float(averaged_variance),
                            'information_content': 'single_scalar'
                        },
                        'temporal_method': {
                            'variance_sequence_diversity': float(temporal_variance_diversity),
                            'distance_pattern_diversity': float(distance_pattern_diversity),
                            'information_content': 'time_series_preserved'
                        },
                        'information_ratio': float(temporal_variance_diversity / max(averaged_variance, 1e-8))
                    }
        
        # Overall methodology validation
        info_ratios = []
        for group_data in comparison_results['temporal_vs_averaging'].values():
            if 'information_ratio' in group_data:
                info_ratios.append(group_data['information_ratio'])
        
        comparison_results['methodology_validation'] = {
            'mean_information_gain': float(np.mean(info_ratios)) if info_ratios else 0.0,
            'temporal_method_superiority': np.mean(info_ratios) > 1.0 if info_ratios else False,
            'explanation': 'Temporal analysis preserves time-series patterns that averaging destroys'
        }
        
        return comparison_results
    
    def _validate_specificity_gradient(self, group_trajectories: Dict[str, List[Dict[str, Any]]], 
                                     prompts_order: List[str], 
                                     prompt_descriptions: List[str]) -> Dict[str, Any]:
        """
        Validate that the analysis correctly identifies the specificity gradient.
        
        This is the key validation - do our methods correctly rank prompt specificity?
        """
        validation = {
            'ground_truth_order': list(zip(prompts_order, prompt_descriptions)),
            'consistency_ranking': {},
            'spatial_separation_ranking': {},
            'correlation_ranking': {},
            'combined_ranking': {},
            'validation_success': {}
        }
        
        # Expected pattern: higher specificity → higher intra-group consistency
        consistency_scores = {}
        for group_name in prompts_order:
            if group_name in group_trajectories:
                # Compute consistency score for this group
                score = self._compute_group_consistency_score(group_trajectories[group_name])
                consistency_scores[group_name] = score
        
        # Rank by consistency
        consistency_ranking = sorted(consistency_scores.items(), key=lambda x: x[1], reverse=True)
        validation['consistency_ranking'] = {
            'ranking': consistency_ranking,
            'expected_order': prompts_order,
            'correlation_with_expected': self._rank_correlation(
                [item[0] for item in consistency_ranking], prompts_order
            )
        }
        
        # Validate against expected gradient
        expected_order = prompts_order  # From random to specific
        observed_order = [item[0] for item in consistency_ranking]
        
        validation['validation_success'] = {
            'spearman_correlation': self._rank_correlation(observed_order, expected_order),
            'top_3_correct': observed_order[-3:] == expected_order[-3:],  # Most specific
            'bottom_3_correct': observed_order[:3] == expected_order[:3],  # Most random
            'monotonic_trend': self._check_monotonic_trend(consistency_scores, prompts_order)
        }
        
        return validation
    
    def _compute_group_consistency_score(self, trajectories: List[Dict[str, Any]]) -> float:
        """Compute overall consistency score for a group of trajectories."""
        
        if len(trajectories) < 2:
            return 0.0
        
        # Extract features for correlation
        features = []
        for traj_data in trajectories:
            feat = self._extract_trajectory_features_for_correlation(traj_data)
            if feat is not None:
                features.append(feat)
        
        if len(features) < 2:
            return 0.0
        
        # Compute pairwise correlations
        correlations = []
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                feat1, feat2 = features[i], features[j]
                min_len = min(len(feat1), len(feat2))
                
                if min_len > 1:
                    try:
                        corr, _ = stats.pearsonr(feat1[:min_len], feat2[:min_len])
                        if not np.isnan(corr):
                            correlations.append(abs(corr))  # Use absolute correlation
                    except Exception:
                        continue
        
        return np.mean(correlations) if correlations else 0.0
    
    def _rank_correlation(self, list1: List[str], list2: List[str]) -> float:
        """Compute Spearman rank correlation between two orderings."""
        
        # Convert to rank indices
        try:
            ranks1 = [list1.index(item) for item in list2 if item in list1]
            ranks2 = list(range(len(ranks1)))
            
            if len(ranks1) > 1:
                corr, _ = stats.spearmanr(ranks1, ranks2)
                return corr if not np.isnan(corr) else 0.0
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _check_monotonic_trend(self, scores: Dict[str, float], expected_order: List[str]) -> bool:
        """Check if scores show monotonic trend with expected order."""
        
        ordered_scores = [scores.get(group, 0.0) for group in expected_order if group in scores]
        
        if len(ordered_scores) < 3:
            return False
        
        # Check if generally increasing (allowing some noise)
        increases = sum(1 for i in range(1, len(ordered_scores)) 
                       if ordered_scores[i] > ordered_scores[i-1])
        
        return increases / (len(ordered_scores) - 1) > 0.6  # 60% of comparisons should increase
    
    def _create_temporal_visualizations(self, group_trajectories: Dict[str, List[Dict[str, Any]]], 
                                      temporal_dynamics: Dict[str, Any],
                                      spatial_analysis: Dict[str, Any],
                                      correlation_analysis: Dict[str, Any],
                                      phase_analysis: Dict[str, Any],
                                      specificity_validation: Dict[str, Any]) -> List[str]:
        """Create comprehensive visualizations for temporal analysis."""
        
        viz_paths = []
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        try:
            # 1. Temporal Dynamics Visualization
            viz_paths.extend(self._create_temporal_dynamics_plots(temporal_dynamics))
            
            # 2. Spatial Analysis Visualization  
            viz_paths.extend(self._create_spatial_analysis_plots(spatial_analysis, group_trajectories))
            
            # 3. Correlation Analysis Visualization
            viz_paths.extend(self._create_correlation_plots(correlation_analysis))
            
            # 4. Phase Analysis Visualization
            viz_paths.extend(self._create_phase_analysis_plots(phase_analysis))
            
            # 5. Specificity Validation Visualization
            viz_paths.extend(self._create_specificity_validation_plots(specificity_validation))
            
            # 6. Comprehensive Summary Dashboard
            viz_paths.extend(self._create_summary_dashboard(
                temporal_dynamics, spatial_analysis, correlation_analysis, 
                phase_analysis, specificity_validation
            ))
            
        except Exception as e:
            self.logger.error(f"Visualization creation failed: {e}")
        
        return viz_paths
    
    def _create_temporal_dynamics_plots(self, temporal_dynamics: Dict[str, Any]) -> List[str]:
        """Create temporal dynamics visualizations."""
        viz_paths = []
        
        # Plot 1: Temporal Consistency Evolution
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            colors = plt.cm.Set1(np.linspace(0, 1, len(temporal_dynamics) - 1))  # -1 for cross_group_analysis
            
            properties = ['variance_sequence', 'velocity_sequence', 'distance_from_start']
            
            for i, prop in enumerate(properties[:3]):  # Plot first 3 properties
                ax = axes.flat[i]
                
                for j, (group_name, group_data) in enumerate(temporal_dynamics.items()):
                    if group_name == 'cross_group_analysis':
                        continue
                    
                    if ('temporal_consistency' in group_data and 
                        prop in group_data['temporal_consistency']):
                        
                        consistency_data = group_data['temporal_consistency'][prop]
                        mean_seq = consistency_data['mean_sequence']
                        std_seq = consistency_data['std_sequence']
                        
                        if len(mean_seq) > 0:
                            x = range(len(mean_seq))
                            ax.plot(x, mean_seq, 'o-', label=group_name, color=colors[j % len(colors)])
                            ax.fill_between(x, 
                                          np.array(mean_seq) - np.array(std_seq),
                                          np.array(mean_seq) + np.array(std_seq),
                                          alpha=0.2, color=colors[j % len(colors)])
                
                ax.set_title(f'{prop.replace("_", " ").title()} Over Time')
                ax.set_xlabel('Diffusion Step')
                ax.set_ylabel('Mean Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Plot 4: Consistency Ranking
            ax = axes.flat[3]
            if 'cross_group_analysis' in temporal_dynamics:
                cross_group = temporal_dynamics['cross_group_analysis']
                if 'consistency_ranking' in cross_group:
                    ranking = cross_group['consistency_ranking']['ranking']
                    groups = [item[0] for item in ranking]
                    scores = [item[1] for item in ranking]
                    
                    bars = ax.bar(range(len(groups)), scores, color=colors[:len(groups)])
                    ax.set_xticks(range(len(groups)))
                    ax.set_xticklabels(groups, rotation=45)
                    ax.set_title('Group Consistency Ranking')
                    ax.set_ylabel('Consistency Score')
                    
                    # Add value labels on bars
                    for bar, score in zip(bars, scores):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{score:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            temporal_path = self.output_dir / "temporal_dynamics_analysis.png"
            plt.savefig(temporal_path, dpi=300, bbox_inches='tight')
            plt.close()
            viz_paths.append(str(temporal_path))
            
        except Exception as e:
            self.logger.warning(f"Failed to create temporal dynamics plot: {e}")
        
        return viz_paths
    
    def _create_spatial_analysis_plots(self, spatial_analysis: Dict[str, Any], 
                                     group_trajectories: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Create spatial analysis visualizations."""
        viz_paths = []
        
        # Plot: PCA Space Group Separation
        try:
            if 'pca_space' in spatial_analysis:
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                
                # Plot 2D and 3D PCA results
                pca_dims = ['2d', '3d']
                
                for i, dim_key in enumerate(pca_dims):
                    if dim_key in spatial_analysis['pca_space']:
                        pca_data = spatial_analysis['pca_space'][dim_key]
                        
                        # Plot explained variance
                        ax = axes[i, 0]
                        explained_var = pca_data['explained_variance_ratio']
                        ax.bar(range(len(explained_var)), explained_var)
                        ax.set_title(f'{dim_key.upper()} PCA Explained Variance')
                        ax.set_xlabel('Component')
                        ax.set_ylabel('Explained Variance Ratio')
                        
                        # Plot group separation metrics
                        ax = axes[i, 1]
                        if 'group_separation' in pca_data:
                            sep_data = pca_data['group_separation']
                            if 'inter_group_distances' in sep_data:
                                distances = list(sep_data['inter_group_distances'].values())
                                labels = list(sep_data['inter_group_distances'].keys())
                                
                                bars = ax.bar(range(len(distances)), distances)
                                ax.set_xticks(range(len(labels)))
                                ax.set_xticklabels([l.replace('_vs_', '\nvs\n') for l in labels], 
                                                 rotation=45, ha='right')
                                ax.set_title(f'{dim_key.upper()} Inter-Group Distances')
                                ax.set_ylabel('Distance')
                
                plt.tight_layout()
                spatial_path = self.output_dir / "spatial_analysis.png"
                plt.savefig(spatial_path, dpi=300, bbox_inches='tight')
                plt.close()
                viz_paths.append(str(spatial_path))
                
        except Exception as e:
            self.logger.warning(f"Failed to create spatial analysis plot: {e}")
        
        return viz_paths
    
    def _create_correlation_plots(self, correlation_analysis: Dict[str, Any]) -> List[str]:
        """Create correlation analysis visualizations."""
        viz_paths = []
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Intra-group correlations
            ax = axes[0, 0]
            if 'intra_group_correlations' in correlation_analysis:
                intra_corrs = correlation_analysis['intra_group_correlations']
                
                groups = []
                mean_correlations = []
                std_correlations = []
                
                for group_name, group_corr in intra_corrs.items():
                    if 'pearson' in group_corr:
                        groups.append(group_name)
                        mean_correlations.append(group_corr['pearson']['mean_correlation'])
                        std_correlations.append(group_corr['pearson']['std_correlation'])
                
                if groups:
                    bars = ax.bar(range(len(groups)), mean_correlations, 
                                 yerr=std_correlations, capsize=5)
                    ax.set_xticks(range(len(groups)))
                    ax.set_xticklabels(groups, rotation=45)
                    ax.set_title('Intra-Group Trajectory Correlations')
                    ax.set_ylabel('Mean Pearson Correlation')
                    ax.grid(True, alpha=0.3)
                    
                    # Add value labels
                    for bar, mean_corr in zip(bars, mean_correlations):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{mean_corr:.3f}', ha='center', va='bottom')
            
            # Plot 2: Inter-group correlations heatmap
            ax = axes[0, 1]
            if 'inter_group_correlations' in correlation_analysis:
                inter_corrs = correlation_analysis['inter_group_correlations']
                
                # Create correlation matrix
                group_names = set()
                for comparison in inter_corrs.keys():
                    g1, g2 = comparison.split('_vs_')
                    group_names.update([g1, g2])
                
                group_names = sorted(list(group_names))
                n_groups = len(group_names)
                
                if n_groups > 1:
                    corr_matrix = np.zeros((n_groups, n_groups))
                    
                    for comparison, corr_data in inter_corrs.items():
                        g1, g2 = comparison.split('_vs_')
                        if g1 in group_names and g2 in group_names:
                            i, j = group_names.index(g1), group_names.index(g2)
                            mean_corr = corr_data.get('mean_inter_correlation', 0.0)
                            corr_matrix[i, j] = mean_corr
                            corr_matrix[j, i] = mean_corr
                    
                    # Set diagonal to 1.0 (self-correlation)
                    np.fill_diagonal(corr_matrix, 1.0)
                    
                    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                    ax.set_xticks(range(n_groups))
                    ax.set_yticks(range(n_groups))
                    ax.set_xticklabels(group_names, rotation=45)
                    ax.set_yticklabels(group_names)
                    ax.set_title('Inter-Group Correlation Matrix')
                    
                    # Add correlation values as text
                    for i in range(n_groups):
                        for j in range(n_groups):
                            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                         ha="center", va="center", color="black")
                    
                    plt.colorbar(im, ax=ax)
            
            # Plot 3: Correlation method comparison
            ax = axes[1, 0]
            if 'intra_group_correlations' in correlation_analysis:
                methods = ['pearson', 'spearman']
                method_scores = {method: [] for method in methods}
                groups = []
                
                for group_name, group_corr in correlation_analysis['intra_group_correlations'].items():
                    groups.append(group_name)
                    for method in methods:
                        if method in group_corr:
                            method_scores[method].append(group_corr[method]['mean_correlation'])
                        else:
                            method_scores[method].append(0.0)
                
                if groups:
                    x = np.arange(len(groups))
                    width = 0.35
                    
                    for i, method in enumerate(methods):
                        ax.bar(x + i*width, method_scores[method], width, 
                              label=method.title(), alpha=0.8)
                    
                    ax.set_xlabel('Groups')
                    ax.set_ylabel('Mean Correlation')
                    ax.set_title('Correlation Method Comparison')
                    ax.set_xticks(x + width/2)
                    ax.set_xticklabels(groups, rotation=45)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            correlation_path = self.output_dir / "correlation_analysis.png"
            plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
            plt.close()
            viz_paths.append(str(correlation_path))
            
        except Exception as e:
            self.logger.warning(f"Failed to create correlation plots: {e}")
        
        return viz_paths
    
    def _create_phase_analysis_plots(self, phase_analysis: Dict[str, Any]) -> List[str]:
        """Create phase-specific analysis visualizations.""" 
        viz_paths = []
        
        try:
            phases = ['early', 'middle', 'late']
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Movement across phases
            ax = axes[0, 0]
            groups = set()
            for phase in phases:
                if phase in phase_analysis:
                    groups.update(phase_analysis[phase].keys())
            
            groups = sorted(list(groups))
            
            if groups:
                x = np.arange(len(phases))
                width = 0.8 / len(groups)
                
                colors = plt.cm.Set1(np.linspace(0, 1, len(groups)))
                
                for i, group in enumerate(groups):
                    movement_values = []
                    for phase in phases:
                        if (phase in phase_analysis and 
                            group in phase_analysis[phase] and
                            'mean_movement' in phase_analysis[phase][group]):
                            movement_values.append(phase_analysis[phase][group]['mean_movement']['mean'])
                        else:
                            movement_values.append(0.0)
                    
                    ax.bar(x + i*width, movement_values, width, 
                          label=group, color=colors[i], alpha=0.8)
                
                ax.set_xlabel('Diffusion Phase')
                ax.set_ylabel('Mean Movement')
                ax.set_title('Movement Across Diffusion Phases')
                ax.set_xticks(x + width * (len(groups)-1) / 2)
                ax.set_xticklabels(phases)
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Plot 2: Movement consistency across phases
            ax = axes[0, 1]
            if groups:
                for i, group in enumerate(groups):
                    consistency_values = []
                    for phase in phases:
                        if (phase in phase_analysis and 
                            group in phase_analysis[phase] and
                            'movement_consistency' in phase_analysis[phase][group]):
                            consistency_values.append(phase_analysis[phase][group]['movement_consistency']['mean'])
                        else:
                            consistency_values.append(0.0)
                    
                    ax.plot(phases, consistency_values, 'o-', 
                           label=group, color=colors[i], linewidth=2, markersize=8)
                
                ax.set_xlabel('Diffusion Phase')
                ax.set_ylabel('Movement Consistency')
                ax.set_title('Movement Consistency Across Phases')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Plot 3: Cross-phase comparison
            ax = axes[1, 0]
            if 'cross_phase_comparison' in phase_analysis:
                cross_phase = phase_analysis['cross_phase_comparison']
                
                # Plot mean_movement evolution for each group
                for group_name, group_data in cross_phase.items():
                    if 'mean_movement' in group_data:
                        phases_data = group_data['mean_movement']
                        phase_names = list(phases_data.keys())
                        phase_values = list(phases_data.values())
                        
                        ax.plot(phase_names, phase_values, 'o-', 
                               label=group_name, linewidth=2, markersize=8)
                
                ax.set_xlabel('Diffusion Phase')
                ax.set_ylabel('Mean Movement')
                ax.set_title('Movement Evolution Across Phases')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            phase_path = self.output_dir / "phase_analysis.png"
            plt.savefig(phase_path, dpi=300, bbox_inches='tight')
            plt.close()
            viz_paths.append(str(phase_path))
            
        except Exception as e:
            self.logger.warning(f"Failed to create phase analysis plots: {e}")
        
        return viz_paths
    
    def _create_specificity_validation_plots(self, specificity_validation: Dict[str, Any]) -> List[str]:
        """Create specificity gradient validation visualizations."""
        viz_paths = []
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Consistency ranking vs expected
            ax = axes[0, 0]
            if 'consistency_ranking' in specificity_validation:
                ranking_data = specificity_validation['consistency_ranking']
                
                observed_ranking = [item[0] for item in ranking_data['ranking']]
                observed_scores = [item[1] for item in ranking_data['ranking']]
                expected_order = ranking_data['expected_order']
                
                # Plot observed ranking
                bars = ax.bar(range(len(observed_ranking)), observed_scores)
                ax.set_xticks(range(len(observed_ranking)))
                ax.set_xticklabels(observed_ranking, rotation=45)
                ax.set_title('Observed Consistency Ranking')
                ax.set_ylabel('Consistency Score')
                
                # Color bars based on expected position
                for i, (bar, group) in enumerate(zip(bars, observed_ranking)):
                    expected_pos = expected_order.index(group) if group in expected_order else -1
                    # Color by expected specificity (later in list = more specific = warmer color)
                    color_intensity = expected_pos / (len(expected_order) - 1) if expected_pos >= 0 else 0
                    bar.set_color(plt.cm.Reds(0.3 + 0.7 * color_intensity))
                
                # Add value labels
                for bar, score in zip(bars, observed_scores):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')
            
            # Plot 2: Expected vs Observed order
            ax = axes[0, 1]
            if 'consistency_ranking' in specificity_validation:
                ranking_data = specificity_validation['consistency_ranking']
                correlation = ranking_data.get('correlation_with_expected', 0.0)
                
                expected_order = ranking_data['expected_order']
                observed_ranking = [item[0] for item in ranking_data['ranking']]
                
                # Create scatter plot of expected vs observed positions
                expected_positions = []
                observed_positions = []
                labels = []
                
                for group in expected_order:
                    if group in observed_ranking:
                        expected_pos = expected_order.index(group)
                        observed_pos = observed_ranking.index(group)
                        expected_positions.append(expected_pos)
                        observed_positions.append(observed_pos)
                        labels.append(group)
                
                if expected_positions:
                    scatter = ax.scatter(expected_positions, observed_positions, 
                                       s=100, alpha=0.7, c=range(len(labels)), cmap='viridis')
                    
                    # Add labels
                    for exp_pos, obs_pos, label in zip(expected_positions, observed_positions, labels):
                        ax.annotate(label, (exp_pos, obs_pos), xytext=(5, 5), 
                                  textcoords='offset points', fontsize=8)
                    
                    # Add diagonal line (perfect correlation)
                    max_pos = max(max(expected_positions), max(observed_positions))
                    ax.plot([0, max_pos], [0, max_pos], 'r--', alpha=0.5, label='Perfect Correlation')
                    
                    ax.set_xlabel('Expected Position (Specificity Order)')
                    ax.set_ylabel('Observed Position (Consistency Ranking)')
                    ax.set_title(f'Expected vs Observed Ranking (r={correlation:.3f})')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            
            # Plot 3: Validation success metrics
            ax = axes[1, 0]
            if 'validation_success' in specificity_validation:
                success_data = specificity_validation['validation_success']
                
                metrics = []
                values = []
                
                for key, value in success_data.items():
                    if isinstance(value, (int, float)):
                        metrics.append(key.replace('_', '\n'))
                        values.append(float(value))
                    elif isinstance(value, bool):
                        metrics.append(key.replace('_', '\n'))
                        values.append(1.0 if value else 0.0)
                
                if metrics:
                    bars = ax.bar(range(len(metrics)), values)
                    ax.set_xticks(range(len(metrics)))
                    ax.set_xticklabels(metrics, rotation=45, ha='right')
                    ax.set_title('Validation Success Metrics')
                    ax.set_ylabel('Score / Success Rate')
                    ax.set_ylim(0, 1.1)
                    
                    # Color bars (green for good, red for bad)
                    for bar, value in zip(bars, values):
                        color = plt.cm.RdYlGn(value)
                        bar.set_color(color)
                    
                    # Add value labels
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                               f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            validation_path = self.output_dir / "specificity_validation.png"
            plt.savefig(validation_path, dpi=300, bbox_inches='tight')
            plt.close()
            viz_paths.append(str(validation_path))
            
        except Exception as e:
            self.logger.warning(f"Failed to create validation plots: {e}")
        
        return viz_paths
    
    def _create_summary_dashboard(self, temporal_dynamics: Dict[str, Any],
                                spatial_analysis: Dict[str, Any],
                                correlation_analysis: Dict[str, Any],
                                phase_analysis: Dict[str, Any],
                                specificity_validation: Dict[str, Any]) -> List[str]:
        """Create comprehensive summary dashboard."""
        viz_paths = []
        
        try:
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
            
            # Title
            fig.suptitle('Temporal Trajectory Analysis: Complete Research Dashboard', 
                        fontsize=20, fontweight='bold')
            
            # Key findings text
            ax_text = fig.add_subplot(gs[0, :])
            ax_text.axis('off')
            
            # Extract key findings
            key_findings = []
            
            # Validation success
            if 'validation_success' in specificity_validation:
                success = specificity_validation['validation_success']
                spearman_corr = success.get('spearman_correlation', 0.0)
                key_findings.append(f"Specificity Gradient Detection: r={spearman_corr:.3f}")
            
            # Consistency ranking
            if ('cross_group_analysis' in temporal_dynamics and 
                'consistency_ranking' in temporal_dynamics['cross_group_analysis']):
                ranking = temporal_dynamics['cross_group_analysis']['consistency_ranking']['ranking']
                most_consistent = ranking[0][0] if ranking else "Unknown"
                least_consistent = ranking[-1][0] if ranking else "Unknown"
                key_findings.append(f"Most Consistent Group: {most_consistent}")
                key_findings.append(f"Least Consistent Group: {least_consistent}")
            
            findings_text = "\n".join([f"• {finding}" for finding in key_findings])
            ax_text.text(0.5, 0.5, findings_text, transform=ax_text.transAxes,
                        fontsize=14, ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
            
            # Add smaller plots from other visualizations
            # This would include key plots from the detailed analysis
            
            plt.tight_layout()
            dashboard_path = self.output_dir / "research_dashboard.png"
            plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
            plt.close()
            viz_paths.append(str(dashboard_path))
            
        except Exception as e:
            self.logger.warning(f"Failed to create dashboard: {e}")
        
        return viz_paths
    
    def _save_temporal_analysis_results(self, results: TemporalTrajectoryAnalysis) -> None:
        """Save comprehensive analysis results."""
        
        # Convert to serializable format
        results_dict = {
            'analysis_metadata': {
                'timestamp': results.analysis_timestamp,
                'groups_analyzed': results.groups_analyzed,
                'total_trajectories': results.total_trajectories,
                'methodology': (
                    "Temporal-aware latent trajectory analysis addressing the fundamental "
                    "limitation that simple averaging destroys critical temporal patterns. "
                    "Implements multi-level analysis: scalar metrics (with caveats), "
                    "temporal dynamics, spatial distributions, correlation analysis, "
                    "and phase-specific behavior."
                )
            },
            'scalar_metrics': results.scalar_metrics,
            'temporal_dynamics': results.temporal_dynamics,
            'spatial_analysis': results.spatial_analysis,
            'correlation_analysis': results.correlation_analysis,
            'phase_analysis': results.phase_analysis,
            'specificity_validation': results.specificity_validation,
            'methodological_comparison': results.methodological_comparison,
            'visualization_paths': results.visualization_paths
        }
        
        # Save main results
        results_path = self.output_dir / "temporal_trajectory_analysis_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        self.logger.info(f"Temporal analysis results saved to: {results_path}")
        
        # Save summary report
        self._create_analysis_report(results)
    
    def _create_analysis_report(self, results: TemporalTrajectoryAnalysis) -> None:
        """Create human-readable analysis report."""
        
        report_path = self.output_dir / "temporal_analysis_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Temporal Latent Trajectory Analysis Report\n\n")
            f.write(f"**Analysis Date:** {results.analysis_timestamp}\n")
            f.write(f"**Groups Analyzed:** {len(results.groups_analyzed)}\n")
            f.write(f"**Total Trajectories:** {results.total_trajectories}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write("This analysis addresses the critical limitation that simple averaging ")
            f.write("of trajectory metrics destroys the most important temporal information. ")
            f.write("We implement a multi-dimensional investigation using temporal-aware methods.\n\n")
            
            # Validation results
            if results.specificity_validation and 'validation_success' in results.specificity_validation:
                f.write("## Specificity Gradient Validation\n\n")
                success = results.specificity_validation['validation_success']
                
                spearman_corr = success.get('spearman_correlation', 0.0)
                f.write(f"**Spearman Rank Correlation with Expected Order:** {spearman_corr:.3f}\n")
                
                if spearman_corr > 0.7:
                    f.write("✅ **STRONG validation** - Analysis correctly identifies specificity gradient\n")
                elif spearman_corr > 0.4:
                    f.write("⚠️ **MODERATE validation** - Analysis partially identifies specificity gradient\n")
                else:
                    f.write("❌ **WEAK validation** - Analysis struggles to identify specificity gradient\n")
                
                f.write(f"\n**Monotonic Trend Detected:** {success.get('monotonic_trend', False)}\n")
                f.write(f"**Top 3 Groups Correct:** {success.get('top_3_correct', False)}\n\n")
            
            # Key findings
            f.write("## Key Findings\n\n")
            
            # Consistency ranking
            if ('temporal_dynamics' in results.__dict__ and 
                'cross_group_analysis' in results.temporal_dynamics):
                cross_group = results.temporal_dynamics['cross_group_analysis']
                if 'consistency_ranking' in cross_group:
                    ranking = cross_group['consistency_ranking']['ranking']
                    f.write("### Trajectory Consistency Ranking\n")
                    f.write("(Higher consistency indicates more similar trajectories within group)\n\n")
                    for i, (group, score) in enumerate(ranking):
                        f.write(f"{i+1}. **{group}**: {score:.4f}\n")
                    f.write("\n")
            
            f.write("## Methodology Notes\n\n")
            f.write("- **Temporal Dynamics**: Preserves time-series information unlike scalar averaging\n")
            f.write("- **Phase Analysis**: Examines early/middle/late diffusion behavior separately\n") 
            f.write("- **Multi-correlation**: Uses Pearson, Spearman, and DTW for comprehensive similarity\n")
            f.write("- **Spatial Analysis**: Multiple PCA dimensions reveal different separation scales\n")
            f.write("- **Validation**: Tests against known specificity gradient for method validation\n\n")
            
            f.write(f"## Visualizations\n\n")
            for viz_path in results.visualization_paths:
                f.write(f"- {Path(viz_path).name}\n")
        
        self.logger.info(f"Analysis report saved to: {report_path}")
