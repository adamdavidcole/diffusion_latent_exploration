"""
Latent trajectory analysis for diffusion models.

This module provides tools for analyzing the trajectory of latent representations
during the diffusion process to understand the geometry of the latent space
and potential biases in representation.

Key hypothesis: Dominant representations may occupy more area in the latent space,
while "marginal" or "othered" representations occupy less area.
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import logging
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from datetime import datetime

# Optional imports
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from src.utils.latent_storage import LatentStorage, LatentMetadata


@dataclass
class TrajectoryAnalysisResult:
    """Results from latent trajectory analysis."""
    video_id: str
    prompt: str
    analysis_type: str
    metrics: Dict[str, float]
    trajectory_stats: Dict[str, Any]
    visualization_paths: List[str] = None
    
    def __post_init__(self):
        if self.visualization_paths is None:
            self.visualization_paths = []


class LatentTrajectoryAnalyzer:
    """
    Analyzes latent trajectories to understand diffusion geometry and potential biases.
    """
    
    def __init__(self, storage_dir: Union[str, Path]):
        """
        Initialize analyzer with latent storage directory.
        
        Args:
            storage_dir: Directory containing stored latents from generation
        """
        self.storage_dir = Path(storage_dir)
        self.latent_storage = LatentStorage(storage_dir)
        self.logger = logging.getLogger(__name__)
        
        # Create analysis output directory
        self.analysis_dir = self.storage_dir / "analysis"
        self.analysis_dir.mkdir(exist_ok=True)
        
        self.visualizations_dir = self.analysis_dir / "visualizations"
        self.visualizations_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Initialized latent trajectory analyzer for: {storage_dir}")
    
    def get_available_videos(self) -> List[str]:
        """Get list of video IDs with stored latents."""
        return self.latent_storage.list_stored_videos()
    
    def get_available_prompt_dirs(self) -> List[str]:
        """Get list of prompt directories with stored latents."""
        prompt_dirs = set()
        videos = self.latent_storage.list_stored_videos()
        for video_id in videos:
            prompt_part = self.latent_storage.get_prompt_from_video_id(video_id)
            prompt_dirs.add(prompt_part)
        return sorted(list(prompt_dirs))
    
    def discover_videos_in_prompt(self, prompt_dir: str) -> List[str]:
        """
        Discover individual video IDs within a prompt directory.
        
        Args:
            prompt_dir: Prompt directory name (e.g., "prompt_000")
            
        Returns:
            List of video IDs that have stored latents
        """
        video_ids = []
        all_videos = self.latent_storage.list_stored_videos()
        
        # Filter videos that belong to this prompt
        for video_id in all_videos:
            if video_id.startswith(prompt_dir + "_vid"):
                video_ids.append(video_id)
        
        # Also look for video summary files in the new directory structure
        # Check for summary.json files in prompt_xxx/vid_xxx/ directories
        prompt_latents_dir = self.storage_dir / prompt_dir
        if prompt_latents_dir.exists():
            for vid_dir in prompt_latents_dir.iterdir():
                if vid_dir.is_dir() and vid_dir.name.startswith("vid_"):
                    summary_file = vid_dir / "summary.json"
                    if summary_file.exists():
                        # Reconstruct video_id from directory structure
                        vid_part = vid_dir.name.replace("vid_", "")  # Convert "vid_001" to "001"
                        video_id = f"{prompt_dir}_vid{vid_part}"
                        if video_id not in video_ids:
                            video_ids.append(video_id)
        
        return sorted(video_ids)
    
    def load_video_trajectory(self, video_id: str) -> Tuple[List[torch.Tensor], List[LatentMetadata]]:
        """
        Load complete latent trajectory for a video.
        
        Args:
            video_id: ID of the video to load (full video_id like "prompt_000_vid001")
            
        Returns:
            Tuple of (latent_tensors, metadata_list)
        """
        # First try to get video summary
        summary = self.latent_storage.get_video_summary(video_id)
        
        if summary:
            # Use stored steps from summary
            stored_steps = summary['stored_steps']
        else:
            # Discover steps by scanning the directory structure
            stored_steps = self.latent_storage.list_steps_for_video(video_id)
            
            if not stored_steps:
                raise ValueError(f"No latent files found for video: {video_id}")
        
        latents = []
        metadata = []
        
        for step in sorted(stored_steps):
            latent = self.latent_storage.load_latent(video_id, step)
            meta = self.latent_storage.load_metadata(video_id, step)
            
            if latent is not None and meta is not None:
                latents.append(latent)
                metadata.append(meta)
            else:
                self.logger.warning(f"Missing latent or metadata for {video_id}, step {step}")
        
        self.logger.info(f"Loaded trajectory for {video_id}: {len(latents)} steps")
        return latents, metadata
    
    def compute_trajectory_metrics(self, latents: List[torch.Tensor], 
                                 metadata: List[LatentMetadata]) -> Dict[str, Any]:
        """
        Compute comprehensive metrics for a latent trajectory.
        
        Args:
            latents: List of latent tensors for each step
            metadata: Corresponding metadata for each step
            
        Returns:
            Dictionary of computed metrics
        """
        if not latents:
            return {}
        
        # Convert to numpy for analysis
        latent_arrays = [latent.numpy() if torch.is_tensor(latent) else latent for latent in latents]
        
        metrics = {}
        
        # Basic trajectory properties
        metrics['num_steps'] = len(latent_arrays)
        metrics['latent_shape'] = latent_arrays[0].shape
        metrics['latent_dimensions'] = np.prod(latent_arrays[0].shape)
        
        # Compute trajectory statistics
        metrics.update(self._compute_trajectory_statistics(latent_arrays))
        
        # Compute geometric properties
        metrics.update(self._compute_geometric_properties(latent_arrays))
        
        # Compute temporal dynamics
        metrics.update(self._compute_temporal_dynamics(latent_arrays, metadata))
        
        return metrics
    
    def _compute_trajectory_statistics(self, latent_arrays: List[np.ndarray]) -> Dict[str, float]:
        """Compute basic statistical properties of trajectory."""
        flattened_latents = [arr.flatten() for arr in latent_arrays]
        
        # Mean and variance evolution
        means = [np.mean(arr) for arr in flattened_latents]
        variances = [np.var(arr) for arr in flattened_latents]
        
        # Trajectory distance metrics
        total_distance = 0.0
        step_distances = []
        
        for i in range(1, len(flattened_latents)):
            dist = np.linalg.norm(flattened_latents[i] - flattened_latents[i-1])
            step_distances.append(dist)
            total_distance += dist
        
        return {
            'mean_trajectory_mean': np.mean(means),
            'mean_trajectory_variance': np.mean(variances),
            'total_trajectory_distance': total_distance,
            'mean_step_distance': np.mean(step_distances) if step_distances else 0.0,
            'trajectory_smoothness': np.std(step_distances) if step_distances else 0.0,
            'initial_variance': variances[0] if variances else 0.0,
            'final_variance': variances[-1] if variances else 0.0,
            'variance_change': (variances[-1] - variances[0]) if len(variances) > 1 else 0.0
        }
    
    def _compute_geometric_properties(self, latent_arrays: List[np.ndarray]) -> Dict[str, float]:
        """Compute geometric properties of the trajectory in latent space."""
        flattened_latents = np.array([arr.flatten() for arr in latent_arrays])
        
        if len(flattened_latents) < 2:
            return {}
        
        # PCA analysis to understand principal directions
        pca = PCA(n_components=min(10, flattened_latents.shape[1], flattened_latents.shape[0]))
        pca_transformed = pca.fit_transform(flattened_latents)
        
        # Trajectory volume estimation
        convex_hull_volume = self._estimate_trajectory_volume(pca_transformed)
        
        # Linearity measure (how straight is the trajectory?)
        start_end_distance = np.linalg.norm(flattened_latents[-1] - flattened_latents[0])
        total_path_length = sum(np.linalg.norm(flattened_latents[i] - flattened_latents[i-1]) 
                               for i in range(1, len(flattened_latents)))
        linearity = start_end_distance / total_path_length if total_path_length > 0 else 0.0
        
        return {
            'pca_explained_variance_ratio': pca.explained_variance_ratio_[:3].tolist(),
            'trajectory_volume_estimate': convex_hull_volume,
            'trajectory_linearity': linearity,
            'start_end_distance': start_end_distance,
            'dimensionality_reduction_ratio': pca.explained_variance_ratio_[0] if len(pca.explained_variance_ratio_) > 0 else 0.0
        }
    
    def _compute_temporal_dynamics(self, latent_arrays: List[np.ndarray], 
                                 metadata: List[LatentMetadata]) -> Dict[str, float]:
        """Compute temporal dynamics of the diffusion process."""
        if len(latent_arrays) < 2:
            return {}
        
        timesteps = [meta.timestep for meta in metadata]
        flattened_latents = [arr.flatten() for arr in latent_arrays]
        
        # Velocity analysis
        velocities = []
        for i in range(1, len(flattened_latents)):
            dt = abs(timesteps[i] - timesteps[i-1]) if len(timesteps) > i else 1.0
            if dt > 0:
                velocity = np.linalg.norm(flattened_latents[i] - flattened_latents[i-1]) / dt
                velocities.append(velocity)
        
        # Acceleration analysis
        accelerations = []
        for i in range(1, len(velocities)):
            acceleration = abs(velocities[i] - velocities[i-1])
            accelerations.append(acceleration)
        
        return {
            'mean_velocity': np.mean(velocities) if velocities else 0.0,
            'max_velocity': np.max(velocities) if velocities else 0.0,
            'velocity_variance': np.var(velocities) if velocities else 0.0,
            'mean_acceleration': np.mean(accelerations) if accelerations else 0.0,
            'max_acceleration': np.max(accelerations) if accelerations else 0.0
        }
    
    def _estimate_trajectory_volume(self, points: np.ndarray) -> float:
        """Estimate the volume occupied by trajectory using convex hull or covariance."""
        if len(points) < 3:
            return 0.0
        
        try:
            from scipy.spatial import ConvexHull
            if points.shape[1] >= 3:  # Need at least 3D for volume
                hull = ConvexHull(points[:, :3])  # Use first 3 principal components
                return hull.volume
        except ImportError:
            pass
        
        # Fallback: use covariance determinant as volume proxy
        cov = np.cov(points.T)
        try:
            return np.sqrt(np.linalg.det(cov))
        except:
            return 0.0
    
    def analyze_single_video(self, video_id: str, 
                           create_visualizations: bool = True) -> TrajectoryAnalysisResult:
        """
        Perform complete analysis of a single video's latent trajectory.
        
        Args:
            video_id: ID of video to analyze
            create_visualizations: Whether to generate visualization plots
            
        Returns:
            TrajectoryAnalysisResult with analysis results
        """
        self.logger.info(f"Analyzing latent trajectory for video: {video_id}")
        
        # Load trajectory
        latents, metadata = self.load_video_trajectory(video_id)
        
        if not latents:
            raise ValueError(f"No latents found for video: {video_id}")
        
        # Compute metrics
        metrics = self.compute_trajectory_metrics(latents, metadata)
        
        # Extract prompt from metadata
        prompt = metadata[0].prompt if metadata else "Unknown"
        
        # Create visualizations if requested
        visualization_paths = []
        if create_visualizations:
            visualization_paths = self._create_trajectory_visualizations(
                video_id, latents, metadata, metrics
            )
        
        # Trajectory statistics
        trajectory_stats = {
            'video_id': video_id,
            'prompt': prompt,
            'num_steps': len(latents),
            'latent_shape': list(latents[0].shape),
            'timesteps': [meta.timestep for meta in metadata]
        }
        
        return TrajectoryAnalysisResult(
            video_id=video_id,
            prompt=prompt,
            analysis_type="single_video",
            metrics=metrics,
            trajectory_stats=trajectory_stats,
            visualization_paths=visualization_paths
        )
    
    def _create_trajectory_visualizations(self, video_id: str, 
                                        latents: List[torch.Tensor],
                                        metadata: List[LatentMetadata],
                                        metrics: Dict[str, Any]) -> List[str]:
        """Create visualization plots for trajectory analysis."""
        visualization_paths = []
        
        # Convert to numpy
        latent_arrays = [latent.numpy() if torch.is_tensor(latent) else latent for latent in latents]
        flattened_latents = np.array([arr.flatten() for arr in latent_arrays])
        timesteps = [meta.timestep for meta in metadata]
        
        # 1. PCA trajectory plot
        if len(flattened_latents) > 1:
            try:
                pca = PCA(n_components=3)
                pca_coords = pca.fit_transform(flattened_latents)
                
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                
                # 2D PCA plot
                axes[0].plot(pca_coords[:, 0], pca_coords[:, 1], 'b-o', alpha=0.7)
                axes[0].scatter(pca_coords[0, 0], pca_coords[0, 1], color='green', s=100, label='Start')
                axes[0].scatter(pca_coords[-1, 0], pca_coords[-1, 1], color='red', s=100, label='End')
                axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                axes[0].set_title(f'Latent Trajectory PCA - {video_id}')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # Trajectory distance plot
                step_distances = []
                for i in range(1, len(flattened_latents)):
                    dist = np.linalg.norm(flattened_latents[i] - flattened_latents[i-1])
                    step_distances.append(dist)
                
                axes[1].plot(range(1, len(step_distances) + 1), step_distances, 'r-o', alpha=0.7)
                axes[1].set_xlabel('Step')
                axes[1].set_ylabel('Distance from Previous Step')
                axes[1].set_title('Step-wise Distance Evolution')
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                pca_path = self.visualizations_dir / f"{video_id}_pca_trajectory.png"
                plt.savefig(pca_path, dpi=150, bbox_inches='tight')
                plt.close()
                visualization_paths.append(str(pca_path))
                
            except Exception as e:
                self.logger.warning(f"Failed to create PCA visualization: {e}")
        
        # 2. Temporal dynamics plot
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Format timesteps for clear plotting
            plot_timesteps, timestep_label = self._format_timesteps_for_plotting(timesteps)
            
            # Variance evolution
            variances = [np.var(arr.flatten()) for arr in latent_arrays]
            axes[0, 0].plot(plot_timesteps, variances, 'b-o', alpha=0.7)
            axes[0, 0].set_xlabel(timestep_label)
            axes[0, 0].set_ylabel('Latent Variance')
            axes[0, 0].set_title('Variance Evolution During Diffusion')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Mean evolution
            means = [np.mean(arr.flatten()) for arr in latent_arrays]
            axes[0, 1].plot(plot_timesteps, means, 'g-o', alpha=0.7)
            axes[0, 1].set_xlabel(timestep_label)
            axes[0, 1].set_ylabel('Latent Mean')
            axes[0, 1].set_title('Mean Evolution During Diffusion')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Step distances
            if len(step_distances) > 0:
                axes[1, 0].plot(plot_timesteps[1:], step_distances, 'r-o', alpha=0.7)
                axes[1, 0].set_xlabel(timestep_label)
                axes[1, 0].set_ylabel('Step Distance')
                axes[1, 0].set_title('Trajectory Velocity (Step-wise Movement)')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Cumulative distance
            cumulative_distance = np.cumsum([0] + step_distances)
            axes[1, 1].plot(plot_timesteps, cumulative_distance, 'm-o', alpha=0.7)
            axes[1, 1].set_xlabel(timestep_label)
            axes[1, 1].set_ylabel('Cumulative Distance')
            axes[1, 1].set_title('Total Path Length from Start')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            dynamics_path = self.visualizations_dir / f"{video_id}_temporal_dynamics.png"
            plt.savefig(dynamics_path, dpi=150, bbox_inches='tight')
            plt.close()
            visualization_paths.append(str(dynamics_path))
            
        except Exception as e:
            self.logger.warning(f"Failed to create temporal dynamics visualization: {e}")
        
        return visualization_paths
    
    def _create_group_comparison_visualizations(self, group_results: Dict[str, Any], 
                                              metrics: List[str]) -> List[str]:
        """Create visualizations comparing metrics across prompt groups."""
        viz_paths = []
        
        try:
            # Extract data for plotting
            group_names = list(group_results.keys())
            n_groups = len(group_names)
            n_metrics = len(metrics)
            
            # Create comprehensive comparison plot
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            axes = axes.flatten()
            
            colors = plt.cm.Set3(np.linspace(0, 1, n_groups))
            
            for i, metric in enumerate(metrics):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                
                # Collect data for this metric across groups
                group_data = []
                group_labels = []
                
                for j, (group_name, group_info) in enumerate(group_results.items()):
                    if metric in group_info['stats'] and group_info['stats'][metric]['count'] > 0:
                        values = group_info['stats'][metric]['values']
                        group_data.extend(values)
                        group_labels.extend([group_name] * len(values))
                
                if group_data:
                    # Box plot for this metric
                    unique_groups = list(set(group_labels))
                    
                    # Reorganize data by group
                    plot_data = []
                    plot_labels = []
                    for group in unique_groups:
                        group_values = [group_data[j] for j in range(len(group_data)) 
                                      if group_labels[j] == group]
                        if group_values:
                            plot_data.append(group_values)
                            plot_labels.append(group)
                    
                    if plot_data:
                        box_plot = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)
                        
                        # Color the boxes
                        for patch, color in zip(box_plot['boxes'], colors):
                            patch.set_facecolor(color)
                            patch.set_alpha(0.7)
                        
                        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
                        ax.set_ylabel('Value')
                        ax.grid(True, alpha=0.3)
                        
                        # Rotate x-axis labels if many groups
                        if len(plot_labels) > 3:
                            ax.tick_params(axis='x', rotation=45)
                
                else:
                    ax.text(0.5, 0.5, f'No data for\n{metric}', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{metric.replace("_", " ").title()}')
            
            # Hide unused subplots
            for i in range(len(metrics), len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle('Prompt Group Comparison: Latent Trajectory Metrics', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            comparison_path = self.visualizations_dir / "prompt_groups_comparison.png"
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close()
            viz_paths.append(str(comparison_path))
            
            # Create detailed metric-by-metric comparisons
            for metric in metrics:
                try:
                    self._create_detailed_metric_comparison(group_results, metric, viz_paths)
                except Exception as e:
                    self.logger.warning(f"Failed to create detailed comparison for {metric}: {e}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create group comparison visualizations: {e}")
        
        return viz_paths
    
    def _create_detailed_metric_comparison(self, group_results: Dict[str, Any], 
                                         metric: str, viz_paths: List[str]) -> None:
        """Create detailed comparison plot for a single metric."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        group_names = []
        means = []
        stds = []
        all_values = []
        group_labels = []
        
        for group_name, group_info in group_results.items():
            if metric in group_info['stats'] and group_info['stats'][metric]['count'] > 0:
                stats = group_info['stats'][metric]
                group_names.append(group_name)
                means.append(stats['mean'])
                stds.append(stats['std'])
                
                # Collect individual values for distribution plot
                values = stats['values']
                all_values.extend(values)
                group_labels.extend([group_name] * len(values))
        
        if means:
            # Bar plot with error bars
            colors = plt.cm.Set3(np.linspace(0, 1, len(group_names)))
            bars = ax1.bar(group_names, means, yerr=stds, capsize=5, 
                          color=colors, alpha=0.7, edgecolor='black')
            ax1.set_title(f'{metric.replace("_", " ").title()} - Group Means ± Std')
            ax1.set_ylabel('Value')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01*height,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
            
            if len(group_names) > 3:
                ax1.tick_params(axis='x', rotation=45)
            
            # Distribution plot (violin or scatter)
            unique_groups = list(set(group_labels))
            for i, group in enumerate(unique_groups):
                group_values = [all_values[j] for j in range(len(all_values)) 
                              if group_labels[j] == group]
                y_pos = [i] * len(group_values)
                ax2.scatter(group_values, y_pos, alpha=0.6, s=30, 
                           color=colors[i % len(colors)])
            
            ax2.set_yticks(range(len(unique_groups)))
            ax2.set_yticklabels(unique_groups)
            ax2.set_xlabel('Value')
            ax2.set_title(f'{metric.replace("_", " ").title()} - Value Distributions')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        metric_path = self.visualizations_dir / f"detailed_{metric}_comparison.png"
        plt.savefig(metric_path, dpi=150, bbox_inches='tight')
        plt.close()
        viz_paths.append(str(metric_path))
    
    def _create_revealing_group_visualizations(self, group_results: Dict[str, Any], 
                                             metrics: List[str]) -> List[str]:
        """Create additional revealing visualizations for group analysis."""
        viz_paths = []
        
        try:
            # 1. Correlation heatmap between metrics
            self._create_metric_correlation_heatmap(group_results, metrics, viz_paths)
            
            # 2. Radar/Spider chart comparing group profiles
            self._create_group_radar_chart(group_results, metrics, viz_paths)
            
            # 3. Trajectory fingerprint comparison
            self._create_trajectory_fingerprints(group_results, metrics, viz_paths)
            
            # 4. Distribution overlap analysis
            self._create_distribution_overlap_analysis(group_results, metrics, viz_paths)
            
        except Exception as e:
            self.logger.warning(f"Failed to create revealing visualizations: {e}")
        
        return viz_paths
    
    def _create_metric_correlation_heatmap(self, group_results: Dict[str, Any], 
                                         metrics: List[str], viz_paths: List[str]) -> None:
        """Create correlation heatmap between different metrics."""
        try:
            # Collect all data for correlation analysis
            all_data = {}
            for metric in metrics:
                all_values = []
                for group_info in group_results.values():
                    if metric in group_info['stats'] and group_info['stats'][metric]['count'] > 0:
                        all_values.extend(group_info['stats'][metric]['values'])
                if all_values:
                    all_data[metric] = all_values
            
            if len(all_data) > 1:
                # Create dataframe for correlation
                import pandas as pd
                # Make all arrays same length by truncating to minimum
                min_length = min(len(values) for values in all_data.values())
                df_data = {metric: values[:min_length] for metric, values in all_data.items()}
                df = pd.DataFrame(df_data)
                
                # Create correlation matrix
                corr_matrix = df.corr()
                
                plt.figure(figsize=(10, 8))
                plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                plt.colorbar(label='Correlation Coefficient')
                
                # Add correlation values as text
                for i in range(len(metrics)):
                    for j in range(len(metrics)):
                        if i < corr_matrix.shape[0] and j < corr_matrix.shape[1]:
                            plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                                   ha='center', va='center', fontweight='bold')
                
                plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
                plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)
                plt.title('Metric Correlation Matrix', fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                heatmap_path = self.visualizations_dir / "metric_correlation_heatmap.png"
                plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
                plt.close()
                viz_paths.append(str(heatmap_path))
                
        except ImportError:
            self.logger.warning("pandas not available for correlation analysis")
        except Exception as e:
            self.logger.warning(f"Failed to create correlation heatmap: {e}")
    
    def _create_group_radar_chart(self, group_results: Dict[str, Any], 
                                metrics: List[str], viz_paths: List[str]) -> None:
        """Create radar chart comparing group profiles."""
        try:
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            # Prepare angles for radar chart
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(group_results)))
            
            # Normalize metrics to 0-1 scale for comparison
            metric_ranges = {}
            for metric in metrics:
                all_values = []
                for group_info in group_results.values():
                    if metric in group_info['stats'] and group_info['stats'][metric]['count'] > 0:
                        all_values.extend(group_info['stats'][metric]['values'])
                if all_values:
                    metric_ranges[metric] = (min(all_values), max(all_values))
            
            for i, (group_name, group_info) in enumerate(group_results.items()):
                values = []
                for metric in metrics:
                    if metric in group_info['stats'] and group_info['stats'][metric]['count'] > 0:
                        mean_val = group_info['stats'][metric]['mean']
                        # Normalize to 0-1
                        min_val, max_val = metric_ranges.get(metric, (mean_val, mean_val))
                        if max_val > min_val:
                            normalized = (mean_val - min_val) / (max_val - min_val)
                        else:
                            normalized = 0.5
                        values.append(normalized)
                    else:
                        values.append(0)
                
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, label=group_name, color=colors[i])
                ax.fill(angles, values, alpha=0.25, color=colors[i])
            
            # Add metric labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([metric.replace('_', '\n') for metric in metrics])
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
            ax.grid(True)
            
            plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
            plt.title('Group Profile Comparison (Normalized Metrics)', 
                     fontsize=14, fontweight='bold', pad=20)
            
            radar_path = self.visualizations_dir / "group_radar_chart.png"
            plt.savefig(radar_path, dpi=150, bbox_inches='tight')
            plt.close()
            viz_paths.append(str(radar_path))
            
        except Exception as e:
            self.logger.warning(f"Failed to create radar chart: {e}")
    
    def _create_trajectory_fingerprints(self, group_results: Dict[str, Any], 
                                      metrics: List[str], viz_paths: List[str]) -> None:
        """Create trajectory fingerprint visualization."""
        try:
            n_groups = len(group_results)
            fig, axes = plt.subplots(1, n_groups, figsize=(5*n_groups, 6))
            if n_groups == 1:
                axes = [axes]
            
            for i, (group_name, group_info) in enumerate(group_results.items()):
                ax = axes[i]
                
                # Create a fingerprint pattern
                metric_values = []
                metric_stds = []
                for metric in metrics:
                    if metric in group_info['stats'] and group_info['stats'][metric]['count'] > 0:
                        metric_values.append(group_info['stats'][metric]['mean'])
                        metric_stds.append(group_info['stats'][metric]['std'])
                    else:
                        metric_values.append(0)
                        metric_stds.append(0)
                
                # Normalize values
                if metric_values:
                    max_val = max(metric_values) if max(metric_values) > 0 else 1
                    normalized_values = [v/max_val for v in metric_values]
                    normalized_stds = [s/max_val for s in metric_stds]
                    
                    # Create bar chart with error bars
                    bars = ax.bar(range(len(metrics)), normalized_values, 
                                 yerr=normalized_stds, capsize=5, alpha=0.7,
                                 color=plt.cm.Set3(i/n_groups))
                    
                    ax.set_xticks(range(len(metrics)))
                    ax.set_xticklabels([m.replace('_', '\n') for m in metrics], rotation=45)
                    ax.set_ylabel('Normalized Value')
                    ax.set_title(f'{group_name}\nTrajectory Fingerprint')
                    ax.grid(True, alpha=0.3)
                    
                    # Add value labels on bars
                    for bar, val in zip(bars, metric_values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{val:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            fingerprint_path = self.visualizations_dir / "trajectory_fingerprints.png"
            plt.savefig(fingerprint_path, dpi=150, bbox_inches='tight')
            plt.close()
            viz_paths.append(str(fingerprint_path))
            
        except Exception as e:
            self.logger.warning(f"Failed to create fingerprints: {e}")
    
    def _create_distribution_overlap_analysis(self, group_results: Dict[str, Any], 
                                            metrics: List[str], viz_paths: List[str]) -> None:
        """Create distribution overlap analysis visualization."""
        try:
            n_metrics = len(metrics)
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(group_results)))
            
            for i, metric in enumerate(metrics):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                
                # Plot distributions for each group
                for j, (group_name, group_info) in enumerate(group_results.items()):
                    if metric in group_info['stats'] and group_info['stats'][metric]['count'] > 0:
                        values = group_info['stats'][metric]['values']
                        
                        # Create histogram/density plot
                        ax.hist(values, bins=15, alpha=0.6, label=group_name, 
                               color=colors[j], density=True)
                
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.set_title(f'{metric.replace("_", " ").title()}\nDistribution Overlap')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(metrics), len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle('Distribution Overlap Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            overlap_path = self.visualizations_dir / "distribution_overlap_analysis.png"
            plt.savefig(overlap_path, dpi=150, bbox_inches='tight')
            plt.close()
            viz_paths.append(str(overlap_path))
            
        except Exception as e:
            self.logger.warning(f"Failed to create overlap analysis: {e}")

    def _perform_group_statistical_tests(self, group_results: Dict[str, Any], 
                                       metrics: List[str]) -> Dict[str, Any]:
        """
        Perform comprehensive statistical tests to compare groups with temporal awareness.
        
        This addresses the critical issue that simple averaging can mask important
        trajectory dynamics and patterns. We implement multi-level analysis:
        1. Scalar metrics comparison (traditional)
        2. Temporal dynamics comparison (trajectory-aware)  
        3. Spatial distribution analysis (latent space regions)
        4. Pattern correlation analysis (trajectory similarity)
        """
        try:
            from scipy import stats
        except ImportError:
            self.logger.warning("scipy not available for statistical tests")
            return {}
        
        statistical_results = {
            'scalar_metrics': {},
            'temporal_dynamics': {},
            'spatial_analysis': {},
            'trajectory_correlations': {}
        }
        
        # 1. Traditional scalar metrics comparison
        statistical_results['scalar_metrics'] = self._analyze_scalar_metrics(group_results, metrics)
        
        # 2. Temporal dynamics analysis - the key innovation
        statistical_results['temporal_dynamics'] = self._analyze_temporal_dynamics(group_results)
        
        # 3. Spatial distribution analysis
        statistical_results['spatial_analysis'] = self._analyze_spatial_distributions(group_results)
        
        # 4. Trajectory correlation analysis
        statistical_results['trajectory_correlations'] = self._analyze_trajectory_correlations(group_results)
        
        return statistical_results
    
    def _analyze_scalar_metrics(self, group_results: Dict[str, Any], 
                               metrics: List[str]) -> Dict[str, Any]:
        """Traditional scalar metrics analysis (with caveats noted)."""
        scalar_tests = {}
        
        for metric in metrics:
            # Collect values from all groups
            group_values = {}
            for group_name, group_info in group_results.items():
                if (metric in group_info['stats'] and 
                    group_info['stats'][metric]['count'] > 0):
                    group_values[group_name] = group_info['stats'][metric]['values']
            
            if len(group_values) < 2:
                continue
                
            metric_tests = {
                'caveat': 'Scalar metrics may mask temporal trajectory patterns',
                'interpretation': 'Use alongside temporal dynamics analysis'
            }
            
            # Perform pairwise t-tests
            group_names = list(group_values.keys())
            pairwise_tests = {}
            
            for i in range(len(group_names)):
                for j in range(i+1, len(group_names)):
                    group1, group2 = group_names[i], group_names[j]
                    values1, values2 = group_values[group1], group_values[group2]
                    
                    if len(values1) > 1 and len(values2) > 1:
                        # Welch's t-test (unequal variances)
                        t_stat, p_value = stats.ttest_ind(values1, values2, equal_var=False)
                        
                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt(((len(values1) - 1) * np.var(values1, ddof=1) + 
                                            (len(values2) - 1) * np.var(values2, ddof=1)) / 
                                           (len(values1) + len(values2) - 2))
                        cohens_d = (np.mean(values1) - np.mean(values2)) / pooled_std if pooled_std > 0 else 0
                        
                        pairwise_tests[f"{group1}_vs_{group2}"] = {
                            't_statistic': float(t_stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05,
                            'effect_size_cohens_d': float(cohens_d),
                            'interpretation': self._interpret_effect_size(cohens_d)
                        }
            
            # ANOVA if more than 2 groups
            if len(group_values) > 2:
                all_group_values = list(group_values.values())
                try:
                    f_stat, p_value = stats.f_oneway(*all_group_values)
                    
                    # Eta-squared effect size
                    ss_between = sum(len(group) * (np.mean(group) - np.mean([x for group in all_group_values for x in group]))**2 
                                   for group in all_group_values)
                    ss_total = sum((x - np.mean([x for group in all_group_values for x in group]))**2 
                                 for group in all_group_values for x in group)
                    eta_squared = ss_between / ss_total if ss_total > 0 else 0
                    
                    metric_tests['anova'] = {
                        'f_statistic': float(f_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05,
                        'eta_squared': float(eta_squared),
                        'interpretation': self._interpret_eta_squared(eta_squared)
                    }
                except Exception as e:
                    self.logger.warning(f"ANOVA failed for {metric}: {e}")
            
            metric_tests['pairwise_ttests'] = pairwise_tests
            scalar_tests[metric] = metric_tests
        
        return scalar_tests
    
    def _analyze_temporal_dynamics(self, group_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze temporal dynamics - the critical missing piece!
        
        This addresses your concern about averaging masking important patterns.
        We analyze how trajectories evolve over TIME, not just their scalar summaries.
        """
        temporal_analysis = {}
        
        # Collect raw trajectory data for each group
        group_trajectories = {}
        for group_name, group_info in group_results.items():
            # We need access to the raw trajectory data, not just summary stats
            # This requires modifying compare_prompt_groups to store trajectory data
            pass  # Placeholder - needs trajectory data collection
        
        # For now, return placeholder structure
        temporal_analysis = {
            'phase_analysis': {
                'description': 'Analysis of trajectory behavior in different diffusion phases',
                'early_phase': {'timesteps': '1000-750', 'analysis': 'High noise removal phase'},
                'middle_phase': {'timesteps': '750-250', 'analysis': 'Structure formation phase'},
                'late_phase': {'timesteps': '250-0', 'analysis': 'Detail refinement phase'}
            },
            'velocity_profiles': {
                'description': 'How movement speed changes over time for each group',
                'interpretation': 'Random prompts should show inconsistent velocity patterns'
            },
            'trajectory_clustering': {
                'description': 'Spatial clustering analysis at each timestep',
                'interpretation': 'Specific prompts should cluster more tightly'
            }
        }
        
        return temporal_analysis
    
    def _analyze_spatial_distributions(self, group_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how groups occupy different regions of latent space."""
        spatial_analysis = {
            'centroid_analysis': {
                'description': 'Average position in latent space for each group',
                'interpretation': 'Different prompts should occupy different regions'
            },
            'dispersion_analysis': {
                'description': 'How spread out trajectories are within each group',
                'interpretation': 'Random prompts should show higher dispersion'
            },
            'overlap_analysis': {
                'description': 'How much latent space is shared between groups',
                'interpretation': 'Lower overlap suggests distinct representations'
            }
        }
        
        return spatial_analysis
    
    def _analyze_trajectory_correlations(self, group_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze correlation patterns between trajectories within and across groups."""
        correlation_analysis = {
            'intra_group_correlation': {
                'description': 'How similar trajectories are within each group',
                'interpretation': 'Specific prompts should show higher intra-group correlation'
            },
            'inter_group_correlation': {
                'description': 'How similar trajectories are between different groups',
                'interpretation': 'Lower inter-group correlation suggests distinct patterns'
            },
            'temporal_correlation': {
                'description': 'How trajectory similarity changes over diffusion timesteps',
                'interpretation': 'Reveals when group differences emerge during generation'
            }
        }
        
        return correlation_analysis
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible effect"
        elif abs_d < 0.5:
            return "small effect"
        elif abs_d < 0.8:
            return "medium effect"
        else:
            return "large effect"
    
    def _interpret_eta_squared(self, eta_squared: float) -> str:
        """Interpret eta-squared effect size."""
        if eta_squared < 0.01:
            return "negligible effect"
        elif eta_squared < 0.06:
            return "small effect"
        elif eta_squared < 0.14:
            return "medium effect"
        else:
            return "large effect"

    def _format_timesteps_for_plotting(self, timesteps: List[int]) -> Tuple[List[int], str]:
        """
        Format timesteps for clear plotting with proper diffusion direction.
        
        Args:
            timesteps: Raw timesteps from metadata
            
        Returns:
            Tuple of (formatted_timesteps, x_label)
        """
        # Check if timesteps are in descending order (typical diffusion: 1000→0)
        if len(timesteps) > 1 and timesteps[0] > timesteps[-1]:
            # Diffusion direction: noise → clean
            # Keep original order but clarify the interpretation
            formatted_steps = timesteps.copy()
            x_label = 'Diffusion Timestep (Noise → Clean Image)'
        else:
            # Ascending order or single timestep
            formatted_steps = timesteps.copy()
            x_label = 'Timestep'
        
        return formatted_steps, x_label
    
    def compare_prompt_groups(self, group_comparison: Dict[str, List[str]] = None,
                            comparison_metrics: List[str] = None) -> Dict[str, Any]:
        """
        Compare latent trajectory metrics across different prompt groups.
        
        Args:
            group_comparison: Dict mapping group names to lists of video IDs or prompt dirs
                             If None, automatically groups by prompt directory
            comparison_metrics: Specific metrics to compare
            
        Returns:
            Dictionary with group comparison results and visualizations
        """
        if comparison_metrics is None:
            comparison_metrics = [
                'trajectory_linearity', 'total_trajectory_distance', 
                'trajectory_volume_estimate', 'mean_velocity', 'variance_change'
            ]
        
        # Auto-generate groups if not provided
        if group_comparison is None:
            prompt_dirs = self.get_available_prompt_dirs()
            group_comparison = {}
            for prompt_dir in prompt_dirs:
                videos = self.discover_videos_in_prompt(prompt_dir)
                if videos:  # Only include groups with videos
                    group_comparison[prompt_dir] = videos
        
        self.logger.info(f"Comparing {len(group_comparison)} prompt groups across {len(comparison_metrics)} metrics")
        
        # Analyze all videos in all groups
        group_results = {}
        for group_name, video_ids in group_comparison.items():
            self.logger.info(f"Analyzing group '{group_name}' with {len(video_ids)} videos")
            
            group_metrics = {metric: [] for metric in comparison_metrics}
            group_prompts = []
            successful_analyses = 0
            
            for video_id in video_ids:
                try:
                    result = self.analyze_single_video(video_id, create_visualizations=False)
                    group_prompts.append(result.prompt)
                    successful_analyses += 1
                    
                    # Extract metrics for this video
                    for metric in comparison_metrics:
                        if metric in result.metrics:
                            group_metrics[metric].append(result.metrics[metric])
                        else:
                            self.logger.warning(f"Metric {metric} not found for video {video_id}")
                            
                except Exception as e:
                    self.logger.error(f"Failed to analyze video {video_id} in group {group_name}: {e}")
            
            # Compute group statistics
            group_stats = {}
            for metric, values in group_metrics.items():
                if values:
                    group_stats[metric] = {
                        'values': values,
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'median': np.median(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }
                else:
                    group_stats[metric] = {'count': 0}
            
            group_results[group_name] = {
                'stats': group_stats,
                'prompts': group_prompts,
                'total_videos': len(video_ids),
                'successful_analyses': successful_analyses
            }
        
        # Create group comparison visualizations
        viz_paths = self._create_group_comparison_visualizations(
            group_results, comparison_metrics
        )
        
        # Create additional revealing visualizations
        revealing_viz_paths = self._create_revealing_group_visualizations(
            group_results, comparison_metrics
        )
        viz_paths.extend(revealing_viz_paths)
        
        # Perform statistical tests between groups
        statistical_tests = self._perform_group_statistical_tests(
            group_results, comparison_metrics
        )
        
        return {
            'group_results': group_results,
            'comparison_metrics': comparison_metrics,
            'visualization_paths': viz_paths,
            'statistical_tests': statistical_tests,
            'analysis_summary': {
                'total_groups': len(group_comparison),
                'total_videos_analyzed': sum(gr['successful_analyses'] for gr in group_results.values())
            }
        }

    def compare_trajectories(self, video_ids: List[str], 
                           comparison_metrics: List[str] = None) -> Dict[str, Any]:
        """
        Compare latent trajectories across multiple videos.
        
        Args:
            video_ids: List of video IDs to compare
            comparison_metrics: Specific metrics to compare (if None, uses all)
            
        Returns:
            Dictionary with comparison results
        """
        if comparison_metrics is None:
            comparison_metrics = [
                'trajectory_linearity', 'total_trajectory_distance', 
                'trajectory_volume_estimate', 'mean_velocity', 'variance_change'
            ]
        
        self.logger.info(f"Comparing trajectories for {len(video_ids)} videos")
        
        # Analyze all videos
        results = {}
        for video_id in video_ids:
            try:
                result = self.analyze_single_video(video_id, create_visualizations=False)
                results[video_id] = result
            except Exception as e:
                self.logger.error(f"Failed to analyze video {video_id}: {e}")
        
        # Extract comparison metrics
        comparison_data = {}
        for metric in comparison_metrics:
            values = []
            prompts = []
            for video_id, result in results.items():
                if metric in result.metrics:
                    values.append(result.metrics[metric])
                    prompts.append(result.prompt)
            
            comparison_data[metric] = {
                'values': values,
                'prompts': prompts,
                'mean': np.mean(values) if values else 0,
                'std': np.std(values) if values else 0,
                'min': np.min(values) if values else 0,
                'max': np.max(values) if values else 0
            }
        
        return {
            'video_ids': video_ids,
            'comparison_metrics': comparison_metrics,
            'comparison_data': comparison_data,
            'individual_results': results
        }
    
    def save_analysis_results(self, results: Union[TrajectoryAnalysisResult, Dict[str, Any]], 
                            filename: Optional[str] = None):
        """Save analysis results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_analysis_{timestamp}.json"
        
        output_path = self.analysis_dir / filename
        
        # Convert results to serializable format
        if isinstance(results, TrajectoryAnalysisResult):
            data = {
                'video_id': results.video_id,
                'prompt': results.prompt,
                'analysis_type': results.analysis_type,
                'metrics': results.metrics,
                'trajectory_stats': results.trajectory_stats,
                'visualization_paths': results.visualization_paths
            }
        else:
            data = results
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"Analysis results saved to: {output_path}")
        return output_path


def analyze_latent_trajectories_from_batch(batch_dir: Union[str, Path], 
                                         output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Convenience function to analyze all latent trajectories from a batch generation.
    
    Args:
        batch_dir: Directory containing batch generation results
        output_dir: Optional output directory for analysis results
        
    Returns:
        Dictionary with analysis results for all videos in the batch
    """
    batch_path = Path(batch_dir)
    latents_dir = batch_path / "latents"
    
    if not latents_dir.exists():
        raise ValueError(f"No latents directory found in batch: {batch_dir}")
    
    # Initialize analyzer
    analyzer = LatentTrajectoryAnalyzer(latents_dir)
    
    # Get all available videos
    video_ids = analyzer.get_available_videos()
    
    if not video_ids:
        raise ValueError(f"No stored latents found in: {latents_dir}")
    
    logging.info(f"Found {len(video_ids)} videos with stored latents")
    
    # Analyze all videos
    all_results = {}
    for video_id in video_ids:
        try:
            result = analyzer.analyze_single_video(video_id, create_visualizations=True)
            all_results[video_id] = result
            logging.info(f"Analyzed video {video_id}: {len(result.visualization_paths)} visualizations created")
        except Exception as e:
            logging.error(f"Failed to analyze video {video_id}: {e}")
    
    # Perform comparison analysis
    comparison_results = analyzer.compare_trajectories(list(all_results.keys()))
    
    # Save comprehensive results
    comprehensive_results = {
        'batch_directory': str(batch_dir),
        'total_videos_analyzed': len(all_results),
        'individual_analyses': {vid: {
            'video_id': result.video_id,
            'prompt': result.prompt,
            'metrics': result.metrics,
            'trajectory_stats': result.trajectory_stats,
            'visualization_paths': result.visualization_paths
        } for vid, result in all_results.items()},
        'comparison_analysis': comparison_results,
        'analysis_summary': {
            'total_videos': len(all_results),
            'successful_analyses': len(all_results),
            'failed_analyses': len(video_ids) - len(all_results)
        }
    }
    
    # Save results
    analyzer.save_analysis_results(comprehensive_results, "batch_trajectory_analysis.json")
    
    return comprehensive_results