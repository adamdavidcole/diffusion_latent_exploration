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
        prompt_dirs = self.latent_storage.list_stored_videos()
        
        # For backward compatibility, return prompt directory names
        # Users can analyze at the prompt level (all videos from same prompt together)
        return prompt_dirs
    
    def get_available_prompt_dirs(self) -> List[str]:
        """Get list of prompt directories with stored latents."""
        return self.latent_storage.list_stored_videos()
    
    def discover_videos_in_prompt(self, prompt_dir: str) -> List[str]:
        """
        Discover individual video IDs within a prompt directory.
        
        Note: Since the new structure stores multiple videos from the same prompt
        in the same directory, this attempts to reconstruct video IDs based on
        video summary files.
        
        Args:
            prompt_dir: Prompt directory name (e.g., "prompt_000")
            
        Returns:
            List of video IDs that have stored latents
        """
        video_ids = []
        
        # Look for video summary files to reconstruct video IDs
        for summary_file in self.storage_dir.glob(f"video_{prompt_dir}_vid*_summary.json"):
            # Extract video ID from summary filename
            filename = summary_file.stem
            if filename.startswith('video_'):
                video_id = filename[6:-8]  # Remove "video_" prefix and "_summary" suffix
                video_ids.append(video_id)
        
        # If no summary files found, just return the prompt directory
        # This allows analysis of all latents in the prompt directory together
        if not video_ids:
            video_ids = [prompt_dir]
            
        return sorted(video_ids)
    
    def load_video_trajectory(self, video_id: str) -> Tuple[List[torch.Tensor], List[LatentMetadata]]:
        """
        Load complete latent trajectory for a video.
        
        Args:
            video_id: ID of the video to load (can be full video_id like "prompt_000_vid001" 
                     or just prompt part like "prompt_000")
            
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
            if "_vid" in video_id:
                prompt_part = video_id.split("_vid")[0]
            else:
                prompt_part = video_id
                
            latents_dir = self.latent_storage.latents_dir / prompt_part
            if not latents_dir.exists():
                raise ValueError(f"No latent directory found for video: {video_id}")
            
            # Find all step files in the directory
            step_files = list(latents_dir.glob("step_*.npy*")) + list(latents_dir.glob("step_*.pt*"))
            if not step_files:
                raise ValueError(f"No latent files found for video: {video_id}")
            
            # Extract step numbers from filenames
            stored_steps = []
            for file_path in step_files:
                # Extract step number from filename like "step_000.npy.gz"
                filename = file_path.stem
                if filename.endswith('.npy') or filename.endswith('.pt'):
                    filename = file_path.with_suffix('').stem
                
                if filename.startswith('step_'):
                    try:
                        step_num = int(filename.split('_')[1])
                        stored_steps.append(step_num)
                    except (IndexError, ValueError):
                        continue
            
            stored_steps = sorted(list(set(stored_steps)))
            
            if not stored_steps:
                raise ValueError(f"No valid step files found for video: {video_id}")
        
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
            
            # Variance evolution
            variances = [np.var(arr.flatten()) for arr in latent_arrays]
            axes[0, 0].plot(timesteps, variances, 'b-o', alpha=0.7)
            axes[0, 0].set_xlabel('Timestep')
            axes[0, 0].set_ylabel('Latent Variance')
            axes[0, 0].set_title('Variance Evolution')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Mean evolution
            means = [np.mean(arr.flatten()) for arr in latent_arrays]
            axes[0, 1].plot(timesteps, means, 'g-o', alpha=0.7)
            axes[0, 1].set_xlabel('Timestep')
            axes[0, 1].set_ylabel('Latent Mean')
            axes[0, 1].set_title('Mean Evolution')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Step distances
            if len(step_distances) > 0:
                axes[1, 0].plot(timesteps[1:], step_distances, 'r-o', alpha=0.7)
                axes[1, 0].set_xlabel('Timestep')
                axes[1, 0].set_ylabel('Step Distance')
                axes[1, 0].set_title('Movement Speed')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Cumulative distance
            cumulative_distance = np.cumsum([0] + step_distances)
            axes[1, 1].plot(timesteps, cumulative_distance, 'm-o', alpha=0.7)
            axes[1, 1].set_xlabel('Timestep')
            axes[1, 1].set_ylabel('Cumulative Distance')
            axes[1, 1].set_title('Total Path Length')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            dynamics_path = self.visualizations_dir / f"{video_id}_temporal_dynamics.png"
            plt.savefig(dynamics_path, dpi=150, bbox_inches='tight')
            plt.close()
            visualization_paths.append(str(dynamics_path))
            
        except Exception as e:
            self.logger.warning(f"Failed to create temporal dynamics visualization: {e}")
        
        return visualization_paths
    
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