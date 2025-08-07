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
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import FFT functions
try:
    from torch.fft import fft, ifft, fft2, ifft2, fftshift
    TORCH_FFT_AVAILABLE = True
except ImportError:
    TORCH_FFT_AVAILABLE = False

@dataclass
class GPUOptimizedAnalysis:
    """Data structure for GPU-optimized analysis results."""
    spatial_patterns: Dict[str, Any]
    temporal_coherence: Dict[str, Any]
    channel_analysis: Dict[str, Any]
    patch_diversity: Dict[str, Any]
    global_structure: Dict[str, Any]
    information_content: Dict[str, Any]
    complexity_measures: Dict[str, Any]
    frequency_patterns: Dict[str, Any]
    group_separability: Dict[str, Any]
    statistical_significance: Dict[str, Any]
    gpu_performance_stats: Dict[str, Any]
    analysis_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'spatial_patterns': self.spatial_patterns,
            'temporal_coherence': self.temporal_coherence,
            'channel_analysis': self.channel_analysis,
            'patch_diversity': self.patch_diversity,
            'global_structure': self.global_structure,
            'information_content': self.information_content,
            'complexity_measures': self.complexity_measures,
            'frequency_patterns': self.frequency_patterns,
            'group_separability': self.group_separability,
            'statistical_significance': self.statistical_significance,
            'gpu_performance_stats': self.gpu_performance_stats,
            'analysis_metadata': self.analysis_metadata
        }


class GPUOptimizedStructureAnalyzer:
    """GPU-accelerated structure-aware latent analysis with trajectory preservation."""
    
    def __init__(self, latents_dir: str, device: str = "auto", 
                 enable_mixed_precision: bool = True, batch_size: int = 32,
                 output_dir: Optional[str] = None):
        """Initialize GPU-optimized analyzer."""
        self.latents_dir = Path(latents_dir)
        self.enable_mixed_precision = enable_mixed_precision
        self.batch_size = batch_size
        
        # Device setup
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print("latents_dir", self.latents_dir)
        
        # Output directory
        if output_dir is None:
            self.output_dir = self.latents_dir.parent / "gpu_optimized_analysis_results"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.performance_stats = {
            'device_used': self.device,
            'mixed_precision_enabled': self.enable_mixed_precision,
            'batch_size': self.batch_size,
            'memory_usage': {}
        }
        
        self.logger.info(f"Initialized GPU analyzer on {self.device}")
        if self.device.startswith("cuda"):
            self.logger.info(f"GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    def _compute_spectral_entropy(self, power_spectrum: torch.Tensor) -> float:
        """Compute spectral entropy of power spectrum."""
        # Normalize to probability distribution
        probs = power_spectrum / (torch.sum(power_spectrum) + 1e-8)
        probs = probs[probs > 1e-8]  # Remove zeros
        
        if len(probs) > 1:
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            return entropy.item()
        return 0.0
    
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
                            prompt_descriptions: Optional[List[str]] = None) -> GPUOptimizedAnalysis:
        """Main analysis entry point with trajectory-aware processing."""
        self.logger.info("Starting GPU-optimized trajectory-aware analysis")
        start_time = time.time()
        
        # 1. Load and batch trajectory data
        self._track_gpu_memory("start")
        group_tensors = self._load_and_batch_trajectory_data(prompt_groups)
        self._track_gpu_memory("data_loaded")
        
        if not group_tensors:
            raise ValueError("No trajectory data loaded")
        
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
            analysis_results['spatial_patterns'] = self._gpu_analyze_spatial_patterns(group_tensors)
            self._track_gpu_memory("spatial_analysis")
            
            self.logger.info("Running temporal coherence analysis...")
            analysis_results['temporal_coherence'] = self._gpu_analyze_temporal_coherence(group_tensors)
            self._track_gpu_memory("temporal_analysis")
            
            self.logger.info("Running channel pattern analysis...")
            analysis_results['channel_analysis'] = self._gpu_analyze_channel_patterns(group_tensors)
            self._track_gpu_memory("channel_analysis")
            
            # Multi-scale analysis
            self.logger.info("Running patch diversity analysis...")
            analysis_results['patch_diversity'] = self._gpu_analyze_patch_diversity(group_tensors)
            
            self.logger.info("Running global structure analysis...")
            analysis_results['global_structure'] = self._gpu_analyze_global_structure(group_tensors)
            
            # Simplified additional analyses
            self.logger.info("Running information content analysis...")
            analysis_results['information_content'] = self._gpu_analyze_information_content(group_tensors)
            
            self.logger.info("Running complexity analysis...")
            analysis_results['complexity_measures'] = self._gpu_analyze_complexity_measures(group_tensors)
            
            self.logger.info("Running frequency analysis...")
            analysis_results['frequency_patterns'] = self._gpu_analyze_frequency_patterns(group_tensors)
            
            # Group separability
            self.logger.info("Running group separability analysis...")
            analysis_results['group_separability'] = self._gpu_analyze_group_separability(group_tensors, prompt_groups)
            
            # Statistical significance
            self.logger.info("Running statistical significance tests...")
            analysis_results['statistical_significance'] = self._gpu_test_statistical_significance(group_tensors, prompt_groups)
        
        self._track_gpu_memory("analysis_complete")
        
        # 3. Create analysis metadata
        total_time = time.time() - start_time
        analysis_metadata = {
            'total_analysis_time_seconds': total_time,
            'prompt_groups': prompt_groups,
            'prompt_descriptions': prompt_descriptions or [],
            'latents_directory': str(self.latents_dir),
            'trajectory_shape': trajectory_shape,
            'device_used': self.device,
            'mixed_precision': self.enable_mixed_precision,
            'batch_size': self.batch_size,
            'analysis_timestamp': time.strftime("%Y%m%d_%H%M%S")
        }
        
        # 4. Save results
        results = GPUOptimizedAnalysis(
            spatial_patterns=analysis_results['spatial_patterns'],
            temporal_coherence=analysis_results['temporal_coherence'],
            channel_analysis=analysis_results['channel_analysis'],
            patch_diversity=analysis_results['patch_diversity'],
            global_structure=analysis_results['global_structure'],
            information_content=analysis_results['information_content'],
            complexity_measures=analysis_results['complexity_measures'],
            frequency_patterns=analysis_results['frequency_patterns'],
            group_separability=analysis_results['group_separability'],
            statistical_significance=analysis_results['statistical_significance'],
            gpu_performance_stats=self.performance_stats,
            analysis_metadata=analysis_metadata
        )
        
        self._save_results(results)
        
        self.logger.info(f"GPU-optimized analysis completed in {total_time:.2f} seconds")
        return results

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
        
        for group_name, data in group_tensors.items():
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
            
            # Store trajectory-aware results
            spatial_analysis['spatial_variance_maps'][group_name] = {
                'mean': float(torch.mean(spatial_vars_per_step).item()),
                'std': float(torch.std(spatial_vars_per_step).item()),
                'distribution': spatial_vars_per_step.flatten().cpu().numpy().tolist()[:1000]  # Limit size
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
        
        for group_name, data in group_tensors.items():
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
                    dominant_powers = [float(mean_power_spectrum[idx].item()) for idx in freq_indices]
                    
                    frequency_analysis = {
                        'dominant_frequencies': dominant_freqs,
                        'dominant_powers': dominant_powers,
                        'spectral_centroid': float(torch.sum(torch.arange(len(mean_power_spectrum)).float().to(self.device) * mean_power_spectrum) / torch.sum(mean_power_spectrum)),
                        'spectral_entropy': float(self._compute_spectral_entropy(mean_power_spectrum))
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
        
        for group_name, data in group_tensors.items():
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
        
        for group_name, data in group_tensors.items():
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
        
        for group_name, data in group_tensors.items():
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
        
        for group_name, data in group_tensors.items():
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
        
        for group_name, data in group_tensors.items():
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
        
        for group_name, data in group_tensors.items():
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
        
        for group_name, data in group_tensors.items():
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
        
        for group_name, data in group_tensors.items():
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

    def _save_results(self, results: GPUOptimizedAnalysis):
        """Save analysis results to disk."""
        # Save main results
        results_file = self.output_dir / "gpu_optimized_analysis_results.json"
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
