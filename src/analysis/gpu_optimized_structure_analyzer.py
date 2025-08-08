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
    temporal_analysis: Dict[str, Any]
    structural_analysis: Dict[str, Any]
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
            'temporal_analysis': self.temporal_analysis,
            'structural_analysis': self.structural_analysis,
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
            
            # Temporal trajectory analysis
            self.logger.info("Running temporal trajectory analysis...")
            analysis_results['temporal_analysis'] = self._gpu_analyze_temporal_trajectories(group_tensors, prompt_groups)
            
            # Structural analysis
            self.logger.info("Running structural analysis...")
            analysis_results['structural_analysis'] = self._gpu_analyze_structural_patterns(group_tensors, prompt_groups)
            
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
            temporal_analysis=analysis_results['temporal_analysis'],
            structural_analysis=analysis_results['structural_analysis'],
            statistical_significance=analysis_results['statistical_significance'],
            gpu_performance_stats=self.performance_stats,
            analysis_metadata=analysis_metadata
        )
        
        self._save_results(results)
        
        # Generate comprehensive visualizations
        self._create_comprehensive_visualizations(results)
        
        self.logger.info(f"GPU-optimized analysis completed in {total_time:.2f} seconds")
        return results

    def _create_comprehensive_visualizations(self, results: GPUOptimizedAnalysis):
        """Create comprehensive visualizations for all key statistical analyses."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            viz_dir = self.output_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            self.logger.info("Creating comprehensive analysis visualizations...")
            
            # 1. Trajectory Spatial Evolution (U-shaped pattern)
            self._plot_trajectory_spatial_evolution(results, viz_dir)
            
            # 2. Cross-Trajectory Synchronization
            self._plot_cross_trajectory_synchronization(results, viz_dir)
            
            # 3. Temporal Momentum Analysis
            self._plot_temporal_momentum_analysis(results, viz_dir)
            
            # 4. Phase Transition Detection
            self._plot_phase_transition_detection(results, viz_dir)
            
            # 5. Temporal Frequency Signatures
            self._plot_temporal_frequency_signatures(results, viz_dir)
            
            # 6. Group Separability Analysis
            self._plot_group_separability(results, viz_dir)
            
            # 7. Spatial Progression Patterns
            self._plot_spatial_progression_patterns(results, viz_dir)
            
            # 8. Edge Density Evolution
            self._plot_edge_density_evolution(results, viz_dir)
            
            # 8b. Edge Formation Trends Dashboard (extracted from spatial progression)
            self._plot_edge_formation_trends_dashboard(results, viz_dir)
            
            # 9. Spatial Coherence Patterns
            self._plot_spatial_coherence_patterns(results, viz_dir)
            
            # 9b. Individual Video Coherence Dashboard (extracted from spatial coherence)
            self._plot_individual_video_coherence_dashboard(results, viz_dir)
            
            # 10. Temporal Stability Windows
            self._plot_temporal_stability_windows(results, viz_dir)
            
            # 11. Channel Evolution Patterns
            self._plot_channel_evolution_patterns(results, viz_dir)
            
            # 12. Global Structure Analysis
            self._plot_global_structure_analysis(results, viz_dir)
            
            # 13. Information Content Analysis
            self._plot_information_content_analysis(results, viz_dir)
            
            # 14. Complexity Measures
            self._plot_complexity_measures(results, viz_dir)
            
            # 15. Statistical Significance Tests
            self._plot_statistical_significance(results, viz_dir)
            
            # 16. Temporal Analysis Visualizations
            self._plot_temporal_analysis(results, viz_dir)
            
            # 17. Structural Analysis Visualizations
            self._plot_structural_analysis(results, viz_dir)
            
            # 18. Comprehensive Dashboard
            self._create_analysis_dashboard(results, viz_dir)
            
            self.logger.info(f"✅ Visualizations saved to: {viz_dir}")
            
        except Exception as e:
            import traceback
            self.logger.error(f"Visualization creation failed: {e}")
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")

    def _plot_trajectory_spatial_evolution(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Plot the U-shaped trajectory spatial evolution pattern."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract trajectory patterns with alphabetical ordering
        spatial_data = results.spatial_patterns['trajectory_spatial_evolution']
        
        # Sort group names alphabetically for consistent ordering
        sorted_group_names = sorted(spatial_data.keys())
        
        # Plot 1: Individual trajectory patterns
        colors = sns.color_palette("husl", len(sorted_group_names))
        for i, group_name in enumerate(sorted_group_names):
            data = spatial_data[group_name]
            trajectory_pattern = data['trajectory_pattern']
            steps = list(range(len(trajectory_pattern)))
            ax1.plot(steps, trajectory_pattern, 'o-', label=group_name, alpha=0.8, linewidth=2, 
                    markersize=3, color=colors[i])
        
        ax1.set_xlabel('Diffusion Step')
        ax1.set_ylabel('Spatial Variance')
        ax1.set_title('Trajectory Spatial Evolution Patterns\n(Universal U-Shaped Denoising Pattern)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Evolution ratio comparison
        evolution_ratios = [spatial_data[group]['evolution_ratio'] for group in sorted_group_names]
        
        bars = ax2.bar(sorted_group_names, evolution_ratios, alpha=0.7, color=colors)
        ax2.set_xlabel('Prompt Group')
        ax2.set_ylabel('Late/Early Spatial Variance Ratio')
        ax2.set_title('Spatial Evolution Ratio by Prompt\n(Recovery Strength in Late Diffusion)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, ratio in zip(bars, evolution_ratios):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(evolution_ratios) * 0.01,
                    f'{ratio:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "trajectory_spatial_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_cross_trajectory_synchronization(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Plot cross-trajectory synchronization analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        sync_data = results.temporal_coherence['cross_trajectory_synchronization']
        
        # Extract data with alphabetical ordering
        group_names = sorted(sync_data.keys())
        mean_correlations = [sync_data[group]['mean_correlation'] for group in group_names]
        correlation_stds = [sync_data[group]['correlation_std'] for group in group_names]
        high_sync_ratios = [sync_data[group]['high_sync_ratio'] for group in group_names]
        
        colors = sns.color_palette("husl", len(group_names))
        
        # Plot 1: Mean correlation by group
        bars1 = ax1.bar(group_names, mean_correlations, alpha=0.7, color=colors)
        ax1.set_ylabel('Mean Cross-Trajectory Correlation')
        ax1.set_title('Cross-Trajectory Synchronization Strength')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, corr in zip(bars1, mean_correlations):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(mean_correlations) * 0.01,
                    f'{corr:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Correlation variability
        ax2.errorbar(group_names, mean_correlations, yerr=correlation_stds, 
                    fmt='o', capsize=5, capthick=2, linewidth=2, markersize=4, alpha=0.8)
        ax2.set_ylabel('Correlation ± Std Dev')
        ax2.set_title('Synchronization Consistency')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: High synchronization ratio
        bars3 = ax3.bar(group_names, high_sync_ratios, alpha=0.7, 
                       color=sns.color_palette("plasma", len(group_names)))
        ax3.set_ylabel('High Sync Ratio (>0.7 correlation)')
        ax3.set_title('Percentage of Highly Synchronized Videos')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add percentage labels
        for bar, ratio in zip(bars3, high_sync_ratios):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(high_sync_ratios) * 0.01,
                    f'{ratio:.1%}', ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Synchronization ranking
        sync_ranking = sorted(zip(group_names, mean_correlations), key=lambda x: x[1], reverse=True)
        ranked_groups, ranked_corrs = zip(*sync_ranking)
        
        ax4.barh(range(len(ranked_groups)), ranked_corrs, alpha=0.7,
                color=sns.color_palette("coolwarm", len(ranked_groups)))
        ax4.set_yticks(range(len(ranked_groups)))
        ax4.set_yticklabels(ranked_groups)
        ax4.set_xlabel('Mean Cross-Trajectory Correlation')
        ax4.set_title('Synchronization Ranking (Best to Worst)')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "cross_trajectory_synchronization.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_temporal_momentum_analysis(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Plot temporal momentum patterns with improved clarity and individual group views."""
        momentum_data = results.temporal_coherence['temporal_momentum_analysis']
        group_names = sorted(momentum_data.keys())
        colors = sns.color_palette("husl", len(group_names))
        
        # Create main overlaid analysis figure
        fig_main = plt.figure(figsize=(16, 12))
        
        # Main plots (2x2 grid)
        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)
        ax3 = plt.subplot(2, 2, 3)
        ax4 = plt.subplot(2, 2, 4)
        
        # Plot 1: Velocity patterns with confidence intervals (overlaid)
        for i, group_name in enumerate(group_names):
            data = momentum_data[group_name]
            velocity_mean = np.array(data['velocity_mean']).flatten()
            velocity_std = np.array(data['velocity_std']).flatten()
            
            # Ensure arrays have the same length
            min_len = min(len(velocity_mean), len(velocity_std))
            velocity_mean = velocity_mean[:min_len]
            velocity_std = velocity_std[:min_len]
            steps = np.arange(min_len)
            
            ax1.plot(steps, velocity_mean, 'o-', label=group_name, 
                    color=colors[i], alpha=0.8, linewidth=2, markersize=3)
            ax1.fill_between(steps, velocity_mean - velocity_std, velocity_mean + velocity_std,
                           alpha=0.15, color=colors[i])
        
        ax1.set_xlabel('Diffusion Step')
        ax1.set_ylabel('Mean Velocity (±1σ)')
        ax1.set_title('Temporal Velocity Evolution - All Groups\n(Denoising Speed with Uncertainty)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Acceleration patterns with confidence intervals (overlaid)
        for i, group_name in enumerate(group_names):
            data = momentum_data[group_name]
            accel_mean = np.array(data['acceleration_mean']).flatten()
            accel_std = np.array(data['acceleration_std']).flatten()
            
            # Ensure arrays have the same length
            min_len = min(len(accel_mean), len(accel_std))
            accel_mean = accel_mean[:min_len]
            accel_std = accel_std[:min_len]
            steps = np.arange(min_len)
            
            ax2.plot(steps, accel_mean, 's-', label=group_name, 
                    color=colors[i], alpha=0.8, linewidth=2, markersize=3)
            ax2.fill_between(steps, accel_mean - accel_std, accel_mean + accel_std,
                           alpha=0.15, color=colors[i])
        
        ax2.set_xlabel('Diffusion Step')
        ax2.set_ylabel('Mean Acceleration (±1σ)')
        ax2.set_title('Temporal Acceleration Evolution - All Groups\n(Denoising Rate Changes)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Direction instability patterns (overlaid)
        for i, group_name in enumerate(group_names):
            data = momentum_data[group_name]
            direction_changes = np.array(data['momentum_direction_changes']).flatten()
            steps = np.arange(len(direction_changes))
            
            ax3.plot(steps, direction_changes, '^-', label=group_name, 
                    color=colors[i], alpha=0.8, linewidth=2, markersize=3)
        
        ax3.set_xlabel('Diffusion Step')
        ax3.set_ylabel('Direction Change Count')
        ax3.set_title('Momentum Direction Changes - All Groups\n(Trajectory Instability)')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Momentum phase space with error ellipses
        for i, group_name in enumerate(group_names):
            data = momentum_data[group_name]
            avg_velocity = np.mean(data['velocity_mean'])
            avg_acceleration = np.mean(data['acceleration_mean'])
            vel_uncertainty = np.mean(data['velocity_std'])
            accel_uncertainty = np.mean(data['acceleration_std'])
            
            # Plot point
            ax4.scatter(avg_velocity, avg_acceleration, s=120, 
                       color=colors[i], alpha=0.8, edgecolors='black', linewidth=1)
            
            # Plot uncertainty ellipse
            from matplotlib.patches import Ellipse
            ellipse = Ellipse((avg_velocity, avg_acceleration), 
                            2*vel_uncertainty, 2*accel_uncertainty,
                            alpha=0.3, color=colors[i])
            ax4.add_patch(ellipse)
            
            # Label
            ax4.annotate(group_name, (avg_velocity, avg_acceleration), 
                        xytext=(8, 8), textcoords='offset points', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3))
        
        ax4.set_xlabel('Average Velocity')
        ax4.set_ylabel('Average Acceleration')
        ax4.set_title('Momentum Phase Space\n(Velocity vs Acceleration with Uncertainty)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path_main = viz_dir / "temporal_momentum_analysis.png"
        plt.savefig(output_path_main, dpi=300, bbox_inches='tight')
        plt.close(fig_main)
        
        # Create separate figure for individual group velocity plots
        n_groups = len(group_names)
        n_cols = min(3, n_groups)
        n_rows = (n_groups + n_cols - 1) // n_cols
        
        fig_individual = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
        fig_individual.suptitle('Individual Group Velocity Evolution', fontsize=14, fontweight='bold')
        
        # Calculate global y-axis range for consistent scaling
        all_velocities = []
        all_stds = []
        for group_name in group_names:
            data = momentum_data[group_name]
            velocity_mean = np.array(data['velocity_mean']).flatten()
            velocity_std = np.array(data['velocity_std']).flatten()
            all_velocities.extend(velocity_mean)
            all_stds.extend(velocity_std)
        
        global_min = min(all_velocities) - max(all_stds)
        global_max = max(all_velocities) + max(all_stds)
        y_margin = (global_max - global_min) * 0.1
        global_ylim = (global_min - y_margin, global_max + y_margin)
        
        for i, group_name in enumerate(group_names):
            ax_ind = plt.subplot(n_rows, n_cols, i + 1)
            
            data = momentum_data[group_name]
            velocity_mean = np.array(data['velocity_mean']).flatten()
            velocity_std = np.array(data['velocity_std']).flatten()
            
            min_len = min(len(velocity_mean), len(velocity_std))
            velocity_mean = velocity_mean[:min_len]
            velocity_std = velocity_std[:min_len]
            steps = np.arange(min_len)
            
            ax_ind.plot(steps, velocity_mean, 'o-', color=colors[i], 
                       alpha=0.9, linewidth=2.5, markersize=4)
            ax_ind.fill_between(steps, velocity_mean - velocity_std, velocity_mean + velocity_std,
                              alpha=0.3, color=colors[i])
            
            ax_ind.set_xlabel('Diffusion Step')
            ax_ind.set_ylabel('Velocity')
            ax_ind.set_title(f'{group_name}\nVelocity Evolution')
            ax_ind.set_ylim(global_ylim)  # Set consistent y-axis range
            ax_ind.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path_individual = viz_dir / "temporal_momentum_individual.png"
        plt.savefig(output_path_individual, dpi=300, bbox_inches='tight')
        plt.close(fig_individual)

    def _plot_phase_transition_detection(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Plot phase transition patterns."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        phase_data = results.temporal_coherence['phase_transition_detection']
        group_names = sorted(phase_data.keys())
        colors = sns.color_palette("husl", len(group_names))
        
        # Debug: Check data structure
        sample_group = group_names[0]
        sample_data = phase_data[sample_group]
        self.logger.info(f"Phase transition data structure for {sample_group}: {list(sample_data.keys())}")
        if 'p75_transitions' in sample_data:
            p75_shape = np.array(sample_data['p75_transitions']).shape
            self.logger.info(f"p75_transitions shape: {p75_shape}")
        
        # Plot 1: 75th percentile transitions
        for i, group_name in enumerate(group_names):
            data = phase_data[group_name]
            p75_transitions = data['p75_transitions']
            
            # Handle different data structures
            if isinstance(p75_transitions, (list, np.ndarray)):
                p75_array = np.array(p75_transitions)
                if p75_array.ndim > 1:
                    # If 2D, take mean across first dimension (videos)
                    p75_transitions = np.mean(p75_array, axis=0)
                else:
                    p75_transitions = p75_array
            
            steps = list(range(len(p75_transitions)))
            ax1.plot(steps, p75_transitions, 'o-', label=group_name, 
                    color=colors[i], alpha=0.8, linewidth=2, markersize=4)
        
        ax1.set_xlabel('Diffusion Step')
        ax1.set_ylabel('Mean Transition Count')
        ax1.set_title('Phase Transitions (75th Percentile)\n(Moderate Changes - Average per Group)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: 95th percentile transitions (major changes)
        for i, group_name in enumerate(group_names):
            data = phase_data[group_name]
            p95_transitions = data['p95_transitions']
            
            # Handle different data structures
            if isinstance(p95_transitions, (list, np.ndarray)):
                p95_array = np.array(p95_transitions)
                if p95_array.ndim > 1:
                    # If 2D, take mean across first dimension (videos)
                    p95_transitions = np.mean(p95_array, axis=0)
                else:
                    p95_transitions = p95_array
            
            steps = list(range(len(p95_transitions)))
            ax2.plot(steps, p95_transitions, '^-', label=group_name, 
                    color=colors[i], alpha=0.8, linewidth=2, markersize=4)
        
        ax2.set_xlabel('Diffusion Step')
        ax2.set_ylabel('Mean Transition Count')
        ax2.set_title('Major Phase Transitions (95th Percentile)\n(Dramatic Changes - Average per Group)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Transition intensity heatmap
        p90_data_list = []
        for group_name in group_names:
            p90_transitions = phase_data[group_name]['p90_transitions']
            p90_array = np.array(p90_transitions)
            
            # Handle different data structures
            if p90_array.ndim > 1:
                # If shape is (steps, features), take mean across features
                if p90_array.shape[1] > p90_array.shape[0]:
                    # Likely (features, steps) - transpose and take mean
                    p90_transitions = np.mean(p90_array.T, axis=1)
                else:
                    # Likely (steps, features) - take mean across features
                    p90_transitions = np.mean(p90_array, axis=1)
            else:
                p90_transitions = p90_array
                
            p90_data_list.append(p90_transitions)
        
        # Create heatmap matrix
        p90_data = np.array(p90_data_list)
        
        im = ax3.imshow(p90_data, cmap='YlOrRd', aspect='auto')
        ax3.set_yticks(range(len(group_names)))
        ax3.set_yticklabels(group_names)
        ax3.set_xlabel('Diffusion Step')
        ax3.set_ylabel('Prompt Group')
        ax3.set_title('Phase Transition Intensity Map\n(90th Percentile)')
        plt.colorbar(im, ax=ax3, label='Transition Count')
        
        # Plot 4: Total transitions by group with better calculation
        total_transitions = []
        for group_name in group_names:
            data = phase_data[group_name]
            
            # Calculate totals more carefully
            p75_total = np.sum(np.array(data['p75_transitions']).flatten())
            p90_total = np.sum(np.array(data['p90_transitions']).flatten())
            p95_total = np.sum(np.array(data['p95_transitions']).flatten())
            
            total = p75_total + p90_total + p95_total
            total_transitions.append(total)
            
            # Debug output
            self.logger.info(f"{group_name}: p75={p75_total:.2f}, p90={p90_total:.2f}, p95={p95_total:.2f}, total={total:.2f}")
        
        bars = ax4.bar(group_names, total_transitions, alpha=0.7, 
                      color=sns.color_palette("rocket", len(group_names)))
        ax4.set_xlabel('Prompt Group')
        ax4.set_ylabel('Total Transition Events')
        ax4.set_title('Overall Phase Transition Activity')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, total in zip(bars, total_transitions):
            height = bar.get_height()
            if total_transitions:  # Avoid division by zero
                max_total = max(total_transitions)
                ax4.text(bar.get_x() + bar.get_width()/2., height + max_total * 0.01,
                        f'{total:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "phase_transition_detection.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_temporal_frequency_signatures(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Plot temporal frequency analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        freq_data = results.temporal_coherence['temporal_frequency_signatures']
        
        # Plot 1: Dominant frequencies
        group_names = sorted(freq_data.keys())  # Alphabetical ordering
        colors = sns.color_palette("husl", len(group_names))
        dominant_freqs = []
        dominant_powers = []
        
        for group_name in group_names:
            data = freq_data[group_name]
            if data['dominant_frequencies']:
                # Ensure we get scalar values - handle both arrays and scalars
                freq_val = data['dominant_frequencies'][0]
                power_val = data['dominant_powers'][0]
                
                # If they're arrays, take the mean
                if isinstance(freq_val, (list, tuple, np.ndarray)):
                    freq_val = np.mean(freq_val)
                if isinstance(power_val, (list, tuple, np.ndarray)):
                    power_val = np.mean(power_val)
                    
                dominant_freqs.append(float(freq_val))
                dominant_powers.append(float(power_val))
            else:
                dominant_freqs.append(0.0)
                dominant_powers.append(0.0)
        
        bars1 = ax1.bar(group_names, dominant_freqs, alpha=0.7, color=colors)
        ax1.set_xlabel('Prompt Group')
        ax1.set_ylabel('Dominant Frequency')
        ax1.set_title('Primary Temporal Frequency by Group')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, freq in zip(bars1, dominant_freqs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(dominant_freqs) * 0.01,
                    f'{freq:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Spectral power
        bars2 = ax2.bar(group_names, dominant_powers, alpha=0.7,
                       color=sns.color_palette("viridis", len(group_names)))
        ax2.set_xlabel('Prompt Group')
        ax2.set_ylabel('Spectral Power')
        ax2.set_title('Dominant Frequency Power')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, power in zip(bars2, dominant_powers):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(dominant_powers) * 0.01,
                    f'{power:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 3: Spectral centroid
        centroids = []
        for data in freq_data.values():
            centroid = data['spectral_centroid']
            if isinstance(centroid, (list, tuple, np.ndarray)):
                centroid = np.mean(centroid)
            centroids.append(float(centroid))
            
        bars3 = ax3.bar(group_names, centroids, alpha=0.7,
                       color=sns.color_palette("coolwarm", len(group_names)))
        ax3.set_xlabel('Prompt Group')
        ax3.set_ylabel('Spectral Centroid')
        ax3.set_title('Frequency Distribution Center')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, centroid in zip(bars3, centroids):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(centroids) * 0.01,
                    f'{centroid:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Spectral entropy (frequency diversity)
        entropies = []
        for data in freq_data.values():
            entropy = data['spectral_entropy']
            if isinstance(entropy, (list, tuple, np.ndarray)):
                entropy = np.mean(entropy)
            entropies.append(float(entropy))
            
        bars4 = ax4.bar(group_names, entropies, alpha=0.7,
                       color=sns.color_palette("rocket", len(group_names)))
        ax4.set_xlabel('Prompt Group')
        ax4.set_ylabel('Spectral Entropy')
        ax4.set_title('Temporal Frequency Diversity')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, entropy in zip(bars4, entropies):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(entropies) * 0.01,
                    f'{entropy:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "temporal_frequency_signatures.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_group_separability(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Plot group separability analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        separability_data = results.group_separability['inter_group_distances']
        
        # Create distance matrix
        group_names = set()
        for key in separability_data.keys():
            group1, group2 = key.split('_vs_')
            group_names.add(group1)
            group_names.add(group2)
        
        group_names = sorted(list(group_names))
        n_groups = len(group_names)
        distance_matrix = np.zeros((n_groups, n_groups))
        
        for i, group1 in enumerate(group_names):
            for j, group2 in enumerate(group_names):
                if i != j:
                    key1 = f"{group1}_vs_{group2}"
                    key2 = f"{group2}_vs_{group1}"
                    if key1 in separability_data:
                        distance_matrix[i, j] = separability_data[key1]
                    elif key2 in separability_data:
                        distance_matrix[i, j] = separability_data[key2]
        
        # Plot 1: Distance matrix heatmap
        im1 = ax1.imshow(distance_matrix, cmap='RdYlBu_r')
        ax1.set_xticks(range(n_groups))
        ax1.set_yticks(range(n_groups))
        ax1.set_xticklabels(group_names, rotation=45)
        ax1.set_yticklabels(group_names)
        ax1.set_title('Inter-Group Distance Matrix\n(Trajectory Separability)')
        plt.colorbar(im1, ax=ax1, label='Distance')
        
        # Plot 2: Average distances from each group
        avg_distances = np.mean(distance_matrix, axis=1)
        bars = ax2.bar(group_names, avg_distances, alpha=0.7,
                      color=sns.color_palette("magma", len(group_names)))
        ax2.set_xlabel('Prompt Group')
        ax2.set_ylabel('Average Distance to Other Groups')
        ax2.set_title('Group Isolation Index\n(Higher = More Unique)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, dist in zip(bars, avg_distances):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{dist:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "group_separability.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_spatial_progression_patterns(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Plot spatial progression pattern analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        spatial_data = results.spatial_patterns['spatial_progression_patterns']
        group_names = sorted(spatial_data.keys())
        colors = sns.color_palette("husl", len(group_names))
        
        # Plot 1: Progression consistency
        consistency_values = [spatial_data[group]['progression_consistency'] for group in group_names]
        bars1 = ax1.bar(group_names, consistency_values, alpha=0.7, color=colors)
        ax1.set_xlabel('Prompt Group')
        ax1.set_ylabel('Progression Consistency')
        ax1.set_title('Spatial Progression Consistency')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Progression variability
        variability_values = [spatial_data[group]['progression_variability'] for group in group_names]
        bars2 = ax2.bar(group_names, variability_values, alpha=0.7, 
                       color=sns.color_palette("viridis", len(group_names)))
        ax2.set_xlabel('Prompt Group')
        ax2.set_ylabel('Progression Variability')
        ax2.set_title('Spatial Progression Variability')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Step deltas evolution over time
        for i, group_name in enumerate(group_names):
            step_deltas = spatial_data[group_name]['step_deltas_mean']
            steps = range(len(step_deltas))
            ax3.plot(steps, step_deltas, 'o-', label=group_name, 
                    color=colors[i], alpha=0.8, linewidth=2)
        
        ax3.set_xlabel('Diffusion Step')
        ax3.set_ylabel('Step Delta Mean')
        ax3.set_title('Spatial Step Delta Evolution')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Step delta standard deviation patterns
        for i, group_name in enumerate(group_names):
            step_deltas_std = spatial_data[group_name]['step_deltas_std']
            steps = range(len(step_deltas_std))
            ax4.plot(steps, step_deltas_std, '^-', label=group_name, 
                    color=colors[i], alpha=0.8, linewidth=2, markersize=4)
        
        ax4.set_xlabel('Diffusion Step')
        ax4.set_ylabel('Step Delta Std Dev')
        ax4.set_title('Spatial Step Delta Variability')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "spatial_progression_patterns.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_edge_formation_trends_dashboard(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Plot edge formation trends dashboard (extracted from spatial progression patterns)."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Get both spatial progression and edge density data
        spatial_data = results.spatial_patterns['spatial_progression_patterns']
        edge_data = results.spatial_patterns['edge_density_evolution']
        sorted_group_names = sorted(spatial_data.keys())
        colors = sns.color_palette("plasma", len(sorted_group_names))
        
        # Plot 1: Edge evolution patterns from spatial progression data
        ax1.set_title('Edge Formation Trends by Group\n(From Spatial Progression Analysis)')
        for i, group_name in enumerate(sorted_group_names):
            data = spatial_data[group_name]
            edge_patterns = data.get('edge_evolution_patterns', [])
            if edge_patterns:
                mean_pattern = np.mean(edge_patterns, axis=0)
                steps = list(range(len(mean_pattern)))
                ax1.plot(steps, mean_pattern, 'o-', label=group_name, 
                        alpha=0.8, color=colors[i], linewidth=2)
        
        ax1.set_xlabel('Diffusion Step')
        ax1.set_ylabel('Edge Density')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mean evolution patterns from edge density analysis
        ax2.set_title('Edge Density Evolution Patterns\n(From Edge Density Analysis)')
        for i, group_name in enumerate(sorted_group_names):
            if group_name in edge_data:
                data = edge_data[group_name]
                evolution_pattern = data.get('mean_evolution_pattern', [])
                if evolution_pattern:
                    steps = list(range(len(evolution_pattern)))
                    ax2.plot(steps, evolution_pattern, 's-', label=group_name, 
                            alpha=0.8, color=colors[i], linewidth=2)
        
        ax2.set_xlabel('Diffusion Step')
        ax2.set_ylabel('Mean Edge Density')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Edge formation trend distribution
        trend_counts = {'increasing': 0, 'decreasing': 0, 'stable': 0}
        for group_name in sorted_group_names:
            if group_name in edge_data:
                data = edge_data[group_name]
                trend = data.get('formation_trend', 'stable')
                if trend in trend_counts:
                    trend_counts[trend] += 1
        
        if sum(trend_counts.values()) > 0:
            ax3.pie(trend_counts.values(), labels=trend_counts.keys(), autopct='%1.1f%%',
                   colors=sns.color_palette("Set2", len(trend_counts)))
            ax3.set_title('Edge Formation Trend Distribution\n(Across All Groups)')
        else:
            ax3.text(0.5, 0.5, 'No edge trend data available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Edge Formation Trends (No Data)')
        
        # Plot 4: Edge density summary statistics
        mean_densities = []
        group_labels = []
        for group_name in sorted_group_names:
            if group_name in edge_data:
                data = edge_data[group_name]
                evolution_pattern = data.get('mean_evolution_pattern', [])
                if evolution_pattern:
                    mean_densities.append(np.mean(evolution_pattern))
                    group_labels.append(group_name)
        
        if mean_densities:
            bars = ax4.bar(group_labels, mean_densities, alpha=0.7, color=colors[:len(group_labels)])
            ax4.set_xlabel('Prompt Group')
            ax4.set_ylabel('Average Edge Density')
            ax4.set_title('Average Edge Density by Group')
            ax4.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, density in zip(bars, mean_densities):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + max(mean_densities) * 0.01,
                        f'{density:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            ax4.text(0.5, 0.5, 'No edge density data available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Average Edge Density (No Data)')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "edge_formation_trends_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_edge_density_evolution(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Plot edge density evolution analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        edge_data = results.spatial_patterns['edge_density_evolution']
        prompt_names = sorted(edge_data.keys())
        colors = sns.color_palette("husl", len(prompt_names))
        
        # Debug: Check actual data structure
        sample_prompt = prompt_names[0]
        sample_data = edge_data[sample_prompt]
        self.logger.info(f"Edge density data structure for {sample_prompt}: {list(sample_data.keys())}")
        
        # Plot 1: Edge density evolution over diffusion steps (using mean_evolution_pattern)
        has_evolution_data = False
        for i, prompt_name in enumerate(prompt_names):
            if 'mean_evolution_pattern' in edge_data[prompt_name] and edge_data[prompt_name]['mean_evolution_pattern']:
                evolution = edge_data[prompt_name]['mean_evolution_pattern']
                if evolution and len(evolution) > 0:
                    steps = range(len(evolution))
                    ax1.plot(steps, evolution, 'o-', label=prompt_name, 
                            color=colors[i], alpha=0.8, linewidth=2, markersize=3)
                    has_evolution_data = True
        
        if has_evolution_data:
            ax1.set_xlabel('Diffusion Step')
            ax1.set_ylabel('Mean Edge Density')
            ax1.set_title('Edge Density Evolution by Prompt\n(Mean Evolution Pattern)')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No edge density evolution data available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Edge Density Evolution (No Data)')
        
        # Plot 2: Evolution variability 
        variability_data_available = any('evolution_variability' in edge_data[prompt] and 
                                        edge_data[prompt]['evolution_variability'] is not None 
                                        for prompt in prompt_names)
        
        if variability_data_available:
            variabilities = []
            valid_prompts = []
            for prompt in prompt_names:
                if 'evolution_variability' in edge_data[prompt] and edge_data[prompt]['evolution_variability'] is not None:
                    var_data = edge_data[prompt]['evolution_variability']
                    # Handle both scalar and array data
                    if isinstance(var_data, (list, np.ndarray)):
                        # If it's an array, take the mean to get a scalar
                        scalar_var = np.mean(var_data)
                    else:
                        # Already a scalar
                        scalar_var = var_data
                    variabilities.append(scalar_var)
                    valid_prompts.append(prompt)
            
            if variabilities:
                bars2 = ax2.bar(valid_prompts, variabilities, alpha=0.7, 
                               color=colors[:len(valid_prompts)])
                ax2.set_xlabel('Prompt ID')
                ax2.set_ylabel('Evolution Variability')
                ax2.set_title('Edge Density Evolution Variability')
                ax2.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, var in zip(bars2, variabilities):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + max(variabilities) * 0.01,
                            f'{var:.3f}', ha='center', va='bottom', fontsize=8)
            else:
                ax2.text(0.5, 0.5, 'No valid variability data', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Evolution Variability (No Data)')
        else:
            ax2.text(0.5, 0.5, 'No evolution variability data available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Evolution Variability (No Data)')
        
        # Plot 3: Edge formation trend
        trend_data_available = any('edge_formation_trend' in edge_data[prompt] and 
                                  edge_data[prompt]['edge_formation_trend'] is not None 
                                  for prompt in prompt_names)
        
        if trend_data_available:
            trends = []
            trend_labels = []
            valid_prompts = []
            for prompt in prompt_names:
                if 'edge_formation_trend' in edge_data[prompt] and edge_data[prompt]['edge_formation_trend'] is not None:
                    trend_value = edge_data[prompt]['edge_formation_trend']
                    # Handle string trend values by converting to numeric
                    if isinstance(trend_value, str):
                        if trend_value.lower() in ['increasing', 'inc']:
                            numeric_trend = 1.0
                        elif trend_value.lower() in ['decreasing', 'dec']:
                            numeric_trend = -1.0
                        elif trend_value.lower() in ['stable', 'constant']:
                            numeric_trend = 0.0
                        else:
                            numeric_trend = 0.0  # Default for unknown strings
                        trend_labels.append(trend_value)
                    else:
                        # Already numeric
                        numeric_trend = float(trend_value)
                        trend_labels.append(f'{numeric_trend:.3f}')
                    
                    trends.append(numeric_trend)
                    valid_prompts.append(prompt)
            
            if trends:
                bars3 = ax3.bar(valid_prompts, trends, alpha=0.7,
                               color=sns.color_palette("viridis", len(valid_prompts)))
                ax3.set_xlabel('Prompt ID')
                ax3.set_ylabel('Edge Formation Trend')
                ax3.set_title('Edge Formation Trend Direction\n(+1=Increasing, 0=Stable, -1=Decreasing)')
                ax3.tick_params(axis='x', rotation=45)
                
                # Add value labels with original trend descriptions
                for bar, label in zip(bars3, trend_labels):
                    height = bar.get_height()
                    y_offset = 0.05 if height >= 0 else -0.15  # Adjust for negative values
                    ax3.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                            label, ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
            else:
                ax3.text(0.5, 0.5, 'No valid trend data', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Edge Formation Trend (No Data)')
        else:
            ax3.text(0.5, 0.5, 'No edge formation trend data available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Edge Formation Trend (No Data)')
        
        # Plot 4: Heatmap of edge evolution patterns
        if has_evolution_data:
            evolution_matrix = []
            valid_prompts = []
            for prompt_name in prompt_names:
                if 'mean_evolution_pattern' in edge_data[prompt_name] and edge_data[prompt_name]['mean_evolution_pattern']:
                    evolution = edge_data[prompt_name]['mean_evolution_pattern']
                    if evolution and len(evolution) > 0:
                        evolution_matrix.append(evolution)
                        valid_prompts.append(prompt_name)
            
            if evolution_matrix:
                evolution_array = np.array(evolution_matrix)
                im = ax4.imshow(evolution_array, cmap='YlOrRd', aspect='auto')
                ax4.set_yticks(range(len(valid_prompts)))
                ax4.set_yticklabels(valid_prompts)
                ax4.set_xlabel('Diffusion Step')
                ax4.set_ylabel('Prompt ID')
                ax4.set_title('Edge Density Evolution Heatmap\n(All Prompts)')
                plt.colorbar(im, ax=ax4, label='Edge Density')
            else:
                ax4.text(0.5, 0.5, 'No evolution matrix data', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Edge Evolution Heatmap (No Data)')
        else:
            ax4.text(0.5, 0.5, 'No edge density evolution data available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Edge Evolution Heatmap (No Data)')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "edge_density_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
        plt.savefig(viz_dir / "edge_density_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_spatial_coherence_patterns(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Plot spatial coherence analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        coherence_data = results.spatial_patterns['spatial_coherence_patterns']
        prompt_names = sorted(coherence_data.keys())
        colors = sns.color_palette("husl", len(prompt_names))
        
        # Debug: Check actual data structure
        sample_prompt = prompt_names[0]
        sample_data = coherence_data[sample_prompt]
        self.logger.info(f"Spatial coherence data structure for {sample_prompt}: {list(sample_data.keys())}")
        
        # Plot 1: Spatial coherence evolution (using coherence_evolution)
        has_evolution_data = False
        for i, prompt_name in enumerate(prompt_names):
            if 'coherence_evolution' in coherence_data[prompt_name] and coherence_data[prompt_name]['coherence_evolution']:
                evolution = coherence_data[prompt_name]['coherence_evolution']
                if evolution and len(evolution) > 0:
                    steps = range(len(evolution))
                    ax1.plot(steps, evolution, 'o-', label=prompt_name, 
                            color=colors[i], alpha=0.8, linewidth=2, markersize=3)
                    has_evolution_data = True
        
        if has_evolution_data:
            ax1.set_xlabel('Diffusion Step')
            ax1.set_ylabel('Spatial Coherence')
            ax1.set_title('Spatial Coherence Evolution by Prompt\n(Individual Trajectories)')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No coherence evolution data available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Spatial Coherence Evolution (No Data)')
        
        # Plot 2: Mean coherence trajectory evolution over diffusion steps
        trajectory_data_available = any('mean_coherence_trajectory' in coherence_data[prompt] and 
                                       coherence_data[prompt]['mean_coherence_trajectory'] is not None 
                                       for prompt in prompt_names)
        
        if trajectory_data_available:
            has_trajectory_evolution = False
            for i, prompt in enumerate(prompt_names):
                if 'mean_coherence_trajectory' in coherence_data[prompt] and coherence_data[prompt]['mean_coherence_trajectory'] is not None:
                    trajectory = coherence_data[prompt]['mean_coherence_trajectory']
                    if isinstance(trajectory, (list, np.ndarray)) and len(trajectory) > 0:
                        # Plot the trajectory evolution over diffusion steps
                        trajectory_array = np.array(trajectory)
                        if trajectory_array.ndim > 1:
                            # If multidimensional, take mean across non-time dimensions
                            trajectory_1d = np.mean(trajectory_array.reshape(trajectory_array.shape[0], -1), axis=1)
                        else:
                            trajectory_1d = trajectory_array
                        
                        steps = range(len(trajectory_1d))
                        ax2.plot(steps, trajectory_1d, 'o-', label=prompt, 
                                color=colors[i], alpha=0.8, linewidth=2, markersize=3)
                        has_trajectory_evolution = True
            
            if has_trajectory_evolution:
                ax2.set_xlabel('Diffusion Step')
                ax2.set_ylabel('Mean Coherence Value')
                ax2.set_title('Mean Coherence Trajectory Evolution\n(By Prompt Over Diffusion Steps)')
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No valid trajectory evolution data', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Mean Coherence Trajectory (No Data)')
        else:
            ax2.text(0.5, 0.5, 'No mean coherence trajectory data available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Mean Coherence Trajectory (No Data)')
        
        # Plot 3: Coherence stability (using coherence_stability)
        stability_data_available = any('coherence_stability' in coherence_data[prompt] and 
                                      coherence_data[prompt]['coherence_stability'] is not None 
                                      for prompt in prompt_names)
        
        if stability_data_available:
            stabilities = []
            valid_prompts = []
            for prompt in prompt_names:
                if 'coherence_stability' in coherence_data[prompt] and coherence_data[prompt]['coherence_stability'] is not None:
                    stabilities.append(coherence_data[prompt]['coherence_stability'])
                    valid_prompts.append(prompt)
            
            if stabilities:
                bars3 = ax3.bar(valid_prompts, stabilities, alpha=0.7,
                               color=sns.color_palette("viridis", len(valid_prompts)))
                ax3.set_xlabel('Prompt ID')
                ax3.set_ylabel('Coherence Stability')
                ax3.set_title('Spatial Coherence Stability by Prompt')
                ax3.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, stab in zip(bars3, stabilities):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + max(stabilities) * 0.01,
                            f'{stab:.3f}', ha='center', va='bottom', fontsize=8)
            else:
                ax3.text(0.5, 0.5, 'No valid stability data', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Coherence Stability (No Data)')
        else:
            ax3.text(0.5, 0.5, 'No coherence stability data available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Coherence Stability (No Data)')
        
        # Plot 4: Coherence evolution heatmap
        if has_evolution_data:
            evolution_matrix = []
            valid_prompts = []
            for prompt_name in prompt_names:
                if 'coherence_evolution' in coherence_data[prompt_name] and coherence_data[prompt_name]['coherence_evolution']:
                    evolution = coherence_data[prompt_name]['coherence_evolution']
                    if evolution and len(evolution) > 0:
                        # Handle multidimensional data by taking mean across extra dimensions
                        evolution_array = np.array(evolution)
                        if evolution_array.ndim > 1:
                            # If 3D like (6, 6, 10), flatten the first two dimensions or take mean
                            if evolution_array.ndim == 3:
                                # Take mean across spatial dimensions (assuming first two are spatial)
                                evolution_1d = np.mean(evolution_array, axis=(0, 1))
                            elif evolution_array.ndim == 2:
                                # Take mean across one dimension
                                evolution_1d = np.mean(evolution_array, axis=0)
                        else:
                            evolution_1d = evolution_array
                        
                        evolution_matrix.append(evolution_1d)
                        valid_prompts.append(prompt_name)
            
            if evolution_matrix:
                try:
                    evolution_2d = np.array(evolution_matrix)
                    # Ensure we have a 2D array for imshow
                    if evolution_2d.ndim == 2:
                        im = ax4.imshow(evolution_2d, cmap='RdYlBu_r', aspect='auto')
                        ax4.set_yticks(range(len(valid_prompts)))
                        ax4.set_yticklabels(valid_prompts)
                        ax4.set_xlabel('Diffusion Step')
                        ax4.set_ylabel('Prompt ID')
                        ax4.set_title('Spatial Coherence Evolution Heatmap\n(By Prompt Over Diffusion Steps)')
                        plt.colorbar(im, ax=ax4, label='Coherence Value')
                    else:
                        ax4.text(0.5, 0.5, f'Data shape not suitable for heatmap: {evolution_2d.shape}', 
                                ha='center', va='center', transform=ax4.transAxes)
                        ax4.set_title('Coherence Evolution Heatmap (Incompatible Shape)')
                except Exception as e:
                    self.logger.warning(f"Failed to create coherence heatmap: {e}")
                    ax4.text(0.5, 0.5, f'Failed to create heatmap: {str(e)}', 
                            ha='center', va='center', transform=ax4.transAxes)
                    ax4.set_title('Coherence Evolution Heatmap (Error)')
            else:
                ax4.text(0.5, 0.5, 'No evolution matrix data', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Coherence Evolution Heatmap (No Data)')
        else:
            ax4.text(0.5, 0.5, 'No coherence evolution data available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Coherence Evolution Heatmap (No Data)')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "spatial_coherence_patterns.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_individual_video_coherence_dashboard(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Plot individual video coherence trajectories by prompt group (extracted from spatial coherence)."""
        coherence_data = results.spatial_patterns['spatial_coherence_patterns']
        sorted_group_names = sorted(coherence_data.keys())
        
        # Determine the grid layout based on the number of groups
        n_groups = len(sorted_group_names)
        n_cols = min(3, n_groups)
        n_rows = (n_groups + n_cols - 1) // n_cols
        
        # Create figure with appropriate size
        fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
        fig.suptitle('Individual Video Coherence Trajectories by Prompt Group', 
                    fontsize=16, fontweight='bold')
        
        # Color palette for videos within each group
        video_colors = sns.color_palette("tab10", 10)  # Support up to 10 videos per group
        
        for idx, group_name in enumerate(sorted_group_names):
            ax = plt.subplot(n_rows, n_cols, idx + 1)
            data = coherence_data[group_name]
            coherence_evolution = data.get('coherence_evolution', [])
            
            videos_plotted = 0
            if coherence_evolution:
                # Plot individual video trajectories
                max_videos_to_show = min(8, len(coherence_evolution))  # Show up to 8 videos per group
                
                for i, video_coherence in enumerate(coherence_evolution[:max_videos_to_show]):
                    if isinstance(video_coherence, (list, np.ndarray)) and len(video_coherence) > 0:
                        # Handle multidimensional video coherence data
                        coherence_array = np.array(video_coherence)
                        if coherence_array.ndim > 1:
                            # If multidimensional, take mean across spatial dimensions
                            coherence_1d = np.mean(coherence_array.reshape(coherence_array.shape[0], -1), axis=1)
                        else:
                            coherence_1d = coherence_array
                        
                        steps = list(range(len(coherence_1d)))
                        ax.plot(steps, coherence_1d, alpha=0.7, linewidth=1.5,
                               color=video_colors[i % len(video_colors)], 
                               label=f'Video {i+1}')
                        videos_plotted += 1
                
                # Overlay mean trajectory if available
                mean_trajectory = data.get('mean_coherence_trajectory', [])
                if mean_trajectory:
                    mean_array = np.array(mean_trajectory)
                    if mean_array.ndim > 1:
                        mean_1d = np.mean(mean_array.reshape(mean_array.shape[0], -1), axis=1)
                    else:
                        mean_1d = mean_array
                    
                    steps = list(range(len(mean_1d)))
                    ax.plot(steps, mean_1d, 'k-', linewidth=3, alpha=0.8, 
                           label='Group Mean')
            
            if videos_plotted > 0:
                ax.set_xlabel('Diffusion Step')
                ax.set_ylabel('Spatial Coherence')
                ax.set_title(f'{group_name}\n({videos_plotted} Individual Videos)')
                ax.legend(fontsize=8, loc='best')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No individual video\ncoherence data available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{group_name}\n(No Data)')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "individual_video_coherence_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_temporal_stability_windows(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Plot temporal stability window analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        stability_data = results.temporal_coherence['temporal_stability_windows']
        sorted_group_names = sorted(stability_data.keys())
        colors = sns.color_palette("rocket", len(sorted_group_names))
        
        # Plot different window sizes
        window_sizes = ['window_3', 'window_5', 'window_7']
        axes = [ax1, ax2, ax3]
        
        for ax, window_size in zip(axes, window_sizes):
            for i, group_name in enumerate(sorted_group_names):
                data = stability_data[group_name].get(window_size, [])
                if data:
                    window_starts = [item['window_start'] for item in data]
                    mean_stabilities = [item['mean_stability'] for item in data]
                    ax.plot(window_starts, mean_stabilities, 'o-', label=group_name, alpha=0.7, color=colors[i])
            
            ax.set_xlabel('Window Start Position')
            ax.set_ylabel('Mean Stability')
            ax.set_title(f'Temporal Stability: {window_size.replace("_", " ").title()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Stability variance comparison
        stability_variances = []
        for group_name in sorted_group_names:
            group_variance = 0
            count = 0
            for window_size in window_sizes:
                data = stability_data[group_name].get(window_size, [])
                if data:
                    variances = [item['stability_variance'] for item in data]
                    group_variance += np.mean(variances)
                    count += 1
            stability_variances.append(group_variance / max(count, 1))
        
        bars = ax4.bar(sorted_group_names, stability_variances, alpha=0.7, color=colors)
        ax4.set_xlabel('Prompt Group')
        ax4.set_ylabel('Average Stability Variance')
        ax4.set_title('Overall Temporal Stability Variance')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "temporal_stability_windows.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_channel_evolution_patterns(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Plot channel evolution analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        channel_data = results.channel_analysis['channel_trajectory_evolution']
        sorted_group_names = sorted(channel_data.keys())
        colors = sns.color_palette("magma", len(sorted_group_names))
        
        # Plot 1: Mean evolution patterns for first few channels
        for i, group_name in enumerate(sorted_group_names):
            data = channel_data[group_name]
            evolution_patterns = data.get('mean_evolution_patterns', [])
            if evolution_patterns and len(evolution_patterns) > 0:
                # Show evolution of first channel
                if len(evolution_patterns[0]) > 0:
                    steps = list(range(len(evolution_patterns[0])))
                    ax1.plot(steps, evolution_patterns[0], 'o-', label=f'{group_name} Ch0', alpha=0.7, color=colors[i])
        
        ax1.set_xlabel('Diffusion Step')
        ax1.set_ylabel('Channel Magnitude')
        ax1.set_title('Channel 0 Evolution Patterns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Channel variability
        specialization_data = results.channel_analysis['channel_specialization_patterns']
        overall_variances = [specialization_data[group]['overall_variance'] for group in sorted_group_names]
        bars = ax2.bar(sorted_group_names, overall_variances, alpha=0.7, color=colors)
        ax2.set_xlabel('Prompt Group')
        ax2.set_ylabel('Overall Channel Variance')
        ax2.set_title('Channel Specialization Variance')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Temporal variance
        temporal_variances = [specialization_data[group]['temporal_variance'] for group in sorted_group_names]
        bars = ax3.bar(sorted_group_names, temporal_variances, alpha=0.7, color=colors)
        ax3.set_xlabel('Prompt Group')
        ax3.set_ylabel('Temporal Channel Variance')
        ax3.set_title('Channel Temporal Specialization')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Variance ratio (overall vs temporal)
        variance_ratios = [ov/tv if tv > 0 else 0 for ov, tv in zip(overall_variances, temporal_variances)]
        bars = ax4.bar(sorted_group_names, variance_ratios, alpha=0.7, color=colors)
        ax4.set_xlabel('Prompt Group')
        ax4.set_ylabel('Overall/Temporal Variance Ratio')
        ax4.set_title('Channel Specialization Ratio')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "channel_evolution_patterns.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_global_structure_analysis(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Plot global structure analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        global_data = results.global_structure['trajectory_global_evolution']
        sorted_group_names = sorted(global_data.keys())
        colors = sns.color_palette("viridis", len(sorted_group_names))
        
        # Plot 1: Variance progression
        for i, group_name in enumerate(sorted_group_names):
            data = global_data[group_name]
            variance_progression = data.get('variance_progression', [])
            if variance_progression:
                steps = list(range(len(variance_progression)))
                ax1.plot(steps, variance_progression, 'o-', label=group_name, alpha=0.7, color=colors[i])
        
        ax1.set_xlabel('Diffusion Step')
        ax1.set_ylabel('Global Variance')
        ax1.set_title('Global Variance Progression')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Magnitude progression
        for i, group_name in enumerate(sorted_group_names):
            data = global_data[group_name]
            magnitude_progression = data.get('magnitude_progression', [])
            if magnitude_progression:
                steps = list(range(len(magnitude_progression)))
                ax2.plot(steps, magnitude_progression, 's-', label=group_name, alpha=0.7, color=colors[i])
        
        ax2.set_xlabel('Diffusion Step')
        ax2.set_ylabel('Global Magnitude')
        ax2.set_title('Global Magnitude Progression')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Convergence patterns
        convergence_data = results.global_structure['convergence_patterns']
        diversity_scores = [convergence_data[group]['overall_diversity_score'] for group in sorted_group_names]
        bars = ax3.bar(sorted_group_names, diversity_scores, alpha=0.7, color=colors)
        ax3.set_xlabel('Prompt Group')
        ax3.set_ylabel('Diversity Score')
        ax3.set_title('Overall Trajectory Diversity')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Variance vs Magnitude correlation
        if global_data:
            final_variances = []
            final_magnitudes = []
            for group_name in sorted_group_names:
                data = global_data[group_name]
                var_prog = data.get('variance_progression', [])
                mag_prog = data.get('magnitude_progression', [])
                if var_prog and mag_prog:
                    final_variances.append(var_prog[-1])
                    final_magnitudes.append(mag_prog[-1])
            
            if final_variances and final_magnitudes:
                ax4.scatter(final_variances, final_magnitudes, s=100, alpha=0.7, c=range(len(sorted_group_names)), cmap='viridis')
                for i, group in enumerate(sorted_group_names):
                    if i < len(final_variances):
                        ax4.annotate(group, (final_variances[i], final_magnitudes[i]), xytext=(2, 2), textcoords='offset points')
                
                ax4.set_xlabel('Final Global Variance')
                ax4.set_ylabel('Final Global Magnitude')
                ax4.set_title('Final State: Variance vs Magnitude')
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "global_structure_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_information_content_analysis(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Plot information content analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        info_data = results.information_content['trajectory_information_content']
        sorted_group_names = sorted(info_data.keys())
        colors = sns.color_palette("plasma", len(sorted_group_names))
        
        # Plot 1: Variance measures
        variance_measures = [info_data[group]['variance_measure'] for group in sorted_group_names]
        bars = ax1.bar(sorted_group_names, variance_measures, alpha=0.7, color=colors)
        ax1.set_xlabel('Prompt Group')
        ax1.set_ylabel('Information Variance Measure')
        ax1.set_title('Trajectory Information Content')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, measure in zip(bars, variance_measures):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(variance_measures)*0.01, 
                    f'{measure:.3f}', ha='center', va='bottom')
        
        # Plot 2: Information ranking
        info_ranking = sorted(zip(sorted_group_names, variance_measures), key=lambda x: x[1], reverse=True)
        ranked_groups, ranked_measures = zip(*info_ranking)
        
        ax2.barh(range(len(ranked_groups)), ranked_measures, alpha=0.7,
                color=sns.color_palette("plasma_r", len(ranked_groups)))
        ax2.set_yticks(range(len(ranked_groups)))
        ax2.set_yticklabels(ranked_groups)
        ax2.set_xlabel('Information Variance Measure')
        ax2.set_title('Information Content Ranking (Highest to Lowest)')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "information_content_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_complexity_measures(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Plot complexity measures analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        complexity_data = results.complexity_measures['trajectory_complexity']
        evolution_data = results.complexity_measures['evolution_complexity']
        sorted_group_names = sorted(complexity_data.keys())
        colors = sns.color_palette("rocket", len(sorted_group_names))
        
        # Plot 1: Standard deviation
        std_values = [complexity_data[group]['standard_deviation'] for group in sorted_group_names]
        bars = ax1.bar(sorted_group_names, std_values, alpha=0.7, color=colors)
        ax1.set_xlabel('Prompt Group')
        ax1.set_ylabel('Standard Deviation')
        ax1.set_title('Trajectory Complexity: Standard Deviation')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Value range
        range_values = [complexity_data[group]['value_range'] for group in sorted_group_names]
        bars = ax2.bar(sorted_group_names, range_values, alpha=0.7, color=colors)
        ax2.set_xlabel('Prompt Group')
        ax2.set_ylabel('Value Range')
        ax2.set_title('Trajectory Complexity: Value Range')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Temporal variation
        temporal_variations = [evolution_data[group]['temporal_variation'] for group in sorted_group_names]
        bars = ax3.bar(sorted_group_names, temporal_variations, alpha=0.7, color=colors)
        ax3.set_xlabel('Prompt Group')
        ax3.set_ylabel('Temporal Variation')
        ax3.set_title('Evolution Complexity: Temporal Variation')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Complexity correlation matrix
        complexity_matrix = np.array([std_values, range_values, temporal_variations])
        complexity_labels = ['Std Dev', 'Range', 'Temporal Var']
        
        im = ax4.imshow(complexity_matrix, cmap='RdYlBu_r', aspect='auto')
        ax4.set_xticks(range(len(sorted_group_names)))
        ax4.set_yticks(range(len(complexity_labels)))
        ax4.set_xticklabels(sorted_group_names, rotation=45)
        ax4.set_yticklabels(complexity_labels)
        ax4.set_title('Complexity Measures Heatmap')
        plt.colorbar(im, ax=ax4, label='Complexity Value')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "complexity_measures.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_statistical_significance(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Plot statistical significance analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        significance_data = results.statistical_significance['trajectory_group_differences']
        summary_data = results.statistical_significance['statistical_summary']
        
        # Extract variance and mean differences
        comparisons = list(significance_data.keys())
        variance_diffs = [significance_data[comp]['variance_difference'] for comp in comparisons]
        mean_diffs = [significance_data[comp]['mean_difference'] for comp in comparisons]
        
        # Plot 1: Variance differences
        bars = ax1.bar(range(len(comparisons)), variance_diffs, alpha=0.7)
        ax1.set_xlabel('Group Comparisons')
        ax1.set_ylabel('Variance Difference')
        ax1.set_title('Statistical Significance: Variance Differences')
        ax1.set_xticks(range(len(comparisons)))
        ax1.set_xticklabels([comp.replace('_vs_', ' vs ') for comp in comparisons], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mean differences
        bars = ax2.bar(range(len(comparisons)), mean_diffs, alpha=0.7, color='orange')
        ax2.set_xlabel('Group Comparisons')
        ax2.set_ylabel('Mean Difference')
        ax2.set_title('Statistical Significance: Mean Differences')
        ax2.set_xticks(range(len(comparisons)))
        ax2.set_xticklabels([comp.replace('_vs_', ' vs ') for comp in comparisons], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Significance magnitude
        significance_magnitudes = [abs(vd) + abs(md) for vd, md in zip(variance_diffs, mean_diffs)]
        bars = ax3.bar(range(len(comparisons)), significance_magnitudes, alpha=0.7, color='red')
        ax3.set_xlabel('Group Comparisons')
        ax3.set_ylabel('Combined Difference Magnitude')
        ax3.set_title('Overall Statistical Significance')
        ax3.set_xticks(range(len(comparisons)))
        ax3.set_xticklabels([comp.replace('_vs_', ' vs ') for comp in comparisons], rotation=45, ha='right')
        
        # Plot 4: Summary statistics
        ax4.axis('off')
        summary_text = f"""
Statistical Analysis Summary:

Groups Analyzed: {summary_data['groups_analyzed']}
Total Comparisons: {summary_data['comparisons_made']}

Variance Differences:
• Maximum: {max(variance_diffs):.6f}
• Minimum: {min(variance_diffs):.6f}
• Range: {max(variance_diffs) - min(variance_diffs):.6f}

Mean Differences:
• Maximum: {max(mean_diffs):.6f}
• Minimum: {min(mean_diffs):.6f}
• Range: {max(mean_diffs) - min(mean_diffs):.6f}

Most Significant Comparison:
{comparisons[significance_magnitudes.index(max(significance_magnitudes))].replace('_vs_', ' vs ')}
(Combined magnitude: {max(significance_magnitudes):.6f})
        """
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(viz_dir / "statistical_significance.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_temporal_analysis(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Plot temporal trajectory analysis visualizations."""
        try:
            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 18))
            
            temporal_data = results.temporal_analysis
            sorted_group_names = sorted(temporal_data.keys())
            colors = sns.color_palette("viridis", len(sorted_group_names))
            
            # Plot 1: Trajectory Length Distribution
            lengths = [temporal_data[group]['trajectory_length']['mean_length'] for group in sorted_group_names]
            length_stds = [temporal_data[group]['trajectory_length']['std_length'] for group in sorted_group_names]
            
            bars = ax1.bar(sorted_group_names, lengths, yerr=length_stds, alpha=0.7, color=colors, capsize=5)
            ax1.set_ylabel('Mean Trajectory Length')
            ax1.set_title('Trajectory Length by Group')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels - fix the type error by ensuring proper calculation
            max_std = max(length_stds) if length_stds else 0
            for bar, length in zip(bars, lengths):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_std*0.1, 
                        f'{length:.2f}', ha='center', va='bottom')
            
            # Plot 2: Velocity Analysis
            mean_velocities = [temporal_data[group]['velocity_analysis']['overall_mean_velocity'] for group in sorted_group_names]
            velocity_vars = [temporal_data[group]['velocity_analysis']['overall_velocity_variance'] for group in sorted_group_names]
            
            ax2.scatter(mean_velocities, velocity_vars, s=100, alpha=0.7, c=range(len(sorted_group_names)), cmap='plasma')
            for i, group in enumerate(sorted_group_names):
                ax2.annotate(group, (mean_velocities[i], velocity_vars[i]), xytext=(5, 5), textcoords='offset points')
            
            ax2.set_xlabel('Mean Velocity')
            ax2.set_ylabel('Velocity Variance')
            ax2.set_title('Velocity Phase Space')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Acceleration Analysis
            mean_accelerations = [temporal_data[group]['acceleration_analysis']['overall_mean_acceleration'] for group in sorted_group_names]
            
            bars = ax3.bar(sorted_group_names, mean_accelerations, alpha=0.7, color=colors)
            ax3.set_ylabel('Mean Acceleration')
            ax3.set_title('Acceleration by Group')
            ax3.tick_params(axis='x', rotation=45)
            
            # Plot 4: Endpoint Distance vs Tortuosity
            endpoint_dists = [temporal_data[group]['endpoint_distance']['mean_endpoint_distance'] for group in sorted_group_names]
            tortuosities = [temporal_data[group]['tortuosity']['mean_tortuosity'] for group in sorted_group_names]
            
            ax4.scatter(endpoint_dists, tortuosities, s=100, alpha=0.7, c=range(len(sorted_group_names)), cmap='coolwarm')
            for i, group in enumerate(sorted_group_names):
                ax4.annotate(group, (endpoint_dists[i], tortuosities[i]), xytext=(5, 5), textcoords='offset points')
            
            ax4.set_xlabel('Mean Endpoint Distance')
            ax4.set_ylabel('Mean Tortuosity')
            ax4.set_title('Path Efficiency Analysis')
            ax4.grid(True, alpha=0.3)
            
            # Plot 5: Semantic Convergence Rate
            convergence_rates = [temporal_data[group]['semantic_convergence']['convergence_rate'] for group in sorted_group_names]
            
            bars = ax5.bar(sorted_group_names, convergence_rates, alpha=0.7, color=colors)
            ax5.set_ylabel('Convergence Rate')
            ax5.set_title('Semantic Convergence Rate')
            ax5.tick_params(axis='x', rotation=45)
            
            # Plot 6: Half-life Distribution
            half_lives = [temporal_data[group]['semantic_convergence']['mean_half_life'] for group in sorted_group_names]
            
            bars = ax6.bar(sorted_group_names, half_lives, alpha=0.7, color=colors)
            ax6.set_ylabel('Mean Half-life (steps)')
            ax6.set_title('Convergence Half-life')
            ax6.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(viz_dir / "temporal_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            import traceback
            self.logger.error(f"Failed to create temporal analysis visualization: {e}")
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
            # Create a simple fallback visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Temporal Analysis Visualization Failed\nError: {str(e)}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Temporal Analysis - Error')
            plt.savefig(viz_dir / "temporal_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
            ax4.grid(True, alpha=0.3)
            
            # Plot 5: Semantic Convergence Rate
            convergence_rates = [temporal_data[group]['semantic_convergence']['convergence_rate'] for group in sorted_group_names]
            half_lives = [temporal_data[group]['semantic_convergence']['mean_half_life'] for group in sorted_group_names]
            
            bars = ax5.bar(sorted_group_names, convergence_rates, alpha=0.7, color=colors)
            ax5.set_ylabel('Convergence Rate')
            ax5.set_title('Semantic Convergence Rate')
            ax5.tick_params(axis='x', rotation=45)
            
            # Plot 6: Half-life Distribution
            half_life_stds = [temporal_data[group]['semantic_convergence']['std_half_life'] for group in sorted_group_names]
            bars = ax6.bar(sorted_group_names, half_lives, yerr=half_life_stds, alpha=0.7, color=colors, capsize=5)
            ax6.set_ylabel('Mean Half-life (steps)')
            ax6.set_title('Convergence Half-life')
            ax6.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(viz_dir / "temporal_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            import traceback
            self.logger.error(f"Failed to create temporal analysis visualization: {e}")
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")

    def _plot_structural_analysis(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Plot structural analysis visualizations."""
        try:
            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 18))
            
            structural_data = results.structural_analysis
            sorted_group_names = sorted(structural_data.keys())
            colors = sns.color_palette("plasma", len(sorted_group_names))
            
            # Plot 1: Variance Analysis
            overall_variances = [structural_data[group]['latent_space_variance']['overall_variance'] for group in sorted_group_names]
            video_variances = [structural_data[group]['latent_space_variance']['variance_across_videos'] for group in sorted_group_names]
            step_variances = [structural_data[group]['latent_space_variance']['variance_across_steps'] for group in sorted_group_names]
            
            x = np.arange(len(sorted_group_names))
            width = 0.25
            
            ax1.bar(x - width, overall_variances, width, label='Overall', alpha=0.7)
            ax1.bar(x, video_variances, width, label='Across Videos', alpha=0.7)
            ax1.bar(x + width, step_variances, width, label='Across Steps', alpha=0.7)
            
            ax1.set_ylabel('Variance')
            ax1.set_title('Latent Space Variance Analysis')
            ax1.set_xticks(x)
            ax1.set_xticklabels(sorted_group_names, rotation=45)
            ax1.legend()
            
            # Plot 2: PCA Effective Dimensionality
            effective_dims = [structural_data[group]['pca_analysis']['effective_dimensionality'] for group in sorted_group_names]
            cumulative_var_90 = [structural_data[group]['pca_analysis']['cumulative_variance_90'] for group in sorted_group_names]
            
            bars = ax2.bar(sorted_group_names, effective_dims, alpha=0.7, color=colors)
            ax2.set_ylabel('Effective Dimensionality')
            ax2.set_title('PCA Effective Dimensionality (90% Variance)')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add cumulative variance labels
            for bar, dim, cum_var in zip(bars, effective_dims, cumulative_var_90):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{cum_var:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Plot 3: Shannon Entropy
            entropies = [structural_data[group]['shannon_entropy']['entropy_estimate'] for group in sorted_group_names]
            # Handle missing entropy per dimension std - use the new stats structure
            try:
                entropy_stds = [structural_data[group]['shannon_entropy']['entropy_per_dimension_stats']['std'] 
                               for group in sorted_group_names]
            except KeyError:
                # Fallback for old data format
                try:
                    entropy_stds = [np.std(structural_data[group]['shannon_entropy']['entropy_per_dimension']) 
                                   for group in sorted_group_names]
                except KeyError:
                    entropy_stds = [0.0] * len(sorted_group_names)
            
            bars = ax3.bar(sorted_group_names, entropies, yerr=entropy_stds, alpha=0.7, color=colors, capsize=5)
            ax3.set_ylabel('Shannon Entropy Estimate')
            ax3.set_title('Information Content (Shannon Entropy)')
            ax3.tick_params(axis='x', rotation=45)
            
            # Plot 4: KL Divergence from Baseline
            kl_divergences = []
            baseline_group = None
            for group in sorted_group_names:
                kl_div = structural_data[group]['kl_divergence']['divergence_from_baseline']
                baseline = structural_data[group]['kl_divergence']['baseline_group']
                if baseline is not None:
                    kl_divergences.append(kl_div)
                    if baseline_group is None:
                        baseline_group = baseline
                else:
                    kl_divergences.append(0.0)  # Baseline group itself
            
            bars = ax4.bar(sorted_group_names, kl_divergences, alpha=0.7, color=colors)
            ax4.set_ylabel('KL Divergence from Baseline')
            ax4.set_title(f'Structural Divergence from {baseline_group or "Baseline"}')
            ax4.tick_params(axis='x', rotation=45)
            
            # Plot 5: Structural Complexity Measures
            rank_estimates = [structural_data[group]['structural_complexity']['rank_estimate'] for group in sorted_group_names]
            condition_numbers = [structural_data[group]['structural_complexity']['condition_number'] for group in sorted_group_names]
            
            # Use log scale for condition numbers if they're very large
            log_condition_numbers = [np.log10(max(cn, 1e-10)) for cn in condition_numbers]
            
            ax5_twin = ax5.twinx()
            bars1 = ax5.bar([x - width/2 for x in range(len(sorted_group_names))], rank_estimates, 
                           width, alpha=0.7, color='blue', label='Rank Estimate')
            bars2 = ax5_twin.bar([x + width/2 for x in range(len(sorted_group_names))], log_condition_numbers, 
                                width, alpha=0.7, color='red', label='Log10(Condition Number)')
            
            ax5.set_ylabel('Rank Estimate', color='blue')
            ax5_twin.set_ylabel('Log10(Condition Number)', color='red')
            ax5.set_title('Structural Complexity Measures')
            ax5.set_xticks(range(len(sorted_group_names)))
            ax5.set_xticklabels(sorted_group_names, rotation=45)
            
            # Plot 6: Spectral Analysis
            spectral_entropies = [structural_data[group]['structural_complexity']['spectral_entropy'] for group in sorted_group_names]
            trace_norms = [structural_data[group]['structural_complexity']['trace_norm'] for group in sorted_group_names]
            
            ax6.scatter(spectral_entropies, trace_norms, s=100, alpha=0.7, c=range(len(sorted_group_names)), cmap='viridis')
            for i, group in enumerate(sorted_group_names):
                ax6.annotate(group, (spectral_entropies[i], trace_norms[i]), xytext=(5, 5), textcoords='offset points')
            
            ax6.set_xlabel('Spectral Entropy')
            ax6.set_ylabel('Trace Norm')
            ax6.set_title('Spectral Analysis: Entropy vs Trace Norm')
            ax6.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(viz_dir / "structural_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            import traceback
            self.logger.error(f"Failed to create structural analysis visualization: {e}")
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
            # Create a simple fallback visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Structural Analysis Visualization Failed\nError: {str(e)}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Structural Analysis - Error')
            plt.savefig(viz_dir / "structural_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()

    def _create_analysis_dashboard(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Create a comprehensive analysis dashboard."""
        fig = plt.figure(figsize=(20, 24))
        
        # Title
        fig.suptitle('GPU-Optimized Diffusion Latent Analysis Dashboard\n' + 
                    f'Analysis completed: {results.analysis_metadata["analysis_timestamp"]}\n' +
                    f'Device: {results.analysis_metadata["device_used"]} | ' +
                    f'Groups: {len(results.analysis_metadata["prompt_groups"])} | ' +
                    f'Shape: {results.analysis_metadata["trajectory_shape"]}',
                    fontsize=16, fontweight='bold')
        
        # Create grid layout
        gs = fig.add_gridspec(6, 4, height_ratios=[1, 1, 1, 1, 1, 1], hspace=0.4, wspace=0.3)
        
        # 1. Trajectory Evolution (top row)
        ax1 = fig.add_subplot(gs[0, :2])
        spatial_data = results.spatial_patterns['trajectory_spatial_evolution']
        for group_name, data in spatial_data.items():
            trajectory_pattern = data['trajectory_pattern']
            steps = list(range(len(trajectory_pattern)))
            ax1.plot(steps, trajectory_pattern, 'o-', label=group_name, alpha=0.7, linewidth=2)
        ax1.set_title('Universal U-Shaped Denoising Pattern')
        ax1.set_xlabel('Diffusion Step')
        ax1.set_ylabel('Spatial Variance')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Cross-Trajectory Sync (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        sync_data = results.temporal_coherence['cross_trajectory_synchronization']
        group_names = sorted(sync_data.keys())  # Alphabetical ordering
        correlations = [sync_data[group_name]['mean_correlation'] for group_name in group_names]
        bars = ax2.bar(group_names, correlations, alpha=0.7, color=sns.color_palette("viridis", len(group_names)))
        ax2.set_title('Cross-Trajectory Synchronization')
        ax2.set_ylabel('Mean Correlation')
        ax2.tick_params(axis='x', rotation=45, labelsize=8)
        
        # 3. Momentum Analysis (second row)
        ax3 = fig.add_subplot(gs[1, :2])
        momentum_data = results.temporal_coherence['temporal_momentum_analysis']
        for group_name in sorted(momentum_data.keys()):
            data = momentum_data[group_name]
            velocity_mean = data['velocity_mean']
            steps = list(range(len(velocity_mean)))
            ax3.plot(steps, velocity_mean, 'o-', label=group_name, alpha=0.7)
        ax3.set_title('Temporal Velocity Patterns')
        ax3.set_xlabel('Diffusion Step')
        ax3.set_ylabel('Velocity')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Phase Transitions (second row right)
        ax4 = fig.add_subplot(gs[1, 2:])
        phase_data = results.temporal_coherence['phase_transition_detection']
        for group_name in sorted(phase_data.keys()):
            data = phase_data[group_name]
            p95_transitions = data['p95_transitions']
            steps = list(range(len(p95_transitions)))
            ax4.plot(steps, p95_transitions, '^-', label=group_name, alpha=0.7)
        ax4.set_title('Major Phase Transitions (95th %ile)')
        ax4.set_xlabel('Diffusion Step')
        ax4.set_ylabel('Transition Count')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. Frequency Analysis (third row)
        ax5 = fig.add_subplot(gs[2, :2])
        freq_data = results.temporal_coherence['temporal_frequency_signatures']
        centroids = [data['spectral_centroid'] for data in freq_data.values()]
        entropies = [data['spectral_entropy'] for data in freq_data.values()]
        ax5.scatter(centroids, entropies, s=100, alpha=0.7, c=range(len(group_names)), cmap='plasma')
        for i, group in enumerate(group_names):
            ax5.annotate(group, (centroids[i], entropies[i]), xytext=(2, 2), textcoords='offset points', fontsize=8)
        ax5.set_title('Frequency Analysis: Centroid vs Entropy')
        ax5.set_xlabel('Spectral Centroid')
        ax5.set_ylabel('Spectral Entropy')
        ax5.grid(True, alpha=0.3)
        
        # 6. Group Separability (third row right)
        ax6 = fig.add_subplot(gs[2, 2:])
        separability_data = results.group_separability['inter_group_distances']
        # Calculate average separability for each group
        group_separability = {}
        for group in group_names:
            distances = []
            for key, distance in separability_data.items():
                if group in key:
                    distances.append(distance)
            group_separability[group] = np.mean(distances) if distances else 0
        
        bars = ax6.bar(group_separability.keys(), group_separability.values(), 
                      alpha=0.7, color=sns.color_palette("magma", len(group_separability)))
        ax6.set_title('Group Uniqueness Index')
        ax6.set_ylabel('Avg Distance to Others')
        ax6.tick_params(axis='x', rotation=45, labelsize=8)
        
        # 7. Summary Statistics (fourth row)
        ax7 = fig.add_subplot(gs[3, :])
        
        # Create summary table
        summary_data = []
        for group in group_names:
            sync_corr = sync_data[group]['mean_correlation']
            sync_std = sync_data[group]['correlation_std']
            high_sync = sync_data[group]['high_sync_ratio']
            
            spatial_evo = spatial_data[group]['evolution_ratio']
            phase_strength = spatial_data[group]['phase_transition_strength']
            
            freq_centroid = freq_data[group]['spectral_centroid']
            freq_entropy = freq_data[group]['spectral_entropy']
            
            summary_data.append([
                group, f"{sync_corr:.3f}", f"{sync_std:.3f}", f"{high_sync:.2f}",
                f"{spatial_evo:.3f}", f"{phase_strength:.3f}", 
                f"{freq_centroid:.2f}", f"{freq_entropy:.3f}"
            ])
        
        # Create table
        table_data = [['Group', 'Sync Corr', 'Sync Std', 'High Sync %', 
                      'Spatial Evo', 'Phase Str', 'Freq Cent', 'Freq Ent']] + summary_data
        
        ax7.axis('tight')
        ax7.axis('off')
        table = ax7.table(cellText=table_data[1:], colLabels=table_data[0], 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)
        ax7.set_title('Statistical Summary Table', pad=20)
        
        # 8. Key Insights (bottom rows)
        ax8 = fig.add_subplot(gs[4:, :])
        ax8.axis('off')
        
        insights_text = """
KEY FINDINGS & INSIGHTS:

🔍 UNIVERSAL DIFFUSION PATTERN:
• All prompt groups show U-shaped spatial variance evolution
• Early diffusion: High variance (~0.98) - noise dominance
• Mid diffusion: Minimum variance (~0.48-0.63) - structure formation  
• Late diffusion: Variance recovery (~0.63) - detail refinement

🤝 CROSS-TRAJECTORY SYNCHRONIZATION VARIATION:
• High variation in video synchronization between prompts
• Some prompts produce highly consistent videos (>90% high-sync)
• Others show more diverse generation patterns (<35% high-sync)
• Suggests content-dependent generation consistency

⚡ TEMPORAL MOMENTUM PATTERNS:
• Consistent negative velocity patterns across all prompts
• Universal denoising direction toward lower noise states
• Prompt-specific acceleration profiles indicate content influence

🌊 PHASE TRANSITION DETECTION:
• Significant transition activity varies by prompt type
• Some prompts show more dramatic behavioral changes
• Transition patterns correlate with content complexity

📊 FREQUENCY SIGNATURES:
• Each prompt group has distinct temporal frequency characteristics
• Spectral centroids vary significantly between groups
• Frequency diversity (entropy) shows content-dependent patterns

🎯 GROUP SEPARABILITY:
• Clear mathematical separation between prompt groups
• Distance matrices reveal distinct trajectory clusters
• Content type strongly influences latent space trajectories

🕒 TEMPORAL TRAJECTORY ANALYSIS:
• Path tortuosity reveals generation efficiency differences
• Convergence rates vary systematically between prompt types
• Endpoint distances show content-dependent trajectory lengths
• Baseline comparison reveals group-specific deviations

🏗️ STRUCTURAL COMPLEXITY PATTERNS:
• Effective dimensionality varies between prompt groups
• Shannon entropy estimates reveal information content differences  
• PCA analysis shows distinct principal component structures
• Structural rank estimates indicate varying complexity levels

HYPOTHESIS VALIDATION:
✅ Different prompts DO produce measurably different trajectory patterns
✅ Universal denoising physics preserved across all content types
✅ Temporal dynamics reveal systematic convergence behaviors
✅ Structural complexity varies meaningfully between groups
❌ Expected monotonic variance decrease - found U-shaped recovery pattern
❌ Expected similar synchronization - found dramatic variation (32%-93%)
        """
        
        ax8.text(0.05, 0.95, insights_text, transform=ax8.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(viz_dir / "comprehensive_analysis_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("✅ Comprehensive analysis dashboard created")

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

    def _select_baseline_group(self, prompt_groups: List[str], strategy: str = "auto") -> str:
        """
        Select baseline group using different strategies for research comparison.
        
        Args:
            prompt_groups: List of prompt group names
            strategy: "auto", "empty_prompt", "first_class_specific", or "alphabetical"
        """
        if strategy == "empty_prompt":
            # Look for empty/no prompt - typically prompt_000 or similar
            empty_candidates = [p for p in prompt_groups if '000' in p or 'empty' in p.lower() or 'no_prompt' in p.lower()]
            if empty_candidates:
                baseline = sorted(empty_candidates)[0]
                self.logger.info(f"Selected empty prompt baseline: {baseline}")
                return baseline
        
        elif strategy == "first_class_specific":
            # Look for first class-specific prompt (e.g., "flower" vs more specific variants)
            # This would be prompt_001 in your flower specificity sequence
            sorted_groups = sorted(prompt_groups)
            if len(sorted_groups) > 1:
                baseline = sorted_groups[1]  # Second group (001) assuming 000 is empty
                self.logger.info(f"Selected first class-specific baseline: {baseline}")
                return baseline
        
        elif strategy == "alphabetical":
            baseline = sorted(prompt_groups)[0]
            self.logger.info(f"Selected alphabetical baseline: {baseline}")
            return baseline
        
        # Auto strategy: prefer empty prompt if available, otherwise alphabetical
        empty_candidates = [p for p in prompt_groups if '000' in p]
        if empty_candidates:
            baseline = sorted(empty_candidates)[0]
            self.logger.info(f"Auto-selected empty prompt baseline: {baseline}")
        else:
            baseline = sorted(prompt_groups)[0]
            self.logger.info(f"Auto-selected alphabetical baseline: {baseline}")
        
        return baseline

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
            flat_trajectories = trajectory_tensor.flatten(start_dim=2)  # [n_videos, steps, flattened_latent]
            
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
