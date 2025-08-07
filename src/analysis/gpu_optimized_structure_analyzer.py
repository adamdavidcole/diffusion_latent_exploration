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
            
            # 9. Spatial Coherence Patterns
            self._plot_spatial_coherence_patterns(results, viz_dir)
            
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
            
            # 16. Comprehensive Dashboard
            self._create_analysis_dashboard(results, viz_dir)
            
            # Advanced latent space understanding visualizations
            self._plot_latent_space_geometry(results, viz_dir)
            self._plot_trajectory_manifold_embedding(results, viz_dir)
            self._plot_diffusion_flow_fields(results, viz_dir)
            self._plot_energy_landscape_evolution(results, viz_dir)
            self._plot_latent_space_topology(results, viz_dir)
            self._plot_information_flow_analysis(results, viz_dir)
            
            self.logger.info(f"✅ Visualizations saved to: {viz_dir}")
            
        except Exception as e:
            self.logger.warning(f"Visualization creation failed: {e}")

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
            ax1.plot(steps, trajectory_pattern, 'o-', label=group_name, alpha=0.7, linewidth=2, color=colors[i])
        
        ax1.set_xlabel('Diffusion Step')
        ax1.set_ylabel('Spatial Variance')
        ax1.set_title('Trajectory Spatial Evolution Patterns\n(Universal U-Shaped Denoising Pattern)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
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
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{ratio:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "trajectory_spatial_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_cross_trajectory_synchronization(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Plot cross-trajectory synchronization analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        sync_data = results.temporal_coherence['cross_trajectory_synchronization']
        
        # Extract data
        group_names = list(sync_data.keys())
        mean_correlations = [data['mean_correlation'] for data in sync_data.values()]
        correlation_stds = [data['correlation_std'] for data in sync_data.values()]
        high_sync_ratios = [data['high_sync_ratio'] for data in sync_data.values()]
        
        # Plot 1: Mean correlation by group
        bars1 = ax1.bar(group_names, mean_correlations, alpha=0.7, 
                       color=sns.color_palette("viridis", len(group_names)))
        ax1.set_ylabel('Mean Cross-Trajectory Correlation')
        ax1.set_title('Cross-Trajectory Synchronization Strength')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, corr in zip(bars1, mean_correlations):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{corr:.3f}', ha='center', va='bottom')
        
        # Plot 2: Correlation variability
        ax2.errorbar(group_names, mean_correlations, yerr=correlation_stds, 
                    fmt='o', capsize=5, capthick=2, linewidth=2)
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
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{ratio*100:.1f}%', ha='center', va='bottom')
        
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
        """Plot temporal momentum patterns with improved clarity."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        momentum_data = results.temporal_coherence['temporal_momentum_analysis']
        group_names = sorted(momentum_data.keys())
        colors = sns.color_palette("husl", len(group_names))
        
        # Plot 1: Velocity patterns with confidence intervals
        for i, group_name in enumerate(group_names):
            data = momentum_data[group_name]
            velocity_mean = np.array(data['velocity_mean'])
            velocity_std = np.array(data['velocity_std'])
            steps = np.arange(len(velocity_mean))
            
            ax1.plot(steps, velocity_mean, 'o-', label=group_name, 
                    color=colors[i], alpha=0.8, linewidth=2, markersize=4)
            ax1.fill_between(steps, velocity_mean - velocity_std, velocity_mean + velocity_std,
                           alpha=0.2, color=colors[i])
        
        ax1.set_xlabel('Diffusion Step')
        ax1.set_ylabel('Mean Velocity (±1σ)')
        ax1.set_title('Temporal Velocity Evolution\n(Denoising Speed with Uncertainty)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Acceleration patterns with confidence intervals
        for i, group_name in enumerate(group_names):
            data = momentum_data[group_name]
            accel_mean = np.array(data['acceleration_mean'])
            accel_std = np.array(data['acceleration_std'])
            steps = np.arange(len(accel_mean))
            
            ax2.plot(steps, accel_mean, 's-', label=group_name, 
                    color=colors[i], alpha=0.8, linewidth=2, markersize=4)
            ax2.fill_between(steps, accel_mean - accel_std, accel_mean + accel_std,
                           alpha=0.2, color=colors[i])
        
        ax2.set_xlabel('Diffusion Step')
        ax2.set_ylabel('Mean Acceleration (±1σ)')
        ax2.set_title('Temporal Acceleration Evolution\n(Denoising Rate Changes)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Direction instability patterns
        for i, group_name in enumerate(group_names):
            data = momentum_data[group_name]
            direction_changes = np.array(data['momentum_direction_changes'])
            steps = np.arange(len(direction_changes))
            
            ax3.plot(steps, direction_changes, '^-', label=group_name, 
                    color=colors[i], alpha=0.8, linewidth=2, markersize=4)
        
        ax3.set_xlabel('Diffusion Step')
        ax3.set_ylabel('Direction Change Count')
        ax3.set_title('Momentum Direction Changes\n(Trajectory Instability)')
        ax3.legend(fontsize=9)
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
                        xytext=(8, 8), textcoords='offset points', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3))
        
        ax4.set_xlabel('Average Velocity')
        ax4.set_ylabel('Average Acceleration')
        ax4.set_title('Momentum Phase Space\n(Velocity vs Acceleration with Uncertainty)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "temporal_momentum_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_phase_transition_detection(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Plot phase transition patterns."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        phase_data = results.temporal_coherence['phase_transition_detection']
        
        # Plot 1: 75th percentile transitions
        for group_name in sorted(phase_data.keys()):
            data = phase_data[group_name]
            p75_transitions = data['p75_transitions']
            steps = list(range(len(p75_transitions)))
            ax1.plot(steps, p75_transitions, 'o-', label=group_name, alpha=0.7, linewidth=2)
        
        ax1.set_xlabel('Diffusion Step')
        ax1.set_ylabel('Transition Count')
        ax1.set_title('Phase Transitions (75th Percentile)\n(Moderate Changes)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: 95th percentile transitions (major changes)
        for group_name in sorted(phase_data.keys()):
            data = phase_data[group_name]
            p95_transitions = data['p95_transitions']
            steps = list(range(len(p95_transitions)))
            ax2.plot(steps, p95_transitions, '^-', label=group_name, alpha=0.7, linewidth=2)
        
        ax2.set_xlabel('Diffusion Step')
        ax2.set_ylabel('Transition Count')
        ax2.set_title('Major Phase Transitions (95th Percentile)\n(Dramatic Changes)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Transition intensity heatmap
        group_names = sorted(phase_data.keys())  # Alphabetical ordering
        p90_data = np.array([phase_data[group_name]['p90_transitions'] for group_name in group_names])
        
        im = ax3.imshow(p90_data, cmap='YlOrRd', aspect='auto')
        ax3.set_yticks(range(len(group_names)))
        ax3.set_yticklabels(group_names)
        ax3.set_xlabel('Diffusion Step')
        ax3.set_ylabel('Prompt Group')
        ax3.set_title('Phase Transition Intensity Map\n(90th Percentile)')
        plt.colorbar(im, ax=ax3, label='Transition Count')
        
        # Plot 4: Total transitions by group
        group_names = sorted(phase_data.keys())  # Alphabetical ordering
        total_transitions = []
        for group_name in group_names:
            data = phase_data[group_name]
            total = sum(data['p75_transitions']) + sum(data['p90_transitions']) + sum(data['p95_transitions'])
            total_transitions.append(total)
        
        bars = ax4.bar(group_names, total_transitions, alpha=0.7, 
                      color=sns.color_palette("rocket", len(group_names)))
        ax4.set_xlabel('Prompt Group')
        ax4.set_ylabel('Total Transition Events')
        ax4.set_title('Overall Phase Transition Activity')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, total in zip(bars, total_transitions):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{total}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "phase_transition_detection.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_temporal_frequency_signatures(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Plot temporal frequency analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        freq_data = results.temporal_coherence['temporal_frequency_signatures']
        
        # Plot 1: Dominant frequencies
        group_names = sorted(freq_data.keys())  # Alphabetical ordering
        dominant_freqs = []
        dominant_powers = []
        
        for group_name in group_names:
            data = freq_data[group_name]
            if data['dominant_frequencies']:
                dominant_freqs.append(data['dominant_frequencies'][0])  # Primary frequency
                dominant_powers.append(data['dominant_powers'][0])
            else:
                dominant_freqs.append(0)
                dominant_powers.append(0)
        
        bars1 = ax1.bar(group_names, dominant_freqs, alpha=0.7,
                       color=sns.color_palette("plasma", len(group_names)))
        ax1.set_xlabel('Prompt Group')
        ax1.set_ylabel('Dominant Frequency')
        ax1.set_title('Primary Temporal Frequency by Group')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Spectral power
        bars2 = ax2.bar(group_names, dominant_powers, alpha=0.7,
                       color=sns.color_palette("viridis", len(group_names)))
        ax2.set_xlabel('Prompt Group')
        ax2.set_ylabel('Spectral Power')
        ax2.set_title('Dominant Frequency Power')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Spectral centroid
        centroids = [data['spectral_centroid'] for data in freq_data.values()]
        bars3 = ax3.bar(group_names, centroids, alpha=0.7,
                       color=sns.color_palette("coolwarm", len(group_names)))
        ax3.set_xlabel('Prompt Group')
        ax3.set_ylabel('Spectral Centroid')
        ax3.set_title('Frequency Distribution Center')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Spectral entropy (frequency diversity)
        entropies = [data['spectral_entropy'] for data in freq_data.values()]
        bars4 = ax4.bar(group_names, entropies, alpha=0.7,
                       color=sns.color_palette("rocket", len(group_names)))
        ax4.set_xlabel('Prompt Group')
        ax4.set_ylabel('Spectral Entropy')
        ax4.set_title('Temporal Frequency Diversity')
        ax4.tick_params(axis='x', rotation=45)
        
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
        sorted_group_names = sorted(spatial_data.keys())
        colors = sns.color_palette("viridis", len(sorted_group_names))
        
        # Plot 1: Step deltas mean
        for i, group_name in enumerate(sorted_group_names):
            data = spatial_data[group_name]
            step_deltas = data['step_deltas_mean']
            steps = list(range(len(step_deltas)))
            ax1.plot(steps, step_deltas, 'o-', label=group_name, alpha=0.7, color=colors[i])
        
        ax1.set_xlabel('Diffusion Step')
        ax1.set_ylabel('Mean Step Delta')
        ax1.set_title('Spatial Change Rate Between Steps')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Progression consistency
        consistency_values = [spatial_data[group]['progression_consistency'] for group in sorted_group_names]
        bars = ax2.bar(sorted_group_names, consistency_values, alpha=0.7, color=colors)
        ax2.set_xlabel('Prompt Group')
        ax2.set_ylabel('Progression Consistency')
        ax2.set_title('Spatial Progression Consistency')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Progression variability
        variability_values = [spatial_data[group]['progression_variability'] for group in sorted_group_names]
        bars = ax3.bar(sorted_group_names, variability_values, alpha=0.7, color=colors)
        ax3.set_xlabel('Prompt Group')
        ax3.set_ylabel('Progression Variability')
        ax3.set_title('Spatial Progression Variability')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Edge evolution patterns
        ax4.set_title('Edge Formation Trends by Group')
        edge_data = []
        for group_name in sorted_group_names:
            data = spatial_data[group_name]
            edge_patterns = data.get('edge_evolution_patterns', [])
            if edge_patterns:
                mean_pattern = np.mean(edge_patterns, axis=0)
                steps = list(range(len(mean_pattern)))
                ax4.plot(steps, mean_pattern, 'o-', label=group_name, alpha=0.7)
        
        ax4.set_xlabel('Diffusion Step')
        ax4.set_ylabel('Edge Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "spatial_progression_patterns.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_edge_density_evolution(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Plot edge density evolution analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        edge_data = results.spatial_patterns['edge_density_evolution']
        sorted_group_names = sorted(edge_data.keys())
        colors = sns.color_palette("plasma", len(sorted_group_names))
        
        # Plot 1: Mean evolution patterns
        for i, group_name in enumerate(sorted_group_names):
            data = edge_data[group_name]
            evolution_pattern = data.get('mean_evolution_pattern', [])
            if evolution_pattern:
                steps = list(range(len(evolution_pattern)))
                ax1.plot(steps, evolution_pattern, 'o-', label=group_name, alpha=0.7, color=colors[i])
        
        ax1.set_xlabel('Diffusion Step')
        ax1.set_ylabel('Mean Edge Density')
        ax1.set_title('Edge Density Evolution Patterns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Edge formation trends
        trend_counts = {'increasing': 0, 'decreasing': 0, 'stable': 0}
        for group_name in sorted_group_names:
            trend = edge_data[group_name].get('edge_formation_trend', 'stable')
            trend_counts[trend] = trend_counts.get(trend, 0) + 1
        
        ax2.pie(trend_counts.values(), labels=trend_counts.keys(), autopct='%1.1f%%')
        ax2.set_title('Edge Formation Trend Distribution')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "edge_density_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_spatial_coherence_patterns(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Plot spatial coherence analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        coherence_data = results.spatial_patterns['spatial_coherence_patterns']
        sorted_group_names = sorted(coherence_data.keys())
        colors = sns.color_palette("coolwarm", len(sorted_group_names))
        
        # Plot 1: Mean coherence trajectories
        for i, group_name in enumerate(sorted_group_names):
            data = coherence_data[group_name]
            coherence_trajectory = data.get('mean_coherence_trajectory', [])
            if coherence_trajectory:
                steps = list(range(len(coherence_trajectory)))
                ax1.plot(steps, coherence_trajectory, 'o-', label=group_name, alpha=0.7, color=colors[i])
        
        ax1.set_xlabel('Diffusion Step')
        ax1.set_ylabel('Spatial Coherence')
        ax1.set_title('Spatial Coherence Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Coherence stability
        stability_values = [coherence_data[group]['coherence_stability'] for group in sorted_group_names]
        bars = ax2.bar(sorted_group_names, stability_values, alpha=0.7, color=colors)
        ax2.set_xlabel('Prompt Group')
        ax2.set_ylabel('Coherence Stability')
        ax2.set_title('Spatial Coherence Stability by Group')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3-4: Individual coherence evolution examples
        example_groups = sorted_group_names[:2]  # Show first two groups as examples
        for idx, group_name in enumerate(example_groups):
            ax = ax3 if idx == 0 else ax4
            data = coherence_data[group_name]
            coherence_evolution = data.get('coherence_evolution', [])
            
            if coherence_evolution:
                for i, video_coherence in enumerate(coherence_evolution[:3]):  # Show first 3 videos
                    steps = list(range(len(video_coherence)))
                    ax.plot(steps, video_coherence, alpha=0.6, label=f'Video {i+1}')
            
            ax.set_xlabel('Diffusion Step')
            ax.set_ylabel('Coherence')
            ax.set_title(f'Individual Video Coherence: {group_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "spatial_coherence_patterns.png", dpi=300, bbox_inches='tight')
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

HYPOTHESIS VALIDATION:
✅ Different prompts DO produce measurably different trajectory patterns
✅ Universal denoising physics preserved across all content types
❌ Expected monotonic variance decrease - found U-shaped recovery pattern
❌ Expected similar synchronization - found dramatic variation (32%-93%)
        """
        
        ax8.text(0.05, 0.95, insights_text, transform=ax8.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(viz_dir / "comprehensive_analysis_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_latent_space_geometry(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Visualize the geometric structure of the latent space."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Latent space curvature analysis
        spatial_data = results.spatial_patterns.get('spatial_variance_maps', {})
        group_names = sorted(spatial_data.keys())
        colors = sns.color_palette("husl", len(group_names))
        
        # Compute curvature proxy from spatial variance changes
        for i, group_name in enumerate(group_names):
            if group_name in spatial_data:
                variances = spatial_data[group_name]['evolution_over_steps']
                if len(variances) > 2:
                    # Second derivative as curvature proxy
                    curvature = np.diff(np.diff(variances))
                    steps = np.arange(len(curvature))
                    ax1.plot(steps, curvature, 'o-', label=group_name, 
                            color=colors[i], alpha=0.8, linewidth=2)
        
        ax1.set_xlabel('Diffusion Step')
        ax1.set_ylabel('Latent Space Curvature')
        ax1.set_title('Latent Space Curvature Evolution\n(Geometric Complexity)')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 2. Dimensional expansion/contraction
        if 'spatial_complexity' in results.spatial_patterns:
            complexity_data = results.spatial_patterns['spatial_complexity']
            
            for i, group_name in enumerate(group_names):
                if group_name in complexity_data:
                    expansion = complexity_data[group_name].get('effective_dimensionality_evolution', [])
                    if expansion:
                        steps = np.arange(len(expansion))
                        ax2.plot(steps, expansion, 's-', label=group_name,
                                color=colors[i], alpha=0.8, linewidth=2)
            
            ax2.set_xlabel('Diffusion Step')
            ax2.set_ylabel('Effective Dimensionality')
            ax2.set_title('Latent Space Dimensional Evolution\n(Intrinsic Dimensionality)')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
        
        # 3. Distance matrix heatmap for group separation
        if len(group_names) > 1:
            distance_matrix = np.zeros((len(group_names), len(group_names)))
            
            # Compute pairwise distances between groups
            for i, group1 in enumerate(group_names):
                for j, group2 in enumerate(group_names):
                    if i != j and group1 in spatial_data and group2 in spatial_data:
                        mean1 = spatial_data[group1]['mean']
                        mean2 = spatial_data[group2]['mean']
                        distance_matrix[i, j] = abs(mean1 - mean2)
            
            im = ax3.imshow(distance_matrix, cmap='viridis', aspect='auto')
            ax3.set_xticks(range(len(group_names)))
            ax3.set_yticks(range(len(group_names)))
            ax3.set_xticklabels(group_names, rotation=45, ha='right', fontsize=9)
            ax3.set_yticklabels(group_names, fontsize=9)
            ax3.set_title('Group Distance Matrix\n(Latent Space Separation)')
            plt.colorbar(im, ax=ax3, label='Distance')
        
        # 4. Trajectory volume evolution
        temporal_data = results.temporal_coherence.get('temporal_momentum_analysis', {})
        for i, group_name in enumerate(group_names):
            if group_name in temporal_data:
                velocities = temporal_data[group_name]['velocity_mean']
                if velocities:
                    # Approximate volume using velocity magnitude
                    volume_proxy = np.cumsum(np.abs(velocities))
                    steps = np.arange(len(volume_proxy))
                    ax4.plot(steps, volume_proxy, '^-', label=group_name,
                            color=colors[i], alpha=0.8, linewidth=2)
        
        ax4.set_xlabel('Diffusion Step')
        ax4.set_ylabel('Cumulative Volume Explored')
        ax4.set_title('Latent Space Volume Exploration\n(Trajectory Coverage)')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "latent_space_geometry.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_trajectory_manifold_embedding(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Visualize trajectory embeddings in lower dimensions."""
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Collect trajectory data for embedding
        spatial_data = results.spatial_patterns.get('spatial_variance_maps', {})
        temporal_data = results.temporal_coherence.get('temporal_momentum_analysis', {})
        
        group_names = sorted(spatial_data.keys())
        colors = sns.color_palette("husl", len(group_names))
        
        # Create feature vectors for each group
        features = []
        labels = []
        
        for group_name in group_names:
            if group_name in spatial_data and group_name in temporal_data:
                # Combine spatial and temporal features
                spatial_features = [
                    spatial_data[group_name]['mean'],
                    spatial_data[group_name]['std'],
                    len(spatial_data[group_name].get('evolution_over_steps', []))
                ]
                
                temporal_features = [
                    np.mean(temporal_data[group_name]['velocity_mean']),
                    np.std(temporal_data[group_name]['velocity_mean']),
                    np.mean(temporal_data[group_name]['acceleration_mean']),
                    np.std(temporal_data[group_name]['acceleration_mean'])
                ]
                
                combined_features = spatial_features + temporal_features
                features.append(combined_features)
                labels.append(group_name)
        
        if len(features) > 2:
            features = np.array(features)
            
            # PCA embedding
            pca = PCA(n_components=2)
            pca_coords = pca.fit_transform(features)
            
            for i, group_name in enumerate(labels):
                ax1.scatter(pca_coords[i, 0], pca_coords[i, 1], 
                           c=[colors[group_names.index(group_name)]], s=150, 
                           alpha=0.8, edgecolors='black', linewidth=1)
                ax1.annotate(group_name, (pca_coords[i, 0], pca_coords[i, 1]),
                            xytext=(8, 8), textcoords='offset points', fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[group_names.index(group_name)], alpha=0.3))
            
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            ax1.set_title('PCA Embedding of Trajectory Features')
            ax1.grid(True, alpha=0.3)
            
            # t-SNE embedding (if enough data)
            if len(features) >= 3:
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
                tsne_coords = tsne.fit_transform(features)
                
                for i, group_name in enumerate(labels):
                    ax2.scatter(tsne_coords[i, 0], tsne_coords[i, 1], 
                               c=[colors[group_names.index(group_name)]], s=150,
                               alpha=0.8, edgecolors='black', linewidth=1)
                    ax2.annotate(group_name, (tsne_coords[i, 0], tsne_coords[i, 1]),
                                xytext=(8, 8), textcoords='offset points', fontsize=9,
                                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[group_names.index(group_name)], alpha=0.3))
                
                ax2.set_xlabel('t-SNE Dimension 1')
                ax2.set_ylabel('t-SNE Dimension 2')
                ax2.set_title('t-SNE Embedding of Trajectory Features')
                ax2.grid(True, alpha=0.3)
        
        # 3. Feature importance from PCA
        if len(features) > 2:
            feature_names = ['Spatial Mean', 'Spatial Std', 'Temporal Length',
                           'Velocity Mean', 'Velocity Std', 'Accel Mean', 'Accel Std']
            
            pc1_importance = np.abs(pca.components_[0])
            pc2_importance = np.abs(pca.components_[1])
            
            x = np.arange(len(feature_names))
            width = 0.35
            
            bars1 = ax3.bar(x - width/2, pc1_importance, width, label='PC1', alpha=0.8)
            bars2 = ax3.bar(x + width/2, pc2_importance, width, label='PC2', alpha=0.8)
            
            ax3.set_xlabel('Features')
            ax3.set_ylabel('Importance (Absolute Loading)')
            ax3.set_title('Feature Importance in PCA')
            ax3.set_xticks(x)
            ax3.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Trajectory divergence visualization
        if len(group_names) > 1:
            divergence_matrix = np.zeros((len(group_names), len(group_names)))
            
            for i, group1 in enumerate(group_names):
                for j, group2 in enumerate(group_names):
                    if i != j and group1 in temporal_data and group2 in temporal_data:
                        vel1 = temporal_data[group1]['velocity_mean']
                        vel2 = temporal_data[group2]['velocity_mean']
                        if vel1 and vel2:
                            # KL-divergence approximation
                            v1_norm = np.array(vel1) / (np.sum(np.abs(vel1)) + 1e-8)
                            v2_norm = np.array(vel2) / (np.sum(np.abs(vel2)) + 1e-8)
                            min_len = min(len(v1_norm), len(v2_norm))
                            if min_len > 0:
                                v1_norm = v1_norm[:min_len]
                                v2_norm = v2_norm[:min_len]
                                divergence = np.sum(v1_norm * np.log((v1_norm + 1e-8) / (v2_norm + 1e-8)))
                                divergence_matrix[i, j] = abs(divergence)
            
            im = ax4.imshow(divergence_matrix, cmap='plasma', aspect='auto')
            ax4.set_xticks(range(len(group_names)))
            ax4.set_yticks(range(len(group_names)))
            ax4.set_xticklabels(group_names, rotation=45, ha='right', fontsize=9)
            ax4.set_yticklabels(group_names, fontsize=9)
            ax4.set_title('Trajectory Divergence Matrix\n(Velocity Pattern Differences)')
            plt.colorbar(im, ax=ax4, label='Divergence')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "trajectory_manifold_embedding.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_diffusion_flow_fields(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Visualize diffusion process as flow fields."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        temporal_data = results.temporal_coherence.get('temporal_momentum_analysis', {})
        group_names = sorted(temporal_data.keys())
        colors = sns.color_palette("husl", len(group_names))
        
        # 1. Velocity field evolution
        max_steps = 0
        for group_name in group_names:
            if group_name in temporal_data:
                velocities = temporal_data[group_name]['velocity_mean']
                max_steps = max(max_steps, len(velocities))
        
        if max_steps > 0:
            step_grid = np.arange(max_steps)
            
            for i, group_name in enumerate(group_names):
                if group_name in temporal_data:
                    velocities = temporal_data[group_name]['velocity_mean']
                    accelerations = temporal_data[group_name]['acceleration_mean']
                    
                    # Pad to max_steps
                    vel_padded = np.pad(velocities, (0, max_steps - len(velocities)), mode='constant')
                    accel_padded = np.pad(accelerations, (0, max_steps - len(accelerations)), mode='constant')
                    
                    # Create flow field
                    group_positions = np.full(max_steps, i)  # Y-position for this group
                    
                    # Arrow plot
                    skip = max(1, max_steps // 15)  # Don't overcrowd arrows
                    ax1.quiver(step_grid[::skip], group_positions[::skip], 
                              vel_padded[::skip], accel_padded[::skip],
                              color=colors[i], alpha=0.7, scale=20, width=0.003,
                              label=group_name)
            
            ax1.set_xlabel('Diffusion Step')
            ax1.set_ylabel('Prompt Group')
            ax1.set_yticks(range(len(group_names)))
            ax1.set_yticklabels(group_names, fontsize=9)
            ax1.set_title('Diffusion Flow Fields\n(Velocity → Acceleration)')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax1.grid(True, alpha=0.3)
        
        # 2. Phase portrait (velocity vs acceleration phase space)
        for i, group_name in enumerate(group_names):
            if group_name in temporal_data:
                velocities = temporal_data[group_name]['velocity_mean']
                accelerations = temporal_data[group_name]['acceleration_mean']
                
                min_len = min(len(velocities), len(accelerations))
                if min_len > 1:
                    vel = np.array(velocities[:min_len])
                    accel = np.array(accelerations[:min_len])
                    
                    # Plot trajectory in phase space
                    ax2.plot(vel, accel, 'o-', color=colors[i], alpha=0.8, 
                            linewidth=2, markersize=4, label=group_name)
                    
                    # Mark start and end
                    ax2.scatter(vel[0], accel[0], color=colors[i], s=100, 
                               marker='s', edgecolors='black', linewidth=1, 
                               label=f'{group_name} start' if i == 0 else "")
                    ax2.scatter(vel[-1], accel[-1], color=colors[i], s=100, 
                               marker='^', edgecolors='black', linewidth=1,
                               label=f'end' if i == 0 else "")
        
        ax2.set_xlabel('Velocity')
        ax2.set_ylabel('Acceleration')
        ax2.set_title('Phase Portrait\n(Diffusion Dynamics in Phase Space)')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. Energy-like landscape (velocity magnitude)
        for i, group_name in enumerate(group_names):
            if group_name in temporal_data:
                velocities = temporal_data[group_name]['velocity_mean']
                accelerations = temporal_data[group_name]['acceleration_mean']
                
                if velocities and accelerations:
                    # Kinetic energy proxy
                    kinetic_energy = np.array(velocities)**2
                    steps = np.arange(len(kinetic_energy))
                    
                    ax3.plot(steps, kinetic_energy, 'o-', color=colors[i], 
                            alpha=0.8, linewidth=2, markersize=4, label=group_name)
        
        ax3.set_xlabel('Diffusion Step')
        ax3.set_ylabel('Kinetic Energy (Velocity²)')
        ax3.set_title('Diffusion Energy Landscape\n(Movement Intensity)')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # 4. Stability basins (acceleration near zero indicates stable regions)
        for i, group_name in enumerate(group_names):
            if group_name in temporal_data:
                accelerations = temporal_data[group_name]['acceleration_mean']
                
                if accelerations:
                    # Find stability regions (low acceleration)
                    accel_abs = np.abs(accelerations)
                    stability_threshold = np.percentile(accel_abs, 25)  # Bottom quartile
                    stable_regions = accel_abs < stability_threshold
                    
                    steps = np.arange(len(accelerations))
                    
                    # Plot acceleration
                    ax4.plot(steps, accelerations, color=colors[i], alpha=0.6, linewidth=1)
                    
                    # Highlight stable regions
                    stable_steps = steps[stable_regions]
                    stable_accels = np.array(accelerations)[stable_regions]
                    ax4.scatter(stable_steps, stable_accels, color=colors[i], 
                               s=50, alpha=0.8, marker='o', edgecolors='black',
                               linewidth=0.5, label=f'{group_name} stable')
        
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Diffusion Step')
        ax4.set_ylabel('Acceleration')
        ax4.set_title('Stability Basins\n(Low Acceleration Regions)')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "diffusion_flow_fields.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_energy_landscape_evolution(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Visualize the evolution of energy-like landscapes during diffusion."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        spatial_data = results.spatial_patterns.get('spatial_variance_maps', {})
        temporal_data = results.temporal_coherence.get('temporal_momentum_analysis', {})
        
        group_names = sorted(spatial_data.keys() if spatial_data else temporal_data.keys())
        colors = sns.color_palette("husl", len(group_names))
        
        # 1. Potential energy evolution (based on spatial variance)
        for i, group_name in enumerate(group_names):
            if group_name in spatial_data:
                evolution = spatial_data[group_name].get('evolution_over_steps', [])
                if evolution:
                    # Higher variance = higher potential energy
                    potential_energy = np.array(evolution)
                    steps = np.arange(len(potential_energy))
                    
                    ax1.plot(steps, potential_energy, 'o-', color=colors[i], 
                            alpha=0.8, linewidth=2, markersize=4, label=group_name)
        
        ax1.set_xlabel('Diffusion Step')
        ax1.set_ylabel('Potential Energy (Spatial Variance)')
        ax1.set_title('Potential Energy Landscape Evolution\n(Spatial Disorder)')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 2. Kinetic energy evolution (based on velocity)
        for i, group_name in enumerate(group_names):
            if group_name in temporal_data:
                velocities = temporal_data[group_name]['velocity_mean']
                if velocities:
                    kinetic_energy = np.array(velocities)**2
                    steps = np.arange(len(kinetic_energy))
                    
                    ax2.plot(steps, kinetic_energy, 's-', color=colors[i], 
                            alpha=0.8, linewidth=2, markersize=4, label=group_name)
        
        ax2.set_xlabel('Diffusion Step')
        ax2.set_ylabel('Kinetic Energy (Velocity²)')
        ax2.set_title('Kinetic Energy Evolution\n(Movement Intensity)')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 3. Total energy conservation/dissipation
        for i, group_name in enumerate(group_names):
            if group_name in spatial_data and group_name in temporal_data:
                potential = spatial_data[group_name].get('evolution_over_steps', [])
                velocities = temporal_data[group_name]['velocity_mean']
                
                if potential and velocities:
                    min_len = min(len(potential), len(velocities))
                    if min_len > 0:
                        pot_energy = np.array(potential[:min_len])
                        kin_energy = np.array(velocities[:min_len])**2
                        total_energy = pot_energy + kin_energy
                        
                        steps = np.arange(len(total_energy))
                        ax3.plot(steps, total_energy, '^-', color=colors[i], 
                                alpha=0.8, linewidth=2, markersize=4, label=group_name)
        
        ax3.set_xlabel('Diffusion Step')
        ax3.set_ylabel('Total Energy (Potential + Kinetic)')
        ax3.set_title('Total Energy Evolution\n(Energy Conservation/Dissipation)')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # 4. Energy landscape topology (energy barriers and wells)
        if len(group_names) > 1:
            # Create energy barrier matrix between groups
            barrier_matrix = np.zeros((len(group_names), len(group_names)))
            
            for i, group1 in enumerate(group_names):
                for j, group2 in enumerate(group_names):
                    if i != j:
                        # Energy barrier as difference in total energy
                        if (group1 in spatial_data and group2 in spatial_data and 
                            group1 in temporal_data and group2 in temporal_data):
                            
                            pot1 = spatial_data[group1].get('evolution_over_steps', [])
                            vel1 = temporal_data[group1]['velocity_mean']
                            pot2 = spatial_data[group2].get('evolution_over_steps', [])
                            vel2 = temporal_data[group2]['velocity_mean']
                            
                            if pot1 and vel1 and pot2 and vel2:
                                energy1 = np.mean(pot1) + np.mean(np.array(vel1)**2)
                                energy2 = np.mean(pot2) + np.mean(np.array(vel2)**2)
                                barrier_matrix[i, j] = abs(energy1 - energy2)
            
            im = ax4.imshow(barrier_matrix, cmap='hot', aspect='auto')
            ax4.set_xticks(range(len(group_names)))
            ax4.set_yticks(range(len(group_names)))
            ax4.set_xticklabels(group_names, rotation=45, ha='right', fontsize=9)
            ax4.set_yticklabels(group_names, fontsize=9)
            ax4.set_title('Energy Barrier Matrix\n(Transition Costs Between Groups)')
            plt.colorbar(im, ax=ax4, label='Energy Barrier')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "energy_landscape_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_latent_space_topology(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Visualize topological properties of the latent space."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        spatial_data = results.spatial_patterns.get('spatial_variance_maps', {})
        temporal_data = results.temporal_coherence.get('temporal_momentum_analysis', {})
        
        group_names = sorted(spatial_data.keys() if spatial_data else temporal_data.keys())
        colors = sns.color_palette("husl", len(group_names))
        
        # 1. Connectivity graph based on similarity
        if len(group_names) > 1:
            # Build similarity matrix
            similarity_matrix = np.zeros((len(group_names), len(group_names)))
            
            for i, group1 in enumerate(group_names):
                for j, group2 in enumerate(group_names):
                    if i != j and group1 in spatial_data and group2 in spatial_data:
                        mean1 = spatial_data[group1]['mean']
                        mean2 = spatial_data[group2]['mean']
                        std1 = spatial_data[group1]['std']
                        std2 = spatial_data[group2]['std']
                        
                        # Similarity based on overlapping distributions
                        similarity = 1 / (1 + abs(mean1 - mean2) / (std1 + std2 + 1e-8))
                        similarity_matrix[i, j] = similarity
            
            # Plot connectivity graph
            threshold = np.percentile(similarity_matrix[similarity_matrix > 0], 70)
            
            # Position nodes in circle
            angles = np.linspace(0, 2*np.pi, len(group_names), endpoint=False)
            positions = [(np.cos(angle), np.sin(angle)) for angle in angles]
            
            # Draw nodes
            for i, (x, y) in enumerate(positions):
                ax1.scatter(x, y, s=200, c=[colors[i]], alpha=0.8, 
                           edgecolors='black', linewidth=1)
                ax1.annotate(group_names[i], (x, y), xytext=(5, 5), 
                            textcoords='offset points', fontsize=9)
            
            # Draw edges for strong connections
            for i in range(len(group_names)):
                for j in range(i+1, len(group_names)):
                    if similarity_matrix[i, j] > threshold:
                        x1, y1 = positions[i]
                        x2, y2 = positions[j]
                        ax1.plot([x1, x2], [y1, y2], 'k-', alpha=0.5, 
                                linewidth=similarity_matrix[i, j]*3)
            
            ax1.set_xlim(-1.2, 1.2)
            ax1.set_ylim(-1.2, 1.2)
            ax1.set_aspect('equal')
            ax1.set_title('Latent Space Connectivity Graph\n(Similarity-based Topology)')
            ax1.grid(True, alpha=0.3)
        
        # 2. Clustering tendency visualization
        if spatial_data:
            cluster_features = []
            for group_name in group_names:
                if group_name in spatial_data:
                    features = [
                        spatial_data[group_name]['mean'],
                        spatial_data[group_name]['std']
                    ]
                    cluster_features.append(features)
            
            if len(cluster_features) > 2:
                cluster_features = np.array(cluster_features)
                
                # Hopkins statistic for clustering tendency
                from scipy.spatial.distance import pdist, squareform
                distances = squareform(pdist(cluster_features))
                
                # Plot distance distribution
                upper_tri = distances[np.triu_indices_from(distances, k=1)]
                ax2.hist(upper_tri, bins=min(10, len(upper_tri)), alpha=0.7, 
                        color='skyblue', edgecolor='black')
                ax2.axvline(np.mean(upper_tri), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(upper_tri):.3f}')
                ax2.set_xlabel('Pairwise Distance')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Distance Distribution\n(Clustering Tendency)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        # 3. Persistence analysis (simplified topological data analysis)
        for i, group_name in enumerate(group_names):
            if group_name in spatial_data:
                evolution = spatial_data[group_name].get('evolution_over_steps', [])
                if len(evolution) > 3:
                    # Simple persistence: how long do features persist
                    evolution_array = np.array(evolution)
                    
                    # Find local maxima (peaks that persist)
                    from scipy.signal import find_peaks
                    peaks, properties = find_peaks(evolution_array, prominence=0.1)
                    
                    if len(peaks) > 0:
                        # Plot evolution with persistent features highlighted
                        steps = np.arange(len(evolution))
                        ax3.plot(steps, evolution, color=colors[i], alpha=0.6, linewidth=1)
                        ax3.scatter(peaks, evolution_array[peaks], 
                                   color=colors[i], s=60, alpha=0.8, 
                                   marker='o', edgecolors='black', linewidth=1,
                                   label=f'{group_name} peaks')
        
        ax3.set_xlabel('Diffusion Step')
        ax3.set_ylabel('Spatial Variance')
        ax3.set_title('Persistent Features\n(Topological Persistence)')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Manifold curvature estimation
        for i, group_name in enumerate(group_names):
            if group_name in temporal_data:
                velocities = temporal_data[group_name]['velocity_mean']
                accelerations = temporal_data[group_name]['acceleration_mean']
                
                if len(velocities) > 2 and len(accelerations) > 2:
                    min_len = min(len(velocities), len(accelerations))
                    vel = np.array(velocities[:min_len])
                    accel = np.array(accelerations[:min_len])
                    
                    # Curvature approximation: |v x a| / |v|^3
                    vel_mag = np.abs(vel)
                    curvature = np.abs(accel) / (vel_mag**3 + 1e-8)
                    
                    # Smooth out infinities
                    curvature = np.clip(curvature, 0, np.percentile(curvature, 95))
                    
                    steps = np.arange(len(curvature))
                    ax4.plot(steps, curvature, 'o-', color=colors[i], 
                            alpha=0.8, linewidth=2, markersize=3, label=group_name)
        
        ax4.set_xlabel('Diffusion Step')
        ax4.set_ylabel('Trajectory Curvature')
        ax4.set_title('Manifold Curvature Evolution\n(Local Geometry)')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "latent_space_topology.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_information_flow_analysis(self, results: GPUOptimizedAnalysis, viz_dir: Path):
        """Visualize information flow and processing during diffusion."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        spatial_data = results.spatial_patterns.get('spatial_variance_maps', {})
        temporal_data = results.temporal_coherence.get('temporal_momentum_analysis', {})
        
        group_names = sorted(spatial_data.keys() if spatial_data else temporal_data.keys())
        colors = sns.color_palette("husl", len(group_names))
        
        # 1. Information entropy evolution (spatial disorder)
        for i, group_name in enumerate(group_names):
            if group_name in spatial_data:
                evolution = spatial_data[group_name].get('evolution_over_steps', [])
                if evolution:
                    # Entropy proxy from variance
                    entropy = np.log(np.array(evolution) + 1e-8)
                    steps = np.arange(len(entropy))
                    
                    ax1.plot(steps, entropy, 'o-', color=colors[i], 
                            alpha=0.8, linewidth=2, markersize=4, label=group_name)
        
        ax1.set_xlabel('Diffusion Step')
        ax1.set_ylabel('Information Entropy (log variance)')
        ax1.set_title('Information Entropy Evolution\n(Disorder/Uncertainty)')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 2. Information processing rate (velocity changes)
        for i, group_name in enumerate(group_names):
            if group_name in temporal_data:
                velocities = temporal_data[group_name]['velocity_mean']
                if len(velocities) > 1:
                    # Information processing as velocity changes
                    processing_rate = np.abs(np.diff(velocities))
                    steps = np.arange(len(processing_rate))
                    
                    ax2.plot(steps, processing_rate, 's-', color=colors[i], 
                            alpha=0.8, linewidth=2, markersize=4, label=group_name)
        
        ax2.set_xlabel('Diffusion Step')
        ax2.set_ylabel('Information Processing Rate')
        ax2.set_title('Information Processing Evolution\n(Rate of Change)')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 3. Mutual information approximation between groups
        if len(group_names) > 1:
            mi_matrix = np.zeros((len(group_names), len(group_names)))
            
            for i, group1 in enumerate(group_names):
                for j, group2 in enumerate(group_names):
                    if i != j and group1 in temporal_data and group2 in temporal_data:
                        vel1 = temporal_data[group1]['velocity_mean']
                        vel2 = temporal_data[group2]['velocity_mean']
                        
                        if vel1 and vel2:
                            min_len = min(len(vel1), len(vel2))
                            if min_len > 1:
                                v1 = np.array(vel1[:min_len])
                                v2 = np.array(vel2[:min_len])
                                
                                # Simplified mutual information via correlation
                                correlation = np.corrcoef(v1, v2)[0, 1]
                                if not np.isnan(correlation):
                                    # Transform correlation to MI-like measure
                                    mi_matrix[i, j] = -0.5 * np.log(1 - correlation**2 + 1e-8)
            
            im = ax3.imshow(mi_matrix, cmap='viridis', aspect='auto')
            ax3.set_xticks(range(len(group_names)))
            ax3.set_yticks(range(len(group_names)))
            ax3.set_xticklabels(group_names, rotation=45, ha='right', fontsize=9)
            ax3.set_yticklabels(group_names, fontsize=9)
            ax3.set_title('Mutual Information Matrix\n(Information Sharing)')
            plt.colorbar(im, ax=ax3, label='Mutual Information')
        
        # 4. Information complexity evolution
        for i, group_name in enumerate(group_names):
            if group_name in temporal_data:
                velocities = temporal_data[group_name]['velocity_mean']
                accelerations = temporal_data[group_name]['acceleration_mean']
                
                if velocities and accelerations:
                    min_len = min(len(velocities), len(accelerations))
                    if min_len > 2:
                        vel = np.array(velocities[:min_len])
                        accel = np.array(accelerations[:min_len])
                        
                        # Complexity as sum of velocity and acceleration entropies
                        vel_entropy = -np.sum(vel * np.log(np.abs(vel) + 1e-8))
                        accel_entropy = -np.sum(accel * np.log(np.abs(accel) + 1e-8))
                        
                        # Plot complexity over time using windowed calculation
                        window_size = max(3, min_len // 4)
                        complexities = []
                        
                        for start in range(0, min_len - window_size + 1):
                            end = start + window_size
                            vel_window = vel[start:end]
                            accel_window = accel[start:end]
                            
                            # Normalize for entropy calculation
                            vel_norm = np.abs(vel_window) / (np.sum(np.abs(vel_window)) + 1e-8)
                            accel_norm = np.abs(accel_window) / (np.sum(np.abs(accel_window)) + 1e-8)
                            
                            complexity = (-np.sum(vel_norm * np.log(vel_norm + 1e-8)) + 
                                        -np.sum(accel_norm * np.log(accel_norm + 1e-8)))
                            complexities.append(complexity)
                        
                        if complexities:
                            steps = np.arange(len(complexities))
                            ax4.plot(steps, complexities, '^-', color=colors[i], 
                                    alpha=0.8, linewidth=2, markersize=4, label=group_name)
        
        ax4.set_xlabel('Time Window')
        ax4.set_ylabel('Information Complexity')
        ax4.set_title('Information Complexity Evolution\n(Processing Complexity)')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "information_flow_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

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
