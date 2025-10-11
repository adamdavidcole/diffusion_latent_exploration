import torch
import numpy as np
from typing import Dict, Any
import logging


from .utils.corrcoef import corrcoef
from .utils.find_peaks import find_peaks
from .utils.compute_spectral_entropy import compute_spectral_entropy


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_temporal_coherence(
        group_tensors: Dict[str, Dict[str, torch.Tensor]],
        device: torch.device
) -> Dict[str, Any]:
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
        logger.info(f"GPU analyzing temporal coherence for {group_name}")
        
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
                
                coherence = corrcoef(step1, step2)
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
        change_percentiles = torch.quantile(trajectory_changes, torch.tensor([0.75, 0.9, 0.95]).to(device), dim=0)
        
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
                    correlation = corrcoef(traj_i, traj_j)
                    if not torch.isnan(correlation):
                        correlations.append(correlation.item())
                    
                    # Phase alignment (using peak positions)
                    if len(traj_i) >= 5:
                        peaks_i = find_peaks(traj_i)
                        peaks_j = find_peaks(traj_j)
                        
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
        # TEMPORARILY DISABLED due to CUDA 12.9/PyTorch compatibility issues
        # TODO: Re-enable after fixing tensor dimension issues in FFT analysis
        frequency_analysis = {
            'dominant_frequencies': [],
            'dominant_powers': [],
            'spectral_centroid': 0.0,
            'spectral_entropy': 0.0,
            'status': 'disabled_cuda_12.9_compatibility'
        }
        
        # Original code commented out due to tensor dimension issues:
        # if n_steps >= 8:
        #     # FFT analysis on trajectory norms
        #     fft_results = torch.fft.fft(trajectory_norms, dim=1)  # [n_videos, steps] -> [n_videos, steps]
        #     power_spectra = torch.abs(fft_results) ** 2  # [n_videos, steps]
        #     mean_power_spectrum = torch.mean(power_spectra, dim=0)  # [steps]
        #     
        #     # Find dominant frequencies (skip DC component)
        #     spectrum_length = mean_power_spectrum.shape[0]  # Use shape[0] instead of len()
        #     if spectrum_length > 4:
        #         freq_slice = mean_power_spectrum[1:spectrum_length//2]
        #         if freq_slice.shape[0] > 0:
        #             # Get indices relative to freq_slice, then adjust for the original spectrum
        #             relative_indices = torch.argsort(freq_slice, descending=True)[:min(3, freq_slice.shape[0])]
        #             freq_indices = relative_indices + 1  # Adjust for skipping DC component
        #             
        #             # Ensure indices are within bounds
        #             freq_indices = freq_indices[freq_indices < spectrum_length]
        #             
        #             dominant_freqs = freq_indices.cpu().numpy().tolist()
        #             # mean_power_spectrum is 1D, so direct indexing should work
        #             dominant_powers = [float(mean_power_spectrum[idx].item()) for idx in freq_indices]
        #         else:
        #             dominant_freqs = []
        #             dominant_powers = []
        #         
        #         frequency_analysis = {
        #             'dominant_frequencies': dominant_freqs,
        #             'dominant_powers': dominant_powers,
        #             'spectral_centroid': float(torch.sum(torch.arange(spectrum_length).float().to(device) * mean_power_spectrum) / torch.sum(mean_power_spectrum)),
        #             'spectral_entropy': float(compute_spectral_entropy(mean_power_spectrum))
        #         }
        #     else:
        #         frequency_analysis = {
        #             'dominant_frequencies': [],
        #             'dominant_powers': [],
        #             'spectral_centroid': 0.0,
        #             'spectral_entropy': 0.0
        #         }

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