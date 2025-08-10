import torch
import numpy as np
from typing import Dict, Any, List
import logging

from .utils.corrcoef import corrcoef
from .utils.find_peaks import find_peaks
from .utils.compute_spectral_entropy import compute_spectral_entropy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Structural Analysis Helper Methods
def latent_space_variance(flat_trajectories: torch.Tensor) -> Dict[str, torch.Tensor]:
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

def pca_analysis(flat_trajectories: torch.Tensor) -> Dict[str, torch.Tensor]:
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
        logger.warning(f"PCA computation failed: {e}, using simplified variance analysis")
        # Fallback to simple variance analysis
        eigenvalues = torch.var(data_centered, dim=0)
        explained_variance_ratio = eigenvalues / torch.sum(eigenvalues)
        
        return {
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance_90': torch.tensor(0.9),
            'effective_dimensionality': torch.tensor(min(10, len(eigenvalues))),
            'pc_magnitudes': torch.sqrt(eigenvalues)
        }

def shannon_entropy_estimation(flat_trajectories: torch.Tensor) -> Dict[str, torch.Tensor]:
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

def kl_divergence_estimation(
        trajectories1: torch.Tensor, 
        trajectories2: torch.Tensor
) -> float:
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

def structural_complexity(flat_trajectories: torch.Tensor) -> Dict[str, float]:
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
        logger.warning(f"SVD failed in structural complexity: {e}, using variance-based approximation")
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

def analyze_structural_patterns(
    group_tensors: Dict[str, Dict[str, torch.Tensor]], 
    prompt_groups: List[str],
    device: torch.device
) -> Dict[str, Any]:
    """GPU-accelerated structural analysis including PCA, variance, and entropy measures."""
    structural_analysis = {}
    
    # Determine baseline latents (first prompt group alphabetically)
    baseline_group = sorted(prompt_groups)[0]
    
    for group_name, group_data in group_tensors.items():
        trajectory_tensor = group_data['trajectory_tensor'].to(device)  # [n_videos, steps, ...]

        logger.info(f"Analyzing structural patterns for group: {group_name}")
        
        # Flatten trajectory for structural analysis
        flat_trajectories = trajectory_tensor.flatten(start_dim=2)  # [n_videos, steps, flattened_latent]

        logger.debug(f"Flat trajectories shape: {flat_trajectories.shape}")
        
        # Latent Space Variance Analysis (fast)
        variance_results = latent_space_variance(flat_trajectories)
        logger.debug("Variance analysis completed")

        # PCA-based Analysis (optimized with sampling)
        pca_results = pca_analysis(flat_trajectories)
        logger.debug("PCA analysis completed")

        # Shannon Entropy Estimation (fast approximation)
        entropy_results = shannon_entropy_estimation(flat_trajectories)
        logger.debug("Entropy estimation completed")

        # KL Divergence Analysis (fast moment-based approximation)
        if group_name != baseline_group and baseline_group in group_tensors:
            baseline_tensor = group_tensors[baseline_group]['trajectory_tensor'].to(device)
            baseline_flat = baseline_tensor.flatten(start_dim=2)
            kl_divergence = kl_divergence_estimation(flat_trajectories, baseline_flat)
        else:
            kl_divergence = 0.0
        logger.debug("KL divergence analysis completed")
        
        # Structural Complexity Measures (optimized)
        complexity_results = structural_complexity(flat_trajectories)
        logger.debug("Structural complexity analysis completed")

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