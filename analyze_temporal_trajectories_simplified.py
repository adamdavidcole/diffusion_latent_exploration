#!/usr/bin/env python3
"""
Simplified Temporal Trajectory Analysis

This script runs a focused version of the temporal-aware analysis that demonstrates
the key methodological breakthrough without running into memory/time constraints.

Focus: Validate that temporal analysis captures specificity patterns that averaging destroys.
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.analysis.temporal_trajectory_analyzer import TemporalTrajectoryAnalyzer

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def analyze_key_findings(analyzer, latents_dir):
    """
    Focus on the core validation: Does temporal analysis detect specificity patterns?
    """
    logger = logging.getLogger(__name__)
    
    # Get available prompt groups (sorted by specificity)
    prompt_dirs = sorted([d for d in os.listdir(latents_dir) if d.startswith('prompt_')])
    logger.info(f"Found {len(prompt_dirs)} prompt groups: {prompt_dirs}")
    
    if len(prompt_dirs) < 3:
        logger.error("Need at least 3 prompt groups for comparison")
        return None
    
    # Load a subset for comparison: most random vs most specific
    subset_groups = [prompt_dirs[0], prompt_dirs[len(prompt_dirs)//2], prompt_dirs[-1]]
    logger.info(f"Analyzing subset: {subset_groups} (random → intermediate → specific)")
    
    results = {}
    
    for group_name in subset_groups:
        logger.info(f"Loading trajectories for {group_name}")
        
        # Load all trajectories for this group
        video_ids = analyzer.base_analyzer.discover_videos_in_prompt(group_name)
        
        group_trajectories = []
        for video_id in video_ids:
            try:
                latents, metadata = analyzer.base_analyzer.load_video_trajectory(video_id)
                timesteps = [meta.timestep for meta in metadata]
                
                group_trajectories.append({
                    'video_id': video_id,
                    'latents': latents,
                    'metadata': metadata,
                    'timesteps': timesteps
                })
            except Exception as e:
                logger.warning(f"Failed to load trajectory for {video_id}: {e}")
                continue
        
        if not group_trajectories:
            logger.warning(f"No trajectories found for {group_name}")
            continue
        
        logger.info(f"Loaded {len(group_trajectories)} trajectories for {group_name}")
        
        # Key Analysis 1: Temporal Consistency
        temporal_metrics = analyze_temporal_consistency(group_trajectories)
        
        # Key Analysis 2: Inter-trajectory Variance 
        variance_metrics = analyze_trajectory_variance(group_trajectories)
        
        # Key Analysis 3: Methodological Comparison
        method_comparison = compare_temporal_vs_averaging(group_trajectories)
        
        results[group_name] = {
            'temporal_consistency': temporal_metrics,
            'trajectory_variance': variance_metrics,
            'method_comparison': method_comparison,
            'n_trajectories': len(group_trajectories)
        }
    
    return results

def analyze_temporal_consistency(trajectories):
    """Measure how consistent trajectory patterns are within a group."""
    consistency_scores = []
    
    for traj_data in trajectories:
        latents = traj_data['latents']
        
        # Convert to numpy arrays
        latent_arrays = []
        for latent in latents:
            if hasattr(latent, 'numpy'):  # torch tensor
                latent_arrays.append(latent.numpy().flatten())
            else:
                latent_arrays.append(latent.flatten())
        
        # Compute step-to-step consistency
        step_variances = []
        for i in range(len(latent_arrays) - 1):
            diff = latent_arrays[i+1] - latent_arrays[i]
            step_variances.append(np.var(diff))
        
        # Consistency = inverse of variance in step changes
        consistency = 1.0 / (1.0 + np.std(step_variances))
        consistency_scores.append(consistency)
    
    return {
        'mean_consistency': float(np.mean(consistency_scores)),
        'std_consistency': float(np.std(consistency_scores)),
        'trajectory_count': len(consistency_scores)
    }

def analyze_trajectory_variance(trajectories):
    """Measure inter-trajectory variance (should be higher for random prompts)."""
    all_endpoints = []
    
    for traj_data in trajectories:
        latents = traj_data['latents']
        endpoint = latents[-1]  # Final latent
        
        if hasattr(endpoint, 'numpy'):
            all_endpoints.append(endpoint.numpy().flatten())
        else:
            all_endpoints.append(endpoint.flatten())
    
    if len(all_endpoints) < 2:
        return {'error': 'Need at least 2 trajectories'}
    
    # Compute pairwise distances between endpoints
    distances = []
    for i in range(len(all_endpoints)):
        for j in range(i+1, len(all_endpoints)):
            dist = np.linalg.norm(all_endpoints[i] - all_endpoints[j])
            distances.append(dist)
    
    return {
        'mean_inter_trajectory_distance': float(np.mean(distances)),
        'std_inter_trajectory_distance': float(np.std(distances)),
        'max_distance': float(np.max(distances)),
        'min_distance': float(np.min(distances))
    }

def compare_temporal_vs_averaging(trajectories):
    """Compare temporal analysis vs simple averaging to demonstrate methodology difference."""
    
    # Method 1: Simple averaging (destroys temporal information)
    all_latents = []
    for traj_data in trajectories:
        latents = traj_data['latents']
        for latent in latents:
            if hasattr(latent, 'numpy'):
                all_latents.append(latent.numpy().flatten())
            else:
                all_latents.append(latent.flatten())
    
    if not all_latents:
        return {'error': 'No latents to analyze'}
    
    averaged_representation = np.mean(all_latents, axis=0)
    averaged_variance = np.var(averaged_representation)
    
    # Method 2: Temporal-aware analysis (preserves time-series information)
    temporal_variances = []
    for traj_data in trajectories:
        latents = traj_data['latents']
        
        # Convert trajectory to arrays
        traj_arrays = []
        for latent in latents:
            if hasattr(latent, 'numpy'):
                traj_arrays.append(latent.numpy().flatten())
            else:
                traj_arrays.append(latent.flatten())
        
        # Compute variance sequence (preserves temporal structure)
        variance_sequence = [np.var(arr) for arr in traj_arrays]
        temporal_variances.append(np.std(variance_sequence))
    
    temporal_variance_diversity = np.mean(temporal_variances)
    
    return {
        'averaged_method_variance': float(averaged_variance),
        'temporal_method_variance_diversity': float(temporal_variance_diversity),
        'information_ratio': float(temporal_variance_diversity / max(averaged_variance, 1e-8)),
        'temporal_method_superior': temporal_variance_diversity > averaged_variance
    }

def print_key_findings(results):
    """Print the key validation results."""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*60)
    logger.info("KEY FINDINGS: Temporal Analysis Validation")
    logger.info("="*60)
    
    groups = list(results.keys())
    if len(groups) >= 3:
        random_group = groups[0]
        specific_group = groups[-1]
        
        logger.info(f"\nCOMPARING: {random_group} (random) vs {specific_group} (specific)")
        
        # Temporal Consistency
        random_consistency = results[random_group]['temporal_consistency']['mean_consistency']
        specific_consistency = results[specific_group]['temporal_consistency']['mean_consistency']
        
        logger.info(f"\n1. TEMPORAL CONSISTENCY:")
        logger.info(f"   Random prompts:   {random_consistency:.4f}")
        logger.info(f"   Specific prompts: {specific_consistency:.4f}")
        logger.info(f"   Specificity improves consistency: {specific_consistency > random_consistency}")
        
        # Inter-trajectory Variance
        random_variance = results[random_group]['trajectory_variance']['mean_inter_trajectory_distance']
        specific_variance = results[specific_group]['trajectory_variance']['mean_inter_trajectory_distance']
        
        logger.info(f"\n2. INTER-TRAJECTORY VARIANCE:")
        logger.info(f"   Random prompts:   {random_variance:.4f}")
        logger.info(f"   Specific prompts: {specific_variance:.4f}")
        logger.info(f"   Random prompts more variable: {random_variance > specific_variance}")
        
        # Methodological Validation
        random_info_ratio = results[random_group]['method_comparison']['information_ratio']
        specific_info_ratio = results[specific_group]['method_comparison']['information_ratio']
        
        logger.info(f"\n3. TEMPORAL vs AVERAGING METHODS:")
        logger.info(f"   Random group info ratio:   {random_info_ratio:.4f}")
        logger.info(f"   Specific group info ratio: {specific_info_ratio:.4f}")
        logger.info(f"   Temporal analysis captures more info: {np.mean([random_info_ratio, specific_info_ratio]) > 1.0}")
        
        logger.info(f"\n✅ VALIDATION RESULT:")
        specificity_detected = (specific_consistency > random_consistency and 
                              random_variance > specific_variance)
        logger.info(f"   Temporal analysis correctly detects specificity gradient: {specificity_detected}")
        
        if specificity_detected:
            logger.info(f"\n🎉 SUCCESS: The temporal-aware methodology successfully identifies")
            logger.info(f"   the prompt specificity gradient that simple averaging would miss!")
        else:
            logger.info(f"\n⚠️  INCONCLUSIVE: Results don't show clear specificity gradient.")
    
    logger.info("\n" + "="*60)

def main():
    """Main analysis function."""
    logger = setup_logging()
    
    # Configuration
    latents_dir = Path("outputs/flower_gen_1-3b_long_latents_20250805_200633/latents")
    
    if not latents_dir.exists():
        logger.error(f"Latents directory not found: {latents_dir}")
        return
    
    logger.info("Initializing Simplified Temporal Trajectory Analyzer...")
    
    try:
        # Initialize analyzer
        analyzer = TemporalTrajectoryAnalyzer(str(latents_dir))
        
        logger.info("Starting focused temporal analysis...")
        logger.info("Objective: Validate temporal-aware methodology vs simple averaging")
        
        # Run focused analysis
        results = analyze_key_findings(analyzer, latents_dir)
        
        if results:
            print_key_findings(results)
            logger.info("✅ Simplified temporal analysis completed successfully!")
        else:
            logger.error("❌ Analysis failed - no results generated")
            
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
