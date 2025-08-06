#!/usr/bin/env python3
"""
Corrected Temporal Analysis Validation

This fixes the flawed metrics in the simplified analysis and provides better validation.
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.analysis.temporal_trajectory_analyzer import TemporalTrajectoryAnalyzer

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def corrected_analysis(analyzer, latents_dir):
    """Run corrected analysis with better metrics."""
    logger = logging.getLogger(__name__)
    
    groups = ['prompt_000', 'prompt_008']  # Random vs Specific
    group_data = {}
    
    for group_name in groups:
        logger.info(f"Loading {group_name}...")
        video_ids = analyzer.base_analyzer.discover_videos_in_prompt(group_name)
        
        trajectories = []
        for video_id in video_ids:
            try:
                latents, metadata = analyzer.base_analyzer.load_video_trajectory(video_id)
                trajectories.append({
                    'video_id': video_id,
                    'latents': latents,
                    'timesteps': [meta.timestep for meta in metadata]
                })
            except Exception as e:
                continue
        
        group_data[group_name] = trajectories
        logger.info(f"Loaded {len(trajectories)} trajectories for {group_name}")
    
    # CORRECTED METRIC 1: Inter-trajectory similarity (should be LOWER for random)
    logger.info("\n" + "="*60)
    logger.info("CORRECTED ANALYSIS: Inter-Trajectory Similarity")
    logger.info("="*60)
    
    for group_name, trajectories in group_data.items():
        if len(trajectories) < 2:
            continue
            
        # Extract trajectory endpoints
        endpoints = []
        for traj in trajectories:
            endpoint = traj['latents'][-1]
            if hasattr(endpoint, 'numpy'):
                endpoints.append(endpoint.numpy().flatten())
            else:
                endpoints.append(endpoint.flatten())
        
        # Compute pairwise correlations between trajectories
        correlations = []
        for i in range(len(endpoints)):
            for j in range(i+1, len(endpoints)):
                corr, _ = pearsonr(endpoints[i], endpoints[j])
                if not np.isnan(corr):
                    correlations.append(corr)
        
        # Compute distances for additional validation
        distances = pdist(endpoints, metric='euclidean')
        
        logger.info(f"{group_name} Inter-Trajectory Analysis:")
        logger.info(f"  Mean Correlation: {np.mean(correlations):.4f} (lower = more diverse)")
        logger.info(f"  Std Correlation:  {np.std(correlations):.4f}")
        logger.info(f"  Mean Distance:    {np.mean(distances):.2f} (higher = more diverse)")
        logger.info(f"  Std Distance:     {np.std(distances):.2f}")
    
    # CORRECTED METRIC 2: Trajectory Path Consistency (full temporal analysis)
    logger.info(f"\nCORRECTED ANALYSIS: Full Trajectory Path Analysis")
    logger.info("="*60)
    
    for group_name, trajectories in group_data.items():
        path_similarities = []
        path_diversities = []
        
        # Convert all trajectories to comparable format
        trajectory_sequences = []
        for traj in trajectories:
            latents = traj['latents']
            sequence = []
            for latent in latents:
                if hasattr(latent, 'numpy'):
                    sequence.append(latent.numpy().flatten())
                else:
                    sequence.append(latent.flatten())
            trajectory_sequences.append(sequence)
        
        # Compare trajectory sequences (not just endpoints)
        for i in range(len(trajectory_sequences)):
            for j in range(i+1, len(trajectory_sequences)):
                seq1, seq2 = trajectory_sequences[i], trajectory_sequences[j]
                
                # Ensure same length
                min_len = min(len(seq1), len(seq2))
                seq1, seq2 = seq1[:min_len], seq2[:min_len]
                
                # Compute step-by-step correlations
                step_correlations = []
                for step in range(min_len):
                    corr, _ = pearsonr(seq1[step], seq2[step])
                    if not np.isnan(corr):
                        step_correlations.append(corr)
                
                if step_correlations:
                    path_similarities.append(np.mean(step_correlations))
                    path_diversities.append(np.std(step_correlations))
        
        logger.info(f"{group_name} Full Path Analysis:")
        logger.info(f"  Mean Path Similarity: {np.mean(path_similarities):.4f} (lower = more diverse)")
        logger.info(f"  Path Diversity:       {np.mean(path_diversities):.4f} (higher = more diverse)")
        logger.info(f"  Trajectory Count:     {len(trajectories)}")
    
    # CORRECTED METRIC 3: Temporal Evolution Patterns
    logger.info(f"\nCORRECTED ANALYSIS: Temporal Evolution Consistency")
    logger.info("="*60)
    
    for group_name, trajectories in group_data.items():
        evolution_patterns = []
        
        for traj in trajectories:
            latents = traj['latents']
            
            # Convert to arrays
            arrays = []
            for latent in latents:
                if hasattr(latent, 'numpy'):
                    arrays.append(latent.numpy().flatten())
                else:
                    arrays.append(latent.flatten())
            
            # Compute evolution pattern (variance over time)
            variances_over_time = [np.var(arr) for arr in arrays]
            evolution_patterns.append(variances_over_time)
        
        if len(evolution_patterns) >= 2:
            # Compare evolution patterns between trajectories
            pattern_correlations = []
            for i in range(len(evolution_patterns)):
                for j in range(i+1, len(evolution_patterns)):
                    corr, _ = pearsonr(evolution_patterns[i], evolution_patterns[j])
                    if not np.isnan(corr):
                        pattern_correlations.append(corr)
            
            logger.info(f"{group_name} Evolution Pattern Analysis:")
            logger.info(f"  Mean Pattern Correlation: {np.mean(pattern_correlations):.4f}")
            logger.info(f"  Pattern Consistency:      {np.std(pattern_correlations):.4f}")
    
    # VALIDATION SUMMARY
    logger.info(f"\n" + "="*60)
    logger.info("CORRECTED VALIDATION SUMMARY")
    logger.info("="*60)
    logger.info("For RANDOM prompts (prompt_000), we expect:")
    logger.info("  - LOWER inter-trajectory correlations")
    logger.info("  - HIGHER trajectory distances") 
    logger.info("  - LOWER path similarities")
    logger.info("  - LOWER evolution pattern correlations")
    logger.info("")
    logger.info("For SPECIFIC prompts (prompt_008), we expect:")
    logger.info("  - HIGHER inter-trajectory correlations")
    logger.info("  - LOWER trajectory distances")
    logger.info("  - HIGHER path similarities") 
    logger.info("  - HIGHER evolution pattern correlations")
    logger.info("")
    logger.info("If these corrected metrics STILL show backwards results,")
    logger.info("then either the flower dataset doesn't have the expected")
    logger.info("specificity gradient, or our prompt assumptions are wrong.")

def main():
    logger = setup_logging()
    
    latents_dir = Path("outputs/flower_gen_1-3b_long_latents_20250805_200633/latents")
    
    if not latents_dir.exists():
        logger.error(f"Latents directory not found: {latents_dir}")
        return
    
    try:
        analyzer = TemporalTrajectoryAnalyzer(str(latents_dir))
        corrected_analysis(analyzer, latents_dir)
        logger.info("✅ Corrected analysis completed!")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
