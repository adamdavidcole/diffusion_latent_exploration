#!/usr/bin/env python3
"""
Run comprehensive temporal trajectory analysis on the flower generation dataset.

This script applies the research-grade temporal-aware analysis to validate
the specificity gradient from random to highly specific prompts.
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    
    # Dataset configuration
    latents_dir = "/home/adam/dev/diffusion_latent_exploration/outputs/flower_gen_1-3b_long_latents_20250805_200633/latents"
    
    # Specificity gradient (your exact setup)
    prompts_specificity_order = [
        "prompt_000",  # Random (empty prompt)
        "prompt_001",  # "(flower)"
        "prompt_002",  # "(flower) blossoming"
        "prompt_003",  # "red (flower) blossoming"
        "prompt_004",  # "red (flower) blossoming on table"
        "prompt_005",  # "red (flower) blossoming on table in front of window"
        "prompt_006",  # "red (flower) blossoming on table in front of window, morning sunlight"
        "prompt_007",  # "red (flower) blossoming on table in front of window, morning sunlight, close-up"
        "prompt_008"   # "red (flower) blossoming on table in front of window, morning sunlight, close-up, nature documentary photography"
    ]
    
    prompt_descriptions = [
        "Empty prompt (random)",
        "(flower)",
        "(flower) blossoming", 
        "red (flower) blossoming",
        "red (flower) blossoming on table",
        "red (flower) blossoming on table in front of window",
        "red (flower) blossoming on table in front of window, morning sunlight",
        "red (flower) blossoming on table in front of window, morning sunlight, close-up",
        "red (flower) blossoming on table in front of window, morning sunlight, close-up, nature documentary photography"
    ]
    
    try:
        # Import and initialize the temporal analyzer
        from src.analysis.temporal_trajectory_analyzer import TemporalTrajectoryAnalyzer
        
        logger.info("Initializing Temporal Trajectory Analyzer...")
        analyzer = TemporalTrajectoryAnalyzer(latents_dir)
        
        logger.info("Starting comprehensive temporal analysis...")
        logger.info(f"Analyzing specificity gradient: {len(prompts_specificity_order)} levels")
        logger.info("Expected pattern: Random → Increasingly Specific Prompts")
        
        # Run the comprehensive analysis
        results = analyzer.analyze_temporal_specificity_gradient(
            prompts_specificity_order=prompts_specificity_order,
            prompt_descriptions=prompt_descriptions
        )
        
        # Print key results
        print("\n" + "="*80)
        print("TEMPORAL TRAJECTORY ANALYSIS - KEY RESULTS")
        print("="*80)
        
        # Validation results
        if results.specificity_validation and 'validation_success' in results.specificity_validation:
            validation = results.specificity_validation['validation_success']
            spearman_corr = validation.get('spearman_correlation', 0.0)
            
            print(f"\n📊 SPECIFICITY GRADIENT VALIDATION:")
            print(f"   Spearman Rank Correlation: {spearman_corr:.4f}")
            
            if spearman_corr > 0.7:
                print("   ✅ STRONG validation - Method successfully detects specificity gradient!")
            elif spearman_corr > 0.4:
                print("   ⚠️  MODERATE validation - Method partially detects gradient")
            else:
                print("   ❌ WEAK validation - Method struggles with gradient detection")
            
            print(f"   Monotonic trend: {validation.get('monotonic_trend', False)}")
            print(f"   Top 3 groups correct: {validation.get('top_3_correct', False)}")
        
        # Consistency ranking
        if 'cross_group_analysis' in results.temporal_dynamics:
            cross_group = results.temporal_dynamics['cross_group_analysis']
            if 'consistency_ranking' in cross_group:
                ranking = cross_group['consistency_ranking']['ranking']
                
                print(f"\n🔄 TRAJECTORY CONSISTENCY RANKING:")
                print("   (Higher = more similar trajectories within group)")
                for i, (group, score) in enumerate(ranking):
                    prompt_desc = prompt_descriptions[prompts_specificity_order.index(group)] if group in prompts_specificity_order else "Unknown"
                    print(f"   {i+1:2d}. {group}: {score:.4f} - {prompt_desc}")
        
        # Methodological insights
        print(f"\n🧪 METHODOLOGICAL INSIGHTS:")
        print(f"   Total trajectories analyzed: {results.total_trajectories}")
        print(f"   Groups analyzed: {len(results.groups_analyzed)}")
        
        if 'scalar_metrics' in results.__dict__ and 'methodology_caveat' in results.scalar_metrics:
            print(f"\n⚠️  IMPORTANT METHODOLOGICAL NOTE:")
            print(f"   {results.scalar_metrics['methodology_caveat']}")
        
        # Visualizations created
        print(f"\n📈 VISUALIZATIONS CREATED:")
        for viz_path in results.visualization_paths:
            print(f"   - {Path(viz_path).name}")
        
        print(f"\n📁 RESULTS SAVED TO:")
        analyzer_output = analyzer.output_dir
        print(f"   {analyzer_output}")
        print(f"   - temporal_trajectory_analysis_results.json (complete data)")
        print(f"   - temporal_analysis_report.md (human-readable summary)")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        
        # Research interpretation
        print(f"\n🔬 RESEARCH INTERPRETATION:")
        print("This analysis validates whether our temporal-aware methods can correctly")
        print("identify the latent space navigation differences between random and specific prompts.")
        print("High validation scores indicate the methods successfully reveal how prompt")
        print("specificity affects diffusion trajectory patterns in latent space.")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
