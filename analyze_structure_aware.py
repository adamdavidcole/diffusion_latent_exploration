#!/usr/bin/env python3
"""
Run structure-aware latent analysis on the flower generation dataset.

This script applies the new structure-aware analysis framework that respects
the 3D video latent structure [batch, channels, frames, height, width].
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
        # Import and initialize the structure-aware analyzer
        from src.analysis.structure_aware_analyzer import StructureAwareLatentAnalyzer
        
        logger.info("Initializing Structure-Aware Latent Analyzer...")
        analyzer = StructureAwareLatentAnalyzer(latents_dir)
        
        logger.info("Starting structure-aware analysis...")
        logger.info(f"Analyzing specificity gradient: {len(prompts_specificity_order)} levels")
        logger.info("Expected pattern: Random → Increasingly Specific Prompts")
        logger.info("Key innovation: Preserving 3D latent structure [batch, channels, frames, height, width]")
        
        # Run the structure-aware analysis
        results = analyzer.analyze_prompt_groups(
            prompt_groups=prompts_specificity_order,
            prompt_descriptions=prompt_descriptions
        )
        
        # Print key results
        print("\n" + "="*80)
        print("STRUCTURE-AWARE LATENT ANALYSIS - KEY RESULTS")
        print("="*80)
        
        print(f"\n📊 ANALYSIS METADATA:")
        print(f"   Latent shape analyzed: {results.latent_shape}")
        print(f"   Total groups: {len(results.groups_analyzed)}")
        print(f"   Analysis timestamp: {results.analysis_timestamp}")
        
        # Spatial patterns
        if results.spatial_patterns and 'spatial_variance_maps' in results.spatial_patterns:
            print(f"\n🎨 SPATIAL PATTERN ANALYSIS:")
            spatial_vars = results.spatial_patterns['spatial_variance_maps']
            
            print("   Spatial variance by group (higher = more spatial complexity):")
            for group in prompts_specificity_order:
                if group in spatial_vars:
                    mean_var = spatial_vars[group]['mean']
                    print(f"   {group}: {mean_var:.6f}")
        
        # Temporal coherence
        if results.temporal_coherence and 'frame_correlation' in results.temporal_coherence:
            print(f"\n🎬 TEMPORAL COHERENCE ANALYSIS:")
            frame_corrs = results.temporal_coherence['frame_correlation']
            
            print("   Frame-to-frame correlation by group (higher = more temporal consistency):")
            for group in prompts_specificity_order:
                if group in frame_corrs:
                    mean_corr = frame_corrs[group]['mean']
                    print(f"   {group}: {mean_corr:.6f}")
        
        # Information content
        if results.information_content and 'information_density' in results.information_content:
            print(f"\n💾 INFORMATION CONTENT ANALYSIS:")
            info_density = results.information_content['information_density']
            
            print("   Information density by group (higher = more complex content):")
            for group in prompts_specificity_order:
                if group in info_density:
                    density = info_density[group]['mean']
                    print(f"   {group}: {density:.6f}")
        
        # Group separability
        if results.group_separability and 'distance_based_separation' in results.group_separability:
            print(f"\n📏 GROUP SEPARABILITY ANALYSIS:")
            separation = results.group_separability['distance_based_separation']
            
            if 'separation_ratio' in separation:
                ratio = separation['separation_ratio']
                print(f"   Inter-group vs Intra-group distance ratio: {ratio:.4f}")
                print(f"   (Higher ratio = better group separation)")
                
                if ratio > 2.0:
                    print("   ✅ STRONG separation - Groups are well-distinguished!")
                elif ratio > 1.5:
                    print("   ⚠️  MODERATE separation - Some group distinction detected")
                else:
                    print("   ❌ WEAK separation - Groups overlap significantly")
        
        # Statistical significance
        if results.statistical_significance and 'group_comparison_tests' in results.statistical_significance:
            print(f"\n📈 STATISTICAL SIGNIFICANCE:")
            tests = results.statistical_significance['group_comparison_tests']
            
            significant_tests = 0
            total_tests = 0
            
            for metric, comparisons in tests.items():
                for comparison, result in comparisons.items():
                    total_tests += 1
                    if result.get('significant', False):
                        significant_tests += 1
            
            if total_tests > 0:
                significance_rate = significant_tests / total_tests
                print(f"   Significant differences: {significant_tests}/{total_tests} ({significance_rate:.1%})")
                
                if significance_rate > 0.5:
                    print("   ✅ HIGH statistical significance - Clear group differences!")
                elif significance_rate > 0.2:
                    print("   ⚠️  MODERATE statistical significance - Some differences detected")
                else:
                    print("   ❌ LOW statistical significance - Limited group differences")
        
        # Complexity measures
        if results.complexity_measures and 'effective_dimensionality' in results.complexity_measures:
            print(f"\n🧮 COMPLEXITY ANALYSIS:")
            eff_dim = results.complexity_measures['effective_dimensionality']
            
            print("   Effective dimensionality by group (higher = more complex representation):")
            for group in prompts_specificity_order:
                if group in eff_dim:
                    dim = eff_dim[group]['mean']
                    print(f"   {group}: {dim:.2f}")
        
        print(f"\n📁 DETAILED RESULTS SAVED TO:")
        analyzer_output = analyzer.output_dir
        print(f"   {analyzer_output}")
        print(f"   - structure_aware_analysis_results.json (complete data)")
        print(f"   - Comprehensive visualizations (if generated)")
        
        print("\n" + "="*80)
        print("STRUCTURE-AWARE ANALYSIS COMPLETE!")
        print("="*80)
        
        # Research interpretation
        print(f"\n🔬 RESEARCH INTERPRETATION:")
        print("This analysis respects the 3D video latent structure and measures:")
        print("• Spatial patterns (how information is organized spatially)")
        print("• Temporal coherence (how consistent video frames are)")
        print("• Channel-specific patterns (different information channels)")
        print("• Information-theoretic content (actual information density)")
        print("• Multi-scale complexity (patch-based and global patterns)")
        print("• Statistical significance of group differences")
        
        return 0
        
    except Exception as e:
        logger.error(f"Structure-aware analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
