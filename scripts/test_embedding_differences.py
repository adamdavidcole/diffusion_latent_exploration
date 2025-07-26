#!/usr/bin/env python3
"""
Quick test to show embedding differences at different weights.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our modules
from src.config.config_manager import ConfigManager
from src.generators.video_generator import WanVideoGenerator
from src.prompts.wan_weighted_embeddings_fixed import WANWeightedEmbeddings


def test_embedding_differences():
    """Test how embedding differences scale with different weights."""
    
    print("🔬 Embedding Differences at Different Weights")
    print("=" * 60)
    
    # Load config
    config_path = "configs/weighted_prompts_example.yaml"
    config_manager = ConfigManager()
    config = config_manager.load_config(config_path)
    
    # Initialize generator to get access to the pipeline
    try:
        generator = WanVideoGenerator(config)
        generator._load_model()  # Load the model
        pipe = generator.pipe
        device = generator.device
        
        print(f"✅ WAN model loaded on device: {device}")
        
    except Exception as e:
        print(f"❌ Failed to load WAN model: {e}")
        return
    
    # Initialize WAN weighted embeddings
    wan_embeddings = WANWeightedEmbeddings(pipe)
    
    # Test prompt
    base_prompt_text = "A person dancing in the park"
    weights_to_test = [1.0, 1.5, 2.0, 3.0, 5.0]
    
    print(f"\n📊 Testing base prompt: '{base_prompt_text}' with different weights")
    print("-" * 60)
    
    # Get baseline embeddings using the SAME method as weighted embeddings for weight=1.0
    # This ensures fair comparison
    baseline_embeddings = wan_embeddings._get_standard_t5_embeddings(base_prompt_text, 512, 1)
    
    print(f"Baseline embeddings shape: {baseline_embeddings.shape}")
    print(f"Baseline stats: mean={baseline_embeddings.mean():.6f}, std={baseline_embeddings.std():.6f}")
    print()
    
    results = []
    
    for weight in weights_to_test:
        print(f"Weight {weight:.1f}:")
        
        # Create weighted prompt using the SAME base text
        if weight == 1.0:
            # For weight 1.0, use the exact same prompt as baseline to ensure 0% difference
            test_prompt = base_prompt_text
        else:
            # For other weights, add weight syntax
            test_prompt = f"A person (dancing:{weight}) in the park"
        
        try:
            # Get weighted embeddings
            weighted_embeddings = wan_embeddings.create_weighted_embeddings(
                prompt=test_prompt,
                max_sequence_length=512,
                num_videos_per_prompt=1,
                weighting_method="multiply"
            )
            
            # Compare with baseline
            diff = weighted_embeddings - baseline_embeddings
            diff_norm = torch.norm(diff).item()
            baseline_norm = torch.norm(baseline_embeddings).item()
            relative_diff = diff_norm / baseline_norm
            
            # Calculate some stats
            mean_diff = diff.mean().item()
            std_diff = diff.std().item()
            max_diff = diff.abs().max().item()
            
            results.append({
                'weight': weight,
                'relative_diff': relative_diff,
                'l2_norm': diff_norm,
                'mean_diff': mean_diff,
                'std_diff': std_diff,
                'max_diff': max_diff
            })
            
            print(f"  Relative difference: {relative_diff:.6f} ({relative_diff*100:.2f}%)")
            print(f"  L2 norm of diff:     {diff_norm:.6f}")
            print(f"  Mean difference:     {mean_diff:.6f}")
            print(f"  Std of difference:   {std_diff:.6f}")
            print(f"  Max abs difference:  {max_diff:.6f}")
            print()
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            print()
    
    # Summary table
    print("📈 Summary Table:")
    print("-" * 60)
    print("Weight | Rel. Diff (%) | L2 Norm   | Max Abs Diff")
    print("-" * 60)
    for result in results:
        print(f"{result['weight']:5.1f}  | {result['relative_diff']*100:10.2f}% | {result['l2_norm']:8.3f} | {result['max_diff']:11.6f}")
    
    print()
    print("📋 Analysis:")
    if len(results) > 1:
        # Check if differences are increasing
        diffs = [r['relative_diff'] for r in results]
        weights = [r['weight'] for r in results]
        
        print(f"• Baseline (weight 1.0): {diffs[0]*100:.2f}% difference")
        print(f"• Highest weight ({weights[-1]}): {diffs[-1]*100:.2f}% difference")
        
        if diffs[-1] > diffs[0]:
            ratio = diffs[-1] / diffs[0] if diffs[0] > 0 else float('inf')
            print(f"• Impact ratio: {ratio:.1f}x stronger effect at highest weight")
        
        # Check for scaling pattern
        scaling_linear = all(diffs[i] <= diffs[i+1] for i in range(len(diffs)-1))
        print(f"• Scaling pattern: {'Linear increase' if scaling_linear else 'Non-linear'}")


if __name__ == "__main__":
    test_embedding_differences()
