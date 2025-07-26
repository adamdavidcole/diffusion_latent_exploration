#!/usr/bin/env python3
"""
Test different weighting methods to see which ones actually work.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our modules
from src.config.config_manager import ConfigManager
from src.generators.video_generator import WanVideoGenerator
from src.prompts.wan_weighted_embeddings_fixed import WANWeightedEmbeddings


def test_weighting_methods():
    """Test different weighting methods to see which ones actually work."""
    
    print("🔬 Testing Different Weighting Methods")
    print("=" * 70)
    
    # Load config
    config_path = "configs/weighted_prompts_example.yaml"
    config_manager = ConfigManager()
    config = config_manager.load_config(config_path)
    
    # Initialize generator
    try:
        generator = WanVideoGenerator(config)
        generator._load_model()
        pipe = generator.pipe
        device = generator.device
        
        print(f"✅ WAN model loaded on device: {device}")
        
    except Exception as e:
        print(f"❌ Failed to load WAN model: {e}")
        return
    
    # Initialize WAN weighted embeddings
    wan_embeddings = WANWeightedEmbeddings(pipe)
    
    # Test prompt
    base_prompt = "A person dancing in the park"
    weights_to_test = [1.0, 1.5, 2.0, 3.0, 5.0]
    methods_to_test = ["multiply", "interpolation", "norm_preserving"]
    
    print(f"\n📊 Base prompt: '{base_prompt}'")
    print("-" * 70)
    
    # Get baseline embeddings
    baseline_embeddings = wan_embeddings._get_standard_t5_embeddings(base_prompt, 512, 1)
    baseline_norm = torch.norm(baseline_embeddings).item()
    
    print(f"Baseline embeddings norm: {baseline_norm:.6f}")
    print()
    
    for method in methods_to_test:
        print(f"🧪 Testing method: {method}")
        print("-" * 50)
        
        method_results = []
        
        for weight in weights_to_test:
            if weight == 1.0:
                test_prompt = base_prompt
            else:
                test_prompt = f"A person (dancing:{weight}) in the park"
            
            try:
                # Get weighted embeddings
                weighted_embeddings = wan_embeddings.create_weighted_embeddings(
                    prompt=test_prompt,
                    max_sequence_length=512,
                    num_videos_per_prompt=1,
                    weighting_method=method
                )
                
                # Compare with baseline
                diff = weighted_embeddings - baseline_embeddings
                diff_norm = torch.norm(diff).item()
                relative_diff = diff_norm / baseline_norm
                
                method_results.append({
                    'weight': weight,
                    'relative_diff': relative_diff,
                    'diff_norm': diff_norm
                })
                
                print(f"  Weight {weight:3.1f}: {relative_diff*100:6.2f}% difference")
                
            except Exception as e:
                print(f"  Weight {weight:3.1f}: ERROR - {e}")
        
        # Analysis for this method
        if len(method_results) > 1:
            diffs = [r['relative_diff'] for r in method_results[1:]]  # Skip weight 1.0
            weights = [r['weight'] for r in method_results[1:]]
            
            # Check if differences increase with weight
            is_increasing = all(diffs[i] <= diffs[i+1] for i in range(len(diffs)-1))
            is_constant = all(abs(diffs[i] - diffs[0]) < 0.001 for i in range(len(diffs)))
            
            if is_constant:
                print(f"  📊 Analysis: CONSTANT differences (~{diffs[0]*100:.2f}%) - method is BROKEN")
            elif is_increasing:
                print(f"  📊 Analysis: INCREASING differences ({diffs[0]*100:.2f}% → {diffs[-1]*100:.2f}%) - method works correctly")
            else:
                print(f"  📊 Analysis: NON-LINEAR scaling - method may work but with complex behavior")
        
        print()


if __name__ == "__main__":
    test_weighting_methods()
