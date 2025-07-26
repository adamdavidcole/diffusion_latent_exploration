#!/usr/bin/env python3
"""
Quick test for the fixed WAN weighted embeddings.
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


def test_fixed_weighted_embeddings():
    """Test the fixed WAN weighted embeddings implementation."""
    
    print("🔧 Testing Fixed WAN Weighted Embeddings")
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
    
    # Test prompts
    test_prompts = [
        "a person (dancing:1.5) in the park",
        "(romantic:1.8) kiss between two people",
    ]
    
    # Test different weighting methods
    methods = ["multiply", "interpolation", "norm_preserving"]
    
    print(f"\n📊 Testing {len(test_prompts)} prompts with {len(methods)} methods...")
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n{i+1}. Testing: '{prompt}'")
        print("-" * 50)
        
        # Get baseline (standard) embeddings
        clean_prompt = "".join(seg.text for seg in wan_embeddings.processor.parse_weighted_prompt(prompt))
        baseline_embeddings = wan_embeddings._get_standard_t5_embeddings(clean_prompt, 512, 1)
        
        print(f"   Baseline: shape={baseline_embeddings.shape}")
        print(f"   Baseline stats: mean={baseline_embeddings.mean():.6f}, std={baseline_embeddings.std():.6f}")
        
        # Test each weighting method
        for method in methods:
            print(f"\n   Method: {method}")
            try:
                weighted_embeddings = wan_embeddings.create_weighted_embeddings(
                    prompt=prompt,
                    max_sequence_length=512,
                    num_videos_per_prompt=1,
                    weighting_method=method
                )
                
                # Compare with baseline
                diff = weighted_embeddings - baseline_embeddings
                diff_norm = torch.norm(diff).item()
                relative_diff = diff_norm / torch.norm(baseline_embeddings).item()
                
                print(f"      ✅ SUCCESS")
                print(f"      Shape: {weighted_embeddings.shape}")
                print(f"      Stats: mean={weighted_embeddings.mean():.6f}, std={weighted_embeddings.std():.6f}")
                print(f"      Relative diff: {relative_diff:.6f} ({relative_diff*100:.2f}%)")
                
                # Quality checks
                has_nan = torch.isnan(weighted_embeddings).any()
                has_inf = torch.isinf(weighted_embeddings).any()
                
                if has_nan or has_inf:
                    print(f"      ❌ Quality issues: NaN={has_nan}, Inf={has_inf}")
                else:
                    print(f"      ✅ Quality: Good")
                
            except Exception as e:
                print(f"      ❌ ERROR: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    test_fixed_weighted_embeddings()
