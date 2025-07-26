#!/usr/bin/env python3
"""
Test script for WAN-specific weighted embeddings.
Compares different weighting methods and analyzes their behavior.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our modules
from src.config.config_manager import ConfigManager
from src.generators.video_generator import WanVideoGenerator
from src.prompts.wan_weighted_embeddings import WANWeightedEmbeddings


def test_wan_weighted_embeddings():
    """Test WAN-specific weighted embeddings implementation."""
    
    print("🔬 WAN Weighted Embeddings Test")
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
        "a person dancing in the park",
        "a person (dancing:1.5) in the park",
        "a person (dancing:2.0) in the park", 
        "(romantic:1.8) kiss between two people",
        "a (beautiful:2.0) and (fast:1.3) car racing"
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
        print(f"   Baseline range: [{baseline_embeddings.min():.6f}, {baseline_embeddings.max():.6f}]")
        
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
                cosine_sim = torch.nn.functional.cosine_similarity(
                    baseline_embeddings.flatten(),
                    weighted_embeddings.flatten(),
                    dim=0
                ).item()
                
                print(f"      Shape: {weighted_embeddings.shape}")
                print(f"      Stats: mean={weighted_embeddings.mean():.6f}, std={weighted_embeddings.std():.6f}")
                print(f"      Range: [{weighted_embeddings.min():.6f}, {weighted_embeddings.max():.6f}]")
                print(f"      Diff norm: {diff_norm:.6f}")
                print(f"      Relative diff: {relative_diff:.6f} ({relative_diff*100:.2f}%)")
                print(f"      Cosine similarity: {cosine_sim:.6f}")
                
                # Quality checks
                has_nan = torch.isnan(weighted_embeddings).any()
                has_inf = torch.isinf(weighted_embeddings).any()
                extreme_values = (torch.abs(weighted_embeddings) > 100).any()
                
                print(f"      Quality: NaN={has_nan}, Inf={has_inf}, Extreme={extreme_values}")
                
                # Rate the method
                if has_nan or has_inf or extreme_values:
                    rating = "❌ FAILED"
                elif relative_diff > 0.5:
                    rating = "⚠️  HIGH CHANGE"
                elif relative_diff > 0.1:
                    rating = "🔶 MODERATE CHANGE"
                elif relative_diff > 0.01:
                    rating = "🟡 SMALL CHANGE"
                else:
                    rating = "🟢 MINIMAL CHANGE"
                
                print(f"      Rating: {rating}")
                
            except Exception as e:
                print(f"      ❌ ERROR: {e}")


def test_token_mapping_accuracy():
    """Test how accurately we map weights to tokens."""
    
    print(f"\n🔍 Token Mapping Accuracy Test")
    print("=" * 60)
    
    # Load config and model
    config_path = "configs/weighted_prompts_example.yaml"
    config_manager = ConfigManager()
    config = config_manager.load_config(config_path)
    
    try:
        generator = WanVideoGenerator(config)
        generator._load_model()
        pipe = generator.pipe
        wan_embeddings = WANWeightedEmbeddings(pipe)
        
        test_cases = [
            "a person (dancing:1.5) in the park",
            "(romantic:1.8) kiss between two people",
            "a (beautiful:2.0) and (fast:1.3) car"
        ]
        
        for prompt in test_cases:
            print(f"\nPrompt: '{prompt}'")
            
            # Parse segments
            segments = wan_embeddings.processor.parse_weighted_prompt(prompt)
            print(f"Segments: {[(seg.text, seg.weight) for seg in segments]}")
            
            # Tokenize full prompt
            clean_prompt = "".join(seg.text for seg in segments)
            tokens = wan_embeddings.tokenizer.encode(clean_prompt, add_special_tokens=True)
            token_texts = [wan_embeddings.tokenizer.decode([t]) for t in tokens]
            
            print(f"Tokens ({len(tokens)}): {token_texts}")
            
            # Show how we would map weights
            current_pos = 1  # Skip start token
            for segment in segments:
                if not segment.text.strip():
                    continue
                    
                segment_tokens = wan_embeddings.tokenizer.encode(segment.text.strip(), add_special_tokens=False)
                segment_token_texts = [wan_embeddings.tokenizer.decode([t]) for t in segment_tokens]
                
                print(f"  '{segment.text}' (weight={segment.weight}) -> tokens {current_pos}:{current_pos+len(segment_tokens)}")
                print(f"    Token texts: {segment_token_texts}")
                
                current_pos += len(segment_tokens)
                
    except Exception as e:
        print(f"❌ Error in token mapping test: {e}")


def recommend_best_method():
    """Provide recommendations based on test results."""
    
    print(f"\n🎯 Recommendations")
    print("=" * 60)
    print("Based on the test results above:")
    print()
    print("1. **INTERPOLATION METHOD** (Recommended)")
    print("   - Uses EOS token as anchor point like SD_embed")
    print("   - More stable than direct multiplication")
    print("   - Preserves semantic relationships")
    print()
    print("2. **NORM_PRESERVING METHOD** (Conservative)")
    print("   - Maintains embedding magnitudes")
    print("   - Safest for model stability")
    print("   - May have weaker effect")
    print()
    print("3. **MULTIPLY METHOD** (Risky)")
    print("   - Direct scaling like our previous attempt")
    print("   - Can cause distribution shift")
    print("   - Only use if others fail")
    print()
    print("💡 **Next Steps:**")
    print("   1. Choose the method with best rating above")
    print("   2. Test actual video generation with that method")
    print("   3. If still produces noise, reduce weight magnitudes")
    print("   4. Consider hybrid approach: light weights + repetition")


if __name__ == "__main__":
    test_wan_weighted_embeddings()
    test_token_mapping_accuracy()
    recommend_best_method()
