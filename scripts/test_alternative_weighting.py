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
    
    print("� Testing Different Weighting Methods")
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


def test_cfg_scaling_approach():
    """Test concept of using different CFG scales for different parts."""
    
    print(f"\n🎛️ CFG Scaling Approach Concept")
    print("=" * 50)
    
    test_prompt = "a person (dancing:1.8) in the park"
    segments = parse_weighted_prompt(test_prompt)
    base_cfg = 7.5
    
    print(f"Prompt: {test_prompt}")
    print(f"Base CFG scale: {base_cfg}")
    
    for text, weight in segments:
        if weight != 1.0:
            # Calculate CFG adjustment
            cfg_multiplier = weight
            adjusted_cfg = base_cfg * cfg_multiplier
            print(f"  '{text}' -> CFG scale {adjusted_cfg:.1f} (multiplier {cfg_multiplier})")
        else:
            print(f"  '{text}' -> CFG scale {base_cfg} (no change)")
    
    print(f"\nNote: This would require modifying the diffusion process")
    print(f"to apply different CFG scales to different prompt regions.")


def test_prompt_interpolation_approach():
    """Test prompt interpolation between base and emphasized versions."""
    
    print(f"\n🔗 Prompt Interpolation Approach")
    print("=" * 50)
    
    test_prompt = "a person (dancing:1.8) in the park"
    segments = parse_weighted_prompt(test_prompt)
    
    # Create base prompt (clean)
    base_prompt = "".join(text for text, _ in segments)
    
    # Create emphasized prompt (stronger language)
    emphasized_words = {
        "dancing": ["energetically dancing", "vibrantly dancing", "passionately dancing"],
        "romantic": ["deeply romantic", "intensely romantic", "passionately romantic"],
        "strong": ["very strong", "extremely strong", "powerfully strong"],
        "beautiful": ["stunningly beautiful", "breathtakingly beautiful", "magnificently beautiful"]
    }
    
    emphasized_prompt = base_prompt
    for text, weight in segments:
        if weight > 1.0 and text.strip() in emphasized_words:
            emphasis_level = min(int((weight - 1.0) * 2), len(emphasized_words[text.strip()]) - 1)
            emphasized_text = emphasized_words[text.strip()][emphasis_level]
            emphasized_prompt = emphasized_prompt.replace(text.strip(), emphasized_text)
    
    print(f"Original:    '{test_prompt}'")
    print(f"Base:        '{base_prompt}'")
    print(f"Emphasized:  '{emphasized_prompt}'")
    
    # Interpolation weights based on original weight
    weight_value = next((w for t, w in segments if w > 1.0), 1.0)
    interpolation_factor = min((weight_value - 1.0), 1.0)  # 0.0 to 1.0
    
    print(f"\nInterpolation factor: {interpolation_factor:.2f}")
    print(f"This could blend between base ({1-interpolation_factor:.1%}) and emphasized ({interpolation_factor:.1%})")


if __name__ == "__main__":
    test_prompt_repetition_approach()
    test_cfg_scaling_approach() 
    test_prompt_interpolation_approach()
    
    print(f"\n🎯 Recommendations:")
    print("=" * 50)
    print("1. PROMPT REPETITION: Simplest and most compatible approach")
    print("   - Works with any model without modification")
    print("   - Natural way to emphasize concepts")
    print("   - Example: 'dancing' -> 'dancing, dancing' for 2x emphasis")
    
    print(f"\n2. ENHANCED LANGUAGE: Use stronger descriptive words")
    print("   - 'dancing' -> 'energetically dancing' for emphasis")
    print("   - More natural than repetition")
    print("   - Requires word mapping dictionary")
    
    print(f"\n3. CURRENT CLEAN APPROACH: Keep using clean prompts")
    print("   - Guaranteed to work without issues")
    print("   - No risk of noise generation")
    print("   - Template variations already provide emphasis")
    
    print(f"\nThe embedding modification approach is too risky - even tiny")
    print(f"changes can disrupt the model's expected input distribution.")
