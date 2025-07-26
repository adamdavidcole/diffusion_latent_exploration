#!/usr/bin/env python3
"""
Debug script to analyze weighted vs unweighted embeddings.
This will help us understand why weighted embeddings produce noise.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import our modules
from src.config.config_manager import ConfigManager
from src.generators.video_generator import WanVideoGenerator, parse_weighted_prompt, create_weighted_embeddings

def analyze_embeddings():
    """Analyze the differences between weighted and unweighted embeddings."""
    
    print("🔍 Weighted Embeddings Debug Analysis")
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
        
        print(f"✅ Model loaded on device: {device}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test prompts
    test_prompts = [
        "a person dancing in the park",
        "a person (dancing:1.5) in the park", 
        "a person (dancing:2.0) in the park",
        "(romantic:1.8) kiss between two people",
        "multiple (strong:2.0) weights and (subtle:1.2) emphasis"
    ]
    
    print(f"\n📊 Analyzing {len(test_prompts)} test prompts...")
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n{i+1}. Analyzing: '{prompt}'")
        print("-" * 50)
        
        # Parse weighted segments
        segments = parse_weighted_prompt(prompt)
        has_weights = any(weight != 1.0 for _, weight in segments)
        
        print(f"   Segments: {segments}")
        print(f"   Has weights: {has_weights}")
        
        if not has_weights:
            print("   → No weights detected, creating baseline embedding")
        
        try:
            # Create embeddings
            with torch.no_grad():
                if has_weights:
                    # Get weighted embeddings
                    weighted_embeddings = create_weighted_embeddings(pipe, prompt, device)
                    
                    # Get clean (unweighted) embeddings for comparison
                    clean_prompt = "".join(text for text, _ in segments)
                    max_length = min(77, getattr(pipe.tokenizer, 'model_max_length', 77))
                    
                    text_inputs = pipe.tokenizer(
                        clean_prompt,
                        padding="max_length",
                        max_length=max_length,
                        truncation=True,
                        return_tensors="pt"
                    )
                    clean_embeddings = pipe.text_encoder(text_inputs.input_ids.to(device))[0]
                    
                    # Compare embeddings
                    print(f"   📏 Shape comparison:")
                    print(f"      Clean:    {clean_embeddings.shape}")
                    print(f"      Weighted: {weighted_embeddings.shape}")
                    
                    print(f"   📊 Statistical comparison:")
                    clean_stats = {
                        'mean': clean_embeddings.mean().item(),
                        'std': clean_embeddings.std().item(), 
                        'min': clean_embeddings.min().item(),
                        'max': clean_embeddings.max().item(),
                        'norm': torch.norm(clean_embeddings).item()
                    }
                    
                    weighted_stats = {
                        'mean': weighted_embeddings.mean().item(),
                        'std': weighted_embeddings.std().item(),
                        'min': weighted_embeddings.min().item(), 
                        'max': weighted_embeddings.max().item(),
                        'norm': torch.norm(weighted_embeddings).item()
                    }
                    
                    print(f"      Clean    - Mean: {clean_stats['mean']:.6f}, Std: {clean_stats['std']:.6f}")
                    print(f"      Weighted - Mean: {weighted_stats['mean']:.6f}, Std: {weighted_stats['std']:.6f}")
                    print(f"      Clean    - Min: {clean_stats['min']:.6f}, Max: {clean_stats['max']:.6f}")
                    print(f"      Weighted - Min: {weighted_stats['min']:.6f}, Max: {weighted_stats['max']:.6f}")
                    print(f"      Clean    - Norm: {clean_stats['norm']:.6f}")
                    print(f"      Weighted - Norm: {weighted_stats['norm']:.6f}")
                    
                    # Calculate difference metrics
                    diff = weighted_embeddings - clean_embeddings
                    relative_change = torch.norm(diff) / torch.norm(clean_embeddings)
                    cosine_sim = torch.nn.functional.cosine_similarity(
                        clean_embeddings.flatten(), 
                        weighted_embeddings.flatten(), 
                        dim=0
                    )
                    
                    print(f"   🔄 Difference analysis:")
                    print(f"      L2 norm of difference: {torch.norm(diff).item():.6f}")
                    print(f"      Relative change: {relative_change.item():.6f} ({relative_change.item()*100:.2f}%)")
                    print(f"      Cosine similarity: {cosine_sim.item():.6f}")
                    
                    # Check for NaN or extreme values
                    has_nan = torch.isnan(weighted_embeddings).any()
                    has_inf = torch.isinf(weighted_embeddings).any()
                    extreme_values = (torch.abs(weighted_embeddings) > 100).any()
                    
                    print(f"   ⚠️  Quality checks:")
                    print(f"      Contains NaN: {has_nan}")
                    print(f"      Contains Inf: {has_inf}")
                    print(f"      Has extreme values (>100): {extreme_values}")
                    
                    # Token-level analysis
                    print(f"   🔤 Token-level analysis:")
                    tokens = text_inputs.input_ids[0].cpu().tolist()
                    token_texts = [pipe.tokenizer.decode([t]) for t in tokens[:10]]  # First 10 tokens
                    print(f"      First 10 tokens: {token_texts}")
                    
                    # Check embedding magnitude per token position
                    token_norms_clean = torch.norm(clean_embeddings[0], dim=-1)[:10]
                    token_norms_weighted = torch.norm(weighted_embeddings[0], dim=-1)[:10]
                    
                    print(f"      Clean token norms:    {[f'{n:.3f}' for n in token_norms_clean.cpu().tolist()]}")
                    print(f"      Weighted token norms: {[f'{n:.3f}' for n in token_norms_weighted.cpu().tolist()]}")
                    
                else:
                    # Just get baseline for unweighted prompts
                    max_length = min(77, getattr(pipe.tokenizer, 'model_max_length', 77))
                    text_inputs = pipe.tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=max_length,
                        truncation=True,
                        return_tensors="pt"
                    )
                    embeddings = pipe.text_encoder(text_inputs.input_ids.to(device))[0]
                    
                    print(f"   📏 Baseline embedding shape: {embeddings.shape}")
                    print(f"   📊 Stats - Mean: {embeddings.mean().item():.6f}, Std: {embeddings.std().item():.6f}")
                    print(f"            - Min: {embeddings.min().item():.6f}, Max: {embeddings.max().item():.6f}")
                    print(f"            - Norm: {torch.norm(embeddings).item():.6f}")
                
        except Exception as e:
            print(f"   ❌ Error analyzing prompt: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎯 Analysis Summary")
    print("=" * 60)
    print("Key things to look for:")
    print("1. Are weighted embeddings dramatically different in magnitude?")
    print("2. Do weighted embeddings contain NaN/Inf values?")
    print("3. Is the cosine similarity between clean/weighted very low?")
    print("4. Are token norms changing in unexpected ways?")
    print("\nIf weighted embeddings have very different statistics or extreme")
    print("values, that explains why they produce noise - the model expects")
    print("embeddings in a specific range/distribution.")


def compare_generation_approaches():
    """Compare different approaches to weighted generation."""
    
    print(f"\n🔬 Comparing Generation Approaches")
    print("=" * 60)
    
    # Load config
    config_path = "configs/weighted_prompts_example.yaml"
    config_manager = ConfigManager()
    config = config_manager.load_config(config_path)
    
    try:
        generator = WanVideoGenerator(config)
        generator._load_model()
        pipe = generator.pipe
        device = generator.device
        
        test_prompt = "a person (dancing:1.8) in the park"
        print(f"Test prompt: {test_prompt}")
        
        # Approach 1: Current weighted embeddings
        print(f"\n1. Current weighted embeddings approach:")
        try:
            weighted_embeds = create_weighted_embeddings(pipe, test_prompt, device)
            print(f"   ✅ Created weighted embeddings: {weighted_embeds.shape}")
            print(f"   📊 Stats: mean={weighted_embeds.mean().item():.6f}, std={weighted_embeds.std().item():.6f}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        # Approach 2: Clean prompt (what we're currently using)
        print(f"\n2. Clean prompt approach (current fallback):")
        segments = parse_weighted_prompt(test_prompt)
        clean_prompt = "".join(text for text, _ in segments)
        print(f"   Clean prompt: '{clean_prompt}'")
        
        max_length = min(77, getattr(pipe.tokenizer, 'model_max_length', 77))
        text_inputs = pipe.tokenizer(
            clean_prompt,
            padding="max_length", 
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        clean_embeds = pipe.text_encoder(text_inputs.input_ids.to(device))[0]
        print(f"   ✅ Created clean embeddings: {clean_embeds.shape}")
        print(f"   📊 Stats: mean={clean_embeds.mean().item():.6f}, std={clean_embeds.std().item():.6f}")
        
        # Approach 3: Simple CFG scaling (hypothetical)
        print(f"\n3. Potential CFG scaling approach:")
        print(f"   Instead of modifying embeddings, we could:")
        print(f"   - Use clean_prompt normally")
        print(f"   - Identify weighted words: {[text for text, weight in segments if weight != 1.0]}")
        print(f"   - Apply higher CFG scale for those regions (not implemented)")
        
        # Approach 4: Multiple prompt interpolation (hypothetical)
        print(f"\n4. Potential prompt interpolation approach:")
        print(f"   - Generate base prompt: '{clean_prompt}'")
        print(f"   - Generate emphasized prompt: 'a person dancing, dancing, dancing in the park'")
        print(f"   - Interpolate between the two embeddings")
        
        print(f"\n💡 Recommendations:")
        print(f"   1. The current clean prompt approach is safe and works")
        print(f"   2. Weighted embeddings may be too aggressive - they change the distribution")
        print(f"   3. Alternative approaches to explore:")
        print(f"      - Prompt repetition for emphasis")
        print(f"      - CFG scale modulation")
        print(f"      - Multi-prompt interpolation") 
        print(f"      - Attention masking at model level")
        
    except Exception as e:
        print(f"❌ Failed to load model for comparison: {e}")


if __name__ == "__main__":
    analyze_embeddings()
    compare_generation_approaches()
