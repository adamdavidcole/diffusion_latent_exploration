#!/usr/bin/env python3
"""
Debug the norm-preserving weighting method to understand why different weights
produce identical differences.
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


def debug_norm_preserving_method():
    """Debug the norm-preserving method in detail."""
    
    print("🔍 Debugging Norm-Preserving Weighting Method")
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
    
    # Test with a simple prompt
    base_prompt = "A person dancing in the park"
    weights_to_test = [1.0, 1.5, 2.0, 3.0, 5.0]
    
    print(f"\n📊 Base prompt: '{base_prompt}'")
    print("-" * 70)
    
    # Get standard embeddings for comparison
    standard_embeddings = wan_embeddings._get_standard_t5_embeddings(base_prompt, 512, 1)
    print(f"Standard embeddings shape: {standard_embeddings.shape}")
    print(f"Standard embeddings stats: mean={standard_embeddings.mean():.6f}, std={standard_embeddings.std():.6f}")
    print()
    
    # Test each weight and analyze the internal process
    for weight in weights_to_test:
        print(f"🔬 Analyzing weight {weight:.1f}:")
        print("-" * 40)
        
        if weight == 1.0:
            weighted_prompt = base_prompt
        else:
            weighted_prompt = f"A person (dancing:{weight}) in the park"
        
        print(f"Weighted prompt: '{weighted_prompt}'")
        
        # Parse the prompt to see what segments we get
        segments = wan_embeddings.processor.parse_weighted_prompt(weighted_prompt)
        print(f"Parsed segments: {[(seg.text, seg.weight) for seg in segments]}")
        
        # Check if weights are detected
        has_weights = any(seg.weight != 1.0 for seg in segments)
        print(f"Has weights detected: {has_weights}")
        
        if not has_weights:
            print("→ Using standard encoding (no weights)")
            result_embeddings = wan_embeddings._get_standard_t5_embeddings(weighted_prompt, 512, 1)
        else:
            print("→ Using weighted encoding")
            
            # Let's manually step through the weighted encoding process
            clean_prompt = "".join(seg.text for seg in segments)
            print(f"Clean prompt: '{clean_prompt}'")
            
            # Tokenize
            text_inputs = wan_embeddings.tokenizer(
                [clean_prompt],
                padding="max_length",
                max_length=512,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            
            input_ids = text_inputs.input_ids[0]
            attention_mask = text_inputs.attention_mask[0]
            seq_len = attention_mask.sum().item()
            
            print(f"Sequence length: {seq_len}")
            
            # Get base embeddings
            with torch.no_grad():
                base_embeddings = wan_embeddings.text_encoder(
                    input_ids.unsqueeze(0).to(device), 
                    attention_mask.unsqueeze(0).to(device)
                ).last_hidden_state
            
            print(f"Base embeddings shape: {base_embeddings.shape}")
            
            # Create weight map
            token_weights = torch.ones(base_embeddings.size(1), device=device, dtype=base_embeddings.dtype)
            
            # Map weights to token positions
            current_pos = 1  # Skip start token
            
            for segment in segments:
                if not segment.text.strip():
                    continue
                    
                segment_tokens, segment_len = wan_embeddings._tokenize_segment(segment.text)
                
                if segment_len > 0 and current_pos + segment_len <= seq_len:
                    end_pos = current_pos + segment_len
                    if segment.weight != 1.0:
                        token_weights[current_pos:end_pos] = segment.weight
                        print(f"  Applied weight {segment.weight} to positions {current_pos}:{end_pos} for '{segment.text.strip()}'")
                    
                    current_pos = end_pos
            
            print(f"Token weights: min={token_weights.min():.3f}, max={token_weights.max():.3f}, mean={token_weights.mean():.3f}")
            
            # Apply norm-preserving method step by step
            print("Applying norm-preserving method:")
            
            # Step 1: Get original norms
            original_norms = torch.norm(base_embeddings, dim=-1, keepdim=True)
            print(f"  Original norms shape: {original_norms.shape}")
            print(f"  Original norms stats: min={original_norms.min():.6f}, max={original_norms.max():.6f}, mean={original_norms.mean():.6f}")
            
            # Step 2: Apply weights
            weighted = base_embeddings * token_weights.unsqueeze(0).unsqueeze(-1)
            print(f"  After weight multiplication:")
            print(f"    Min value: {weighted.min():.6f}, Max value: {weighted.max():.6f}")
            
            # Step 3: Get new norms
            weighted_norms = torch.norm(weighted, dim=-1, keepdim=True)
            print(f"  Weighted norms stats: min={weighted_norms.min():.6f}, max={weighted_norms.max():.6f}, mean={weighted_norms.mean():.6f}")
            
            # Step 4: Restore original norms
            result_embeddings = weighted * (original_norms / (weighted_norms + 1e-8))
            
            print(f"  After norm restoration:")
            print(f"    Min value: {result_embeddings.min():.6f}, Max value: {result_embeddings.max():.6f}")
            
            # Check if norms are actually preserved
            final_norms = torch.norm(result_embeddings, dim=-1, keepdim=True)
            norm_preservation_error = torch.mean(torch.abs(final_norms - original_norms)).item()
            print(f"  Norm preservation error: {norm_preservation_error:.10f}")
            
            # Finalize following WAN pattern
            actual_seq_len = seq_len
            trimmed = result_embeddings[0, :actual_seq_len]
            hidden_dim = trimmed.size(-1)
            padded = torch.zeros(512, hidden_dim, dtype=trimmed.dtype, device=device)
            padded[:actual_seq_len] = trimmed
            result_embeddings = padded.unsqueeze(0)
        
        # Compare with standard embeddings
        diff = result_embeddings - standard_embeddings
        diff_norm = torch.norm(diff).item()
        standard_norm = torch.norm(standard_embeddings).item()
        relative_diff = diff_norm / standard_norm
        
        print(f"Final comparison:")
        print(f"  Relative difference: {relative_diff:.6f} ({relative_diff*100:.2f}%)")
        print(f"  L2 norm of difference: {diff_norm:.6f}")
        print(f"  Standard embeddings norm: {standard_norm:.6f}")
        
        # Analyze the difference pattern
        diff_stats = {
            'mean': diff.mean().item(),
            'std': diff.std().item(),
            'min': diff.min().item(),
            'max': diff.max().item(),
            'abs_mean': diff.abs().mean().item()
        }
        print(f"  Difference stats: {diff_stats}")
        print()


if __name__ == "__main__":
    debug_norm_preserving_method()
