#!/usr/bin/env python3
"""
Investigate the diffusion callback timing by looking at when callbacks are actually called.
"""
import sys
import torch
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

def analyze_callback_timing():
    """Analyze when callbacks are actually called during diffusion."""
    
    print("="*60)
    print("DIFFUSION CALLBACK TIMING ANALYSIS") 
    print("="*60)
    
    # Check if there's a pattern in the timesteps
    experiment_dir = Path('outputs/SpecificityTests/birdv2_gen_1-3b_long_latents_attn_20250820_113133/latents/prompt_000/vid_001')
    
    import json
    timesteps = []
    sigmas = []
    
    for step in range(20):
        metadata_file = experiment_dir / f'step_{step:03d}_metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            timesteps.append(metadata['timestep'])
            sigmas.append(metadata.get('sigma', 0))
    
    print(f"Stored timesteps: {timesteps}")
    print(f"Timestep progression: {timesteps[0]} -> {timesteps[-1]}")
    print(f"Sigma progression: {sigmas[0]:.3f} -> {sigmas[-1]:.3f}")
    
    # The key insight: in diffusion, we START with noise (timestep 1000) and END with clean image (timestep 0)
    # So if step 19 has timestep 208, there should be more steps to get to timestep 0
    
    print("\n" + "="*60)
    print("DIFFUSION PROCESS ANALYSIS")
    print("="*60)
    
    print("Standard diffusion process:")
    print("  1. Start with pure noise (timestep ~1000)")
    print("  2. Apply denoising steps to gradually reduce noise")
    print("  3. End with clean image (timestep 0)")
    print()
    print(f"Our captured process:")
    print(f"  - Step 0: timestep {timesteps[0]} (after 1st denoising step)")
    print(f"  - Step 19: timestep {timesteps[-1]} (after 20th denoising step)")
    print()
    
    if timesteps[-1] > 50:  # If final timestep is still high
        print("❌ ISSUE DETECTED: Final timestep is still high!")
        print(f"   Expected final timestep: ~0-10")
        print(f"   Actual final timestep: {timesteps[-1]}")
        print()
        print("This suggests one of:")
        print("  1. The callback is called BEFORE the final result")
        print("  2. There are additional steps after step 19")
        print("  3. The scheduling is different than expected")
    else:
        print("✅ Timestep progression looks normal")
    
    # Let's also check: could the issue be in VIDEO vs IMAGE generation?
    # Video generation might have additional post-processing
    
    print("\n" + "="*60)
    print("VIDEO GENERATION SPECIFICS")
    print("="*60)
    
    print("WAN Video Generation process might include:")
    print("  1. Diffusion denoising (what we capture)")
    print("  2. VAE spatial decode (latent -> pixel space)")
    print("  3. Video temporal processing")
    print("  4. Format conversion/normalization")
    print("  5. Compression/encoding")
    
    return timesteps, sigmas

def check_generation_config():
    """Check the generation configuration for clues."""
    
    print("\n" + "="*60)
    print("GENERATION CONFIGURATION")
    print("="*60)
    
    import yaml
    config_file = Path('outputs/SpecificityTests/birdv2_gen_1-3b_long_latents_attn_20250820_113133/configs/generation_config.yaml')
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Model: {config['model_settings']['model_id']}")
    print(f"Steps: {config['model_settings']['steps']}")
    print(f"CFG Scale: {config['model_settings']['cfg_scale']}")
    print(f"Seed: {config['model_settings']['seed']}")
    print(f"Video frames: {config['video_settings']['frames']}")
    print(f"Resolution: {config['video_settings']['width']}x{config['video_settings']['height']}")

if __name__ == "__main__":
    timesteps, sigmas = analyze_callback_timing()
    check_generation_config()
    
    print("\n" + "="*60)
    print("NEXT STEPS FOR DEBUGGING")
    print("="*60)
    print("1. Check diffusion literature for WAN scheduling")
    print("2. Examine WAN pipeline source code") 
    print("3. Add debugging to capture the TRUE final latent")
    print("4. Compare our callback timing with standard diffusion")
