#!/usr/bin/env python3
"""
Debug script to compare final generation output with decoded step 19 latent.
This will help identify if there's a difference between:
1. The final video from generation
2. The decoded video from step 19 latent
"""
import sys
import torch
import numpy as np
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.visualization.latent_visualizer import LatentDecoder

# Setup logging
logging.basicConfig(level=logging.INFO)

def compare_videos():
    """Compare final generated video with decoded step 19 latent."""
    
    # Paths
    experiment_dir = Path("outputs/SpecificityTests/birdv2_gen_1-3b_long_latents_attn_20250820_113133")
    final_video_path = experiment_dir / "videos" / "prompt_000" / "video_001.mp4"  # Note: video_001, not vid_001
    decoded_step19_path = experiment_dir / "decoded_latents" / "prompt_000" / "vid_001" / "step_019.mp4"
    
    print(f"Experiment directory: {experiment_dir}")
    print(f"Final video exists: {final_video_path.exists()}")
    print(f"Decoded step 19 exists: {decoded_step19_path.exists()}")
    
    if not final_video_path.exists():
        print("❌ Final video not found")
        return
        
    if not decoded_step19_path.exists():
        print("❌ Decoded step 19 not found")
        return
    
    # Get file sizes for basic comparison
    final_size = final_video_path.stat().st_size
    decoded_size = decoded_step19_path.stat().st_size
    
    print(f"Final video size: {final_size:,} bytes")
    print(f"Decoded step 19 size: {decoded_size:,} bytes")
    print(f"Size difference: {abs(final_size - decoded_size):,} bytes ({abs(final_size - decoded_size)/final_size:.2%})")
    
    # Now let's check the raw latent data and see if we can manually decode it
    # and compare with the final pipeline output
    
    print("\n" + "="*60)
    print("MANUAL LATENT DECODE TEST")
    print("="*60)
    
    # Initialize decoder
    decoder = LatentDecoder("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "cuda:0")
    decoder.load_vae()
    
    # Load step 19 latent
    step19_latent_path = experiment_dir / "latents" / "prompt_000" / "vid_001" / "step_019.npy.gz"
    latent_tensor, metadata = decoder.load_latent_step(step19_latent_path)
    
    print(f"Step 19 latent shape: {latent_tensor.shape}")
    print(f"Step 19 latent dtype: {latent_tensor.dtype}")
    print(f"Step 19 latent range: [{latent_tensor.min():.3f}, {latent_tensor.max():.3f}]")
    print(f"Step 19 metadata: {metadata}")
    
    # Decode the latent manually
    print("\nDecoding latent manually...")
    decoded_frames = decoder.decode_latent_to_frames(latent_tensor)
    print(f"Decoded frames shape: {decoded_frames.shape}")
    print(f"Decoded frames dtype: {decoded_frames.dtype}")
    print(f"Decoded frames range: [{decoded_frames.min():.3f}, {decoded_frames.max():.3f}]")
    
    # Save a test decode
    test_output = experiment_dir / "test_manual_decode_step19.mp4"
    success = decoder.frames_to_video(decoded_frames, test_output, fps=12)
    print(f"Manual decode saved: {success} -> {test_output}")
    
    if success and test_output.exists():
        manual_size = test_output.stat().st_size
        print(f"Manual decode size: {manual_size:,} bytes")
        print(f"Difference from final: {abs(final_size - manual_size):,} bytes ({abs(final_size - manual_size)/final_size:.2%})")
        print(f"Difference from auto decode: {abs(decoded_size - manual_size):,} bytes ({abs(decoded_size - manual_size)/decoded_size:.2%})")
    
    decoder.cleanup()
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    if abs(final_size - decoded_size) > final_size * 0.1:  # More than 10% difference
        print("❌ SIGNIFICANT DIFFERENCE detected between final video and decoded step 19")
        print("   This suggests there may be an issue with:")
        print("   1. When latents are captured during generation")
        print("   2. How latents are stored/loaded")
        print("   3. Differences in VAE decoding process")
    else:
        print("✅ File sizes are similar - differences might be due to:")
        print("   1. Video compression differences")
        print("   2. Minor floating point precision differences")
        print("   3. Different encoding settings")

if __name__ == "__main__":
    compare_videos()
