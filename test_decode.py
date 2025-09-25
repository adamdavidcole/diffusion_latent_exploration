#!/usr/bin/env python3
"""
Quick test script to debug latent decoding
"""
import sys
import logging
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.visualization.latent_visualizer import LatentDecoder

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Test decoding
decoder = LatentDecoder("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "cuda:0")
decoder.load_vae()

# Load and decode single step
latent_path = Path('outputs/SpecificityTests/birdv2_gen_1-3b_long_latents_attn_20250820_113133/latents/prompt_000/vid_001/step_000.npy.gz')
output_path = Path('test_decode.mp4')

print("Loading latent...")
latent_tensor, metadata = decoder.load_latent_step(latent_path)
print(f"Loaded latent: {latent_tensor.shape}, {latent_tensor.dtype}")
print(f"Metadata: {metadata}")

print("Decoding to frames...")
decoded_frames = decoder.decode_latent_to_frames(latent_tensor)
print(f"Decoded frames: {decoded_frames.shape}, {decoded_frames.dtype}")
print(f"Frames min/max: {decoded_frames.min():.3f} / {decoded_frames.max():.3f}")

print("Exporting to video...")
success = decoder.frames_to_video(decoded_frames, output_path, fps=12)
print(f"Export success: {success}")

decoder.cleanup()
