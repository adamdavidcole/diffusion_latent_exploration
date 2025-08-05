#!/usr/bin/env python3
"""
Test script to verify thumbnail generation integration in video generator.
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, '/home/adam/dev/diffusion_latent_exploration')

from src.generators.video_generator import generate_thumbnail

def test_thumbnail_integration():
    """Test thumbnail generation on an existing video."""
    
    # Use the correct path structure
    test_video = "/home/adam/dev/diffusion_latent_exploration/outputs/14b_hero_rescue_weighted_5s_20250727_223151/videos/prompt_000/video_001.mp4"
    
    print(f"Testing thumbnail generation integration...")
    print(f"Test video: {test_video}")
    
    # Check if test video exists
    if not Path(test_video).exists():
        print(f"❌ Test video not found: {test_video}")
        return False
    
    print(f"✅ Test video found")
    
    # Test thumbnail generation
    print(f"Generating thumbnail...")
    success = generate_thumbnail(test_video)
    
    if success:
        # Check if thumbnail was created
        thumbnail_path = Path(test_video).with_suffix('.jpg')
        if thumbnail_path.exists():
            print(f"✅ Thumbnail generated successfully: {thumbnail_path}")
            print(f"✅ Integration test passed!")
            return True
        else:
            print(f"❌ Thumbnail generation reported success but file not found: {thumbnail_path}")
            return False
    else:
        print(f"❌ Thumbnail generation failed")
        return False

if __name__ == "__main__":
    success = test_thumbnail_integration()
    sys.exit(0 if success else 1)
