#!/usr/bin/env python3
"""
Simple debug script to test VLM model loading and basic inference.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.vlm_analysis.vlm_model_loader import VLMModelLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test basic model loading."""
    logger.info("Testing model loading...")
    
    try:
        # Initialize model loader
        model_loader = VLMModelLoader()
        
        # Load model
        model_loader.load_model()
        
        # Check model info
        info = model_loader.get_model_info()
        logger.info(f"Model info: {info}")
        
        return model_loader
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise

def test_simple_inference(model_loader, video_path):
    """Test simple inference with minimal prompt."""
    logger.info(f"Testing inference with video: {video_path}")
    
    try:
        # Simple prompt
        prompt = "Describe this video in one sentence."
        
        # Analyze
        response = model_loader.analyze_video(
            video_path=video_path,
            text_prompt=prompt,
            max_new_tokens=100,     # Increase tokens
            do_sample=True,         # Enable sampling
            fps=12.0,
            max_pixels=50000        # Very low resolution for speed
        )
        
        logger.info(f"✅ Inference successful!")
        logger.info(f"Response: {response}")
        
        return response
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise

def main():
    video_path = "/home/adam/dev/diffusion_latent_exploration/outputs/14b_kiss_latent_attention_no_weight_20250811_003646/videos/prompt_001/video_001.mp4"
    
    if not Path(video_path).exists():
        logger.error(f"Video not found: {video_path}")
        return 1
    
    try:
        # Test model loading
        model_loader = test_model_loading()
        
        # Test simple inference
        test_simple_inference(model_loader, video_path)
        
        # Cleanup
        model_loader.cleanup()
        
        logger.info("✅ All tests passed!")
        return 0
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
