#!/usr/bin/env python3
"""
Simple test to debug attention storage hook registration.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from src.config import ConfigManager
from src.generators import WAN13BVideoGenerator


def test_hook_registration():
    """Test attention hook registration on WAN model."""
    
    # Create minimal config for testing
    config_manager = ConfigManager()
    config = config_manager.create_default_config()
    
    # Use 1.3B model instead of 14B
    config.model_settings.model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    config.model_settings.device = "cuda:1"  # Use cuda:1
    
    # Use small settings for fast testing
    config.model_settings.steps = 5
    config.video_settings.width = 480
    config.video_settings.height = 480
    config.video_settings.duration = 1.0  # Just 1 second
    
    print("Loading WAN model for hook testing...")
    
    try:
        # Create generator and load model
        generator = WAN13BVideoGenerator(config)
        
        if not generator.is_real_model():
            print("❌ Using mock model, cannot test attention hooks")
            return False
        
        actual_generator = generator.model  # Get the actual WanVideoGenerator
        
        print(f"Generator type: {type(actual_generator)}")
        print(f"Has pipe: {hasattr(actual_generator, 'pipe')}")
        
        # Force model loading if pipe is None
        if hasattr(actual_generator, 'pipe') and actual_generator.pipe is None:
            print("Pipe is None, attempting to load model...")
            if hasattr(actual_generator, '_load_model'):
                actual_generator._load_model()
            else:
                print("No _load_model method found")
        
        if hasattr(actual_generator, 'pipe') and actual_generator.pipe is not None:
            print(f"Model loaded: {type(actual_generator.pipe.transformer).__name__}")
            print(f"Model device: {next(actual_generator.pipe.transformer.parameters()).device}")
        else:
            print("❌ Model pipe is still None, model failed to load")
            print(f"Available methods: {[m for m in dir(actual_generator) if not m.startswith('_')]}")
            return False
        
        # Test attention storage
        from src.utils.attention_storage import AttentionStorage
        
        storage = AttentionStorage(
            storage_dir="/tmp/test_attention_hooks",
            tokenizer_name="google/umt5-xxl",
            storage_interval=1  # Store every step for testing
        )
        
        # Test prompt with token
        test_prompt = "(flower)"
        storage.start_video_storage(
            video_id="test_vid",
            prompt=test_prompt
        )

        # print(pipe.transformer)
        
        print(f"\nTokens found: {storage.target_tokens}")
        
        # Register attention module wrappers
        print("\nRegistering attention module wrappers...")
        success = storage.register_attention_hooks(actual_generator.pipe.transformer)
        
        print(f"Modules wrapped: {len(storage.original_modules) if hasattr(storage, 'original_modules') else 0}")
        
        if not success or len(storage.original_modules) == 0:
            print("❌ No modules wrapped! This is the problem.")
            success = False
        else:
            print("✅ Modules wrapped successfully!")
            success = True
        
        # Clean up
        storage.remove_attention_hooks(actual_generator.pipe.transformer)
        
        return success
        
    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_hook_registration()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
