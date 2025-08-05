#!/usr/bin/env python3
"""
Test script to validate attention storage integration.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import ConfigManager, GenerationConfig
from src.orchestrator import VideoGenerationOrchestrator


def test_attention_integration():
    """Test that attention storage integrates properly with the system."""
    
    # Load default config
    config_manager = ConfigManager()
    config = config_manager.create_default_config()
    
    # Enable attention storage with correct settings
    config.attention_analysis_settings.store_attention = True
    config.attention_analysis_settings.tokenizer_name = "google/umt5-xxl"  # Correct T5 tokenizer
    config.attention_analysis_settings.storage_interval = 5  # Store every 5 steps
    config.attention_analysis_settings.storage_dtype = "float16"  # Use FP16 for efficiency
    config.attention_analysis_settings.compress_attention = True
    
    # Create orchestrator
    orchestrator = VideoGenerationOrchestrator(config)
    
    # Test prompt with parenthetical tokens
    test_template = "a beautiful (landscape) with (mountains) and trees"
    
    # Preview the batch to ensure attention tokens are detected
    print("Testing attention storage integration...")
    print(f"Template: {test_template}")
    
    try:
        preview = orchestrator.preview_batch(test_template, max_preview=2)
        
        print("\nPreview successful!")
        print(f"Total variations: {preview['total_variations']}")
        print(f"Total videos would be generated: {preview['total_videos']}")
        
        # Check if attention validation works
        preview_variations = preview['preview_variations']
        for i, prompt_text in enumerate(preview_variations):
            print(f"\nVariation {i+1}: {prompt_text}")
            
            # Test tokenizer detection
            from src.utils.attention_storage import AttentionStorage
            storage = AttentionStorage(
                storage_dir="/tmp/test_attention",
                tokenizer_name="google/umt5-xxl"
            )
            
            tokens = storage.parse_parenthetical_tokens(prompt_text)
            print(f"  Detected parenthetical tokens: {tokens}")
            
            if tokens:
                for word, token_ids in tokens.items():
                    print(f"    '{word}' -> token IDs: {token_ids}")
        
        print("\nAttention storage integration test passed!")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_attention_integration()
    sys.exit(0 if success else 1)
