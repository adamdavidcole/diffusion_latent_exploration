#!/usr/bin/env python3
"""
Debug script to check config loading for weighted embeddings.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config.config_manager import ConfigManager

def debug_config():
    """Debug the config loading to see if weighted embeddings are enabled."""
    
    print("🔍 Config Debug for Weighted Embeddings")
    print("=" * 50)
    
    # Load config
    config_path = "configs/weighted_prompts_example.yaml"
    config_manager = ConfigManager()
    config = config_manager.load_config(config_path)
    
    print(f"Config loaded from: {config_path}")
    print(f"Config type: {type(config)}")
    
    print(f"\nPrompt settings:")
    print(f"  Has prompt_settings: {hasattr(config, 'prompt_settings')}")
    
    if hasattr(config, 'prompt_settings'):
        prompt_settings = config.prompt_settings
        print(f"  prompt_settings type: {type(prompt_settings)}")
        print(f"  enable_weighting: {getattr(prompt_settings, 'enable_weighting', 'NOT FOUND')}")
        print(f"  use_weighted_embeddings: {getattr(prompt_settings, 'use_weighted_embeddings', 'NOT FOUND')}")
        print(f"  embedding_method: {getattr(prompt_settings, 'embedding_method', 'NOT FOUND')}")
        print(f"  variation_weight: {getattr(prompt_settings, 'variation_weight', 'NOT FOUND')}")
        
        # Try all attributes
        print(f"\nAll prompt_settings attributes:")
        for attr in dir(prompt_settings):
            if not attr.startswith('_'):
                value = getattr(prompt_settings, attr)
                print(f"    {attr}: {value}")
    else:
        print("  ❌ No prompt_settings found!")
    
    print(f"\nAll config attributes:")
    for attr in dir(config):
        if not attr.startswith('_'):
            value = getattr(config, attr)
            print(f"  {attr}: {type(value)}")


if __name__ == "__main__":
    debug_config()
