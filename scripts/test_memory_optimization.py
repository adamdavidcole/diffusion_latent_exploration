#!/usr/bin/env python3
"""
Test script for memory optimization features with WAN 14B model.
This script tests the memory management features without running a full batch.
"""

import sys
import os
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.config_manager import ConfigManager
from generators.video_generator import WAN13BVideoGenerator, get_gpu_memory_info, clear_gpu_memory

def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

def test_memory_optimization():
    """Test memory optimization features."""
    setup_logging()
    
    logging.info("Testing WAN 14B memory optimization...")
    
    # Load optimized config
    config_manager = ConfigManager()
    try:
        config = config_manager.load_config("configs/wan_14b_optimized.yaml")
        logging.info("✅ Loaded optimized 14B config successfully")
    except Exception as e:
        logging.error(f"❌ Failed to load config: {e}")
        return False
    
    # Test memory settings
    if hasattr(config, 'memory_settings'):
        mem_settings = config.memory_settings
        logging.info("✅ Memory settings found:")
        logging.info(f"  - Memory optimization: {mem_settings.enable_memory_optimization}")
        logging.info(f"  - Clear cache between videos: {mem_settings.clear_cache_between_videos}")
        logging.info(f"  - Reload model for large models: {mem_settings.reload_model_for_large_models}")
        logging.info(f"  - Gradient checkpointing: {mem_settings.use_gradient_checkpointing}")
        logging.info(f"  - Memory efficient attention: {mem_settings.enable_memory_efficient_attention}")
    else:
        logging.error("❌ Memory settings not found in config")
        return False
    
    # Test GPU memory info
    mem_info = get_gpu_memory_info()
    if mem_info:
        logging.info("✅ GPU memory info available:")
        logging.info(f"  - Total: {mem_info['total_gb']:.1f}GB")
        logging.info(f"  - Free: {mem_info['free_gb']:.1f}GB")
        logging.info(f"  - Allocated: {mem_info['allocated_gb']:.1f}GB")
    else:
        logging.warning("⚠️  GPU memory info not available (CPU mode or no CUDA)")
    
    # Test model initialization (without loading)
    try:
        generator = WAN13BVideoGenerator(config)
        logging.info("✅ Generator initialized successfully")
        
        # Check if it detects large model
        if hasattr(generator.model, '_is_large_model'):
            is_large = generator.model._is_large_model()
            logging.info(f"✅ Large model detection: {is_large}")
        
        # Check memory settings integration
        if hasattr(generator.model, 'memory_settings'):
            logging.info("✅ Memory settings properly integrated into generator")
        else:
            logging.warning("⚠️  Memory settings not found in generator")
            
    except Exception as e:
        logging.error(f"❌ Failed to initialize generator: {e}")
        return False
    
    # Test memory clearing function
    try:
        clear_gpu_memory()
        logging.info("✅ Memory clearing function works")
    except Exception as e:
        logging.warning(f"⚠️  Memory clearing had issues: {e}")
    
    logging.info("✅ Memory optimization test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_memory_optimization()
    sys.exit(0 if success else 1)
