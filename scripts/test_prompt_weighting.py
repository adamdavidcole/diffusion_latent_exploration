#!/usr/bin/env python3
"""
Test script to verify prompt weighting functionality is working correctly.
This creates a side-by-side comparison between weighted and unweighted prompts.
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.orchestrator import VideoGenerationOrchestrator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_prompt_weighting():
    """Test prompt weighting with a clear comparison."""
    
    print("🧪 Prompt Weighting Test")
    print("=" * 50)
    
    # Test 1: Load configuration and verify settings
    print("\n1. Testing Configuration Loading")
    print("-" * 30)
    
    config_manager = ConfigManager()
    config = config_manager.load_config("configs/weighted_prompts_example.yaml")
    
    print(f"✅ Config loaded successfully")
    print(f"   Model: {config.model_settings.model_id}")
    print(f"   Device: {config.model_settings.device}")
    print(f"   Weighting enabled: {config.prompt_settings.enable_weighting}")
    print(f"   Variation weight: {config.prompt_settings.variation_weight}")
    print(f"   Video size: {config.video_settings.width}x{config.video_settings.height}")
    print(f"   Duration: {config.video_settings.duration}s at {config.video_settings.fps}fps")
    
    # Test 2: Create orchestrator and test prompt processing
    print("\n2. Testing Prompt Processing")
    print("-" * 30)
    
    orchestrator = VideoGenerationOrchestrator(config)
    
    # Test template that should show clear differences with weighting
    test_template = "a person [walking|running|dancing|jumping] in a park"
    
    print(f"Test template: {test_template}")
    variations = orchestrator.process_prompt_template(test_template, max_variations=4)
    
    print(f"\nGenerated {len(variations)} variations:")
    for i, var in enumerate(variations, 1):
        print(f"  {i}. Standard: {var.text}")
        if var.weighted_text:
            print(f"     Weighted: {var.weighted_text}")
            print(f"     🎯 Difference: The variation word gets {config.prompt_settings.variation_weight}x emphasis")
        else:
            print(f"     ❌ No weighted version found!")
    
    # Test 3: Manual prompt weight detection
    print("\n3. Testing Manual Weight Detection")
    print("-" * 30)
    
    from src.generators.video_generator import parse_weighted_prompt, WAN_AVAILABLE
    
    test_prompts = [
        "a simple prompt without weights",
        "a prompt with (emphasis:1.5) on specific words",
        "multiple (strong:2.0) weights and (subtle:1.2) emphasis",
        "a person (dancing:1.8) in the park"
    ]
    
    for prompt in test_prompts:
        segments = parse_weighted_prompt(prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Parsed segments: {segments}")
        has_weights = any(weight != 1.0 for _, weight in segments)
        print(f"Has weights: {'✅ Yes' if has_weights else '❌ No'}")
    
    # Test 4: Check if WAN model components are available
    print("\n4. Testing WAN Model Availability")
    print("-" * 30)
    
    if WAN_AVAILABLE:
        print("✅ WAN dependencies available")
        try:
            # Test model initialization (but don't actually load it)
            print(f"   Model ID configured: {config.model_settings.model_id}")
            print(f"   Target device: {config.model_settings.device}")
            
            # Check if the device is available
            import torch
            if config.model_settings.device.startswith('cuda'):
                device_id = int(config.model_settings.device.split(':')[1])
                if torch.cuda.is_available() and device_id < torch.cuda.device_count():
                    print(f"   ✅ Device {config.model_settings.device} is available")
                else:
                    print(f"   ❌ Device {config.model_settings.device} is not available")
                    print(f"   Available devices: {torch.cuda.device_count()}")
            else:
                print(f"   ✅ CPU device selected")
                
        except Exception as e:
            print(f"   ⚠️  Model check failed: {e}")
    else:
        print("❌ WAN dependencies not available - will use mock generator")
    
    # Test 5: Quick generation test (if requested)
    print("\n5. Generation Test Options")
    print("-" * 30)
    print("To test actual video generation with weighted prompts:")
    print(f"Option A (Single video):")
    print(f'  python main.py --config configs/weighted_prompts_example.yaml \\')
    print(f'    --template "a person (dancing:1.8) in the park" \\')
    print(f'    --videos-per-variation 1')
    
    print(f"\nOption B (Template with variations):")
    print(f'  python main.py --config configs/weighted_prompts_example.yaml \\')
    print(f'    --template "{test_template}" \\')
    print(f'    --videos-per-variation 1')
    
    print(f"\nOption C (Quick test with mock generator):")
    print(f'  # Edit configs/weighted_prompts_example.yaml and comment out WAN imports')
    print(f'  # This will use mock generator to test the prompt processing pipeline')
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 Test Summary")
    print("=" * 50)
    
    # Check what's working
    issues_found = []
    
    if not hasattr(config, 'prompt_settings'):
        issues_found.append("❌ Prompt settings not found in config")
    elif not config.prompt_settings.enable_weighting:
        issues_found.append("❌ Prompt weighting not enabled in config")
    else:
        print("✅ Prompt weighting configuration is correct")
    
    if not any(var.weighted_text for var in variations):
        issues_found.append("❌ No weighted prompts were generated")
    else:
        weighted_count = sum(1 for var in variations if var.weighted_text)
        print(f"✅ Generated {weighted_count}/{len(variations)} weighted prompts")
    
    if not WAN_AVAILABLE:
        print("⚠️  WAN dependencies not available (will use mock generator)")
    else:
        print("✅ WAN dependencies are available")
    
    if issues_found:
        print("\n🚨 Issues Found:")
        for issue in issues_found:
            print(f"   {issue}")
    else:
        print("\n🎉 All tests passed! Prompt weighting is ready to use.")
        print("\nNext step: Run one of the generation commands above to see it in action!")

if __name__ == "__main__":
    test_prompt_weighting()
