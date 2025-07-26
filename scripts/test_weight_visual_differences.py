#!/usr/bin/env python3
"""
Test visual differences with different prompt weights.
Generates videos with the same prompt but different weights to verify visual impact.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our modules
from src.config.config_manager import ConfigManager
from src.generators.video_generator import WanVideoGenerator


def test_weight_visual_differences():
    """Test visual differences with different prompt weights."""
    
    print("🎯 Testing Visual Differences with Different Weights")
    print("=" * 70)
    
    # Load config
    config_path = "configs/weighted_prompts_example.yaml"
    config_manager = ConfigManager()
    config = config_manager.load_config(config_path)
    
    # Ensure weighted embeddings are enabled
    config.prompt_settings.use_weighted_embeddings = True
    config.prompt_settings.embedding_method = "norm_preserving"
    
    # Create output directory for this test
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    test_output_dir = Path(f"outputs/weight_test_{timestamp}")
    test_output_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir = str(test_output_dir)
    
    print(f"📁 Output directory: {test_output_dir}")
    
    # Test weights to compare
    weights = [1.0, 1.5, 2.0, 3.0, 5.0]
    
    # Test prompt - using a concept that should show clear visual differences
    base_prompt = "A person dancing"
    target_word = "dancing"
    
    print(f"\n🎭 Base prompt: '{base_prompt}'")
    print(f"🎯 Target word: '{target_word}'")
    print(f"⚖️  Testing weights: {weights}")
    
    # Fixed seed for consistent comparison
    config.model_settings.seed = 12345
    
    # Initialize generator
    try:
        generator = WanVideoGenerator(config)
        print(f"✅ WAN generator initialized")
        
    except Exception as e:
        print(f"❌ Failed to initialize WAN generator: {e}")
        return
    
    print(f"\n🎬 Generating {len(weights)} videos for comparison...")
    print("-" * 50)
    
    results = []
    
    for i, weight in enumerate(weights):
        print(f"\n{i+1}/{len(weights)} - Weight: {weight}")
        
        # Create weighted prompt
        if weight == 1.0:
            prompt = base_prompt  # No weighting syntax for baseline
            video_name = f"weight_{weight:.1f}_baseline"
        else:
            prompt = f"A person ({target_word}:{weight}) "
            video_name = f"weight_{weight:.1f}_enhanced"
        
        print(f"   Prompt: '{prompt}'")
        
        # Create individual output directory for this weight
        weight_dir = test_output_dir / f"weight_{weight:.1f}"
        weight_dir.mkdir(exist_ok=True)
        
        try:
            # Generate video
            start_time = time.time()
            
            # Use the generate method directly
            video_path = generator.generate(
                prompt=prompt,
                output_path=weight_dir / f"{video_name}.mp4",
                seed=config.model_settings.seed
            )
            
            generation_time = time.time() - start_time
            
            print(f"   ✅ Generated: {video_path}")
            print(f"   ⏱️  Time: {generation_time:.1f}s")
            
            results.append({
                'weight': weight,
                'prompt': prompt,
                'video_path': video_path,
                'generation_time': generation_time
            })
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print(f"\n📊 Test Results Summary")
    print("=" * 50)
    print(f"Total videos generated: {len(results)}")
    print(f"Output directory: {test_output_dir}")
    
    if results:
        print(f"\nGenerated videos:")
        for result in results:
            rel_path = Path(result['video_path']).relative_to(test_output_dir)
            print(f"  Weight {result['weight']:4.1f}: {rel_path} ({result['generation_time']:.1f}s)")
        
        print(f"\n🔍 Visual Comparison Guide:")
        print(f"1. Open all videos in a video player")
        print(f"2. Compare the intensity/prominence of '{target_word}' motion")
        print(f"3. Higher weights should show more pronounced dancing")
        print(f"4. Weight 1.0 is the baseline (no enhancement)")
        
        # Create a comparison script
        comparison_script = test_output_dir / "compare_videos.py"
        with open(comparison_script, 'w') as f:
            f.write(f"""#!/usr/bin/env python3
# Quick comparison script for generated videos
import subprocess
import sys
from pathlib import Path

videos = [
""")
            for result in results:
                rel_path = Path(result['video_path']).relative_to(test_output_dir)
                f.write(f'    ("{result["weight"]}", "{rel_path}"),\n')
            
            f.write(f"""]

print("Generated videos with different '{target_word}' weights:")
for weight, path in videos:
    print(f"  Weight {{weight:4.1f}}: {{path}}")

print("\\nTo view videos:")
for weight, path in videos:
    print(f"  vlc '{{path}}' &")
""")
        
        print(f"\n📝 Created comparison script: {comparison_script}")
        
    else:
        print("❌ No videos were successfully generated")
    
    print(f"\n✅ Weight visual difference test completed!")


if __name__ == "__main__":
    test_weight_visual_differences()
