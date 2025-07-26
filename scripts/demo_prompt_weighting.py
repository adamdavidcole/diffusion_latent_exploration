#!/usr/bin/env python3
"""
Demonstration of the new prompt weighting feature for WAN video generation.

This script shows how to use weighted prompts to emphasize specific variations
in your prompt templates, leading to more focused and distinct video outputs.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.prompts import PromptManager, WeightingConfig
from src.orchestrator import VideoGenerationOrchestrator
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demonstrate_prompt_weighting():
    """Demonstrate the prompt weighting feature with examples."""
    
    print("🎬 WAN Video Generation - Prompt Weighting Demo")
    print("=" * 60)
    
    # Example 1: Basic weighted prompts
    print("\n1. Basic Prompt Weighting Examples")
    print("-" * 40)
    
    # Create a prompt manager
    prompt_manager = PromptManager()
    
    # Example prompts with different weighting scenarios
    examples = [
        {
            "name": "Strong Variation Emphasis",
            "template": "a romantic kiss between [two people|two men|two women|a man and a woman]",
            "weighting": WeightingConfig(enable_weighting=True, variation_weight=1.8, base_weight=1.0)
        },
        {
            "name": "Subtle Variation Emphasis", 
            "template": "a peaceful landscape with [mountains|lakes|forests|deserts] in the background",
            "weighting": WeightingConfig(enable_weighting=True, variation_weight=1.3, base_weight=1.0)
        },
        {
            "name": "Extreme Variation Focus",
            "template": "an action scene with [cars|motorcycles|airplanes|boats] racing",
            "weighting": WeightingConfig(enable_weighting=True, variation_weight=2.0, base_weight=0.8)
        }
    ]
    
    for example in examples:
        print(f"\n{example['name']}:")
        print(f"Template: {example['template']}")
        
        # Load template with weighting
        template = prompt_manager.load_template(example["template"], example["weighting"])
        variations = template.generate_variations()
        
        print(f"Generated {len(variations)} variations:")
        for i, var in enumerate(variations[:3]):  # Show first 3
            print(f"  {i+1}. Standard: {var.text}")
            if var.weighted_text:
                print(f"     Weighted: {var.weighted_text}")
        if len(variations) > 3:
            print(f"     ... and {len(variations) - 3} more variations")
    
    # Example 2: Configuration-based weighting
    print("\n\n2. Configuration-Based Weighting")
    print("-" * 40)
    
    # Load the weighted prompts example config
    config_manager = ConfigManager()
    try:
        config = config_manager.load_config("configs/weighted_prompts_example.yaml")
        print("✅ Loaded weighted prompts configuration")
        
        print(f"Prompt weighting enabled: {config.prompt_settings.enable_weighting}")
        print(f"Variation weight: {config.prompt_settings.variation_weight}")
        print(f"Base weight: {config.prompt_settings.base_weight}")
        
        # Create orchestrator with this config
        orchestrator = VideoGenerationOrchestrator(config)
        
        # Process a template using the config settings
        template = "a beautiful scene with [sunrise|sunset|moonlight|starlight] illumination"
        variations = orchestrator.process_prompt_template(template, max_variations=4)
        
        print(f"\nProcessed template with config weighting:")
        print(f"Template: {template}")
        for i, var in enumerate(variations):
            print(f"  {i+1}. Standard: {var.text}")
            if var.weighted_text:
                print(f"     Weighted: {var.weighted_text}")
                
    except Exception as e:
        logger.error(f"Could not load weighted prompts config: {e}")
    
    # Example 3: Manual prompt weighting syntax
    print("\n\n3. Manual Prompt Weighting Syntax")
    print("-" * 40)
    
    manual_examples = [
        "a (romantic:1.5) kiss between two people",
        "a landscape with (beautiful mountains:1.3) and (clear lakes:1.2)",
        "(dramatic:2.0) action scene with (fast cars:1.8) racing",
        "a peaceful (forest:1.4) with (morning sunlight:1.6) filtering through trees"
    ]
    
    print("You can also manually specify weights in prompts using (text:weight) syntax:")
    for prompt in manual_examples:
        print(f"  • {prompt}")
    
    print("\n\n4. Benefits of Prompt Weighting")
    print("-" * 40)
    print("✨ More focused outputs: Variations get stronger emphasis")
    print("✨ Better distinction: Each variation becomes more unique") 
    print("✨ Creative control: Fine-tune the balance between base and variation")
    print("✨ Automatic integration: Works seamlessly with existing templates")
    
    print("\n\n5. Usage Tips")
    print("-" * 40)
    print("💡 Start with weights around 1.3-1.8 for subtle emphasis")
    print("💡 Use higher weights (2.0+) for dramatic differences")
    print("💡 Lower base weights (0.8) can help variation stand out more")
    print("💡 Test different weights to find what works for your content")
    
    print("\n" + "=" * 60)
    print("Ready to generate videos with weighted prompts!")
    print("Use: python main.py --config configs/weighted_prompts_example.yaml --template 'your template here'")

if __name__ == "__main__":
    demonstrate_prompt_weighting()
