#!/usr/bin/env python3
"""
Test nested template syntax with weighting.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.prompts.prompt_manager import PromptTemplate, WeightingConfig
from src.prompts.prompt_weighting import PromptWeightingProcessor

def test_nested_syntax():
    """Test how nested template variations work with weighting."""
    
    print("🔍 Testing Nested Template Syntax with Weighting")
    print("=" * 60)
    
    # Your proposed template
    template = "a romantic kiss between ([two people | two men | two women | a man and woman]:10)"
    
    print(f"Template: {template}")
    print()
    
    # Test 1: Check if the prompt parser can handle this syntax
    print("🧪 Test 1: Raw Prompt Weight Parsing")
    print("-" * 40)
    
    processor = PromptWeightingProcessor()
    segments = processor.parse_weighted_prompt(template)
    
    print("Parsed segments:")
    for i, segment in enumerate(segments):
        print(f"  {i+1}. '{segment.text}' (weight: {segment.weight})")
    
    print()
    
    # Test 2: Check template variation parsing
    print("🧪 Test 2: Template Variation Generation")
    print("-" * 40)
    
    # Set up weighting config
    weighting_config = WeightingConfig(
        enable_weighting=True,
        variation_weight=10.0,  # The weight from your template
        base_weight=1.0
    )
    
    # Create prompt template
    prompt_template = PromptTemplate(template, weighting_config)
    
    print(f"Template variables found: {list(prompt_template.variables.keys())}")
    print(f"Template variables: {prompt_template.variables}")
    print()
    
    # Generate variations
    variations = prompt_template.generate_variations()
    
    print(f"Generated {len(variations)} variations:")
    for i, var in enumerate(variations):
        print(f"  {i+1}. ID: {var.variation_id}")
        print(f"     Text: '{var.text}'")
        if var.weighted_text:
            print(f"     Weighted: '{var.weighted_text}'")
        else:
            print(f"     Weighted: None")
        print()
    
    # Test 3: Check if final prompts would work with weighted embeddings
    print("🧪 Test 3: Weighted Embedding Compatibility")
    print("-" * 40)
    
    for i, var in enumerate(variations):
        final_prompt = var.weighted_text or var.text
        segments = processor.parse_weighted_prompt(final_prompt)
        
        has_weights = any(seg.weight != 1.0 for seg in segments)
        max_weight = max(seg.weight for seg in segments) if segments else 1.0
        
        print(f"Variation {i+1}: '{final_prompt}'")
        print(f"  Has weights: {has_weights}")
        print(f"  Max weight: {max_weight}")
        print(f"  Segments: {len(segments)}")
        for j, seg in enumerate(segments):
            if seg.weight != 1.0:
                print(f"    - '{seg.text}' (weight: {seg.weight})")
        print()


def test_alternative_syntaxes():
    """Test alternative ways to structure the nested syntax."""
    
    print("🔧 Alternative Syntax Options")
    print("=" * 60)
    
    alternatives = [
        # Option 1: Weight outside the brackets
        ("(a romantic kiss between [two people | two men | two women | a man and woman]:10)", "Weight outside variation"),
        
        # Option 2: Weight inside each option
        ("a romantic kiss between [(two people:10) | (two men:10) | (two women:10) | (a man and woman:10)]", "Weight inside each option"),
        
        # Option 3: Separate the weighting from variation
        ("a romantic kiss between [two people | two men | two women | a man and woman]", "No weight - relies on automatic weighting"),
        
        # Option 4: Mix of both approaches
        ("a (romantic:1.5) kiss between ([two people | two men | two women | a man and woman]:10)", "Mixed weighting"),
    ]
    
    processor = PromptWeightingProcessor()
    
    for template, description in alternatives:
        print(f"📝 {description}")
        print(f"Template: {template}")
        
        # Parse for weights
        segments = processor.parse_weighted_prompt(template)
        print("Weight parsing result:")
        for seg in segments:
            if seg.weight != 1.0:
                print(f"  - '{seg.text}' (weight: {seg.weight})")
        
        # Try template parsing
        weighting_config = WeightingConfig(enable_weighting=True, variation_weight=2.0)
        try:
            prompt_template = PromptTemplate(template, weighting_config)
            variations = prompt_template.generate_variations()
            print(f"Template parsing: ✅ Generates {len(variations)} variations")
            if variations:
                print(f"  First variation: '{variations[0].text}'")
                if variations[0].weighted_text:
                    print(f"  First weighted: '{variations[0].weighted_text}'")
        except Exception as e:
            print(f"Template parsing: ❌ Error: {e}")
        
        print()


if __name__ == "__main__":
    test_nested_syntax()
    test_alternative_syntaxes()
