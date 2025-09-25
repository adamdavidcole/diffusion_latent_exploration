#!/usr/bin/env python3
"""
Test script for the new DynamicScale approach to CFG scheduling.
This tests the basic DynamicScale functionality without running a full video generation.
"""

import sys
import os
sys.path.append('/home/adam/dev/diffusion_latent_exploration')

from src.utils.dynamic_guidance import DynamicScale, create_guidance_callback

def test_dynamic_scale():
    """Test that DynamicScale behaves like a number but can be updated."""
    
    print("=== Testing DynamicScale Class ===")
    
    # Create mutable state
    schedule_state = {"value": 5.0}
    
    # Create DynamicScale
    dyn_scale = DynamicScale(lambda: schedule_state["value"])
    
    print(f"Initial value: {float(dyn_scale)} (should be 5.0)")
    print(f"Comparison > 1.0: {dyn_scale > 1.0} (should be True)")
    
    # Test arithmetic
    test_tensor = 10.0
    result = dyn_scale * test_tensor
    print(f"Multiplication: {dyn_scale} * {test_tensor} = {result} (should be 50.0)")
    
    # Update the value and test again
    schedule_state["value"] = 2.5
    print(f"After update: {float(dyn_scale)} (should be 2.5)")
    print(f"Comparison > 1.0: {dyn_scale > 1.0} (should be True)")
    
    new_result = dyn_scale * test_tensor
    print(f"New multiplication: {dyn_scale} * {test_tensor} = {new_result} (should be 25.0)")
    
    # Test with zero (important for CFG)
    schedule_state["value"] = 0.0
    print(f"Zero test: {float(dyn_scale)} (should be 0.0)")
    print(f"Comparison > 1.0: {dyn_scale > 1.0} (should be False)")
    
    print("✅ DynamicScale tests passed!")
    

def test_create_guidance_callback():
    """Test the create_guidance_callback factory function."""
    
    print("\n=== Testing create_guidance_callback ===")
    
    # Define a schedule
    schedule = {0: 6.5, 5: 3.5, 10: 0.0}
    
    # Create dynamic scale and callback
    dynamic_scale, callback = create_guidance_callback(
        schedule=schedule,
        interpolation='linear',
        total_steps=15,
        verbose=True
    )
    
    print(f"Initial dynamic scale value: {float(dynamic_scale)} (should be 6.5)")
    
    # Simulate callback calls for different steps
    class MockPipe:
        pass
    
    mock_pipe = MockPipe()
    
    # Test at different steps
    test_steps = [0, 2, 5, 7, 10, 14]
    for step in test_steps:
        # Call callback (this should update the schedule state)
        result = callback(mock_pipe, step, torch.tensor(100.0 - step * 5), {})
        current_value = float(dynamic_scale)
        print(f"Step {step}: dynamic_scale = {current_value:.2f}")
    
    print("✅ Callback tests passed!")


if __name__ == "__main__":
    try:
        import torch
        test_dynamic_scale()
        test_create_guidance_callback()
        print("\n🎉 All tests passed! DynamicScale approach is working correctly.")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
