#!/usr/bin/env python3
"""
Test script to verify attention maps are captured during WAN generation.
"""
import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

import torch
from src.generators import WAN13BVideoGenerator  
from src.config import GenerationConfig
from src.utils.attention_storage import AttentionStorage
import tempfile
import shutil

def test_attention_capture():
    """Test that attention maps are captured during generation."""
    
    # Use a temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            print("Creating generation config...")
            # Create a minimal config 
            config = GenerationConfig(
                output_dir=temp_dir
            )
            # Set device and model
            config.model_settings.device = "cuda:1"
            config.model_settings.model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
            
            print("Loading WAN generator...")
            generator = WAN13BVideoGenerator(config)
            
            print(f"Generator model type: {type(generator.model)}")
            print(f"WAN_AVAILABLE: {True}")  # Should be True based on imports
            
            # Ensure model is loaded
            if generator.model.pipe is None:
                print("Model pipe is None, loading model...")
                generator.model._load_model()
            else:
                print("Model pipe already loaded")
                
            print(f"Model loaded successfully: {type(generator.model.pipe)}")
            
            # Create attention storage
            attention_storage = AttentionStorage(
                storage_dir=temp_dir,
                tokenizer_name="google/umt5-xxl"  # Use proper tokenizer name
            )
            
            # Set up attention storage for a simple prompt
            prompt = "A beautiful (flower) blooming in spring"
            print(f"Testing with prompt: {prompt}")
            
            attention_storage.start_video_storage("test_flower", prompt)
            
            # Register attention wrappers
            print("Registering attention wrappers...")
            success = attention_storage.register_attention_hooks(generator.model.pipe.transformer)
            
            if not success:
                print("❌ Failed to register attention wrappers")
                return False
                
            print(f"✅ Wrapped {len(attention_storage.original_modules)} modules")
            
            # Generate a very short video (1 frame, 2 inference steps)
            print("Starting generation...")
            
            try:
                video_path = generator.generate(
                    prompt=prompt,
                    num_frames=1,  # Minimal
                    num_inference_steps=2,  # Minimal 
                    guidance_scale=7.5,
                    height=320,  # Small resolution for speed
                    width=576,
                    fps=8,
                    output_path=os.path.join(temp_dir, "test_flower.mp4"),
                    attention_storage=attention_storage  # Pass attention storage
                )
                
                print(f"✅ Generation completed: {video_path}")
                
                # Check if attention maps were captured
                print(f"Attention maps captured: {len(attention_storage.current_attention_maps)}")
                
                if attention_storage.current_attention_maps:
                    print("✅ SUCCESS: Attention maps were captured!")
                    for key, attention_map in attention_storage.current_attention_maps.items():
                        print(f"  Block {key}: {attention_map.shape}")
                    return True
                else:
                    print("❌ FAILED: No attention maps captured")
                    return False
                    
            except Exception as e:
                print(f"❌ Generation failed: {e}")
                import traceback
                traceback.print_exc()
                return False
                
            finally:
                # Clean up
                print("Cleaning up...")
                attention_storage.remove_attention_hooks(generator.model.pipe.transformer)
                
        except Exception as e:
            print(f"❌ Test setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    print("Testing attention capture during WAN generation...")
    success = test_attention_capture()
    
    if success:
        print("\n🎉 Test PASSED: Attention maps are being captured!")
        sys.exit(0)
    else:
        print("\n💥 Test FAILED: Attention maps are not being captured")
        sys.exit(1)
