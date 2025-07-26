#!/usr/bin/env python3
"""
Debug script to investigate the tokenizer issue with weighted prompts.
"""

import sys
from pathlib import Path
import logging
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.generators.video_generator import WanVideoGenerator

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_tokenizer():
    """Debug the tokenizer issue."""
    
    print("🔍 Debugging WAN Tokenizer for Weighted Prompts")
    print("=" * 50)
    
    # Load config
    config_manager = ConfigManager()
    config = config_manager.load_config("configs/weighted_prompts_example.yaml")
    
    # Create generator (but don't load model yet)
    generator = WanVideoGenerator(config)
    
    print("Loading WAN model components...")
    generator._load_model()
    
    # Check tokenizer properties
    tokenizer = generator.pipe.tokenizer
    text_encoder = generator.pipe.text_encoder
    
    print(f"\nTokenizer Info:")
    print(f"  Type: {type(tokenizer)}")
    print(f"  Model max length: {tokenizer.model_max_length}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    
    # Test simple tokenization
    test_prompts = [
        "a person dancing in the park",
        "a person (dancing:1.8) in the park"
    ]
    
    for prompt in test_prompts:
        print(f"\nTesting prompt: {prompt}")
        
        try:
            # Try basic tokenization
            tokens = tokenizer.encode(prompt)
            print(f"  Tokens length: {len(tokens)}")
            print(f"  Max token value: {max(tokens) if tokens else 'N/A'}")
            
            # Try tokenization with padding
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=min(77, tokenizer.model_max_length),  # Use safe max length
                truncation=True,
                return_tensors="pt"
            )
            print(f"  Input IDs shape: {text_inputs.input_ids.shape}")
            print(f"  Max input ID value: {text_inputs.input_ids.max().item()}")
            
            # Try text encoding
            device = generator.device
            embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
            print(f"  Embeddings shape: {embeddings.shape}")
            print(f"  Embeddings dtype: {embeddings.dtype}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            print(f"  Error type: {type(e)}")

if __name__ == "__main__":
    debug_tokenizer()
