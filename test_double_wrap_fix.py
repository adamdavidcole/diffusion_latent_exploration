#!/usr/bin/env python3
"""
Test script to verify double wrapping prevention fix
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.attention_storage import AttentionStorage
import torch.nn as nn

class MockAttn2Module(nn.Module):
    """Mock attention module for testing"""
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(64, 64)
        self.to_k = nn.Linear(64, 64)
        self.to_v = nn.Linear(64, 64)

class MockTransformer(nn.Module):
    """Mock transformer model for testing"""
    def __init__(self):
        super().__init__()
        self.transformer_block_0 = nn.Module()
        self.transformer_block_0.attn2 = MockAttn2Module()
        self.transformer_block_1 = nn.Module() 
        self.transformer_block_1.attn2 = MockAttn2Module()

def test_double_wrap_prevention():
    """Test that double wrapping is prevented"""
    print("Testing double wrapping prevention...")
    
    model = MockTransformer()
    
    # First registration - should wrap modules
    attention_storage_1 = AttentionStorage()
    success_1 = attention_storage_1.register_attention_hooks(model)
    print(f"First registration: {success_1}")
    
    # Check that modules are wrapped
    attn2_0 = model.transformer_block_0.attn2
    attn2_1 = model.transformer_block_1.attn2
    print(f"Block 0 attn2 type after first wrap: {type(attn2_0).__name__}")
    print(f"Block 1 attn2 type after first wrap: {type(attn2_1).__name__}")
    
    # Second registration - should reuse existing wrappers
    attention_storage_2 = AttentionStorage()
    success_2 = attention_storage_2.register_attention_hooks(model)
    print(f"Second registration: {success_2}")
    
    # Check that modules are still WanAttentionWrapper (not double wrapped)
    attn2_0_after = model.transformer_block_0.attn2
    attn2_1_after = model.transformer_block_1.attn2
    print(f"Block 0 attn2 type after second wrap: {type(attn2_0_after).__name__}")
    print(f"Block 1 attn2 type after second wrap: {type(attn2_1_after).__name__}")
    
    # Verify they are the same instances (not double wrapped)
    assert attn2_0 is attn2_0_after, "Module should be reused, not double wrapped"
    assert attn2_1 is attn2_1_after, "Module should be reused, not double wrapped"
    
    # Test that we can access original attributes through wrapper
    try:
        # Should work - accessing through wrapper to original module
        assert hasattr(attn2_0.original_module, 'to_q'), "Should be able to access to_q through original_module"
        print("✅ Can access to_q through original_module")
    except AttributeError as e:
        print(f"❌ Error accessing to_q: {e}")
        
    # Test restoration
    attention_storage_2.remove_attention_hooks(model)
    
    # Check that modules are restored
    restored_attn2_0 = model.transformer_block_0.attn2
    restored_attn2_1 = model.transformer_block_1.attn2
    print(f"Block 0 attn2 type after restoration: {type(restored_attn2_0).__name__}")
    print(f"Block 1 attn2 type after restoration: {type(restored_attn2_1).__name__}")
    
    # Verify restoration worked
    assert type(restored_attn2_0).__name__ == "MockAttn2Module", "Module should be restored to original"
    assert type(restored_attn2_1).__name__ == "MockAttn2Module", "Module should be restored to original"
    
    print("✅ All tests passed! Double wrapping prevention is working correctly.")

if __name__ == "__main__":
    test_double_wrap_prevention()
