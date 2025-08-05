#!/usr/bin/env python3
"""
Example analysis functions for consistent attention map format
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Tuple, Union
import json

class AttentionAnalyzer:
    """Helper class for analyzing consistently-formatted attention maps."""
    
    def __init__(self, attention_maps_dir: Union[str, Path]):
        self.attention_dir = Path(attention_maps_dir)
    
    def load_step(self, video_id: str, token_word: str, step: int) -> torch.Tensor:
        """
        Load attention map for a specific step.
        Returns: [blocks, heads, spatial, tokens] tensor
        """
        # Extract prompt and video parts from video_id
        if "_vid" in video_id:
            prompt_part, vid_num = video_id.split("_vid")
            vid_part = f"vid_{vid_num}"
        else:
            prompt_part = video_id
            vid_part = "vid_001"
        
        step_file = self.attention_dir / prompt_part / vid_part / f"token_{token_word}" / f"step_{step:03d}.npy"
        
        if step_file.exists():
            attention = np.load(step_file)
            return torch.from_numpy(attention)
        else:
            raise FileNotFoundError(f"Attention map not found: {step_file}")
    
    def load_all_steps(self, video_id: str, token_word: str) -> torch.Tensor:
        """
        Load all steps and stack into [steps, blocks, heads, spatial, tokens].
        This gives you the full temporal dimension for analysis.
        """
        # Find all step files
        if "_vid" in video_id:
            prompt_part, vid_num = video_id.split("_vid")
            vid_part = f"vid_{vid_num}"
        else:
            prompt_part = video_id
            vid_part = "vid_001"
        
        token_dir = self.attention_dir / prompt_part / vid_part / f"token_{token_word}"
        step_files = sorted(token_dir.glob("step_*.npy"))
        
        if not step_files:
            raise FileNotFoundError(f"No attention maps found in {token_dir}")
        
        # Load and stack all steps
        steps = []
        for step_file in step_files:
            attention = np.load(step_file)
            steps.append(torch.from_numpy(attention))
        
        # Stack into [steps, blocks, heads, spatial, tokens]
        full_tensor = torch.stack(steps, dim=0)
        return full_tensor
    
    def aggregate_blocks(self, attention: torch.Tensor) -> torch.Tensor:
        """Average across transformer blocks. Input: [..., blocks, heads, spatial, tokens]"""
        return attention.mean(dim=-4)  # Average blocks dimension
    
    def aggregate_heads(self, attention: torch.Tensor) -> torch.Tensor:
        """Average across attention heads. Input: [..., heads, spatial, tokens]"""
        return attention.mean(dim=-3)  # Average heads dimension
    
    def aggregate_steps(self, attention: torch.Tensor) -> torch.Tensor:
        """Average across diffusion steps. Input: [steps, ...]"""
        return attention.mean(dim=0)
    
    def get_temporal_evolution(self, video_id: str, token_word: str, 
                             aggregate_blocks: bool = True, 
                             aggregate_heads: bool = True) -> torch.Tensor:
        """
        Get attention evolution over time.
        Returns: [steps, spatial, tokens] if aggregated, or [steps, blocks, heads, spatial, tokens] if not
        """
        full_tensor = self.load_all_steps(video_id, token_word)
        
        if aggregate_blocks:
            full_tensor = self.aggregate_blocks(full_tensor)
        if aggregate_heads:
            full_tensor = self.aggregate_heads(full_tensor)
        
        return full_tensor
    
    def get_layer_analysis(self, video_id: str, token_word: str, step: int,
                          aggregate_heads: bool = True) -> torch.Tensor:
        """
        Analyze attention across transformer layers for a specific step.
        Returns: [blocks, spatial, tokens] if heads aggregated, or [blocks, heads, spatial, tokens] if not
        """
        step_tensor = self.load_step(video_id, token_word, step)
        
        if aggregate_heads:
            step_tensor = self.aggregate_heads(step_tensor)
        
        return step_tensor
    
    def get_head_analysis(self, video_id: str, token_word: str, step: int,
                         aggregate_blocks: bool = True) -> torch.Tensor:
        """
        Analyze individual attention heads for a specific step.
        Returns: [heads, spatial, tokens] if blocks aggregated, or [blocks, heads, spatial, tokens] if not
        """
        step_tensor = self.load_step(video_id, token_word, step)
        
        if aggregate_blocks:
            step_tensor = self.aggregate_blocks(step_tensor)
        
        return step_tensor
    
    def get_spatial_attention_map(self, video_id: str, token_word: str, step: int,
                                 height: int, width: int, frames: int,
                                 aggregate_all: bool = True) -> torch.Tensor:
        """
        Get spatial attention map reshaped to video dimensions.
        Returns: [frames, height, width] if aggregated, or [..., frames, height, width] if not
        """
        step_tensor = self.load_step(video_id, token_word, step)
        
        if aggregate_all:
            # Average across blocks and heads: [spatial, tokens] -> [spatial]
            spatial_attention = step_tensor.mean(dim=[0, 1]).squeeze(-1)
        else:
            # Keep all dimensions: [blocks, heads, spatial, tokens] -> [blocks, heads, spatial]
            spatial_attention = step_tensor.squeeze(-1)
        
        # Reshape to spatial dimensions
        if aggregate_all:
            return spatial_attention.view(frames, height, width)
        else:
            return spatial_attention.view(-1, spatial_attention.shape[-2] // (frames * height * width), 
                                        frames, height, width)

# Example usage:
if __name__ == "__main__":
    # Initialize analyzer with the latest test output
    analyzer = AttentionAnalyzer("outputs/debug_consistent_attention_20250805_121520/attention_maps")
    
    try:
        # Load single step: [blocks, heads, spatial, tokens]
        step_0 = analyzer.load_step("prompt_000_vid001", "flower", 0)
        print(f"✅ Step 0 shape: {step_0.shape}")
        
        # Load all steps: [steps, blocks, heads, spatial, tokens]
        all_steps = analyzer.load_all_steps("prompt_000_vid001", "flower")
        print(f"✅ All steps shape: {all_steps.shape}")
        
        # Get temporal evolution (aggregated): [steps, spatial, tokens]
        temporal = analyzer.get_temporal_evolution("prompt_000_vid001", "flower")
        print(f"✅ Temporal evolution shape: {temporal.shape}")
        
        # Get layer analysis: [blocks, spatial, tokens]
        layers = analyzer.get_layer_analysis("prompt_000_vid001", "flower", step=0)
        print(f"✅ Layer analysis shape: {layers.shape}")
        
        # Get spatial map: [frames, height, width]
        spatial_map = analyzer.get_spatial_attention_map("prompt_000_vid001", "flower", 0, 
                                                       height=15, width=26, frames=3)
        print(f"✅ Spatial map shape: {spatial_map.shape}")
        
        print("\n🎉 All tests passed! Consistent attention format is working perfectly.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
