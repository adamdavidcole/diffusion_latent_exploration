#!/usr/bin/env python3
"""
Enhanced attention map analyzer with comprehensive loading and analysis capabilities.
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
import json
import logging
from dataclasses import dataclass


@dataclass
class AttentionMapInfo:
    """Information about an attention map from metadata."""
    video_id: str
    step: int
    timestep: float
    total_steps: int
    token_word: str
    token_ids: List[int]
    token_texts: List[str]
    aggregation_method: str
    attention_shape: Tuple[int, ...]
    spatial_resolution: Tuple[int, int]
    num_blocks: int
    num_heads: int
    video_width: Optional[int] = None
    video_height: Optional[int] = None
    video_frames: Optional[int] = None
    threshold_applied: Optional[float] = None
    dtype: str = "float32"
    prompt: str = ""
    seed: Optional[int] = None
    cfg_scale: Optional[float] = None


class AttentionAnalyzer:
    """Enhanced analyzer for attention maps with metadata support."""
    
    def __init__(self, attention_maps_dir: Union[str, Path]):
        self.attention_dir = Path(attention_maps_dir)
        self.logger = logging.getLogger(__name__)
        
    def _parse_video_id(self, video_id: str) -> Tuple[str, str]:
        """Parse video_id into prompt_part and video_part."""
        if "_vid" in video_id:
            prompt_part, vid_num = video_id.rsplit("_vid", 1)
            vid_part = f"vid{vid_num}"
        else:
            prompt_part = video_id
            vid_part = "vid001"
        return prompt_part, vid_part
    
    def get_token_dir(self, video_id: str, token_word: str) -> Path:
        """Get the directory path for a specific token."""
        prompt_part, vid_part = self._parse_video_id(video_id)
        return self.attention_dir / prompt_part / vid_part / f"token_{token_word}"
    
    def load_metadata(self, video_id: str, token_word: str, step: int) -> AttentionMapInfo:
        """Load metadata for a specific attention map."""
        token_dir = self.get_token_dir(video_id, token_word)
        metadata_file = token_dir / f"step_{step:03d}_metadata.json"
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        return AttentionMapInfo(
            video_id=metadata.get('video_id', video_id),
            step=metadata.get('step', step),
            timestep=metadata.get('timestep', 0.0),
            total_steps=metadata.get('total_steps', 1),
            token_word=metadata.get('token_word', token_word),
            token_ids=metadata.get('token_ids', []),
            token_texts=metadata.get('token_texts', []),
            aggregation_method=metadata.get('aggregation_method', 'unknown'),
            attention_shape=tuple(metadata.get('attention_shape', [])),
            spatial_resolution=tuple(metadata.get('spatial_resolution', [0, 0])),
            num_blocks=metadata.get('num_blocks', 0),
            num_heads=metadata.get('num_heads', 0),
            video_width=metadata.get('video_width'),
            video_height=metadata.get('video_height'),
            video_frames=metadata.get('video_frames'),
            threshold_applied=metadata.get('threshold_applied'),
            dtype=metadata.get('dtype', 'float32'),
            prompt=metadata.get('prompt', ''),
            seed=metadata.get('seed'),
            cfg_scale=metadata.get('cfg_scale')
        )
    
    def load_step(self, video_id: str, token_word: str, step: int) -> Tuple[torch.Tensor, AttentionMapInfo]:
        """
        Load attention map and metadata for a specific step.
        Returns: (attention_tensor, metadata)
        """
        token_dir = self.get_token_dir(video_id, token_word)
        step_file = token_dir / f"step_{step:03d}.npy"
        
        # Check for compressed version if uncompressed doesn't exist
        if not step_file.exists():
            step_file_gz = token_dir / f"step_{step:03d}.npy.gz"
            if step_file_gz.exists():
                import gzip
                with gzip.open(step_file_gz, 'rb') as f:
                    attention = np.load(f)
            else:
                raise FileNotFoundError(f"Attention map not found: {step_file} or {step_file_gz}")
        else:
            attention = np.load(step_file)
        
        metadata = self.load_metadata(video_id, token_word, step)
        
        return torch.from_numpy(attention), metadata
    
    def load_all_steps(self, video_id: str, token_word: str) -> Tuple[torch.Tensor, List[AttentionMapInfo]]:
        """
        Load all steps and stack into temporal tensor.
        Returns: (stacked_tensor, metadata_list)
        Shape: [steps, blocks/1, heads/1, spatial, tokens]
        """
        token_dir = self.get_token_dir(video_id, token_word)
        # Look for both compressed and uncompressed files
        step_files = sorted(token_dir.glob("step_*.npy")) + sorted(token_dir.glob("step_*.npy.gz"))
        # Remove duplicates (prefer .npy over .npy.gz)
        step_files_dict = {}
        for step_file in step_files:
            if step_file.name.endswith('.npy.gz'):
                step_num = int(step_file.name.split('_')[1].split('.')[0])
            else:
                step_num = int(step_file.stem.split('_')[1])
            if step_num not in step_files_dict or not step_file.name.endswith('.gz'):
                step_files_dict[step_num] = step_file
        
        step_files = [step_files_dict[k] for k in sorted(step_files_dict.keys())]
        
        if not step_files:
            raise FileNotFoundError(f"No attention maps found in {token_dir}")
        
        # Load all steps and metadata
        steps = []
        metadata_list = []
        
        for step_file in step_files:
            # Handle both .npy and .npy.gz files
            if step_file.name.endswith('.npy.gz'):
                step_num = int(step_file.name.split('_')[1].split('.')[0])
            else:
                step_num = int(step_file.stem.split('_')[1])
            
            # Load attention data (handle compression)
            if step_file.name.endswith('.gz'):
                import gzip
                with gzip.open(step_file, 'rb') as f:
                    attention = np.load(f)
            else:
                attention = np.load(step_file)
            
            metadata = self.load_metadata(video_id, token_word, step_num)
            
            steps.append(torch.from_numpy(attention))
            metadata_list.append(metadata)
        
        # Stack into [steps, blocks, heads, spatial, tokens]
        full_tensor = torch.stack(steps, dim=0)
        return full_tensor, metadata_list
    
    def get_available_videos(self) -> List[str]:
        """Get list of available video IDs."""
        video_ids = []
        for prompt_dir in self.attention_dir.iterdir():
            if prompt_dir.is_dir() and prompt_dir.name.startswith('prompt_'):
                for vid_dir in prompt_dir.iterdir():
                    if vid_dir.is_dir() and vid_dir.name.startswith('vid'):
                        # Reconstruct video_id: prompt_000_vid001
                        video_id = f"{prompt_dir.name}_{vid_dir.name}"
                        video_ids.append(video_id)
        return sorted(video_ids)
    
    def get_available_tokens(self, video_id: str) -> List[str]:
        """Get list of available tokens for a video."""
        prompt_part, vid_part = self._parse_video_id(video_id)
        video_dir = self.attention_dir / prompt_part / vid_part
        
        if not video_dir.exists():
            return []
        
        tokens = []
        for token_dir in video_dir.iterdir():
            if token_dir.is_dir() and token_dir.name.startswith('token_'):
                token_name = token_dir.name[6:]  # Remove 'token_' prefix
                tokens.append(token_name)
        
        return sorted(tokens)
    
    def get_video_info(self, video_id: str) -> Dict[str, Any]:
        """Get comprehensive info about a video including all available tokens."""
        prompt_part, vid_part = self._parse_video_id(video_id)
        video_dir = self.attention_dir / prompt_part / vid_part
        
        if not video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {video_dir}")
        
        # Load summary if available
        summary_file = video_dir / "summary.json"
        summary = {}
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        
        # Get available tokens and their step counts
        tokens = {}
        for token_dir in video_dir.iterdir():
            if token_dir.is_dir() and token_dir.name.startswith('token_'):
                token_name = token_dir.name[6:]
                step_files = list(token_dir.glob("step_*.npy*"))  # Include .gz files
                tokens[token_name] = len(step_files)
        
        return {
            'video_id': video_id,
            'video_dir': str(video_dir),
            'tokens': tokens,
            'summary': summary
        }
    
    # Aggregation methods
    def aggregate_blocks(self, attention: torch.Tensor) -> torch.Tensor:
        """Average across transformer blocks. Input: [..., blocks, heads, spatial, tokens]"""
        return attention.mean(dim=-4)
    
    def aggregate_heads(self, attention: torch.Tensor) -> torch.Tensor:
        """Average across attention heads. Input: [..., heads, spatial, tokens]"""
        return attention.mean(dim=-3)
    
    def aggregate_steps(self, attention: torch.Tensor) -> torch.Tensor:
        """Average across diffusion steps. Input: [steps, ...]"""
        return attention.mean(dim=0)
    
    def aggregate_tokens(self, attention: torch.Tensor) -> torch.Tensor:
        """Average across tokens. Input: [..., spatial, tokens]"""
        return attention.mean(dim=-1)
    
    # Analysis methods
    def get_temporal_evolution(self, video_id: str, token_word: str, 
                             aggregate_blocks: bool = True, 
                             aggregate_heads: bool = True) -> Tuple[torch.Tensor, List[AttentionMapInfo]]:
        """
        Get attention evolution over time.
        Returns: (tensor, metadata_list)
        Tensor shape: [steps, spatial, tokens] if aggregated, or [steps, blocks, heads, spatial, tokens] if not
        """
        full_tensor, metadata_list = self.load_all_steps(video_id, token_word)
        
        if aggregate_blocks:
            full_tensor = self.aggregate_blocks(full_tensor)
        if aggregate_heads:
            full_tensor = self.aggregate_heads(full_tensor)
        
        return full_tensor, metadata_list
    
    def get_layer_analysis(self, video_id: str, token_word: str, step: int,
                          aggregate_heads: bool = True) -> Tuple[torch.Tensor, AttentionMapInfo]:
        """
        Analyze attention across transformer layers for a specific step.
        Returns: (tensor, metadata)
        Tensor shape: [blocks, spatial, tokens] if heads aggregated, or [blocks, heads, spatial, tokens] if not
        """
        step_tensor, metadata = self.load_step(video_id, token_word, step)
        
        if aggregate_heads:
            step_tensor = self.aggregate_heads(step_tensor)
        
        return step_tensor, metadata
    
    def get_head_analysis(self, video_id: str, token_word: str, step: int,
                         aggregate_blocks: bool = True) -> Tuple[torch.Tensor, AttentionMapInfo]:
        """
        Analyze individual attention heads for a specific step.
        Returns: (tensor, metadata)
        Tensor shape: [heads, spatial, tokens] if blocks aggregated, or [blocks, heads, spatial, tokens] if not
        """
        step_tensor, metadata = self.load_step(video_id, token_word, step)
        
        if aggregate_blocks:
            step_tensor = self.aggregate_blocks(step_tensor)
        
        return step_tensor, metadata
    
    def get_spatial_attention_map(self, video_id: str, token_word: str, step: int,
                                 aggregate_all: bool = True) -> Tuple[torch.Tensor, AttentionMapInfo]:
        """
        Get spatial attention map with automatic reshaping based on metadata.
        Returns: (tensor, metadata)
        Tensor shape depends on video dimensions from metadata and aggregation settings
        """
        step_tensor, metadata = self.load_step(video_id, token_word, step)
        
        if aggregate_all:
            # Average across blocks and heads: [spatial, tokens] -> [spatial]
            spatial_attention = step_tensor.mean(dim=[0, 1]).squeeze(-1)
        else:
            # Keep all dimensions: [blocks, heads, spatial, tokens] -> [blocks, heads, spatial]
            spatial_attention = step_tensor.squeeze(-1)
        
        # Use metadata to reshape if video dimensions are available
        if metadata.video_frames and metadata.video_height and metadata.video_width:
            # Calculate latent dimensions based on VAE downsampling
            # From your visualization code:
            # - Frames: (latent_num_frames - 1) * 4 + 1 = orig_frames
            #   So: latent_frames = (orig_frames - 1) // 4 + 1
            # - Width/Height: latent_dim * 16 = orig_dim  
            #   So: latent_dim = orig_dim // 16
            latent_frames = (metadata.video_frames - 1) // 4 + 1
            latent_height = metadata.video_height // 16
            latent_width = metadata.video_width // 16
            
            expected_spatial_size = latent_frames * latent_height * latent_width
            actual_spatial_size = spatial_attention.numel()
            
            self.logger.info(f"Reshaping attention map:")
            self.logger.info(f"  Original video: {metadata.video_frames}×{metadata.video_height}×{metadata.video_width}")
            self.logger.info(f"  Calculated latent: {latent_frames}×{latent_height}×{latent_width}")
            self.logger.info(f"  Expected elements: {expected_spatial_size}")
            self.logger.info(f"  Actual elements: {actual_spatial_size}")
            
            if aggregate_all and actual_spatial_size == expected_spatial_size:
                # Reshape to [frames, height, width]
                try:
                    spatial_attention = spatial_attention.view(latent_frames, latent_height, latent_width)
                    self.logger.info(f"✅ Successfully reshaped to latent dimensions: {spatial_attention.shape}")
                except RuntimeError as e:
                    self.logger.warning(f"Could not reshape spatial attention to latent dimensions: {e}")
            elif not aggregate_all and actual_spatial_size == expected_spatial_size:
                # Reshape preserving block/head dimensions
                batch_size = spatial_attention.shape[0] * spatial_attention.shape[1]
                try:
                    spatial_attention = spatial_attention.view(batch_size, latent_frames, latent_height, latent_width)
                    self.logger.info(f"✅ Successfully reshaped with batch size {batch_size}")
                except RuntimeError as e:
                    self.logger.warning(f"Could not reshape spatial attention with batch size {batch_size}: {e}")
            else:
                self.logger.warning(f"❌ Spatial size mismatch: got {actual_spatial_size}, expected {expected_spatial_size}")
                self.logger.warning(f"Cannot reshape - keeping original shape: {spatial_attention.shape}")
                self.logger.warning(f"Downsampling factors - VAE: {vae_downsample}, Frames: {frame_downsample}, Attention: {attention_downsample}")
        
        return spatial_attention, metadata
    
    def compute_average_attention(self, video_id: str, token_word: str,
                                aggregate_blocks: bool = True,
                                aggregate_heads: bool = True,
                                aggregate_steps: bool = True) -> Tuple[torch.Tensor, List[AttentionMapInfo]]:
        """
        Compute average attention across specified dimensions.
        Returns: (averaged_tensor, metadata_list)
        """
        temporal_tensor, metadata_list = self.get_temporal_evolution(
            video_id, token_word, aggregate_blocks, aggregate_heads
        )
        
        if aggregate_steps:
            averaged_tensor = self.aggregate_steps(temporal_tensor)
            return averaged_tensor, metadata_list
        else:
            return temporal_tensor, metadata_list


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        attention_dir = sys.argv[1]
    else:
        # Use latest test output if available
        latest_dirs = sorted(Path("outputs").glob("*attention*"), key=lambda x: x.stat().st_mtime, reverse=True)
        if latest_dirs:
            attention_dir = latest_dirs[0] / "attention_maps"
        else:
            print("No attention maps directory found. Please specify one as an argument.")
            sys.exit(1)
    
    print(f"Analyzing attention maps in: {attention_dir}")
    
    try:
        analyzer = AttentionAnalyzer(attention_dir)
        
        # Get available videos
        videos = analyzer.get_available_videos()
        print(f"Available videos: {videos}")
        
        if videos:
            video_id = videos[0]
            print(f"\nAnalyzing video: {video_id}")
            
            # Get video info
            video_info = analyzer.get_video_info(video_id)
            print(f"Video info: {video_info}")
            
            # Get available tokens
            tokens = analyzer.get_available_tokens(video_id)
            print(f"Available tokens: {tokens}")
            
            if tokens:
                token = tokens[0]
                print(f"\nAnalyzing token: {token}")
                
                # Load single step with metadata
                step_tensor, metadata = analyzer.load_step(video_id, token, 0)
                print(f"Step 0 shape: {step_tensor.shape}")
                print(f"Metadata: {metadata}")
                
                # Load all steps
                all_steps, all_metadata = analyzer.load_all_steps(video_id, token)
                print(f"All steps shape: {all_steps.shape}")
                
                # Get temporal evolution
                temporal, _ = analyzer.get_temporal_evolution(video_id, token)
                print(f"Temporal evolution shape: {temporal.shape}")
                
                # Get spatial map with automatic reshaping
                spatial, spatial_metadata = analyzer.get_spatial_attention_map(video_id, token, 0)
                print(f"Spatial map shape: {spatial.shape}")
                
                print("\n✅ Enhanced AttentionAnalyzer working correctly!")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
