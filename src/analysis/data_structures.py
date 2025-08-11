from pathlib import Path
import torch

from typing import Dict, List, Any, Optional, Tuple, TypedDict
from dataclasses import dataclass

from dataclasses import asdict

@dataclass
class LatentTrajectoryAnalysis:
    """Data structure for GPU-optimized analysis results."""
    spatial_patterns: Dict[str, Any]
    temporal_coherence: Dict[str, Any]
    channel_analysis: Dict[str, Any]
    patch_diversity: Dict[str, Any]
    global_structure: Dict[str, Any]
    information_content: Dict[str, Any]
    complexity_measures: Dict[str, Any]
    frequency_patterns: Dict[str, Any]
    group_separability: Dict[str, Any]
    temporal_analysis: Dict[str, Any]
    structural_analysis: Dict[str, Any]
    statistical_significance: Dict[str, Any]
    # New advanced geometric analysis components
    convex_hull_analysis: Dict[str, Any]
    functional_pca_analysis: Dict[str, Any]
    individual_trajectory_geometry: Dict[str, Any]
    intrinsic_dimension_analysis: Dict[str, Any]
    gpu_performance_stats: Dict[str, Any]
    analysis_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LatentTrajectoryAnalysis':
        """Create an instance of LatentTrajectoryAnalysis from a dictionary."""
        return cls(**data)


# TypedDict approach for backwards compatibility
class GroupTensor(TypedDict):
    """Type definition for group tensor dictionary structure - maintains dict behavior."""
    trajectory_tensor: torch.Tensor  # [n_videos, steps, 1, 16, frames, H, W]
    trajectory_metadata: List[Dict[str, Any]]
    n_videos: int
    n_steps: int
    latent_shape: Tuple[int, ...]  # [16, frames, H, W]
    full_shape: Tuple[int, ...]    # [n_videos, steps, 1, 16, frames, H, W]


# Type alias for the complete group tensors structure
GroupTensors = Dict[str, GroupTensor]

#TODO: implements group tensors as a dataclass
# @dataclass
# class GroupTensor:
#     """Data structure for batched trajectory tensors and metadata for a single group."""
#     trajectory_tensor: torch.Tensor  # [n_videos, steps, 1, 16, frames, H, W]
#     trajectory_metadata: List[Dict[str, Any]]
#     n_videos: int
#     n_steps: int
#     latent_shape: Tuple[int, ...]  # [16, frames, H, W]
#     full_shape: Tuple[int, ...]    # [n_videos, steps, 1, 16, frames, H, W]
    
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> 'GroupTensor':
#         """Create GroupTensor instance from dictionary (e.g., from load_and_batch_trajectory_data)."""
#         return cls(
#             trajectory_tensor=data['trajectory_tensor'],
#             trajectory_metadata=data['trajectory_metadata'],
#             n_videos=data['n_videos'],
#             n_steps=data['n_steps'],
#             latent_shape=data['latent_shape'],
#             full_shape=data['full_shape']
#         )


# @dataclass
# class GroupTensors:
#     """Collection of GroupTensor instances for all prompt groups."""
#     groups: Dict[str, GroupTensor]
    
#     def __getitem__(self, group_name: str) -> GroupTensor:
#         """Allow dict-like access to groups."""
#         return self.groups[group_name]
    
#     def __iter__(self):
#         """Allow iteration over group names."""
#         return iter(self.groups)
    
#     def items(self):
#         """Allow dict-like items() access."""
#         return self.groups.items()
    
#     def keys(self):
#         """Allow dict-like keys() access."""
#         return self.groups.keys()
    
#     def values(self):
#         """Allow dict-like values() access."""
#         return self.groups.values()
    
#     @classmethod
#     def from_dict(cls, data: Dict[str, Dict[str, Any]]) -> 'GroupTensors':
#         """Create GroupTensors from dictionary of group data."""
#         groups = {name: GroupTensor.from_dict(group_data) for name, group_data in data.items()}
#         return cls(groups=groups)

class NormCfg(TypedDict):
    """Type definition for normalization configuration - maintains dict behavior."""
    per_step_whiten: bool
    per_channel_standardize: bool
    snr_normalize: bool


# TODO: implement NormConfig as a dataclass
# @dataclass
# class NormConfig:
#     """Configuration for normalization strategies."""
#     per_step_whiten: bool = False
#     per_channel_standardize: bool = False
#     snr_normalize: bool = False