from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from attrs import asdict

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