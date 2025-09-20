"""
Video Similarity Analysis Module

This module provides tools for analyzing visual similarity between video prompts
in diffusion model experiments, specifically designed for bias detection analysis.
"""

from .video_similarity_analyzer import VideoSimilarityAnalyzer
from .similarity_metrics import SimilarityMetrics
from .frame_extractor import FrameExtractor

__all__ = [
    'VideoSimilarityAnalyzer',
    'SimilarityMetrics', 
    'FrameExtractor'
]