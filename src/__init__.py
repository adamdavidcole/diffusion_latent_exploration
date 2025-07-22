"""
WAN 1.3B Video Generation Project
A comprehensive tool for batch video generation with prompt variations.
"""

__version__ = "1.0.0"
__author__ = "Video Generation Team"
__description__ = "Batch video generation using WAN 1.3B with prompt variations"

from src.config import ConfigManager, GenerationConfig
from src.orchestrator import VideoGenerationOrchestrator
from src.prompts import PromptManager, PromptTemplate
from src.generators import WAN13BVideoGenerator, BatchVideoGenerator

__all__ = [
    'ConfigManager',
    'GenerationConfig', 
    'VideoGenerationOrchestrator',
    'PromptManager',
    'PromptTemplate',
    'WAN13BVideoGenerator',
    'BatchVideoGenerator'
]
