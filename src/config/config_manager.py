"""
Configuration management for WAN 1.3B video generation.
Handles loading, validation, and management of generation settings.
"""
import yaml
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from pathlib import Path


@dataclass
class ModelSettings:
    """Model-specific configuration settings."""
    seed: int = 42
    sampler: str = "unipc"     # unipc recommended for WAN
    cfg_scale: float = 7.5
    steps: int = 50
    eta: float = 0.0
    clip_skip: int = 1
    model_id: str = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    device: str = "auto"  # Device to use: "auto", "cuda:0", "cuda:1", "cpu", etc.
    

@dataclass
class MemorySettings:
    """Memory optimization settings for large models."""
    enable_memory_optimization: bool = True
    clear_cache_between_videos: bool = True
    reload_model_for_large_models: bool = True
    use_gradient_checkpointing: bool = True
    enable_memory_efficient_attention: bool = True
    

@dataclass
class PromptSettings:
    """Prompt weighting and processing settings."""
    enable_weighting: bool = False
    variation_weight: float = 1.5
    base_weight: float = 1.0
    enable_prompt_weighting: bool = True  # Allow disabling globally
    
    # Advanced weighted embeddings (experimental)
    use_weighted_embeddings: bool = False
    embedding_method: str = "norm_preserving"  # "multiply", "interpolation", or "norm_preserving"


@dataclass
class VideoSettings:
    """Video generation parameters."""
    width: int = 512
    height: int = 512
    fps: int = 24
    duration: float = 4.0  # seconds
    frames: Optional[int] = None  # calculated from fps * duration if None
    

@dataclass
class GenerationConfig:
    """Complete configuration for video generation batch."""
    model_settings: ModelSettings = field(default_factory=ModelSettings)
    memory_settings: MemorySettings = field(default_factory=MemorySettings)
    video_settings: VideoSettings = field(default_factory=VideoSettings)
    prompt_settings: PromptSettings = field(default_factory=PromptSettings)
    videos_per_variation: int = 3
    output_dir: str = "outputs"
    batch_name: Optional[str] = None
    use_timestamp: bool = True
    
    def __post_init__(self):
        """Calculate frames if not specified."""
        if self.video_settings.frames is None:
            self.video_settings.frames = int(self.video_settings.fps * self.video_settings.duration)


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self):
        self.config_dir = Path("configs")
        self.config_dir.mkdir(exist_ok=True)
    
    def load_config(self, config_path: Union[str, Path]) -> GenerationConfig:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return self._parse_config(config_data)
    
    def save_config(self, config: GenerationConfig, config_path: Union[str, Path]):
        """Save configuration to YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self._config_to_dict(config)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def create_default_config(self, save_path: Optional[Union[str, Path]] = None) -> GenerationConfig:
        """Create and optionally save a default configuration."""
        config = GenerationConfig()
        
        if save_path:
            self.save_config(config, save_path)
        
        return config
    
    def _parse_config(self, config_data: Dict) -> GenerationConfig:
        """Parse dictionary data into GenerationConfig object."""
        model_data = config_data.get('model_settings', {})
        memory_data = config_data.get('memory_settings', {})
        video_data = config_data.get('video_settings', {})
        prompt_data = config_data.get('prompt_settings', {})
        
        model_settings = ModelSettings(
            seed=model_data.get('seed', 42),
            sampler=model_data.get('sampler', 'unipc'),
            cfg_scale=model_data.get('cfg_scale', 7.5),
            steps=model_data.get('steps', 50),
            eta=model_data.get('eta', 0.0),
            clip_skip=model_data.get('clip_skip', 1),
            model_id=model_data.get('model_id', 'Wan-AI/Wan2.1-T2V-14B-Diffusers'),
            device=model_data.get('device', 'auto')
        )
        
        memory_settings = MemorySettings(
            enable_memory_optimization=memory_data.get('enable_memory_optimization', True),
            clear_cache_between_videos=memory_data.get('clear_cache_between_videos', True),
            reload_model_for_large_models=memory_data.get('reload_model_for_large_models', True),
            use_gradient_checkpointing=memory_data.get('use_gradient_checkpointing', True),
            enable_memory_efficient_attention=memory_data.get('enable_memory_efficient_attention', True)
        )
        
        video_settings = VideoSettings(
            width=video_data.get('width', 512),
            height=video_data.get('height', 512),
            fps=video_data.get('fps', 24),
            duration=video_data.get('duration', 4.0),
            frames=video_data.get('frames')
        )
        
        prompt_settings = PromptSettings(
            enable_weighting=prompt_data.get('enable_weighting', False),
            variation_weight=prompt_data.get('variation_weight', 1.5),
            base_weight=prompt_data.get('base_weight', 1.0),
            enable_prompt_weighting=prompt_data.get('enable_prompt_weighting', True),
            use_weighted_embeddings=prompt_data.get('use_weighted_embeddings', False),
            embedding_method=prompt_data.get('embedding_method', 'norm_preserving')
        )
        
        return GenerationConfig(
            model_settings=model_settings,
            memory_settings=memory_settings,
            video_settings=video_settings,
            prompt_settings=prompt_settings,
            videos_per_variation=config_data.get('videos_per_variation', 3),
            output_dir=config_data.get('output_dir', 'outputs'),
            batch_name=config_data.get('batch_name'),
            use_timestamp=config_data.get('use_timestamp', True)
        )
    
    def _config_to_dict(self, config: GenerationConfig) -> Dict:
        """Convert GenerationConfig to dictionary for YAML serialization."""
        return {
            'model_settings': {
                'seed': config.model_settings.seed,
                'sampler': config.model_settings.sampler,
                'cfg_scale': config.model_settings.cfg_scale,
                'steps': config.model_settings.steps,
                'eta': config.model_settings.eta,
                'clip_skip': config.model_settings.clip_skip,
                'model_id': config.model_settings.model_id
            },
            'memory_settings': {
                'enable_memory_optimization': config.memory_settings.enable_memory_optimization,
                'clear_cache_between_videos': config.memory_settings.clear_cache_between_videos,
                'reload_model_for_large_models': config.memory_settings.reload_model_for_large_models,
                'use_gradient_checkpointing': config.memory_settings.use_gradient_checkpointing,
                'enable_memory_efficient_attention': config.memory_settings.enable_memory_efficient_attention
            },
            'video_settings': {
                'width': config.video_settings.width,
                'height': config.video_settings.height,
                'fps': config.video_settings.fps,
                'duration': config.video_settings.duration,
                'frames': config.video_settings.frames
            },
            'prompt_settings': {
                'enable_weighting': config.prompt_settings.enable_weighting,
                'variation_weight': config.prompt_settings.variation_weight,
                'base_weight': config.prompt_settings.base_weight,
                'enable_prompt_weighting': config.prompt_settings.enable_prompt_weighting,
                'use_weighted_embeddings': config.prompt_settings.use_weighted_embeddings,
                'embedding_method': config.prompt_settings.embedding_method
            },
            'videos_per_variation': config.videos_per_variation,
            'output_dir': config.output_dir,
            'batch_name': config.batch_name,
            'use_timestamp': config.use_timestamp,
            'note': config.note
        }
    
    def validate_config(self, config: GenerationConfig) -> bool:
        """Validate configuration parameters."""
        errors = []
        
        # Validate model settings
        if config.model_settings.seed < 0:
            errors.append("Seed must be non-negative")
        
        if config.model_settings.cfg_scale < 0:
            errors.append("CFG scale must be non-negative")
        
        if config.model_settings.steps < 1:
            errors.append("Steps must be at least 1")
        
        # Validate video settings
        if config.video_settings.width < 64 or config.video_settings.width % 8 != 0:
            errors.append("Width must be at least 64 and divisible by 8")
        
        if config.video_settings.height < 64 or config.video_settings.height % 8 != 0:
            errors.append("Height must be at least 64 and divisible by 8")
        
        if config.video_settings.fps < 1:
            errors.append("FPS must be at least 1")
        
        if config.video_settings.duration <= 0:
            errors.append("Duration must be positive")
        
        if config.videos_per_variation < 1:
            errors.append("Videos per variation must be at least 1")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))
        
        return True
