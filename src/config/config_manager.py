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
    video_settings: VideoSettings = field(default_factory=VideoSettings)
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
        video_data = config_data.get('video_settings', {})
        
        model_settings = ModelSettings(
            seed=model_data.get('seed', 42),
            sampler=model_data.get('sampler', 'unipc'),
            cfg_scale=model_data.get('cfg_scale', 7.5),
            steps=model_data.get('steps', 50),
            eta=model_data.get('eta', 0.0),
            clip_skip=model_data.get('clip_skip', 1),
            model_id=model_data.get('model_id', 'Wan-AI/Wan2.1-T2V-14B-Diffusers')
        )
        
        video_settings = VideoSettings(
            width=video_data.get('width', 512),
            height=video_data.get('height', 512),
            fps=video_data.get('fps', 24),
            duration=video_data.get('duration', 4.0),
            frames=video_data.get('frames')
        )
        
        return GenerationConfig(
            model_settings=model_settings,
            video_settings=video_settings,
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
            'video_settings': {
                'width': config.video_settings.width,
                'height': config.video_settings.height,
                'fps': config.video_settings.fps,
                'duration': config.video_settings.duration,
                'frames': config.video_settings.frames
            },
            'videos_per_variation': config.videos_per_variation,
            'output_dir': config.output_dir,
            'batch_name': config.batch_name,
            'use_timestamp': config.use_timestamp
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
