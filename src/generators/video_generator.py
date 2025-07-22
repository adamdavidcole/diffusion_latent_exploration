"""
Video generation engine using WAN 1.3B model.
Handles the actual video generation process with proper error handling and progress tracking.
"""
import os
import time
import logging
from typing import List, Optional, Dict, Any, Callable
from pathlib import Path
from dataclasses import dataclass
import json

# WAN model imports
try:
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video
    import numpy as np
    import cv2
    WAN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"WAN model dependencies not available: {e}. Using mock implementation.")
    WAN_AVAILABLE = False


@dataclass
class GenerationResult:
    """Result of a single video generation."""
    success: bool
    video_path: Optional[str] = None
    error_message: Optional[str] = None
    generation_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class WanVideoGenerator:
    """Real WAN model implementation for video generation."""
    
    def __init__(self, config):
        self.config = config
        self.model_id = config.model_settings.model_id  # Use model_id from config
        self.pipe = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    def _load_model(self):
        """Load the WAN model components."""
        if not WAN_AVAILABLE:
            raise RuntimeError("WAN model dependencies not available")
            
        logging.info(f"Loading WAN model: {self.model_id}")
        logging.info(f"Using device: {self.device}")
        
        try:
            # Load VAE
            vae = AutoencoderKLWan.from_pretrained(
                self.model_id, 
                subfolder="vae", 
                torch_dtype=torch.float32
            )
            
            # Configure scheduler based on resolution
            # 5.0 for 720P, 3.0 for 480P
            flow_shift = 5.0 if max(self.config.video_settings.width, 
                                   self.config.video_settings.height) >= 720 else 3.0
            
            scheduler = UniPCMultistepScheduler(
                prediction_type='flow_prediction',
                use_flow_sigmas=True,
                num_train_timesteps=1000,
                flow_shift=flow_shift
            )
            
            # Load pipeline
            self.pipe = WanPipeline.from_pretrained(
                self.model_id,
                vae=vae,
                torch_dtype=torch.bfloat16
            )
            self.pipe.scheduler = scheduler
            self.pipe.to(self.device)
            
            logging.info("WAN model loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to load WAN model: {e}")
            raise
    
    def generate(self, prompt: str, output_path: str, **kwargs) -> GenerationResult:
        """Generate a video using the WAN model."""
        start_time = time.time()
        
        try:
            if self.pipe is None:
                self._load_model()
            
            # Extract generation parameters
            width = kwargs.get('width', self.config.video_settings.width)
            height = kwargs.get('height', self.config.video_settings.height)
            num_frames = kwargs.get('frames', self.config.video_settings.frames)
            num_inference_steps = kwargs.get('steps', self.config.model_settings.steps)
            guidance_scale = kwargs.get('cfg_scale', self.config.model_settings.cfg_scale)
            generator_seed = kwargs.get('seed', self.config.model_settings.seed)
            
            # Create generator for reproducibility
            generator = torch.Generator(device=self.device).manual_seed(generator_seed)
            
            logging.info(f"Generating video: {width}x{height}, {num_frames} frames")
            logging.info(f"Prompt: {prompt}")
            logging.info(f"Seed: {generator_seed}")
            
            # Generate video
            with torch.no_grad():
                video_frames = self.pipe(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator
                ).frames[0]
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Export video to file
            video_path = output_path
            if not video_path.endswith(('.mp4', '.avi', '.mov')):
                video_path += '.mp4'
            
            export_to_video(video_frames, video_path, fps=self.config.video_settings.fps)
            
            generation_time = time.time() - start_time
            
            return GenerationResult(
                success=True,
                video_path=video_path,
                generation_time=generation_time,
                metadata={
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "num_frames": num_frames,
                    "steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": generator_seed,
                    "model_id": self.model_id,
                    "device": self.device
                }
            )
            
        except Exception as e:
            generation_time = time.time() - start_time
            logging.error(f"WAN video generation failed: {e}")
            
            return GenerationResult(
                success=False,
                error_message=str(e),
                generation_time=generation_time
            )


class MockVideoGenerator:
    """Mock implementation for development/testing when WAN is not available."""
    
    def __init__(self, config):
        self.config = config
        
    def generate(self, prompt: str, output_path: str, **kwargs) -> GenerationResult:
        """Mock video generation that creates a placeholder file."""
        import random
        
        # Simulate generation time
        generation_time = random.uniform(10, 30)
        time.sleep(min(generation_time / 10, 3))  # Shortened for testing
        
        try:
            # Create a placeholder file
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Create a simple text file as placeholder
            placeholder_content = f"""Mock Video Generated
Prompt: {prompt}
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
Settings: {json.dumps(kwargs, indent=2)}
"""
            
            with open(output_path + ".txt", "w") as f:
                f.write(placeholder_content)
            
            return GenerationResult(
                success=True,
                video_path=output_path + ".txt",
                generation_time=generation_time,
                metadata={
                    "prompt": prompt,
                    "settings": kwargs,
                    "mock": True
                }
            )
            
        except Exception as e:
            return GenerationResult(
                success=False,
                error_message=str(e),
                generation_time=generation_time
            )


class WAN13BVideoGenerator:
    """
    WAN Video Generator wrapper.
    Automatically chooses between real WAN implementation and mock based on availability.
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the WAN model or fallback to mock."""
        try:
            if WAN_AVAILABLE and torch.cuda.is_available():
                logging.info("Initializing real WAN model")
                self.model = WanVideoGenerator(self.config)
            else:
                if not WAN_AVAILABLE:
                    logging.warning("WAN dependencies not available, using mock implementation")
                else:
                    logging.warning("CUDA not available, using mock implementation")
                self.model = MockVideoGenerator(self.config)
                
        except Exception as e:
            logging.error(f"Failed to initialize WAN model: {e}")
            logging.info("Falling back to mock implementation")
            self.model = MockVideoGenerator(self.config)
    
    def generate(self, prompt: str, output_path: str, **kwargs) -> GenerationResult:
        """Generate a single video."""
        return self.model.generate(prompt, output_path, **kwargs)
    
    def is_available(self) -> bool:
        """Check if the model is properly loaded and available."""
        return self.model is not None
    
    def is_real_model(self) -> bool:
        """Check if using real WAN model (not mock)."""
        return isinstance(self.model, WanVideoGenerator)


class BatchVideoGenerator:
    """Handles batch generation of videos with progress tracking."""
    
    def __init__(self, generator: WAN13BVideoGenerator):
        self.generator = generator
        self.progress_callback: Optional[Callable] = None
        self.logger = logging.getLogger(__name__)
    
    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """Set callback function for progress updates."""
        self.progress_callback = callback
    
    def generate_batch(self, 
                      prompts: List[str],
                      output_dir: str,
                      videos_per_prompt: int = 1,
                      filename_template: str = "{prompt_id}_{video_num:03d}",
                      **generation_kwargs) -> Dict[str, List[GenerationResult]]:
        """
        Generate multiple videos for each prompt.
        
        Args:
            prompts: List of prompt strings
            output_dir: Base output directory
            videos_per_prompt: Number of videos to generate per prompt
            filename_template: Template for video filenames
            **generation_kwargs: Additional generation parameters
        
        Returns:
            Dictionary mapping prompts to their generation results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        total_videos = len(prompts) * videos_per_prompt
        current_video = 0
        
        for prompt_idx, prompt in enumerate(prompts):
            prompt_results = []
            prompt_id = f"prompt_{prompt_idx:03d}"
            
            # Create subfolder for this prompt
            prompt_dir = output_dir / prompt_id
            prompt_dir.mkdir(exist_ok=True)
            
            # Save prompt to file
            with open(prompt_dir / "prompt.txt", "w", encoding="utf-8") as f:
                f.write(prompt)
            
            for video_num in range(videos_per_prompt):
                current_video += 1
                
                # Update progress
                if self.progress_callback:
                    self.progress_callback(current_video, total_videos, prompt)
                
                # Generate filename
                filename = filename_template.format(
                    prompt_id=prompt_id,
                    video_num=video_num + 1,
                    prompt_idx=prompt_idx
                )
                
                # Add seed variation for multiple videos
                # For each prompt, restart seed sequence from base + video_num
                current_kwargs = generation_kwargs.copy()
                
                # Always set seed with variation, regardless of whether it was passed in
                base_seed = generation_kwargs.get('seed', self.generator.config.model_settings.seed)
                current_kwargs['seed'] = base_seed + video_num
                
                output_path = prompt_dir / filename
                
                self.logger.info(f"Generating video {current_video}/{total_videos}: {filename}")
                self.logger.info(f"Using seed: {current_kwargs['seed']} (base: {base_seed} + video_num: {video_num})")
                
                # Generate video
                result = self.generator.generate(
                    prompt=prompt,
                    output_path=str(output_path),
                    **current_kwargs
                )
                
                # Log result
                if result.success:
                    self.logger.info(f"Successfully generated: {result.video_path}")
                else:
                    self.logger.error(f"Generation failed: {result.error_message}")
                
                prompt_results.append(result)
            
            results[prompt] = prompt_results
        
        return results
    
    def generate_summary_report(self, 
                               results: Dict[str, List[GenerationResult]], 
                               output_path: str):
        """Generate a summary report of the batch generation."""
        total_videos = sum(len(prompt_results) for prompt_results in results.values())
        successful_videos = sum(
            sum(1 for result in prompt_results if result.success)
            for prompt_results in results.values()
        )
        
        total_time = sum(
            sum(result.generation_time for result in prompt_results)
            for prompt_results in results.values()
        )
        
        report = {
            "summary": {
                "total_prompts": len(results),
                "total_videos": total_videos,
                "successful_videos": successful_videos,
                "failed_videos": total_videos - successful_videos,
                "success_rate": successful_videos / total_videos if total_videos > 0 else 0,
                "total_generation_time": total_time,
                "average_time_per_video": total_time / total_videos if total_videos > 0 else 0
            },
            "prompt_details": {}
        }
        
        for prompt, prompt_results in results.items():
            successful = sum(1 for result in prompt_results if result.success)
            report["prompt_details"][prompt] = {
                "total_videos": len(prompt_results),
                "successful_videos": successful,
                "failed_videos": len(prompt_results) - successful,
                "generation_times": [result.generation_time for result in prompt_results],
                "errors": [result.error_message for result in prompt_results if not result.success]
            }
        
        # Save report
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report
