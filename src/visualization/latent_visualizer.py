"""
Latent Visualizer for decoding and visualizing latent representations.
Provides functionality to decode latent tensors back to videos using VAE models.
"""
import logging
import numpy as np
import torch
import gzip
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

# Import video generation components
try:
    from diffusers import AutoencoderKLWan
    from diffusers.utils import export_to_video
    from diffusers.video_processor import VideoProcessor
    DIFFUSERS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Diffusers not available: {e}")
    DIFFUSERS_AVAILABLE = False


@dataclass
class LatentDecodeResult:
    """Result of latent decoding operation."""
    success: bool
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    decode_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LatentDecoder:
    """Handles decoding of latent tensors back to videos using VAE models."""
    
    def __init__(self, model_id: str, device: str = "auto"):
        """
        Initialize the latent decoder.
        
        Args:
            model_id: HuggingFace model ID (e.g., "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
            device: Device to run on ("auto", "cuda:0", "cpu", etc.)
        """
        self.model_id = model_id
        self.vae = None
        self.video_processor = None
        self._setup_device(device)
        
    def _setup_device(self, device: str):
        """Setup the compute device."""
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logging.info(f"LatentDecoder using device: {self.device}")
        
        # Set CUDA device if specified
        if self.device.startswith('cuda:'):
            device_id = int(self.device.split(':')[1])
            torch.cuda.set_device(device_id)
    
    def load_vae(self):
        """Load the VAE model and video processor."""
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("Diffusers not available - cannot load VAE")
            
        if self.vae is not None:
            logging.info("VAE already loaded")
            return
            
        logging.info(f"Loading VAE from model: {self.model_id}")
        start_time = time.time()
        
        try:
            # Load VAE with memory optimization
            self.vae = AutoencoderKLWan.from_pretrained(
                self.model_id,
                subfolder="vae",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Load video processor for postprocessing
            self.vae_scale_factor_spatial = 8
            self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
            
            # Move to device
            self.vae.to(self.device)
            self.vae.eval()  # Set to evaluation mode
            
            load_time = time.time() - start_time
            logging.info(f"VAE and video processor loaded successfully in {load_time:.2f}s")
            
        except Exception as e:
            logging.error(f"Failed to load VAE: {e}")
            raise
    
    def unload_vae(self):
        """Unload VAE and video processor to free memory."""
        if self.vae is not None:
            del self.vae
            self.vae = None
            
        if self.video_processor is not None:
            del self.video_processor
            self.video_processor = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logging.info("VAE and video processor unloaded and memory cleared")
    
    def cleanup(self):
        """Clean up resources (alias for unload_vae)."""
        self.unload_vae()
    
    def load_latent_step(self, latent_path: Path) -> Tuple[torch.Tensor, Dict]:
        """
        Load a single latent step from compressed numpy file.
        
        Args:
            latent_path: Path to the .npy.gz file
            
        Returns:
            Tuple of (latent_tensor, metadata)
        """
        # Load compressed numpy array
        with gzip.open(latent_path, 'rb') as f:
            latent_array = np.load(f)
        
        # Convert to tensor
        latent_tensor = torch.from_numpy(latent_array)
        
        # Load metadata if available
        metadata_path = latent_path.with_name(latent_path.stem.replace('.npy', '_metadata.json'))
        metadata = {}
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        logging.debug(f"Loaded latent: {latent_tensor.shape} from {latent_path}")
        
        return latent_tensor, metadata
    
    def decode_latent_to_frames(self, latent_tensor: torch.Tensor) -> torch.Tensor:
        """
        Decode a latent tensor to video frames using VAE with proper WAN pipeline normalization.
        
        Args:
            latent_tensor: Latent tensor to decode [B, C, T, H, W]
            
        Returns:
            Decoded video frames tensor
        """
        if self.vae is None:
            raise RuntimeError("VAE not loaded. Call load_vae() first.")
        
        if self.video_processor is None:
            raise RuntimeError("Video processor not loaded. Call load_vae() first.")
        
        # Move latent to device and ensure proper dtype
        latent_tensor = latent_tensor.to(self.device)
        
        # Convert to VAE dtype (typically float32)
        latent_tensor = latent_tensor.to(self.vae.dtype)
        
        logging.debug(f"Decoding latent tensor: {latent_tensor.shape}, dtype: {latent_tensor.dtype}")
        
        # Apply WAN pipeline normalization before VAE decode
        with torch.no_grad():
            # Apply latents normalization from WAN pipeline
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latent_tensor.device, latent_tensor.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latent_tensor.device, latent_tensor.dtype
            )
            
            # Normalize latents: latents = latents / latents_std + latents_mean
            normalized_latents = latent_tensor / latents_std + latents_mean
            
            logging.debug(f"Applied latents normalization: mean={latents_mean.flatten()[:3].tolist()}, std_factor={latents_std.flatten()[:3].tolist()}")
            
            # VAE decode expects [B, C, T, H, W] format
            video = self.vae.decode(normalized_latents, return_dict=False)[0]
            
            # Apply video postprocessing from WAN pipeline
            video = self.video_processor.postprocess_video(video, output_type="np")
        
        logging.debug(f"Decoded video shape after postprocessing: {video.shape}")
        
        return video
    
    def frames_to_video(self, frames: Union[torch.Tensor, np.ndarray], output_path: Path, fps: int = 12) -> bool:
        """
        Export decoded frames to video file using diffusers export_to_video.
        
        Args:
            frames: Postprocessed video frames from decode_latent_to_frames()
            output_path: Output video file path
            fps: Frames per second
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # VideoProcessor.postprocess_video returns [B, T, H, W, C] format
            # Remove batch dimension for export_to_video which expects [T, H, W, C]
            if isinstance(frames, np.ndarray) and len(frames.shape) == 5:  # [B, T, H, W, C]
                frames = frames.squeeze(0)  # Remove batch dimension -> [T, H, W, C]
            elif hasattr(frames, 'squeeze') and len(frames.shape) == 5:  # torch tensor [B, T, H, W, C]
                frames = frames.squeeze(0).cpu().numpy()  # -> [T, H, W, C] numpy
            
            logging.debug(f"Exporting video: shape={frames.shape}, dtype={frames.dtype}")
            
            export_to_video(frames, str(output_path), fps=fps)
            
            logging.debug(f"Video exported to: {output_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to export video: {e}")
            return False
    
    def decode_latent_step_to_video(self, 
                                   latent_path: Path, 
                                   output_path: Path, 
                                   fps: int = 12) -> LatentDecodeResult:
        """
        Decode a single latent step file to video.
        
        Args:
            latent_path: Path to .npy.gz latent file
            output_path: Output video path
            fps: Video frames per second
            
        Returns:
            LatentDecodeResult with operation details
        """
        start_time = time.time()
        
        try:
            # Load latent step
            latent_tensor, metadata = self.load_latent_step(latent_path)
            
            # Decode to frames
            decoded_frames = self.decode_latent_to_frames(latent_tensor)
            
            # Export to video
            success = self.frames_to_video(decoded_frames, output_path, fps)
            
            if not success:
                return LatentDecodeResult(
                    success=False,
                    error_message="Failed to export video",
                    decode_time=time.time() - start_time
                )
            
            decode_time = time.time() - start_time
            
            return LatentDecodeResult(
                success=True,
                output_path=str(output_path),
                decode_time=decode_time,
                metadata={
                    **metadata,
                    "latent_shape": list(latent_tensor.shape),
                    "output_shape": list(decoded_frames.shape),
                    "fps": fps
                }
            )
            
        except Exception as e:
            decode_time = time.time() - start_time
            logging.error(f"Failed to decode latent step: {e}")
            
            return LatentDecodeResult(
                success=False,
                error_message=str(e),
                decode_time=decode_time
            )


class ExperimentLatentDecoder:
    """Handles decoding latents for entire experiments with batch processing."""
    
    def __init__(self, experiment_dir: Path, device: str = "auto"):
        """
        Initialize experiment decoder.
        
        Args:
            experiment_dir: Path to experiment directory containing configs/ and latents/
            device: Device to run on
        """
        self.experiment_dir = Path(experiment_dir)
        self.device = device
        self.decoder = None
        
        # Load configuration
        self.config = self._load_experiment_config()
        
    def _load_experiment_config(self) -> Dict:
        """Load experiment configuration from configs/generation_config.yaml."""
        config_path = self.experiment_dir / "configs" / "generation_config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration not found: {config_path}")
        
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logging.info(f"Loaded experiment config from: {config_path}")
        return config
    
    def get_model_id(self) -> str:
        """Extract model ID from experiment configuration."""
        model_id = self.config.get('model_settings', {}).get('model_id')
        if not model_id:
            raise ValueError("model_id not found in experiment configuration")
        return model_id
    
    def get_video_settings(self) -> Dict:
        """Extract video settings from experiment configuration."""
        return self.config.get('video_settings', {})
    
    def initialize_decoder(self):
        """Initialize the latent decoder with experiment model."""
        if self.decoder is None:
            model_id = self.get_model_id()
            self.decoder = LatentDecoder(model_id, self.device)
            self.decoder.load_vae()
            logging.info(f"Initialized decoder for model: {model_id}")
    
    def find_latent_directories(self) -> List[Path]:
        """Find all video latent directories in the experiment."""
        latents_dir = self.experiment_dir / "latents"
        
        if not latents_dir.exists():
            raise FileNotFoundError(f"Latents directory not found: {latents_dir}")
        
        video_dirs = []
        
        # Search pattern: latents/prompt_XXX/vid_YYY/
        for prompt_dir in latents_dir.glob("prompt_*"):
            if prompt_dir.is_dir():
                for vid_dir in prompt_dir.glob("vid_*"):
                    if vid_dir.is_dir():
                        video_dirs.append(vid_dir)
        
        logging.info(f"Found {len(video_dirs)} video latent directories")
        return sorted(video_dirs)
    
    def find_latent_steps(self, video_dir: Path) -> List[Path]:
        """Find all latent step files in a video directory."""
        step_files = list(video_dir.glob("step_*.npy.gz"))
        return sorted(step_files)
    
    def decode_video_directory(self, 
                              video_dir: Path, 
                              output_dir: Optional[Path] = None,
                              fps: Optional[int] = None,
                              step_filter: Optional[str] = None) -> Dict[str, LatentDecodeResult]:
        """
        Decode all latent steps in a video directory.
        
        Args:
            video_dir: Path to video latent directory (e.g., latents/prompt_000/vid_001/)
            output_dir: Output directory for decoded videos (default: latents_videos/)
            fps: Video FPS (default: from config)
            step_filter: Filter for step files (e.g., "step_000")
            
        Returns:
            Dictionary mapping step names to decode results
        """
        if self.decoder is None:
            self.initialize_decoder()
        
        # Setup output directory
        if output_dir is None:
            output_dir = self.experiment_dir / "latents_videos"
        
        # Preserve directory structure: prompt_XXX/vid_YYY/
        relative_path = video_dir.relative_to(self.experiment_dir / "latents")
        video_output_dir = output_dir / relative_path
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get FPS from config if not specified
        if fps is None:
            fps = self.get_video_settings().get('fps', 12)
        
        # Find all latent steps
        step_files = self.find_latent_steps(video_dir)
        
        # Apply step filter if specified
        if step_filter:
            step_filters = [f.strip() for f in step_filter.split(',')]
            step_files = [f for f in step_files 
                         if any(f.stem.replace('.npy', '') == sf for sf in step_filters)]
        
        if not step_files:
            logging.warning(f"No latent steps found in {video_dir}")
            return {}
        
        logging.info(f"Decoding {len(step_files)} steps from {video_dir}")
        
        results = {}
        
        for step_file in step_files:
            step_name = step_file.stem.replace('.npy', '')  # e.g., "step_000"
            output_path = video_output_dir / f"{step_name}.mp4"
            
            result = self.decoder.decode_latent_step_to_video(
                latent_path=step_file,
                output_path=output_path,
                fps=fps
            )
            
            results[step_name] = result
            
            if result.success:
                logging.debug(f"✅ Decoded {step_name} in {result.decode_time:.2f}s")
            else:
                logging.error(f"❌ Failed to decode {step_name}: {result.error_message}")
        
        return results
    
    def decode_experiment(self, 
                         output_dir: Optional[Path] = None,
                         prompt_filter: Optional[str] = None,
                         video_filter: Optional[str] = None,
                         step_filter: Optional[str] = None) -> Dict[str, Dict[str, LatentDecodeResult]]:
        """
        Decode all latents in the experiment.
        
        Args:
            output_dir: Output directory for decoded videos
            prompt_filter: Filter for prompt directories (e.g., "prompt_000")
            video_filter: Filter for video directories (e.g., "vid_001")
            step_filter: Filter for step files (e.g., "step_000")
            
        Returns:
            Nested dictionary: {video_path: {step_name: result}}
        """
        if self.decoder is None:
            self.initialize_decoder()
        
        video_dirs = self.find_latent_directories()
        
        # Apply filters
        if prompt_filter:
            video_dirs = [d for d in video_dirs if prompt_filter in str(d)]
        if video_filter:
            video_dirs = [d for d in video_dirs if video_filter in str(d)]
        
        if not video_dirs:
            logging.warning("No video directories found after filtering")
            return {}
        
        logging.info(f"Decoding {len(video_dirs)} video directories")
        
        all_results = {}
        
        for video_dir in video_dirs:
            logging.info(f"Processing video directory: {video_dir}")
            
            try:
                results = self.decode_video_directory(video_dir, output_dir, step_filter=step_filter)
                
                all_results[str(video_dir)] = results
                
            except Exception as e:
                logging.error(f"Failed to process {video_dir}: {e}")
                all_results[str(video_dir)] = {}
        
        return all_results
    
    def cleanup(self):
        """Clean up resources."""
        if self.decoder:
            self.decoder.unload_vae()
            self.decoder = None


def create_decode_summary_report(results: Dict[str, Dict[str, LatentDecodeResult]], 
                                output_path: Path) -> Dict:
    """
    Create a summary report of decoding results.
    
    Args:
        results: Nested results dictionary
        output_path: Path to save the report
        
    Returns:
        Summary statistics dictionary
    """
    total_steps = 0
    successful_steps = 0
    total_time = 0.0
    
    video_summaries = {}
    
    for video_path, video_results in results.items():
        video_total = len(video_results)
        video_successful = sum(1 for r in video_results.values() if r.success)
        video_time = sum(r.decode_time for r in video_results.values())
        
        video_summaries[video_path] = {
            "total_steps": video_total,
            "successful_steps": video_successful,
            "failed_steps": video_total - video_successful,
            "total_decode_time": video_time,
            "avg_decode_time": video_time / video_total if video_total > 0 else 0,
            "success_rate": video_successful / video_total if video_total > 0 else 0
        }
        
        total_steps += video_total
        successful_steps += video_successful
        total_time += video_time
    
    summary = {
        "total_steps": total_steps,
        "successful_steps": successful_steps,
        "failed_steps": total_steps - successful_steps,
        "success_rate": successful_steps / total_steps if total_steps > 0 else 0,
        "total_decode_time": total_time,
        "avg_decode_time": total_time / total_steps if total_steps > 0 else 0,
        "video_summaries": video_summaries
    }
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"Decode summary saved to: {output_path}")
    return summary
