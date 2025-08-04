"""
Utilities for storing and managing latent representations during diffusion generation.
"""
import os
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import logging
import json
from dataclasses import dataclass, asdict
import gzip
import pickle


@dataclass
class LatentMetadata:
    """Metadata for stored latents."""
    video_id: str
    step: int
    timestep: float
    total_steps: int
    shape: tuple
    dtype: str
    prompt: str
    seed: Optional[int] = None
    cfg_scale: Optional[float] = None
    generation_time: Optional[float] = None


class LatentStorage:
    """Manages storage and retrieval of latent representations during diffusion."""
    
    def __init__(self, 
                 storage_dir: Union[str, Path], 
                 storage_format: str = "numpy",
                 compress: bool = True,
                 storage_interval: int = 1,
                 storage_dtype: str = "float32"):
        """
        Initialize latent storage manager.
        
        Args:
            storage_dir: Directory to store latent files
            storage_format: Format for storage ("numpy" or "torch")
            compress: Whether to compress stored latents
            storage_interval: Store every N steps (1 = all steps)
            storage_dtype: Data type for storage ("float32" or "float16")
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.storage_format = storage_format
        self.compress = compress
        self.storage_interval = storage_interval
        self.storage_dtype = storage_dtype
        
        # Use storage_dir directly as the latents directory
        self.latents_dir = self.storage_dir
        
        self.logger = logging.getLogger(__name__)
        
        # Storage tracking
        self.current_video_id = None
        self.current_video_dir = None  # Track the current video's directory
        self.stored_steps = []
        
    def start_video_storage(self, video_id: str, prompt: str, **generation_params):
        """Start storing latents for a new video."""
        self.current_video_id = video_id
        self.current_prompt = prompt
        self.current_generation_params = generation_params
        self.stored_steps = []
        
        # Extract prompt and video parts from video_id (e.g., "prompt_000_vid001" -> "prompt_000", "vid001")
        # Video IDs are formatted as: prompt_XXX_vidYYY
        if "_vid" in video_id:
            prompt_part, vid_part = video_id.split("_vid")
            vid_part = f"vid_{vid_part}"  # Convert "001" to "vid_001"
        else:
            # Fallback if format is different
            prompt_part = video_id
            vid_part = "vid_001"  # Default video number
        
        # Create video-specific directory structure: prompt_000/vid_001/
        self.current_video_dir = self.latents_dir / prompt_part / vid_part
        self.current_video_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting latent storage for video: {video_id}")
        self.logger.info(f"Latents will be stored in: {self.current_video_dir}")
    
    def store_latent(self, 
                    latent: torch.Tensor, 
                    step: int, 
                    timestep: float, 
                    total_steps: int) -> bool:
        """
        Store a latent representation for the current video.
        
        Args:
            latent: The latent tensor to store
            step: Current denoising step
            timestep: Current timestep value
            total_steps: Total number of steps
            
        Returns:
            bool: True if latent was stored, False if skipped
        """
        if self.current_video_id is None:
            self.logger.warning("No active video for latent storage")
            return False
        
        # Check if we should store this step based on interval
        if step % self.storage_interval != 0:
            return False
        
        try:
            # Store latent data in the video-specific directory first
            latent_cpu = latent.detach().cpu()
            
            # Apply dtype conversion if specified
            if self.storage_dtype == "float16":
                latent_cpu = latent_cpu.half()
            elif self.storage_dtype == "float32":
                latent_cpu = latent_cpu.float()
            else:
                self.logger.warning(f"Unknown storage_dtype: {self.storage_dtype}, using original dtype")
            
            # Create metadata with only valid LatentMetadata parameters
            valid_params = {}
            if 'seed' in self.current_generation_params:
                valid_params['seed'] = self.current_generation_params['seed']
            if 'cfg_scale' in self.current_generation_params:
                valid_params['cfg_scale'] = self.current_generation_params['cfg_scale']
            
            metadata = LatentMetadata(
                video_id=self.current_video_id,
                step=step,
                timestep=timestep,
                total_steps=total_steps,
                shape=tuple(latent.shape),
                dtype=str(latent_cpu.dtype),  # Use the actual stored dtype
                prompt=self.current_prompt,
                **valid_params
            )
            
            # Generate simplified filename: step_000.npy.gz (no video_id prefix)
            filename_base = f"step_{step:03d}"
            
            if self.storage_format == "numpy":
                latent_array = latent_cpu.numpy()
                latent_file = self.current_video_dir / f"{filename_base}.npy"
                
                if self.compress:
                    latent_file = latent_file.with_suffix(".npy.gz")
                    with gzip.open(latent_file, 'wb') as f:
                        np.save(f, latent_array)
                else:
                    np.save(latent_file, latent_array)
                    
            elif self.storage_format == "torch":
                latent_file = self.current_video_dir / f"{filename_base}.pt"
                
                if self.compress:
                    latent_file = latent_file.with_suffix(".pt.gz")
                    with gzip.open(latent_file, 'wb') as f:
                        torch.save(latent_cpu, f)
                else:
                    torch.save(latent_cpu, latent_file)
            else:
                raise ValueError(f"Unsupported storage format: {self.storage_format}")
            
            # Store metadata in the video-specific directory
            metadata_file = self.current_video_dir / f"{filename_base}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)
            
            self.stored_steps.append(step)
            self.logger.debug(f"Stored latent for step {step} at {latent_file}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing latent for step {step}: {e}")
            return False
    
    def finish_video_storage(self) -> Dict[str, Any]:
        """Finish storing latents for the current video and return summary."""
        if self.current_video_id is None:
            return {}
        
        summary = {
            "video_id": self.current_video_id,
            "stored_steps": self.stored_steps.copy(),
            "total_stored": len(self.stored_steps),
            "storage_dir": str(self.storage_dir),
            "storage_format": self.storage_format,
            "compressed": self.compress
        }
        
        # Save video summary to the individual video directory
        summary_file = self.current_video_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Finished storing latents for {self.current_video_id}: {len(self.stored_steps)} steps stored")
        
        # Reset for next video
        self.current_video_id = None
        self.current_video_dir = None
        self.current_prompt = None
        self.current_generation_params = {}
        self.stored_steps = []
        
        return summary
    
    def load_latent(self, video_id: str, step: int) -> Optional[torch.Tensor]:
        """Load a stored latent representation."""
        # Extract prompt and video parts from video_id (e.g., "prompt_000_vid001" -> "prompt_000", "vid_001")
        if "_vid" in video_id:
            prompt_part, vid_num = video_id.split("_vid")
            vid_part = f"vid_{vid_num}"  # Convert "001" to "vid_001"
        else:
            prompt_part = video_id
            vid_part = "vid_001"  # Default fallback
        
        video_dir = self.latents_dir / prompt_part / vid_part
        filename_base = f"step_{step:03d}"
        
        # Try different file extensions
        possible_files = [
            video_dir / f"{filename_base}.npy",
            video_dir / f"{filename_base}.npy.gz",
            video_dir / f"{filename_base}.pt",
            video_dir / f"{filename_base}.pt.gz"
        ]
        
        for latent_file in possible_files:
            if latent_file.exists():
                try:
                    if latent_file.suffix == '.gz':
                        with gzip.open(latent_file, 'rb') as f:
                            if '.npy' in latent_file.name:
                                latent_array = np.load(f)
                                tensor = torch.from_numpy(latent_array)
                                # Convert back to float32 for analysis if it was stored as float16
                                if tensor.dtype == torch.float16:
                                    tensor = tensor.float()
                                return tensor
                            elif '.pt' in latent_file.name:
                                tensor = torch.load(f)
                                # Convert back to float32 for analysis if it was stored as float16
                                if tensor.dtype == torch.float16:
                                    tensor = tensor.float()
                                return tensor
                    else:
                        if latent_file.suffix == '.npy':
                            latent_array = np.load(latent_file)
                            tensor = torch.from_numpy(latent_array)
                            # Convert back to float32 for analysis if it was stored as float16
                            if tensor.dtype == torch.float16:
                                tensor = tensor.float()
                            return tensor
                        elif latent_file.suffix == '.pt':
                            tensor = torch.load(latent_file)
                            # Convert back to float32 for analysis if it was stored as float16
                            if tensor.dtype == torch.float16:
                                tensor = tensor.float()
                            return tensor
                except Exception as e:
                    self.logger.error(f"Error loading latent from {latent_file}: {e}")
                    continue
        
        self.logger.warning(f"Latent not found for video {video_id}, step {step}")
        return None
    
    def load_metadata(self, video_id: str, step: int) -> Optional[LatentMetadata]:
        """Load metadata for a stored latent."""
        # Extract prompt and video parts from video_id (e.g., "prompt_000_vid001" -> "prompt_000", "vid_001")
        if "_vid" in video_id:
            prompt_part, vid_num = video_id.split("_vid")
            vid_part = f"vid_{vid_num}"  # Convert "001" to "vid_001"
        else:
            prompt_part = video_id
            vid_part = "vid_001"  # Default fallback
        
        video_dir = self.latents_dir / prompt_part / vid_part
        filename_base = f"step_{step:03d}"
        metadata_file = video_dir / f"{filename_base}_metadata.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                return LatentMetadata(**metadata_dict)
            except Exception as e:
                self.logger.error(f"Error loading metadata from {metadata_file}: {e}")
        
        return None
    
    def get_video_summary(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get storage summary for a video."""
        # Extract prompt and video parts from video_id
        if "_vid" in video_id:
            prompt_part, vid_part = video_id.split("_vid")
            vid_part = f"vid_{vid_part}"  # Convert "001" to "vid_001"
        else:
            prompt_part = video_id
            vid_part = "vid_001"
        
        # Look for summary in the individual video directory
        video_dir = self.latents_dir / prompt_part / vid_part
        summary_file = video_dir / "summary.json"
        
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading video summary: {e}")
        
        return None
    
    def list_stored_videos(self) -> list:
        """List all videos with stored latents."""
        videos = set()
        
        # Scan prompt directories in the storage directory
        for prompt_dir in self.latents_dir.iterdir():
            if prompt_dir.is_dir() and prompt_dir.name.startswith('prompt_'):
                # Scan video directories within each prompt directory
                for vid_dir in prompt_dir.iterdir():
                    if vid_dir.is_dir() and vid_dir.name.startswith('vid_'):
                        # Check if this directory has any latent files
                        latent_files = list(vid_dir.glob("step_*.npy*")) + list(vid_dir.glob("step_*.pt*"))
                        if latent_files:
                            # Reconstruct video ID: prompt_000_vid001
                            vid_num = vid_dir.name.replace('vid_', '')  # Extract "001" from "vid_001"
                            video_id = f"{prompt_dir.name}_vid{vid_num}"
                            videos.add(video_id)
        
        return sorted(list(videos))
    
    def get_prompt_from_video_id(self, video_id: str) -> str:
        """Extract prompt directory name from video ID."""
        if "_vid" in video_id:
            return video_id.split("_vid")[0]
        else:
            return video_id
    
    def list_steps_for_video(self, video_id: str) -> List[int]:
        """List all available steps for a given video ID."""
        prompt_part = self.get_prompt_from_video_id(video_id)
        
        # Extract video part
        if "_vid" in video_id:
            _, vid_num = video_id.split("_vid")
            vid_part = f"vid_{vid_num}"
        else:
            vid_part = "vid_001"  # Default fallback
            
        video_dir = self.latents_dir / prompt_part / vid_part
        
        if not video_dir.exists():
            return []
        
        # Find all step files
        step_files = list(video_dir.glob("step_*.npy*")) + list(video_dir.glob("step_*.pt*"))
        steps = []
        
        for file_path in step_files:
            # Extract step number from filename like "step_000.npy.gz"
            filename = file_path.stem
            if filename.endswith('.npy') or filename.endswith('.pt'):
                filename = file_path.with_suffix('').stem
            
            if filename.startswith('step_'):
                try:
                    step_num = int(filename.split('_')[1])
                    steps.append(step_num)
                except (IndexError, ValueError):
                    continue
        
        return sorted(list(set(steps)))
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get overall storage statistics."""
        videos = self.list_stored_videos()
        
        # Count all files in all prompt/video directories
        total_files = 0
        total_size = 0
        for prompt_dir in self.latents_dir.iterdir():
            if prompt_dir.is_dir():
                for vid_dir in prompt_dir.iterdir():
                    if vid_dir.is_dir():
                        for file_path in vid_dir.glob("*"):
                            if file_path.is_file():
                                total_files += 1
                                total_size += file_path.stat().st_size
        
        return {
            "total_videos": len(videos),
            "total_latent_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "storage_format": self.storage_format,
            "storage_dtype": self.storage_dtype,
            "compressed": self.compress,
            "videos": videos
        }


def create_denoising_callback(latent_storage: LatentStorage):
    """
    Create a callback function for the diffusion pipeline to store latents.
    
    Args:
        latent_storage: LatentStorage instance to use for storing
        
    Returns:
        Callback function compatible with diffusion pipelines
    """
    def callback(step: int, timestep: torch.Tensor, latents: torch.Tensor):
        """Callback function to store latents during denoising."""
        try:
            # Convert timestep to float if it's a tensor
            if torch.is_tensor(timestep):
                timestep_val = timestep.item()
            else:
                timestep_val = float(timestep)
            
            # We don't know total_steps here, so we'll estimate or set a default
            # This can be improved by passing total_steps to the callback creation
            total_steps = 50  # Default assumption, should be configured
            
            latent_storage.store_latent(
                latent=latents,
                step=step,
                timestep=timestep_val,
                total_steps=total_steps
            )
            
        except Exception as e:
            logging.error(f"Error in denoising callback: {e}")
    
    return callback
