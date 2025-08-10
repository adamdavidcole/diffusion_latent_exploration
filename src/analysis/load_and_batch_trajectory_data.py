from pathlib import Path
import torch
import numpy as np
from typing import Dict, List
import logging
import json


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)    

def load_and_batch_trajectory_data(
    latents_dir: Path,
    prompt_groups: List[str],
    device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Load and batch trajectory data preserving diffusion step structure."""
    import gzip
    
    group_tensors = {}
    
    for group_name in prompt_groups:
        logger.info(f"Loading trajectory-structured latents for group: {group_name}")
        
        group_dir = latents_dir / group_name
        if not group_dir.exists():
            logger.warning(f"Group directory not found: {group_dir}")
            continue
        
        # Find all video directories
        video_dirs = sorted([d for d in group_dir.iterdir() if d.is_dir() and d.name.startswith('vid_')])
        
        if not video_dirs:
            logger.warning(f"No video directories found in {group_dir}")
            continue
        
        video_trajectories = []
        trajectory_metadata = []
        
        for video_dir in video_dirs:
            try:
                # Find all step files for this video
                step_files = sorted([f for f in video_dir.glob("step_*.npy.gz")])
                
                if not step_files:
                    logger.warning(f"No step files found in {video_dir}")
                    continue
                
                # Load trajectory preserving diffusion step order
                video_steps = []
                step_metadata = []
                
                for step_file in step_files:
                    # Load compressed numpy array
                    with gzip.open(step_file, 'rb') as f:
                        step_latent = np.load(f)
                    
                    # Convert to tensor: [1, 16, frames, H, W]
                    step_tensor = torch.from_numpy(step_latent).float()
                    video_steps.append(step_tensor)
                    
                    # Load metadata if available
                    metadata_file = step_file.with_name(step_file.stem.replace('.npy', '_metadata.json'))
                    if metadata_file.exists():
                        with open(metadata_file) as f:
                            step_meta = json.load(f)
                            step_metadata.append(step_meta)
                
                if video_steps:
                    # Stack steps to create video trajectory: [steps, 1, 16, frames, H, W]
                    video_trajectory = torch.stack(video_steps, dim=0)
                    video_trajectories.append(video_trajectory)
                    
                    trajectory_metadata.append({
                        'video_id': video_dir.name,
                        'n_steps': len(video_steps),
                        'step_metadata': step_metadata,
                        'trajectory_shape': video_trajectory.shape
                    })
                    
                    logger.debug(f"Loaded video {video_dir.name}: {video_trajectory.shape}")
                    
            except Exception as e:
                logger.warning(f"Failed to load trajectory for {video_dir.name}: {e}")
        
        if video_trajectories:
            try:
                # Ensure all trajectories have same number of steps
                min_steps = min(traj.shape[0] for traj in video_trajectories)
                logger.info(f"Truncating all trajectories to {min_steps} steps for consistency")
                
                # Truncate and stack: [n_videos, steps, 1, 16, frames, H, W]
                truncated_trajectories = [traj[:min_steps] for traj in video_trajectories]
                batched_trajectories = torch.stack(truncated_trajectories, dim=0)
                
                # Move to device
                batched_trajectories = batched_trajectories.to(device)
                
                group_tensors[group_name] = {
                    'trajectory_tensor': batched_trajectories,  # [n_videos, steps, 1, 16, frames, H, W]
                    'trajectory_metadata': trajectory_metadata,
                    'n_videos': len(video_trajectories),
                    'n_steps': min_steps,
                    'latent_shape': batched_trajectories.shape[3:],  # [16, frames, H, W]
                    'full_shape': batched_trajectories.shape
                }
                
                logger.info(f"✅ Loaded {len(video_trajectories)} trajectory videos for {group_name}")
                logger.info(f"   Shape: {batched_trajectories.shape} [videos, steps, batch, channels, frames, H, W]")
                logger.info(f"   Preserving trajectory structure for diffusion step analysis")
                logger.info(f"   Device: {batched_trajectories.device}")
                
            except Exception as e:
                logger.error(f"Failed to batch trajectories for {group_name}: {e}")
    
    return group_tensors