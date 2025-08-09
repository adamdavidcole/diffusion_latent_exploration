#!/usr/bin/env python3
"""
Backfill existing latent metadata with timesteps and sigma values.

This script loads the WAN model scheduler, captures the timestep/sigma schedules,
and updates existing metadata files to include these values.
"""

import json
import logging
import sys
import torch
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.latent_storage import LatentStorage, LatentMetadata
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('backfill_latent_metadata.log')
        ]
    )


def create_scheduler_and_capture_schedule(num_inference_steps: int = 20) -> tuple:
    """
    Create a UniPCMultistepScheduler and capture timesteps/sigmas.
    
    Args:
        num_inference_steps: Number of inference steps (typically 20)
        
    Returns:
        Tuple of (timesteps_list, sigmas_list)
    """
    try:
        # Create the same scheduler configuration as used in WAN generation
        scheduler = UniPCMultistepScheduler(
            beta_schedule="linear",
            rescale_betas_zero_snr=True,
            num_train_timesteps=1000,
            use_flow_sigmas=True,
            flow_shift=7.0
        )
        
        # Set timesteps to capture the schedule
        scheduler.set_timesteps(num_inference_steps)
        
        # Extract timesteps and sigmas
        timesteps = scheduler.timesteps.detach().cpu().numpy().tolist()
        sigmas = scheduler.sigmas.detach().cpu().numpy().tolist()
        
        logging.info(f"Created scheduler with {len(timesteps)} timesteps and {len(sigmas)} sigmas")
        
        return timesteps, sigmas
        
    except Exception as e:
        logging.error(f"Failed to create scheduler: {e}")
        return None, None


def backfill_metadata_file(metadata_file: Path, timesteps: List[float], sigmas: List[float]) -> bool:
    """
    Backfill a single metadata file with timestep and sigma values.
    
    Args:
        metadata_file: Path to the metadata JSON file
        timesteps: List of timestep values from scheduler
        sigmas: List of sigma values from scheduler
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load existing metadata
        with open(metadata_file, 'r') as f:
            metadata_dict = json.load(f)
        
        # Check if sigma is already present
        if 'sigma' in metadata_dict:
            logging.debug(f"Sigma already present in {metadata_file}")
            return True
        
        # Get the step number
        step = metadata_dict.get('step')
        if step is None:
            logging.warning(f"No step found in metadata file: {metadata_file}")
            return False
        
        # Add sigma if available for this step
        if step < len(sigmas):
            metadata_dict['sigma'] = float(sigmas[step])
            logging.debug(f"Added sigma={sigmas[step]} for step {step}")
        else:
            logging.warning(f"Step {step} >= available sigmas ({len(sigmas)}) for {metadata_file}")
            return False
        
        # Update timestep if it looks like it needs updating (sometimes stored as int)
        if step < len(timesteps):
            # Update with the exact timestep from scheduler
            metadata_dict['timestep'] = float(timesteps[step])
            logging.debug(f"Updated timestep={timesteps[step]} for step {step}")
        
        # Write back the updated metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        logging.debug(f"Successfully updated: {metadata_file}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to update {metadata_file}: {e}")
        return False


def find_metadata_files(latents_dir: Path) -> List[Path]:
    """
    Find all metadata files in the latents directory.
    
    Args:
        latents_dir: Path to the latents directory
        
    Returns:
        List of metadata file paths
    """
    metadata_files = []
    
    # Scan for metadata files in the expected structure: prompt_XXX/vid_YYY/step_ZZZ_metadata.json
    for prompt_dir in latents_dir.iterdir():
        if prompt_dir.is_dir() and prompt_dir.name.startswith('prompt_'):
            for vid_dir in prompt_dir.iterdir():
                if vid_dir.is_dir() and vid_dir.name.startswith('vid_'):
                    for metadata_file in vid_dir.glob("step_*_metadata.json"):
                        metadata_files.append(metadata_file)
    
    return sorted(metadata_files)


def get_step_counts_by_total_steps(metadata_files: List[Path]) -> Dict[int, int]:
    """
    Group metadata files by their total_steps value to understand what schedules to create.
    
    Args:
        metadata_files: List of metadata file paths
        
    Returns:
        Dictionary mapping total_steps -> count of files
    """
    step_counts = {}
    
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            total_steps = metadata.get('total_steps', 20)  # Default to 20
            step_counts[total_steps] = step_counts.get(total_steps, 0) + 1
            
        except Exception as e:
            logging.warning(f"Failed to read {metadata_file}: {e}")
    
    return step_counts


def main():
    """Main backfill function."""
    setup_logging()
    
    if len(sys.argv) < 2:
        print("Usage: python backfill_latent_metadata.py <latents_directory>")
        print("Example: python backfill_latent_metadata.py /path/to/outputs/batch/latents")
        sys.exit(1)
    
    latents_dir = Path(sys.argv[1])
    
    if not latents_dir.exists():
        logging.error(f"Latents directory does not exist: {latents_dir}")
        sys.exit(1)
    
    logging.info(f"Starting backfill for latents directory: {latents_dir}")
    
    # Find all metadata files
    metadata_files = find_metadata_files(latents_dir)
    logging.info(f"Found {len(metadata_files)} metadata files to process")
    
    if not metadata_files:
        logging.warning("No metadata files found. Nothing to backfill.")
        return
    
    # Analyze what step counts we have
    step_counts = get_step_counts_by_total_steps(metadata_files)
    logging.info(f"Found metadata with these total_steps: {step_counts}")
    
    # Process each unique total_steps value
    schedules_cache = {}  # Cache schedules to avoid recreating
    
    updated_count = 0
    failed_count = 0
    
    for metadata_file in metadata_files:
        try:
            # Load metadata to get total_steps
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            total_steps = metadata.get('total_steps', 20)
            
            # Get or create schedule for this total_steps
            if total_steps not in schedules_cache:
                logging.info(f"Creating scheduler for {total_steps} steps")
                timesteps, sigmas = create_scheduler_and_capture_schedule(total_steps)
                
                if timesteps is None or sigmas is None:
                    logging.error(f"Failed to create schedule for {total_steps} steps")
                    failed_count += 1
                    continue
                
                schedules_cache[total_steps] = (timesteps, sigmas)
            
            timesteps, sigmas = schedules_cache[total_steps]
            
            # Backfill this metadata file
            if backfill_metadata_file(metadata_file, timesteps, sigmas):
                updated_count += 1
            else:
                failed_count += 1
                
        except Exception as e:
            logging.error(f"Failed to process {metadata_file}: {e}")
            failed_count += 1
    
    logging.info(f"Backfill completed: {updated_count} files updated, {failed_count} files failed")
    
    # Print summary
    print(f"\nBackfill Summary:")
    print(f"Total metadata files found: {len(metadata_files)}")
    print(f"Successfully updated: {updated_count}")
    print(f"Failed to update: {failed_count}")
    print(f"Schedules created for steps: {list(schedules_cache.keys())}")


if __name__ == "__main__":
    main()
