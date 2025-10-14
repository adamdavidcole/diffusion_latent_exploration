#!/usr/bin/env python3
"""
Backfill script to generate missing prompt_schedule.json files for existing batch outputs.

This script:
1. Recursively scans the outputs folder for batch directories
2. Checks if generation_config.yaml indicates prompt scheduling was used
3. Generates the missing prompt_schedule.json file if needed

Usage:
    python scripts/backfill_prompt_schedules.py
    python scripts/backfill_prompt_schedules.py --dry-run
    python scripts/backfill_prompt_schedules.py --output-dir outputs/PromptInterpolation
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def is_valid_batch_directory(path: Path) -> bool:
    """
    Check if a directory is a valid batch output directory.
    
    A valid batch directory should have:
    - configs/ subdirectory with generation_config.yaml
    - videos/ subdirectory
    
    Args:
        path: Path to check
        
    Returns:
        True if valid batch directory
    """
    if not path.is_dir():
        return False
    
    configs_dir = path / "configs"
    videos_dir = path / "videos"
    generation_config = configs_dir / "generation_config.yaml"
    
    return (
        configs_dir.exists() and
        videos_dir.exists() and
        generation_config.exists()
    )


def load_generation_config(config_path: Path) -> Optional[dict]:
    """
    Load and parse the generation_config.yaml file.
    
    Args:
        config_path: Path to generation_config.yaml
        
    Returns:
        Parsed config dictionary or None if error
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Failed to load config {config_path}: {e}")
        return None


def extract_prompt_schedule_settings(config: dict) -> Optional[Tuple[Dict[int, str], str, int]]:
    """
    Extract prompt schedule settings from generation config.
    
    Args:
        config: Parsed generation config dictionary
        
    Returns:
        Tuple of (schedule, interpolation, total_steps) or None if not using prompt scheduling
    """
    # Check if prompt scheduling is enabled
    prompt_schedule_settings = config.get('prompt_schedule_settings', {})
    
    if not prompt_schedule_settings:
        return None
    
    enabled = prompt_schedule_settings.get('enabled', False)
    if not enabled:
        return None
    
    schedule = prompt_schedule_settings.get('schedule', {})
    if not schedule:
        return None
    
    interpolation = prompt_schedule_settings.get('interpolation', 'slerp')
    
    # Get total steps from model settings
    model_settings = config.get('model_settings', {})
    total_steps = model_settings.get('steps', 20)
    
    # Convert schedule keys to integers (YAML might load them as strings)
    schedule_int_keys = {}
    for key, value in schedule.items():
        try:
            schedule_int_keys[int(key)] = value
        except (ValueError, TypeError):
            logging.warning(f"Invalid schedule key: {key}")
            continue
    
    if not schedule_int_keys:
        return None
    
    return (schedule_int_keys, interpolation, total_steps)


def save_prompt_schedule(schedule: Dict[int, str], 
                        interpolation: str,
                        total_steps: int,
                        output_path: Path) -> bool:
    """
    Save the prompt schedule configuration to a JSON file.
    
    This matches the format used in video_generator.py.
    
    Args:
        schedule: Dictionary mapping steps to prompts
        interpolation: Interpolation method used
        total_steps: Total number of steps
        output_path: Path to save JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        schedule_data = {
            "schedule": schedule,
            "interpolation": interpolation,
            "total_steps": total_steps,
            "keyframes": sorted(schedule.keys())
        }
        
        with open(output_path, 'w') as f:
            json.dump(schedule_data, f, indent=2)
        
        return True
    except Exception as e:
        logging.error(f"Failed to save prompt schedule to {output_path}: {e}")
        return False


def find_batch_directories(root_dir: Path) -> List[Path]:
    """
    Recursively find all valid batch directories under root_dir.
    
    Args:
        root_dir: Root directory to search
        
    Returns:
        List of valid batch directory paths
    """
    batch_dirs = []
    
    # Walk through all directories
    for path in root_dir.rglob('*'):
        if is_valid_batch_directory(path):
            batch_dirs.append(path)
    
    return sorted(batch_dirs)


def process_batch_directory(batch_dir: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """
    Process a single batch directory and generate prompt_schedule.json if needed.
    
    Args:
        batch_dir: Path to batch directory
        dry_run: If True, don't actually write files
        
    Returns:
        Tuple of (success, message)
    """
    configs_dir = batch_dir / "configs"
    generation_config_path = configs_dir / "generation_config.yaml"
    prompt_schedule_path = configs_dir / "prompt_schedule.json"
    
    # Check if prompt_schedule.json already exists
    if prompt_schedule_path.exists():
        return (True, "prompt_schedule.json already exists")
    
    # Load generation config
    config = load_generation_config(generation_config_path)
    if config is None:
        return (False, "Failed to load generation_config.yaml")
    
    # Extract prompt schedule settings
    schedule_info = extract_prompt_schedule_settings(config)
    if schedule_info is None:
        return (True, "No prompt scheduling enabled")
    
    schedule, interpolation, total_steps = schedule_info
    
    # Log what we found
    keyframes = sorted(schedule.keys())
    logging.info(f"  Found prompt schedule with {len(keyframes)} keyframes: {keyframes}")
    logging.info(f"  Interpolation: {interpolation}, Total steps: {total_steps}")
    
    if dry_run:
        return (True, f"Would create prompt_schedule.json with {len(keyframes)} keyframes")
    
    # Save the prompt schedule
    if save_prompt_schedule(schedule, interpolation, total_steps, prompt_schedule_path):
        return (True, f"Created prompt_schedule.json with {len(keyframes)} keyframes")
    else:
        return (False, "Failed to save prompt_schedule.json")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill missing prompt_schedule.json files for existing batch outputs"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Root output directory to scan (default: outputs)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually creating files'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Resolve output directory
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        logging.error(f"Output directory does not exist: {output_dir}")
        sys.exit(1)
    
    if not output_dir.is_dir():
        logging.error(f"Output path is not a directory: {output_dir}")
        sys.exit(1)
    
    logging.info(f"Scanning for batch directories in: {output_dir}")
    if args.dry_run:
        logging.info("DRY RUN MODE - No files will be created")
    
    # Find all batch directories
    batch_dirs = find_batch_directories(output_dir)
    logging.info(f"Found {len(batch_dirs)} batch directories")
    
    if not batch_dirs:
        logging.info("No batch directories found")
        sys.exit(0)
    
    # Process each batch directory
    stats = {
        'total': len(batch_dirs),
        'created': 0,
        'already_exists': 0,
        'no_schedule': 0,
        'failed': 0
    }
    
    for i, batch_dir in enumerate(batch_dirs, 1):
        relative_path = batch_dir.relative_to(output_dir)
        logging.info(f"\n[{i}/{len(batch_dirs)}] Processing: {relative_path}")
        
        success, message = process_batch_directory(batch_dir, dry_run=args.dry_run)
        logging.info(f"  Result: {message}")
        
        # Update statistics
        if not success:
            stats['failed'] += 1
        elif "already exists" in message:
            stats['already_exists'] += 1
        elif "No prompt scheduling" in message:
            stats['no_schedule'] += 1
        elif "Created" in message or "Would create" in message:
            stats['created'] += 1
    
    # Print summary
    logging.info("\n" + "="*80)
    logging.info("SUMMARY")
    logging.info("="*80)
    logging.info(f"Total batch directories scanned: {stats['total']}")
    logging.info(f"Prompt schedules created: {stats['created']}")
    logging.info(f"Already had prompt_schedule.json: {stats['already_exists']}")
    logging.info(f"No prompt scheduling enabled: {stats['no_schedule']}")
    logging.info(f"Failed: {stats['failed']}")
    
    if args.dry_run:
        logging.info("\nDRY RUN completed - No files were created")
    else:
        logging.info("\nBackfill completed successfully!")
    
    sys.exit(0 if stats['failed'] == 0 else 1)


if __name__ == "__main__":
    main()
