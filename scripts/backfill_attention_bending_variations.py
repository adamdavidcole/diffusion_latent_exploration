#!/usr/bin/env python3
"""
Backfill script to reconstruct attention_bending_variations configuration
from video_metadata.json files.

This script analyzes the bending_metadata in video_metadata.json to rebuild
the original attention_bending_variations config that was used to generate
the experiment. This is useful for experiments that were generated before
attention_bending_variations was saved to generation_config.yaml.

Usage:
    python scripts/backfill_attention_bending_variations.py <experiment_path>
    python scripts/backfill_attention_bending_variations.py outputs/dual_gpu_comprehensive_20260129_023925
"""

import sys
import json
import yaml
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def load_video_metadata(metadata_path: Path) -> Optional[Dict]:
    """Load video_metadata.json file."""
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {metadata_path}: {e}")
        return None


def analyze_bending_variations(videos: List[Dict]) -> Dict[str, Any]:
    """
    Analyze video metadata to extract unique bending variations.
    
    Returns a dict organizing variations by operation type and parameter.
    Key format: "operation:parameter_name" to support multiple parameters per operation.
    """
    # Group videos by operation + parameter name to handle operations with multiple parameters
    operations_map = defaultdict(lambda: {
        'operation': None,
        'parameter_values': set(),
        'target_tokens': set(),
        'timestep_ranges': set(),
        'layer_ranges': set(),
        'parameter_name': None,
        'examples': []
    })
    
    baseline_count = 0
    
    for video in videos:
        bending_meta = video.get('bending_metadata')
        
        # Skip if no bending_metadata key or if it's None (baseline)
        if not bending_meta or bending_meta is None:
            baseline_count += 1
            continue
        
        operation = bending_meta.get('transformation_type')  # Note: uses 'transformation_type', not 'operation'
        if not operation:
            baseline_count += 1
            continue
        
        # Extract transformation parameters
        transform_params = bending_meta.get('transformation_params', {})
        param_name = list(transform_params.keys())[0] if transform_params else None
        param_value = list(transform_params.values())[0] if transform_params else None
        target_token = bending_meta.get('target_token', 'ALL')
        timestep_range = bending_meta.get('timestep_range', 'ALL')
        layer_range = bending_meta.get('layer_indices')
        
        # Normalize flip operations: flip_horizontal/flip_vertical -> flip
        normalized_operation = operation
        if operation in ['flip_horizontal', 'flip_vertical']:
            normalized_operation = 'flip'
        
        # Create composite key to handle multiple parameters per operation
        op_key = f"{operation}:{param_name}" if param_name else operation
        
        # Store in operations map
        op_data = operations_map[op_key]
        op_data['operation'] = normalized_operation
        op_data['parameter_name'] = param_name
        
        if param_value is not None:
            op_data['parameter_values'].add(param_value)
        
        op_data['target_tokens'].add(target_token)
        
        # Convert timestep_range to hashable format
        if timestep_range is None or timestep_range == 'ALL':
            timestep_str = 'ALL'
        elif isinstance(timestep_range, list) and len(timestep_range) == 2:
            timestep_str = f"{timestep_range[0]}-{timestep_range[1]}"
        else:
            timestep_str = str(timestep_range)
        op_data['timestep_ranges'].add(timestep_str)
        
        # Convert layer indices to range string
        if layer_range:
            if isinstance(layer_range, list):
                if len(layer_range) > 2:
                    # Check if continuous range
                    sorted_layers = sorted(layer_range)
                    if sorted_layers == list(range(sorted_layers[0], sorted_layers[-1] + 1)):
                        layer_str = f"{sorted_layers[0]}-{sorted_layers[-1]}"
                    else:
                        layer_str = str(sorted_layers)
                elif len(layer_range) == 1:
                    layer_str = str(layer_range[0])
                else:
                    layer_str = f"{layer_range[0]}-{layer_range[-1]}"
            else:
                layer_str = str(layer_range)
            op_data['layer_ranges'].add(layer_str)
        else:
            op_data['layer_ranges'].add('ALL')
        
        # Keep some examples
        if len(op_data['examples']) < 3:
            op_data['examples'].append({
                'display_name': bending_meta.get('display_name'),
                'filename': video.get('filename')
            })
    
    logger.info(f"Found {baseline_count} baseline videos (no bending)")
    logger.info(f"Found {len(operations_map)} unique operations")
    
    return dict(operations_map)


def reconstruct_variations_config(operations_map: Dict) -> Dict[str, Any]:
    """
    Reconstruct the attention_bending_variations config from analyzed data.
    Operations will be ordered by their first appearance in the metadata.
    
    Handles special cases:
    - If scale_x and scale_y exist with matching ranges, merge to scale_factor
    - Multiple parameters per operation (translate_x, translate_y)
    """
    # Start with baseline config
    config = {
        'enabled': True,
        'generate_baseline': True,  # Assume true if we found baseline videos
        'renormalize': False,  # Default, can't determine from metadata
        'operations': []
    }
    
    # Check for scale_x and scale_y to merge into scale_factor
    scale_operations = {}
    for op_key, data in operations_map.items():
        if data['operation'] == 'scale' and data['parameter_name'] in ['scale_x', 'scale_y']:
            scale_operations[data['parameter_name']] = {
                'key': op_key,
                'data': data,
                'values': sorted(list(data['parameter_values']))
            }
    
    # Determine if we should merge scale_x and scale_y
    merge_scale = False
    if 'scale_x' in scale_operations and 'scale_y' in scale_operations:
        scale_x_vals = scale_operations['scale_x']['values']
        scale_y_vals = scale_operations['scale_y']['values']
        # Merge if they have the same parameter values
        if scale_x_vals == scale_y_vals:
            merge_scale = True
    
    # Build operation configs, preserving order from operations_map
    # (which uses dict and preserves insertion order in Python 3.7+)
    scale_merged = False
    for op_key, data in operations_map.items():
        # Skip scale_y if we're merging (we'll use scale_x entry for merged version)
        if merge_scale and data['operation'] == 'scale':
            if data['parameter_name'] == 'scale_y':
                continue
            elif data['parameter_name'] == 'scale_x' and not scale_merged:
                # Use scale_factor instead
                op_config = {
                    'operation': data['operation'],
                    'parameter_name': 'scale_factor'
                }
                scale_merged = True
            else:
                # Other scale parameter (shouldn't happen but handle it)
                op_config = {
                    'operation': data['operation'],
                    'parameter_name': data['parameter_name']
                }
        else:
            op_config = {
                'operation': data['operation'],
                'parameter_name': data['parameter_name']
            }
        
        # Add parameter range if values were found
        param_values = sorted(list(data['parameter_values']))
        if param_values:
            if len(param_values) > 1:
                op_config['range'] = [min(param_values), max(param_values)]
                op_config['steps'] = len(param_values)
            # Note: For boolean operations (flip), parameter_value might be True/False
            # which won't have a range
        
        # Add target tokens as array of quoted strings
        # Rule: Always array, but omit field if only contains "ALL" (default)
        tokens = sorted(list(data['target_tokens']))
        if not (len(tokens) == 1 and tokens[0] == 'ALL'):
            # Include if multiple tokens OR if single token is not ALL
            op_config['target_token'] = tokens  # Will be formatted as array of quoted strings
        # If only "ALL", omit the field (it's the default)
        
        # Add timestep ranges as array of quoted strings
        # Rule: Always array if present, omit if only "ALL" (default)
        timesteps = sorted(list(data['timestep_ranges']))
        if not (len(timesteps) == 1 and timesteps[0] == 'ALL'):
            # Include if multiple ranges OR if single range is not ALL
            op_config['apply_to_timesteps'] = timesteps
        # If only ALL, omit the field (it's the default)
        
        # Add layer ranges as array of strings
        # Rule: Always array if present, omit if only "ALL" (default)
        layers = sorted(list(data['layer_ranges']))
        if not (len(layers) == 1 and layers[0] == 'ALL'):
            # Include if multiple layers OR if single layer is not ALL
            op_config['apply_to_layers'] = layers
        # If only ALL, omit the field (it's the default)
        
        # Default strength and padding
        op_config['strength'] = 1.0
        if data['operation'] in ['scale', 'rotate', 'translate', 'blur', 'sharpen']:
            op_config['padding_mode'] = 'border'
        
        config['operations'].append(op_config)
    
    return config


def save_reconstructed_config(config: Dict, output_path: Path):
    """Save reconstructed config to YAML file with proper formatting."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Custom YAML dumper to format ranges in flow style
        class CustomDumper(yaml.SafeDumper):
            def write_line_break(self, data=None):
                super().write_line_break(data)
        
        def represent_none(self, _):
            return self.represent_scalar('tag:yaml.org,2002:null', '')
        
        # Add custom representer to handle ranges and target_token strings
        def dict_representer(dumper, data):
            return dumper.represent_dict(data.items())
        
        CustomDumper.add_representer(type(None), represent_none)
        CustomDumper.add_representer(dict, dict_representer)
        
        # Format config to have proper string/list representations
        formatted_config = format_config_for_yaml(config)
        
        # First pass: dump to string
        yaml_str = yaml.dump(
            {'attention_bending_variations': formatted_config}, 
            Dumper=CustomDumper,
            default_flow_style=False, 
            indent=2, 
            sort_keys=False,
            width=float("inf")
        )
        
        # Post-process to fix formatting
        import re
        
        # Convert range arrays to flow style: range: [a, b]
        yaml_str = re.sub(
            r'range:\s*\n\s*-\s*(-?\d+(?:\.\d+)?)\s*\n\s*-\s*(-?\d+(?:\.\d+)?)',
            r'range: [\1, \2]',
            yaml_str
        )
        
        # Quote string values in arrays (ALL, layer ranges, timestep ranges)
        # Match unquoted ALL and quote it
        yaml_str = re.sub(
            r'- (ALL)(?=\s*\n)',
            r'- "\1"',
            yaml_str
        )
        
        # Quote timestep/layer range strings (e.g., 0-5, 13-18)
        yaml_str = re.sub(
            r'- (\d+-\d+)(?=\s*\n)',
            r'- "\1"',
            yaml_str
        )
        
        # Quote comma-separated token strings (e.g., "rose, horse, ship")
        # Match strings that contain commas
        yaml_str = re.sub(
            r"- ([a-zA-Z][a-zA-Z0-9, ]+)(?=\s*\n)",
            r'- "\1"',
            yaml_str
        )
        
        # Write final formatted YAML
        with open(output_path, 'w') as f:
            f.write(yaml_str)
            
        logger.info(f"✓ Saved reconstructed config to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save config: {e}")


def format_config_for_yaml(config: Dict) -> Dict:
    """Format config to ensure proper YAML output."""
    return config


def update_generation_config(generation_config_path: Path, variations_config: Dict):
    """Update existing generation_config.yaml with attention_bending_variations."""
    try:
        # Load existing config
        with open(generation_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Add or update attention_bending_variations
        config['attention_bending_variations'] = variations_config
        
        # Custom YAML dumper
        class CustomDumper(yaml.SafeDumper):
            pass
        
        def represent_none(self, _):
            return self.represent_scalar('tag:yaml.org,2002:null', '')
        
        CustomDumper.add_representer(type(None), represent_none)
        
        # First pass: dump to string
        yaml_str = yaml.dump(
            config, 
            Dumper=CustomDumper,
            default_flow_style=False, 
            indent=2, 
            sort_keys=False,
            width=float("inf")
        )
        
        # Post-process to fix formatting
        import re
        
        # Convert range arrays to flow style: range: [a, b]
        yaml_str = re.sub(
            r'range:\s*\n\s*-\s*(-?\d+(?:\.\d+)?)\s*\n\s*-\s*(-?\d+(?:\.\d+)?)',
            r'range: [\1, \2]',
            yaml_str
        )
        
        # Quote string values in arrays (ALL, layer ranges, timestep ranges)
        yaml_str = re.sub(
            r'- (ALL)(?=\s*\n)',
            r'- "\1"',
            yaml_str
        )
        yaml_str = re.sub(
            r'- (\d+-\d+)(?=\s*\n)',
            r'- "\1"',
            yaml_str
        )
        yaml_str = re.sub(
            r"- ([a-zA-Z][a-zA-Z0-9, ]+)(?=\s*\n)",
            r'- "\1"',
            yaml_str
        )
        
        # Write back with proper formatting
        with open(generation_config_path, 'w') as f:
            f.write(yaml_str)
        
        logger.info(f"✓ Updated {generation_config_path} with attention_bending_variations")
        return True
    except Exception as e:
        logger.error(f"Failed to update generation_config.yaml: {e}")
        return False


def backfill_experiment(experiment_path: Path, update_original: bool = False) -> bool:
    """
    Backfill attention_bending_variations for a single experiment.
    
    Args:
        experiment_path: Path to experiment directory
        update_original: If True, update the original generation_config.yaml
                        If False, only create a separate reconstructed config file
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Processing: {experiment_path.name}")
    logger.info(f"{'='*70}")
    
    # Check required files
    configs_dir = experiment_path / "configs"
    metadata_path = configs_dir / "video_metadata.json"
    generation_config_path = configs_dir / "generation_config.yaml"
    
    if not configs_dir.exists():
        logger.error(f"Configs directory not found: {configs_dir}")
        return False
    
    # Check for video_metadata.json or GPU-specific files
    metadata_files = []
    if metadata_path.exists():
        metadata_files = [metadata_path]
    else:
        # Look for GPU-specific metadata files
        gpu_files = list(configs_dir.glob("video_metadata_*_gpu*.json"))
        if gpu_files:
            logger.info(f"Found {len(gpu_files)} GPU-specific metadata files")
            metadata_files = gpu_files
        else:
            logger.error(f"No video_metadata.json files found in: {configs_dir}")
            return False
    
    # Load and merge video metadata from all files
    logger.info(f"Loading metadata from {len(metadata_files)} file(s)...")
    all_videos = []
    for mf in metadata_files:
        logger.info(f"  Loading: {mf.name}")
        metadata = load_video_metadata(mf)
        if metadata:
            videos = metadata.get('videos', [])
            all_videos.extend(videos)
            logger.info(f"    Found {len(videos)} videos")
    
    if not all_videos:
        logger.error("No videos found in metadata files")
        return False
    
    logger.info(f"Total videos loaded: {len(all_videos)}")
    
    # Analyze bending variations
    logger.info("\nAnalyzing bending variations...")
    operations_map = analyze_bending_variations(all_videos)
    
    if not operations_map:
        logger.warning("No bending variations found in metadata")
        return False
    
    # Print analysis summary
    logger.info(f"\n{'─'*70}")
    logger.info("ANALYSIS SUMMARY")
    logger.info(f"{'─'*70}")
    for operation, data in sorted(operations_map.items()):
        logger.info(f"\nOperation: {operation}")
        logger.info(f"  Parameter: {data['parameter_name']}")
        logger.info(f"  Values: {sorted(list(data['parameter_values']))}")
        logger.info(f"  Target tokens: {sorted(list(data['target_tokens']))}")
        logger.info(f"  Timestep ranges: {sorted(list(data['timestep_ranges']))}")
        logger.info(f"  Layer ranges: {sorted(list(data['layer_ranges']))}")
        logger.info(f"  Example videos:")
        for ex in data['examples']:
            logger.info(f"    - {ex['filename']}")
    
    # Reconstruct config
    logger.info(f"\n{'─'*70}")
    logger.info("RECONSTRUCTING CONFIG")
    logger.info(f"{'─'*70}")
    variations_config = reconstruct_variations_config(operations_map)
    
    # Save reconstructed config
    reconstructed_path = configs_dir / "attention_bending_variations_reconstructed.yaml"
    save_reconstructed_config(variations_config, reconstructed_path)
    
    # Optionally update original generation_config.yaml
    if update_original and generation_config_path.exists():
        logger.info("\nUpdating original generation_config.yaml...")
        update_generation_config(generation_config_path, variations_config)
    elif update_original:
        logger.warning(f"generation_config.yaml not found at {generation_config_path}")
    
    logger.info(f"\n{'='*70}")
    logger.info("✓ Backfill complete!")
    logger.info(f"{'='*70}\n")
    
    return True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nError: Missing experiment path")
        sys.exit(1)
    
    experiment_path = Path(sys.argv[1])
    
    # Check if path exists
    if not experiment_path.exists():
        logger.error(f"Experiment path not found: {experiment_path}")
        sys.exit(1)
    
    # Determine if we should update the original config
    update_original = '--update' in sys.argv or '-u' in sys.argv
    
    if update_original:
        logger.info("Mode: UPDATE - Will modify original generation_config.yaml")
    else:
        logger.info("Mode: SAFE - Will only create reconstructed config file")
        logger.info("  (Use --update or -u to modify original generation_config.yaml)")
    
    # Process experiment
    success = backfill_experiment(experiment_path, update_original)
    
    if success:
        logger.info("✓ SUCCESS")
        sys.exit(0)
    else:
        logger.error("✗ FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
