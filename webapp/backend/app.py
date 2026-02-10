#!/usr/bin/env python3
"""
WAN Video Matrix Viewer - Flask Backend API
Modern restructured version of the video viewer webapp
"""

import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify, send_file, render_template, send_from_directory
from flask_cors import CORS  # Install with: pip install flask-cors

# Import utilities (simplified - remove if not needed)
# from utils.file_utils import safe_filename


class VideoAnalyzer:
    """Analyzes video generation outputs and organizes them for the web interface"""
    
    def __init__(self, outputs_dir):
        self.outputs_dir = Path(outputs_dir)
        self.experiments = {}
    
    @staticmethod
    @staticmethod
    def apply_resolved_tokens_to_display(display_name, bending_metadata):
        """Update display name to show resolved tokens instead of full comma-separated spec."""
        if not bending_metadata:
            return display_name
        
        # Check if we have resolved tokens
        resolved_tokens = bending_metadata.get('resolved_tokens', {})
        target_token = bending_metadata.get('target_token')
        
        # If no comma-separated tokens or no resolution, return as-is
        if not target_token or ',' not in target_token or not resolved_tokens:
            return display_name
        
        # Get the resolved list for this token spec
        resolved_list = resolved_tokens.get(target_token, [])
        
        # Build the display string
        if resolved_list:
            resolved_str = ', '.join(resolved_list)
        else:
            resolved_str = '(none active)'
        
        # Replace the full token spec with resolved tokens in the display name
        # The display_name format is "Token: <tokens> | Operation..." 
        updated_display = display_name.replace(target_token, resolved_str)
        
        return updated_display
    
    @staticmethod
    def format_bending_label(bending_id):
        """Convert technical bending_id to human-readable label."""
        if bending_id == 'baseline' or bending_id is None:
            return 'Baseline (No Bending)'
        
        # Parse the bending_id: operation_value_[token]_tTIMESTEPS_lLAYERS
        # Handle compound operations like flip_horizontal, translate_x
        parts = bending_id.split('_')
        if len(parts) < 2:
            return bending_id  # Return as-is if can't parse
        
        # Check for compound operation names (flip_horizontal, flip_vertical, translate_x, translate_y)
        if parts[0] in ['flip', 'translate'] and len(parts) > 2 and parts[1] in ['horizontal', 'vertical', 'x', 'y']:
            operation = f"{parts[0]}_{parts[1]}"
            value = parts[2] if len(parts) > 2 else ''
            value_offset = 3  # Parts after value start at index 3
        else:
            operation = parts[0]
            value = parts[1] if len(parts) > 1 else ''
            value_offset = 2  # Parts after value start at index 2
        
        # Extract token, timestep and layer info
        token = None
        timesteps = None
        layers = None
        
        for i, part in enumerate(parts[value_offset:], start=value_offset):
            if part.startswith('t'):
                timesteps = part[1:]  # Remove 't' prefix
            elif part.startswith('l'):
                # Handle remaining parts as layer spec
                layer_part = '_'.join(parts[i:])  # Rejoin in case of 'l14,15'
                layers = layer_part[1:]  # Remove 'l' prefix
                break
            elif not part.startswith('t') and not part.startswith('l'):
                # This is the token (appears before timestep/layer specs)
                # For comma-separated tokens, rejoin parts that don't start with 't' or 'l'
                token_parts = [part]
                # Check if next parts are also token parts (for comma-separated like "kiss, rose, ship")
                j = i + 1
                while j < len(parts) and not parts[j].startswith('t') and not parts[j].startswith('l'):
                    token_parts.append(parts[j])
                    j += 1
                token = '_'.join(token_parts)  # Rejoin token parts
                # Skip ahead past token parts
                for _ in range(len(token_parts) - 1):
                    value_offset += 1
        
        # Format operation-specific labels
        op_formatters = {
            'scale': lambda v: f"Scale {v}×",
            'rotate': lambda v: f"Rotate {v}°",
            'translate_x': lambda v: f"Translate X {v}",
            'translate_y': lambda v: f"Translate Y {v}",
            'flip_horizontal': lambda v: f"Flip H: {v}",
            'flip_vertical': lambda v: f"Flip V: {v}",
            'amplify': lambda v: f"Amplify {v}×",
        }
        
        formatter = op_formatters.get(operation, lambda v: f"{operation.capitalize()} {v}")
        
        label_parts = []
        
        # Add token info FIRST (only if not "ALL" - for brevity)
        if token and token != 'ALL':
            label_parts.append(f"Token: {token}")
        
        # Add operation and value
        label_parts.append(formatter(value))
        
        if timesteps:
            if timesteps == 'ALL':
                label_parts.append("All Steps")
            else:
                label_parts.append(f"Steps {timesteps}")
        
        if layers:
            if layers == 'ALL':
                label_parts.append("All Layers")
            elif ',' in layers:
                label_parts.append(f"Layers {layers}")
            else:
                label_parts.append(f"Layer {layers}")
        
        return ' | '.join(label_parts)
        
    def scan_outputs(self, summary_only=False):
        """
        Scan the outputs directory and build hierarchical experiment tree.
        
        Args:
            summary_only: If True, only load minimal data for tree view (fast).
                         If False, load full experiment data (slow but complete).
        """
        print(f"Scanning outputs directory: {self.outputs_dir} (summary_only={summary_only})")
        
        if not self.outputs_dir.exists():
            print(f"Outputs directory not found: {self.outputs_dir}")
            return {"type": "folder", "name": "outputs", "path": "", "children": []}
        
        # Build the tree structure
        tree = self._build_tree(self.outputs_dir, "", summary_only=summary_only)
        print(f"Built experiment tree with {self._count_experiments(tree)} experiments")
        return tree
    
    def _build_tree(self, directory, relative_path, summary_only=False):
        """
        Recursively build tree structure from directory.
        
        Args:
            directory: Directory to scan
            relative_path: Path relative to outputs root
            summary_only: If True, use fast summary analysis instead of full analysis
        """
        node = {
            "type": "folder",
            "name": directory.name or "outputs",
            "path": relative_path,
            "children": [],
            "created_timestamp": directory.stat().st_ctime
        }
        
        items = []
        
        # Collect all items in this directory
        for item in directory.iterdir():
            if item.name.startswith('.'):
                continue
                
            if item.is_dir():
                item_relative_path = str(Path(relative_path) / item.name) if relative_path else item.name
                
                # Try to analyze as experiment first (using appropriate method)
                if summary_only:
                    exp_data = self._analyze_experiment_summary(item)
                else:
                    exp_data = self._analyze_experiment(item)
                if exp_data:
                    # This is an experiment directory
                    experiment_node = {
                        "type": "experiment",
                        "name": item.name,
                        "path": item_relative_path,
                        "created_timestamp": item.stat().st_ctime,
                        "experiment_data": exp_data
                    }
                    items.append(experiment_node)
                else:
                    # This is a regular folder, recurse into it
                    folder_node = self._build_tree(item, item_relative_path, summary_only=summary_only)
                    # Only include folders that have experiments (directly or in subfolders)
                    if self._has_experiments(folder_node):
                        items.append(folder_node)
        
        # Sort: folders first, then experiments, both by creation date (newest first)
        folders = [item for item in items if item["type"] == "folder"]
        experiments = [item for item in items if item["type"] == "experiment"]
        
        folders.sort(key=lambda x: x["created_timestamp"], reverse=True)
        experiments.sort(key=lambda x: x["created_timestamp"], reverse=True)
        
        node["children"] = folders + experiments
        return node
    
    def _has_experiments(self, node):
        """Check if a folder node contains any experiments (recursively)"""
        if node["type"] == "experiment":
            return True
        
        for child in node.get("children", []):
            if self._has_experiments(child):
                return True
        
        return False
    
    def _count_experiments(self, node):
        """Count total experiments in tree"""
        if node["type"] == "experiment":
            return 1
        
        count = 0
        for child in node.get("children", []):
            count += self._count_experiments(child)
        
        return count
    
    def get_experiment_by_path(self, experiment_path):
        """Get experiment data by hierarchical path"""
        tree = self.scan_outputs()
        return self._find_experiment_in_tree(tree, experiment_path.split('/'))
    
    def _find_experiment_in_tree(self, node, path_parts):
        """Find experiment in tree by path parts"""
        if not path_parts:
            return None
            
        if len(path_parts) == 1:
            # Last part - look for experiment
            for child in node.get("children", []):
                if child["type"] == "experiment" and child["name"] == path_parts[0]:
                    return child["experiment_data"]
        else:
            # Look for folder and recurse
            for child in node.get("children", []):
                if child["type"] == "folder" and child["name"] == path_parts[0]:
                    return self._find_experiment_in_tree(child, path_parts[1:])
        
        return None
    
    def _analyze_experiment_summary(self, exp_dir):
        """
        Fast experiment analysis for tree view - only config files, no video enumeration.
        Returns minimal data needed for sidebar display and filtering.
        """
        try:
            # Get creation timestamp from filesystem
            creation_time = exp_dir.stat().st_ctime
            creation_datetime = datetime.fromtimestamp(creation_time)
            
            # Try to load base prompt from prompt_template.txt first (fast)
            prompt_template_path = exp_dir / 'configs' / 'prompt_template.txt'
            base_prompt = "Unknown prompt"
            
            if prompt_template_path.exists():
                try:
                    with open(prompt_template_path, 'r') as f:
                        base_prompt = f.read().strip()
                except Exception:
                    pass
            
            # Load basic config to check if this is a valid experiment
            config_path = exp_dir / 'configs' / 'generation_config.yaml'
            if not config_path.exists():
                return None  # Not a valid experiment
            
            # Read minimal config data
            model_id = "Unknown model"
            duration_seconds = None
            attention_bending_settings = None
            
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    
                    # Get base prompt from config if not found in template file
                    if base_prompt == "Unknown prompt":
                        base_prompt = config.get('base_prompt', 'Unknown prompt')
                    
                    model_settings = config.get('model_settings', {})
                    model_id = model_settings.get('model_id', 'Unknown model')
                    
                    # Extract duration
                    video_settings = config.get('video_settings', {})
                    duration_seconds = video_settings.get('duration')
                    if duration_seconds is None:
                        frames = video_settings.get('frames')
                        fps = video_settings.get('fps')
                        if frames and fps:
                            try:
                                duration_seconds = float(frames) / float(fps)
                            except Exception:
                                duration_seconds = None
                    
                    # Check attention bending (just enabled flag)
                    attention_bending_raw = config.get('attention_bending_settings')
                    if attention_bending_raw and attention_bending_raw.get('enabled'):
                        attention_bending_settings = {'enabled': True}
            except Exception as e:
                print(f"Warning: Could not parse config for {exp_dir.name}: {e}")
                return None
            
            # Fast video count - just count files without parsing
            videos_dir = exp_dir / 'videos'
            videos_count = 0
            if videos_dir.exists():
                # Count .mp4 files (both old and new format)
                videos_count = len(list(videos_dir.glob('**/*.mp4')))
            
            if videos_count == 0:
                return None  # Skip experiments with no videos
            
            # Fast checks for analysis availability (just directory/file existence)
            vlm_analysis_dir = exp_dir / 'vlm_analysis'
            has_vlm_analysis = vlm_analysis_dir.exists() and any(vlm_analysis_dir.glob('prompt_*/aggregated_results.json'))
            
            trajectory_analysis_dir = exp_dir / 'latent_trajectory_analysis'
            has_trajectory_analysis = trajectory_analysis_dir.exists()
            
            # Check for attention videos
            attention_videos_dir = exp_dir / 'attention_videos'
            attention_videos_available = attention_videos_dir.exists() and len(list(attention_videos_dir.glob('**/*.mp4'))) > 0
            
            return {
                'name': exp_dir.name,
                'base_prompt': base_prompt,
                'model_id': model_id,
                'videos_count': videos_count,
                'duration_seconds': duration_seconds,
                'path': str(exp_dir),
                'created_at': creation_datetime.isoformat(),
                'created_timestamp': creation_time,
                'has_vlm_analysis': has_vlm_analysis,
                'has_trajectory_analysis': has_trajectory_analysis,
                'attention_videos': {'available': attention_videos_available} if attention_videos_available else None,
                'attention_bending_settings': attention_bending_settings
            }
            
        except Exception as e:
            print(f"Error analyzing experiment summary {exp_dir.name}: {e}")
            return None
    
    def _analyze_experiment(self, exp_dir):
        """Analyze a single experiment directory"""
        print(f"\n{'='*70}")
        print(f"🔍 Analyzing experiment: {exp_dir.name}")
        print(f"{'='*70}")
        try:
            # Get creation timestamp from filesystem
            creation_time = exp_dir.stat().st_ctime
            creation_datetime = datetime.fromtimestamp(creation_time)
            
            # Try to load base prompt from prompt_template.txt first
            prompt_template_path = exp_dir / 'configs' / 'prompt_template.txt'
            base_prompt = "Unknown prompt"
            model_id = "Unknown model"
            
            if prompt_template_path.exists():
                with open(prompt_template_path, 'r') as f:
                    base_prompt = f.read().strip()
            
            # Load model info and other config from YAML
            config_path = exp_dir / 'configs' / 'generation_config.yaml'
            duration_seconds = None
            cfg_scale = None
            cfg_schedule_settings = None
            cfg_schedule_data = None
            attention_bending_settings = None
            video_settings = {}  # Initialize with empty dict
            model_settings = {}  # Initialize with empty dict
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    if not prompt_template_path.exists():
                        base_prompt = config.get('base_prompt', 'Unknown prompt')
                    model_id = config.get('model_settings', {}).get('model_id', 'Unknown model')
                    
                    # Extract CFG scale from model_settings
                    model_settings = config.get('model_settings', {})
                    cfg_scale = model_settings.get('cfg_scale')
                    
                    # Extract CFG schedule settings if present
                    cfg_schedule_settings = config.get('cfg_schedule_settings')
                    
                    # Extract attention bending settings if present
                    attention_bending_raw = config.get('attention_bending_settings')
                    if attention_bending_raw and attention_bending_raw.get('enabled'):
                        # Extract and format attention bending info - pass through ALL parameters
                        configs = attention_bending_raw.get('configs', [])
                        attention_bending_settings = {
                            'enabled': True,
                            'apply_to_output': attention_bending_raw.get('apply_to_output', False),
                            'num_configs': len(configs),
                            'configs': [
                                {
                                    # Core identification
                                    'token': cfg.get('token'),
                                    'mode': cfg.get('mode'),
                                    
                                    # Pass through all parameters dynamically
                                    # Amplify parameters (for AMPLIFY mode)
                                    'amplify_factor': cfg.get('amplify_factor'),
                                    
                                    # Spatial transformation parameters (scale_factor for SCALE mode)
                                    'angle': cfg.get('angle'),
                                    'crop_rotated': cfg.get('crop_rotated'),
                                    'translate_x': cfg.get('translate_x'),
                                    'translate_y': cfg.get('translate_y'),
                                    'flip_horizontal': cfg.get('flip_horizontal'),
                                    'flip_vertical': cfg.get('flip_vertical'),
                                    'scale_factor': cfg.get('scale_factor'),
                                    
                                    # Blur/sharpen parameters
                                    'kernel_size': cfg.get('kernel_size'),
                                    'sigma': cfg.get('sigma'),
                                    'sharpen_amount': cfg.get('sharpen_amount'),
                                    
                                    # Regional mask parameters
                                    'region': cfg.get('region'),
                                    'region_feather': cfg.get('region_feather'),
                                    
                                    # Control parameters
                                    'strength': cfg.get('strength'),
                                    'apply_to_layers': cfg.get('apply_to_layers'),
                                    'apply_to_timesteps': cfg.get('apply_to_timesteps'),
                                    'padding_mode': cfg.get('padding_mode'),
                                    
                                    # Stability parameters
                                    'renormalize': cfg.get('renormalize'),
                                    'preserve_sparsity': cfg.get('preserve_sparsity'),
                                }
                                for cfg in configs
                            ]
                        }
                    
                    # Extract duration from config
                    video_settings = config.get('video_settings', {})
            
            # Load detailed CFG schedule data if available
            cfg_schedule_file = exp_dir / 'configs' / 'cfg_schedule.json'
            if cfg_schedule_file.exists():
                try:
                    with open(cfg_schedule_file, 'r') as f:
                        cfg_schedule_data = json.load(f)
                except Exception as e:
                    print(f"Failed to load CFG schedule data: {e}")
                    cfg_schedule_data = None
            
            # Load prompt schedule data if available
            prompt_schedule_data = None
            prompt_schedule_file = exp_dir / 'configs' / 'prompt_schedule.json'
            if prompt_schedule_file.exists():
                try:
                    with open(prompt_schedule_file, 'r') as f:
                        prompt_schedule_data = json.load(f)
                except Exception as e:
                    print(f"Failed to load prompt schedule data: {e}")
                    prompt_schedule_data = None
            
            # Extract duration from config
            duration_seconds = video_settings.get('duration')
            if duration_seconds is None:
                frames = video_settings.get('frames')
                fps = video_settings.get('fps')
                if frames and fps:
                    try:
                        duration_seconds = float(frames) / float(fps)
                    except Exception:
                        duration_seconds = None
            
            # Load prompt variations to get actual variation names
            variations_data = {}
            variations_path = exp_dir / 'configs' / 'prompt_variations.json'
            if variations_path.exists():
                with open(variations_path, 'r') as f:
                    variations_list = json.load(f)
                    # Create a mapping from index to variation data
                    for i, variation in enumerate(variations_list):
                        variations_data[str(i).zfill(3)] = variation  # "000", "001", etc.
            
            # Load video metadata to get bending display names
            video_metadata_map = {}
            metadata_file = exp_dir / 'configs' / 'video_metadata.json'
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        # Create map from video_id to metadata
                        for video in metadata.get('videos', []):
                            video_id = video.get('video_id')
                            if video_id:
                                video_metadata_map[video_id] = {
                                    'bending_metadata': video.get('bending_metadata'),
                                    'prompt_variation': video.get('prompt_variation', {}).get('text')
                                }
                except Exception as e:
                    print(f"Warning: Could not load video metadata for {exp_dir.name}: {e}")
            
            # Find all videos
            videos_dir = exp_dir / 'videos'
            videos = []
            
            if videos_dir.exists():
                print(f"📂 Scanning videos directory: {videos_dir}")
                
                # Check for NEW structure (videos directly in videos/ directory)
                # Filename format: video_0001_p00_baseline_s00.mp4
                direct_videos = list(videos_dir.glob('video_*.mp4'))
                if direct_videos:
                    print(f"  ✓ Found {len(direct_videos)} videos in NEW format (direct in videos/)")
                    for video_file in direct_videos:
                        video_info = self._extract_video_metadata_new_format(
                            video_file,
                            variations_data,
                            video_settings=video_settings,
                            model_settings=model_settings,
                            cfg_scale=cfg_scale,
                            video_metadata_map=video_metadata_map
                        )
                        if video_info:
                            videos.append(video_info)
                        else:
                            print(f"  ⚠ Failed to extract metadata from: {video_file.name}")
                
                # Check for OLD structure (videos in prompt_XXX subdirectories)
                prompt_dirs = [d for d in videos_dir.iterdir() if d.is_dir() and d.name.startswith('prompt_')]
                if prompt_dirs:
                    print(f"  ✓ Found {len(prompt_dirs)} prompt subdirectories in OLD format")
                    for prompt_dir in prompt_dirs:
                        variation_num = prompt_dir.name.split('_')[1]
                        
                        # Get the actual variation text from the JSON data
                        variation_info = variations_data.get(variation_num, {})
                        
                        # Use the full 'text' field for complete variation text (untruncated)
                        # Falls back to 'prompt' field, then 'var_0', then generic text
                        variation_text = (
                            variation_info.get('text', '') or 
                            variation_info.get('prompt', '') or
                            variation_info.get('variables', {}).get('var_0', f"Variation {variation_num}")
                        )
                        
                        variation_id = variation_info.get('id', f"variation_{variation_num}")
                        
                        # Find video files in this prompt directory
                        for video_file in prompt_dir.glob('video_*.mp4'):
                            video_info = self._extract_video_metadata(
                                video_file, 
                                variation_num, 
                                variation_text, 
                                variation_id,
                                video_settings=video_settings,
                                model_settings=model_settings,
                                cfg_scale=cfg_scale
                            )
                            if video_info:
                                videos.append(video_info)
                
                if not direct_videos and not prompt_dirs:
                    print(f"  ⚠ No videos found in either OLD or NEW format")
                    print(f"  Directory contents: {[item.name for item in videos_dir.iterdir()]}")
            
            if not videos:
                print(f"  ✗ No videos found - experiment will be filtered out")
                print(f"  Checked: {videos_dir}")
                return None
            
            print(f"  ✓ Total videos found: {len(videos)}")
                
            # Organize videos by variation and seed
            video_grid = self._organize_videos(videos)
            
            # Extract unique seeds and variations (by variation_num to avoid duplicates)
            seeds = sorted(list(set(v['seed'] for v in videos)))
            unique_variations = {}
            for v in videos:
                unique_variations[v['variation_num']] = v['variation']
            variations = [unique_variations[var_num] for var_num in sorted(unique_variations.keys())]
            
            # Scan for attention videos
            attention_videos = self._scan_attention_videos(exp_dir)
            
            # Check for VLM analysis
            vlm_analysis_dir = exp_dir / 'vlm_analysis'
            has_vlm_analysis = vlm_analysis_dir.exists() and any(vlm_analysis_dir.glob('prompt_*/aggregated_results.json'))
            
            # Check for trajectory analysis
            trajectory_analysis_dir = exp_dir / 'latent_trajectory_analysis'
            has_trajectory_analysis = trajectory_analysis_dir.exists()

            # Check for similarity analysis
            has_similarity_analysis, similarity_analysis_data = self._load_similarity_analysis(exp_dir)

            # Check for latent videos
            has_latent_videos = self._validate_latent_videos(exp_dir)
            
            # Check for attention videos
            has_attention_videos = self._validate_attention_videos(exp_dir)
            
            result = {
                'name': exp_dir.name,
                'base_prompt': base_prompt,
                'model_id': model_id,
                # 'videos': videos,  # REMOVED: 100% redundant with video_grid, saves 47.6% payload size
                'video_grid': video_grid,
                'seeds': seeds,
                'variations': variations,
                'videos_count': len(videos),
                'variations_count': len(variations),
                'seeds_count': len(seeds),
                'duration_seconds': duration_seconds,
                'cfg_scale': cfg_scale,
                'cfg_schedule_settings': cfg_schedule_settings,
                'cfg_schedule_data': cfg_schedule_data,
                'prompt_schedule_data': prompt_schedule_data,
                'attention_bending_settings': attention_bending_settings,
                'path': str(exp_dir),
                'created_at': creation_datetime.isoformat(),
                'created_timestamp': creation_time,
                'attention_videos': attention_videos,
                'has_vlm_analysis': has_vlm_analysis,
                'has_trajectory_analysis': has_trajectory_analysis,
                'has_similarity_analysis': has_similarity_analysis,
                'has_latent_videos': has_latent_videos,
                'has_attention_videos': has_attention_videos
            }
            
            # Add similarity analysis data if available
            if has_similarity_analysis and similarity_analysis_data:
                result['similarity_analysis'] = similarity_analysis_data
            
            print(f"  ✓ Experiment analysis complete:")
            print(f"    - Videos: {len(videos)}")
            print(f"    - Variations: {len(variations)}")
            print(f"    - Seeds: {len(seeds)}")
            print(f"{'='*70}\n")
            
            return result
            
        except Exception as e:
            print(f"  ✗ Error analyzing experiment {exp_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            print(f"{'='*70}\n")
            return None
    
    def _extract_video_metadata_new_format(self, video_path, variations_data, 
                                          video_settings=None, model_settings=None, cfg_scale=None,
                                          video_metadata_map=None):
        """
        Extract metadata from NEW filename format.
        
        Format: video_{num:04d}_p{prompt:02d}_{bending_id}_s{seed:02d}.mp4
        Examples:
          - video_0001_p00_baseline_s00.mp4
          - video_0002_p00_scale_0.75_t0-10_lALL_s00.mp4
        """
        try:
            filename = video_path.stem
            parts = filename.split('_')
            
            # Parse: ['video', '0001', 'p00', 'baseline', 's00'] or
            #        ['video', '0002', 'p00', 'scale', '0.75', 't0-10', 'lALL', 's00']
            if len(parts) < 5:
                print(f"    ⚠ Invalid NEW format filename (too few parts): {filename}")
                return None
            
            video_number = int(parts[1])  # '0001' -> 1
            prompt_idx = int(parts[2][1:])  # 'p00' -> 0
            
            # Find seed offset (last part starting with 's')
            seed_offset = 0
            for i in range(len(parts) - 1, -1, -1):
                if parts[i].startswith('s'):
                    try:
                        seed_offset = int(parts[i][1:])
                        break
                    except ValueError:
                        pass
            
            # Bending ID is everything between prompt and seed
            bending_parts = parts[3:-1]  # Everything between p00 and s00
            bending_id = '_'.join(bending_parts)
            is_baseline = bending_id == 'baseline'
            
            # Get variation info from variations_data
            variation_num = f"{prompt_idx:03d}"
            variation_info = variations_data.get(variation_num, {})
            variation_text = (
                variation_info.get('text', '') or 
                variation_info.get('prompt', '') or
                f"Prompt {prompt_idx}"
            )
            variation_id = variation_info.get('id', f"variation_{variation_num}")
            
            # Extract settings
            video_settings = video_settings or {}
            model_settings = model_settings or {}
            
            width = video_settings.get('width', 1024)
            height = video_settings.get('height', 576)
            num_frames = video_settings.get('frames', 25)
            steps = model_settings.get('steps', 20)
            cfg_scale_value = cfg_scale if cfg_scale is not None else model_settings.get('cfg_scale', 6.5)
            
            # Calculate seed
            base_seed = model_settings.get('seed', 42)
            actual_seed = base_seed + seed_offset
            
            # Create video_id for metadata lookup (3-digit format)
            video_id = f"p{prompt_idx:03d}_b{video_number - 1:03d}_s{seed_offset:03d}"
            
            # Get readable label from bending_metadata if available
            readable_label = self.format_bending_label(bending_id)
            if video_metadata_map and video_id in video_metadata_map:
                bending_meta = video_metadata_map[video_id].get('bending_metadata', {})
                if bending_meta and 'display_name' in bending_meta:
                    readable_label = bending_meta['display_name']
                    # Apply resolved tokens to update the display
                    readable_label = self.apply_resolved_tokens_to_display(readable_label, bending_meta)
            
            metadata = {
                'video_path': str(video_path.relative_to(self.outputs_dir)),
                'variation': bending_id if bending_id else 'baseline',
                'variation_text': readable_label,  # Human-readable label
                'variation_id': variation_id,
                'variation_num': variation_num,
                'filename': video_path.name,
                'video_number': video_number,
                'seed': actual_seed,
                'seed_offset': seed_offset,
                'steps': steps,
                'cfg_scale': cfg_scale_value,
                'width': width,
                'height': height,
                'num_frames': num_frames,
                'bending_id': bending_id,
                'is_baseline': is_baseline,
                'format': 'new'  # Flag to indicate new format
            }
            
            print(f"    ✓ Extracted: video={video_number}, prompt={prompt_idx}, bending={bending_id}, seed={actual_seed}")
            return metadata
            
        except Exception as e:
            print(f"    ✗ Error extracting NEW format metadata from {video_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_video_metadata(self, video_path, variation_num, variation_text, variation_id, 
                               video_settings=None, model_settings=None, cfg_scale=None):
        """Extract metadata from video filename and path"""
        try:
            # Parse filename for metadata (format: video_001.mp4, video_002.mp4, etc.)
            filename = video_path.stem
            
            # Extract video number from filename (video_001 -> 001)
            video_number = 1  # default
            if filename.startswith('video_'):
                try:
                    video_number = int(filename.split('_')[1])
                except (IndexError, ValueError):
                    video_number = 1
            
            # Extract actual values from config or use defaults
            video_settings = video_settings or {}
            model_settings = model_settings or {}
            
            width = video_settings.get('width', 1024)
            height = video_settings.get('height', 576)
            num_frames = video_settings.get('frames', 25)
            steps = model_settings.get('steps', 20)
            cfg_scale_value = cfg_scale if cfg_scale is not None else model_settings.get('cfg_scale', 6.5)
            
            # Calculate actual seed: starting_seed + (video_number - 1)
            # video_number is 1-indexed, so subtract 1 to get 0-indexed offset
            base_seed = model_settings.get('seed', 999)  # Default seed is 999
            actual_seed = base_seed + (video_number - 1)
            
            metadata = {
                'video_path': str(video_path.relative_to(self.outputs_dir)),
                'variation': variation_text,  # Use actual variation text instead of generic "Variation X"
                'variation_id': variation_id,  # Add variation ID for reference
                'variation_num': variation_num,  # Keep the numeric identifier
                'filename': video_path.name,
                'video_number': video_number,
                'seed': actual_seed,  # Calculated from base_seed + (video_number - 1)
                'steps': steps,  # From model_settings
                'cfg_scale': cfg_scale_value,  # From model_settings or passed parameter
                'width': width,  # From video_settings
                'height': height,  # From video_settings
                'num_frames': num_frames  # From video_settings
            }
            
            return metadata
            
        except Exception as e:
            print(f"Error extracting metadata from {video_path}: {e}")
            return None
    
    def _organize_videos(self, videos):
        """Organize videos into a grid structure by variation and seed"""
        # For NEW format videos with bending variations, organize by variation AND bending_id
        # Group by (variation_num, bending_id) to keep bending variations separate
        variations = {}
        
        for video in videos:
            var_num = video['variation_num']
            bending_id = video.get('bending_id', 'baseline')  # Default to baseline for old format
            
            # Create composite key for grouping
            group_key = (var_num, bending_id)
            
            if group_key not in variations:
                # Use the formatted variation_text from metadata if available
                # This contains the human-readable formatted label from format_bending_label()
                display_text = video.get('variation_text', video['variation'])
                
                variations[group_key] = {
                    'videos': [],
                    'variation_text': display_text,
                    'variation_num': var_num,
                    'bending_id': bending_id,
                    'is_baseline': video.get('is_baseline', True)
                }
            variations[group_key]['videos'].append(video)
        
        # Create grid structure, sorted by variation number, then bending_id
        # This keeps baseline first, then bending variations
        grid = []
        for group_key in sorted(variations.keys(), key=lambda k: (k[0], '' if variations[k]['is_baseline'] else k[1])):
            row = {
                'variation': variations[group_key]['variation_text'],
                'variation_num': variations[group_key]['variation_num'],
                'bending_id': variations[group_key]['bending_id'],
                'is_baseline': variations[group_key]['is_baseline'],
                'videos': sorted(variations[group_key]['videos'], key=lambda x: x['seed'])
            }
            grid.append(row)
            
        return grid
    
    def _scan_attention_videos(self, exp_dir):
        """Scan experiment directory for attention videos and return metadata"""
        attention_videos_dir = exp_dir / 'attention_videos'
        
        if not attention_videos_dir.exists():
            return {
                'available': False,
                'prompts': {},
                'total_count': 0
            }
        
        attention_data = {
            'available': True,
            'prompts': {},
            'total_count': 0
        }
        
        try:
            # Scan each prompt directory
            for prompt_dir in sorted(attention_videos_dir.iterdir()):
                if not prompt_dir.is_dir() or not prompt_dir.name.startswith('prompt_'):
                    continue
                
                prompt_id = prompt_dir.name
                attention_data['prompts'][prompt_id] = {
                    'videos': {}
                }
                
                # Scan each video directory within the prompt
                for video_dir in sorted(prompt_dir.iterdir()):
                    if not video_dir.is_dir():
                        continue
                    
                    # Support both old format (vid001) and new format (p000_b001_s000)
                    video_id = video_dir.name
                    if video_id.startswith('vid'):
                        # Old format: vid001 -> video_num = 1
                        video_num = int(video_id.replace('vid', '').lstrip('0') or '1')
                    elif video_id.startswith('p') and '_b' in video_id and '_s' in video_id:
                        # New format: p000_b001_s000 -> extract bending variation number
                        # p000_b001_s000 -> bending_num = 1
                        try:
                            bending_part = video_id.split('_b')[1].split('_')[0]
                            video_num = int(bending_part.lstrip('0') or '0')
                        except:
                            video_num = 0
                    else:
                        # Unknown format, skip
                        continue
                    
                    attention_data['prompts'][prompt_id]['videos'][video_id] = {
                        'video_number': video_num,
                        'tokens': {}
                    }
                    
                    # Scan each token directory
                    for token_dir in sorted(video_dir.iterdir()):
                        if not token_dir.is_dir() or not token_dir.name.startswith('token_'):
                            continue
                        
                        token_name = token_dir.name.replace('token_', '')
                        
                        # Check for attention video files
                        aggregate_overlay = token_dir / 'aggregate_overlay.mp4'
                        aggregate_attention = token_dir / 'aggregate_attention.mp4'
                        
                        if aggregate_overlay.exists():
                            # Create relative path from outputs directory
                            rel_path = aggregate_overlay.relative_to(self.outputs_dir)
                            attention_data['prompts'][prompt_id]['videos'][video_id]['tokens'][token_name] = {
                                'aggregate_overlay_path': str(rel_path),
                                'has_aggregate_attention': aggregate_attention.exists()
                            }
                            
                            if aggregate_attention.exists():
                                rel_attention_path = aggregate_attention.relative_to(self.outputs_dir)
                                attention_data['prompts'][prompt_id]['videos'][video_id]['tokens'][token_name]['aggregate_attention_path'] = str(rel_attention_path)
                            
                            attention_data['total_count'] += 1
        
        except Exception as e:
            print(f"Error scanning attention videos for {exp_dir.name}: {e}")
            return {
                'available': False,
                'prompts': {},
                'total_count': 0,
                'error': str(e)
            }
        
        return attention_data
    
    def _validate_attention_videos(self, exp_dir):
        """Validate attention_videos folder structure and check for completeness"""
        attention_videos_dir = exp_dir / 'attention_videos'
        
        if not attention_videos_dir.exists():
            return False
        
        try:
            # Check if there are prompt subdirectories
            prompt_dirs = [d for d in attention_videos_dir.iterdir() if d.is_dir() and d.name.startswith('prompt_')]
            
            if not prompt_dirs:
                return False
            
            # For each prompt directory, validate structure
            for prompt_dir in prompt_dirs:
                # Check for video subdirectories
                video_dirs = [d for d in prompt_dir.iterdir() if d.is_dir() and d.name.startswith('vid')]
                
                if not video_dirs:
                    continue  # Skip prompts with no videos
                
                # For each video directory, check for token directories
                for video_dir in video_dirs:
                    token_dirs = [d for d in video_dir.iterdir() if d.is_dir() and d.name.startswith('token_')]
                    
                    if not token_dirs:
                        continue  # Skip videos with no tokens
                    
                    # For each token directory, check for step files
                    for token_dir in token_dirs:
                        step_files = list(token_dir.glob('step_*.mp4'))
                        
                        if not step_files:
                            continue  # Skip tokens with no step files
                        
                        # Check that each mp4 has a corresponding jpg
                        for mp4_file in step_files:
                            jpg_file = mp4_file.with_suffix('.jpg')
                            if not jpg_file.exists():
                                return False  # Missing corresponding image
            
            # If we made it here, structure is valid
            return True
            
        except Exception as e:
            print(f"Error validating attention videos for {exp_dir.name}: {e}")
            return False
    
    def _validate_latent_videos(self, exp_dir):
        """Validate latents_videos folder structure and check for completeness"""
        latent_videos_dir = exp_dir / 'latents_videos'  # Note: using 'latents_videos' (plural)
        
        if not latent_videos_dir.exists():
            return False
        
        try:
            # Check if there are prompt subdirectories
            prompt_dirs = [d for d in latent_videos_dir.iterdir() if d.is_dir() and d.name.startswith('prompt_')]
            
            if not prompt_dirs:
                return False
            
            # For each prompt directory, validate structure
            for prompt_dir in prompt_dirs:
                # Check for video subdirectories
                video_dirs = [d for d in prompt_dir.iterdir() if d.is_dir() and d.name.startswith('vid_')]
                
                if not video_dirs:
                    continue  # Skip prompts with no videos
                
                # For each video directory, check for step files
                for video_dir in video_dirs:
                    step_files = list(video_dir.glob('step_*.mp4'))
                    
                    if not step_files:
                        continue  # Skip videos with no step files
                    
                    # Check that each mp4 has a corresponding jpg
                    for mp4_file in step_files:
                        jpg_file = mp4_file.with_suffix('.jpg')
                        if not jpg_file.exists():
                            return False  # Missing corresponding image
            
            # If we made it here, structure is valid
            return True
            
        except Exception as e:
            print(f"Error validating latent videos for {exp_dir.name}: {e}")
            return False
    
    def _load_vlm_analysis(self, exp_dir):
        """Load VLM analysis data for an experiment"""
        vlm_analysis_dir = exp_dir / 'vlm_analysis'
        
        if not vlm_analysis_dir.exists():
            return {
                'has_vlm_analysis': False,
                'has_trajectory_analysis': False,
                'vlm_analysis': None
            }
        
        try:
            # Check for trajectory analysis
            trajectory_analysis_dir = exp_dir / 'latent_trajectory_analysis'
            has_trajectory_analysis = trajectory_analysis_dir.exists()
            
            # Load individual prompt group results
            prompt_groups = {}
            for prompt_dir in vlm_analysis_dir.iterdir():
                if prompt_dir.is_dir() and prompt_dir.name.startswith('prompt_'):
                    aggregated_file = prompt_dir / 'aggregated_results.json'
                    if aggregated_file.exists():
                        with open(aggregated_file, 'r') as f:
                            prompt_groups[prompt_dir.name] = json.load(f)
            
            # Load overall results
            overall_file = vlm_analysis_dir / 'aggregated_results.json'
            overall_data = None
            if overall_file.exists():
                with open(overall_file, 'r') as f:
                    overall_data = json.load(f)
            
            return {
                'has_vlm_analysis': bool(prompt_groups),
                'has_trajectory_analysis': has_trajectory_analysis,
                'vlm_analysis': {
                    'prompt_groups': prompt_groups,
                    'overall': overall_data
                } if prompt_groups else None
            }
            
        except Exception as e:
            print(f"Error loading VLM analysis for {exp_dir.name}: {e}")
            return {
                'has_vlm_analysis': False,
                'has_trajectory_analysis': False,
                'vlm_analysis': None
            }

    def _load_trajectory_analysis(self, exp_dir):
        """Load trajectory analysis data for an experiment"""
        trajectory_analysis_dir = exp_dir / 'latent_trajectory_analysis'
        
        if not trajectory_analysis_dir.exists():
            return {
                'has_trajectory_analysis': False,
                'trajectory_analysis': None
            }
        
        try:
            # Load trajectory analysis data from each normalization subfolder
            trajectory_data = {}
            
            for norm_dir in trajectory_analysis_dir.iterdir():
                if norm_dir.is_dir():
                    analysis_file = norm_dir / 'latent_trajectory_analysis.json'
                    if analysis_file.exists():
                        with open(analysis_file, 'r') as f:
                            trajectory_data[norm_dir.name] = {
                                'ok': True,
                                'data': json.load(f)
                            }
            
            return {
                'has_trajectory_analysis': bool(trajectory_data),
                'trajectory_analysis': trajectory_data if trajectory_data else None
            }
            
        except Exception as e:
            print(f"Error loading trajectory analysis for {exp_dir.name}: {e}")
            return {
                'has_trajectory_analysis': False,
                'trajectory_analysis': None
            }

    def _load_similarity_analysis(self, exp_dir):
        """Load similarity analysis data for an experiment"""
        similarity_analysis_dir = exp_dir / 'similarity_analysis'
        
        if not similarity_analysis_dir.exists():
            return False, None
        
        try:
            # Find all similarity analysis JSON files
            json_files = list(similarity_analysis_dir.glob('similarity_analysis_*.json'))
            
            if not json_files:
                return False, None
            
            # Use most recent file by modification time
            most_recent_file = max(json_files, key=lambda f: f.stat().st_mtime)
            
            with open(most_recent_file, 'r') as f:
                similarity_data = json.load(f)
            
            # Validate required fields
            required_fields = ['rankings', 'analysis_config', 'baseline_prompt']
            for field in required_fields:
                if field not in similarity_data:
                    print(f"Error: Similarity analysis missing required field '{field}' in {most_recent_file}")
                    return False, None
            
            # Extract just the data we need for frontend
            frontend_data = {
                'rankings': similarity_data['rankings'],
                'detailed_similarities': similarity_data.get('detailed_similarities', {}),
                'analysis_config': similarity_data['analysis_config'],
                'baseline_prompt': similarity_data['baseline_prompt'],
                'metrics_used': similarity_data['analysis_config'].get('metrics', []),
                'weights_used': similarity_data['analysis_config'].get('weights', {}),
                'analysis_file': most_recent_file.name
            }
            
            return True, frontend_data
            
        except Exception as e:
            print(f"Error loading similarity analysis for {exp_dir.name}: {e}")
            return False, None

    def _load_attention_videos(self, exp_dir):
        """Load attention videos data for an experiment"""
        attention_videos_dir = exp_dir / 'attention_videos'
        
        if not attention_videos_dir.exists():
            return {
                'has_attention_videos': False,
                'attention_videos': None,
                'video_metadata_map': {}
            }
        
        # Load video metadata to get bending labels (may have GPU suffix)
        video_metadata_map = {}
        metadata_files = list((exp_dir / 'configs').glob('video_metadata*.json'))
        
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    # Create map from video_id to metadata
                    for video in metadata.get('videos', []):
                        video_id = video.get('video_id')
                        if video_id:
                            video_metadata_map[video_id] = {
                                'bending_metadata': video.get('bending_metadata'),
                                'prompt_variation': video.get('prompt_variation', {}).get('text')
                            }
            except Exception as e:
                print(f"Warning: Could not load video metadata from {metadata_file.name} for {exp_dir.name}: {e}")
        
        try:
            attention_videos_data = {}
            tiers_available = set()
            available_layers = set()
            available_heads = set()
            
            # Process each prompt directory
            for prompt_dir in sorted(attention_videos_dir.iterdir()):
                if not prompt_dir.is_dir() or not prompt_dir.name.startswith('prompt_'):
                    continue
                
                prompt_id = prompt_dir.name
                attention_videos_data[prompt_id] = {}
                
                # Process each video directory within this prompt
                for video_dir in sorted(prompt_dir.iterdir()):
                    if not video_dir.is_dir():
                        continue
                    
                    # Support both old format (vid001) and new format (p000_b001_s000)
                    video_id = video_dir.name
                    if not (video_id.startswith('vid') or 
                            (video_id.startswith('p') and '_b' in video_id and '_s' in video_id)):
                        # Unknown format, skip
                        continue
                    
                    attention_videos_data[prompt_id][video_id] = {}
                    
                    # Process each token directory within this video
                    for token_dir in sorted(video_dir.iterdir()):
                        if not token_dir.is_dir() or not token_dir.name.startswith('token_'):
                            continue
                        
                        # Extract token name (remove 'token_' prefix)
                        token_name = token_dir.name.replace('token_', '')
                        attention_videos_data[prompt_id][video_id][token_name] = {}
                        
                        # Process each step file within this token and parse tier structure
                        step_files = sorted(token_dir.glob('step_*.mp4'))
                        for mp4_file in step_files:
                            filename = mp4_file.stem  # Gets "step_000" or "step_000_layer_01_head_02"
                            
                            # Parse filename to detect tier
                            # Tier 1: step_XXX
                            # Tier 2: step_XXX_layer_YY
                            # Tier 3: step_XXX_layer_YY_head_ZZ
                            tier = 1
                            layer_num = None
                            head_num = None
                            base_step = filename
                            
                            if '_layer_' in filename:
                                parts = filename.split('_layer_')
                                base_step = parts[0]  # e.g., "step_000"
                                remainder = parts[1]  # e.g., "01" or "01_head_02"
                                
                                if '_head_' in remainder:
                                    # Tier 3: per-layer-head
                                    tier = 3
                                    layer_part, head_part = remainder.split('_head_')
                                    layer_num = int(layer_part)
                                    head_num = int(head_part)
                                    available_layers.add(layer_num)
                                    available_heads.add(head_num)
                                else:
                                    # Tier 2: per-layer average
                                    tier = 2
                                    layer_num = int(remainder)
                                    available_layers.add(layer_num)
                            
                            tiers_available.add(tier)
                            
                            # Create relative path from outputs directory for video serving
                            rel_video_path = mp4_file.relative_to(self.outputs_dir)
                            
                            # Check for corresponding image file
                            jpg_file = mp4_file.with_suffix('.jpg')
                            rel_image_path = None
                            if jpg_file.exists():
                                rel_image_path = jpg_file.relative_to(self.outputs_dir)
                            
                            # Store with tier metadata
                            attention_videos_data[prompt_id][video_id][token_name][filename] = {
                                'video_path': str(rel_video_path),
                                'image_path': str(rel_image_path) if rel_image_path else None,
                                'tier': tier,
                                'step': base_step,
                                'layer': layer_num,
                                'head': head_num
                            }
            
            return {
                'has_attention_videos': bool(attention_videos_data),
                'attention_videos': attention_videos_data if attention_videos_data else None,
                'video_metadata_map': video_metadata_map,
                'tiers_available': sorted(list(tiers_available)),
                'available_layers': sorted(list(available_layers)),
                'available_heads': sorted(list(available_heads))
            }
            
        except Exception as e:
            print(f"Error loading attention videos for {exp_dir.name}: {e}")
            return {
                'has_attention_videos': False,
                'attention_videos': None
            }

    def _load_latent_videos(self, exp_dir):
        """Load latent videos data for an experiment"""
        latent_videos_dir = exp_dir / 'latents_videos'  # Fixed: use 'latents_videos' (plural)
        
        if not latent_videos_dir.exists():
            return {
                'has_latent_videos': False,
                'latent_videos': None,
                'video_metadata_map': {}
            }
        
        # Load video metadata to get bending labels
        video_metadata_map = {}
        metadata_file = exp_dir / 'configs' / 'video_metadata.json'
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    # Create map from video_id to metadata
                    for video in metadata.get('videos', []):
                        video_id = video.get('video_id')
                        if video_id:
                            video_metadata_map[video_id] = {
                                'bending_metadata': video.get('bending_metadata'),
                                'prompt_variation': video.get('prompt_variation', {}).get('text')
                            }
            except Exception as e:
                print(f"Warning: Could not load video metadata for {exp_dir.name}: {e}")
        
        try:
            latent_videos_data = {}
            
            # Process each prompt directory
            for prompt_dir in sorted(latent_videos_dir.iterdir()):
                if not prompt_dir.is_dir() or not prompt_dir.name.startswith('prompt_'):
                    continue
                
                prompt_id = prompt_dir.name
                latent_videos_data[prompt_id] = {}
                
                # Process each video directory within this prompt
                for video_dir in sorted(prompt_dir.iterdir()):
                    if not video_dir.is_dir() or not video_dir.name.startswith('vid_'):
                        continue
                    
                    video_id = video_dir.name
                    latent_videos_data[prompt_id][video_id] = {}
                    
                    # Process each step file within this video
                    step_files = sorted(video_dir.glob('step_*.mp4'))
                    for mp4_file in step_files:
                        step_name = mp4_file.stem  # Gets "step_000" from "step_000.mp4"
                        
                        # Create relative path from outputs directory for video serving
                        rel_video_path = mp4_file.relative_to(self.outputs_dir)
                        
                        # Check for corresponding image file
                        jpg_file = mp4_file.with_suffix('.jpg')
                        rel_image_path = None
                        if jpg_file.exists():
                            rel_image_path = jpg_file.relative_to(self.outputs_dir)
                        
                        latent_videos_data[prompt_id][video_id][step_name] = {
                            'video_path': str(rel_video_path),
                            'image_path': str(rel_image_path) if rel_image_path else None
                        }
            
            return {
                'has_latent_videos': bool(latent_videos_data),
                'latent_videos': latent_videos_data if latent_videos_data else None,
                'video_metadata_map': video_metadata_map
            }
            
        except Exception as e:
            print(f"Error loading latent videos: {e}")
            return {
                'has_latent_videos': False,
                'latent_videos': None,
                'video_metadata_map': None
            }

    def _load_attention_bending_data(self, exp_dir):
        """
        Load and structure attention bending data for the visualization UI.
        Returns organized data with baseline videos, bending videos, and filter options.
        """
        try:
            # Load video metadata for bending information
            video_metadata_files = list((exp_dir / 'configs').glob('video_metadata*.json'))
            
            if not video_metadata_files:
                return {
                    'available': False,
                    'error': 'No video metadata found'
                }
            
            all_videos = []
            for metadata_file in video_metadata_files:
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        all_videos.extend(metadata.get('videos', []))
                except Exception as e:
                    print(f"Warning: Could not load {metadata_file.name}: {e}")
            
            if not all_videos:
                return {
                    'available': False,
                    'error': 'No videos found in metadata'
                }
            
            # Separate baseline and bending videos
            baseline_videos = []
            bending_videos = []
            
            # Track unique values for filters
            operations = set()
            tokens = set()
            timestep_ranges = set()
            layer_ranges = set()
            prompts_dict = {}
            seeds = set()
            
            for video in all_videos:
                if not video.get('success', False):
                    continue
                
                bending_meta = video.get('bending_metadata', {})
                prompt_var = video.get('prompt_variation', {})
                
                # Categorize baseline vs bending
                is_baseline = not bending_meta or bending_meta.get('variation_id') == 'baseline'
                
                video_data = {
                    'video_id': video.get('video_id'),
                    'filename': video.get('filename'),
                    'video_path': str(Path(exp_dir.name) / 'videos' / video.get('filename')),
                    'seed': video.get('seed'),
                    'prompt_index': prompt_var.get('index'),
                    'prompt_text': prompt_var.get('text', ''),
                    'prompt_variation': prompt_var,
                    'bending_metadata': bending_meta
                }
                
                if is_baseline:
                    baseline_videos.append(video_data)
                else:
                    bending_videos.append(video_data)
                    
                    # Extract filter values from bending metadata
                    transform_type = bending_meta.get('transformation_type', 'unknown')
                    operations.add(transform_type)
                    
                    # Use resolved_tokens if available, otherwise fall back to target_token
                    target_token = bending_meta.get('target_token', 'ALL')
                    resolved_token_list = bending_meta.get('resolved_tokens', {}).get(target_token, [])
                    
                    if resolved_token_list:
                        # Add each resolved token individually
                        for token in resolved_token_list:
                            tokens.add(token)
                    else:
                        # Fall back to the target token (could be comma-separated or "ALL")
                        if target_token and ',' not in target_token:
                            tokens.add(target_token)
                        elif target_token == 'ALL' or not target_token:
                            tokens.add('ALL')
                    
                    timestep_range = bending_meta.get('timestep_range')
                    if timestep_range:
                        # Format: "0-2" or "ALL"
                        if isinstance(timestep_range, list) and len(timestep_range) == 2:
                            range_str = f"{timestep_range[0]}-{timestep_range[1]}"
                        else:
                            range_str = "ALL"
                        timestep_ranges.add(range_str)
                    else:
                        timestep_ranges.add("ALL")
                    
                    # Get layer information - check multiple possible field names
                    layer_indices = bending_meta.get('layer_indices') or bending_meta.get('apply_to_layers')
                    if layer_indices:
                        # Format: "0-5" or "ALL"
                        if isinstance(layer_indices, list) and len(layer_indices) >= 2:
                            range_str = f"{min(layer_indices)}-{max(layer_indices)}"
                            layer_ranges.add(range_str)
                            print(f"DEBUG: Added layer range: {range_str} from video {video.get('filename')}")
                        elif layer_indices == "ALL" or (isinstance(layer_indices, str) and layer_indices.upper() == "ALL"):
                            layer_ranges.add("ALL")
                            print(f"DEBUG: Added ALL (explicit) from video {video.get('filename')}")
                        else:
                            layer_ranges.add("ALL")
                            print(f"DEBUG: Added ALL (list too short or unexpected format) from video {video.get('filename')}: {layer_indices}")
                    else:
                        layer_ranges.add("ALL")
                        print(f"DEBUG: Added ALL (no layer info) from video {video.get('filename')}, bending_meta keys: {list(bending_meta.keys())}")
                
                # Track prompts and seeds
                prompt_idx = prompt_var.get('index', 0)
                if prompt_idx not in prompts_dict:
                    prompts_dict[prompt_idx] = {
                        'index': prompt_idx,
                        'text': prompt_var.get('text', f'Prompt {prompt_idx}'),
                        'id': f'p{prompt_idx}'
                    }
                
                seeds.add(video.get('seed'))
            
            # Sort and format filter options
            filter_options = {
                'operations': sorted(list(operations)),
                'tokens': sorted(list(tokens)),
                'timestep_ranges': sorted(list(timestep_ranges), key=lambda x: (x != "ALL", x)),
                'layer_ranges': sorted(list(layer_ranges), key=lambda x: (x != "ALL", x)),
                'prompts': [prompts_dict[idx] for idx in sorted(prompts_dict.keys())],
                'seeds': sorted(list(seeds))
            }
            
            return {
                'available': True,
                'baseline_videos': baseline_videos,
                'bending_videos': bending_videos,
                'filter_options': filter_options,
                'video_count': {
                    'total': len(all_videos),
                    'baseline': len(baseline_videos),
                    'bending': len(bending_videos)
                }
            }
            
        except Exception as e:
            print(f"Error loading attention bending data: {e}")
            import traceback
            traceback.print_exc()
            return {
                'available': False,
                'error': str(e)
            }

        except Exception as e:
            print(f"Error loading latent videos for {exp_dir.name}: {e}")
            return {
                'has_latent_videos': False,
                'latent_videos': None,
                'video_metadata_map': {}
            }


def create_app():
    """Create and configure the Flask application"""
    # Determine if we're in production (dist) or development
    current_dir = Path(__file__).parent
    is_production = current_dir.name == 'backend' and current_dir.parent.name == 'dist'
    
    if is_production:
        # Production paths (from dist/backend/)
        template_folder = '../templates'
        static_folder = '../static'
        outputs_path = current_dir.parent.parent.parent / 'outputs'
    else:
        # Development paths (from webapp/backend/)
        template_folder = '../react-frontend/dist'
        static_folder = '../react-frontend/dist'
        outputs_path = current_dir.parent.parent / 'outputs'
    
    app = Flask(__name__, 
                template_folder=template_folder,
                static_folder=static_folder)
    
    # Set environment for template rendering
    if is_production:
        app.config['ENV'] = 'production'
    else:
        app.config['ENV'] = 'development'
    
    # Enable CORS for API endpoints - allow frontend domain
    CORS(app, resources={
        r"/api/*": {
            "origins": [
                "http://localhost:5174",  # Development
                "http://localhost:3000",  # Alternative dev port
                "https://*.netlify.app",  # Netlify deploy previews
                "https://diffusion-exploration.netlify.app"  # Your actual Netlify domain
            ]
        }
    })
    
    # Configuration
    app.config['VIDEO_OUTPUTS_DIR'] = str(outputs_path)
    
    # Initialize video analyzer
    analyzer = VideoAnalyzer(app.config['VIDEO_OUTPUTS_DIR'])
    
    # Routes
    @app.route('/')
    def index():
        """Serve the React application"""
        try:
            return send_from_directory(app.template_folder, 'index.html')
        except FileNotFoundError:
            # If React build doesn't exist, show development message
            return '''
            <h1>WAN Video Viewer</h1>
            <p>React build not found. Run <code>npm run build</code> to build the frontend.</p>
            <p>For development, use <code>npm run dev</code> to run the Vite dev server.</p>
            ''', 404
    
    @app.route('/api/experiments')
    def get_experiments():
        """
        Get hierarchical experiment tree with full details.
        WARNING: Slow for large datasets. Use /api/experiments/summary for tree view.
        """
        try:
            tree = analyzer.scan_outputs(summary_only=False)
            return jsonify(tree)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/experiments/summary')
    def get_experiments_summary():
        """
        Get hierarchical experiment tree with minimal data (fast).
        Returns only name, path, counts, and flags - no video lists or metadata.
        Ideal for initial tree view rendering.
        """
        try:
            tree = analyzer.scan_outputs(summary_only=True)
            return jsonify(tree)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/experiment/<path:experiment_path>')
    def get_experiment(experiment_path):
        """Get detailed data for a specific experiment by hierarchical path"""
        try:
            experiment_data = analyzer.get_experiment_by_path(experiment_path)
            
            if not experiment_data:
                return jsonify({'error': 'Experiment not found'}), 404
            
            return jsonify(experiment_data)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/experiment/<path:experiment_path>/analysis')
    def get_experiment_analysis(experiment_path):
        """Get VLM analysis data for a specific experiment"""
        try:
            # Find the experiment directory
            full_experiment_path = Path(app.config['VIDEO_OUTPUTS_DIR']) / experiment_path
            
            if not full_experiment_path.exists():
                return jsonify({'error': 'Experiment not found'}), 404
            
            # Load analysis data
            analysis_data = analyzer._load_vlm_analysis(full_experiment_path)
            
            return jsonify(analysis_data)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/experiment/<path:experiment_path>/trajectory-analysis')
    def get_experiment_trajectory_analysis(experiment_path):
        """Get trajectory analysis data for a specific experiment"""
        try:
            # Find the experiment directory
            full_experiment_path = Path(app.config['VIDEO_OUTPUTS_DIR']) / experiment_path
            
            if not full_experiment_path.exists():
                return jsonify({'error': 'Experiment not found'}), 404
            
            # Load trajectory analysis data
            trajectory_analysis_data = analyzer._load_trajectory_analysis(full_experiment_path)
            
            return jsonify(trajectory_analysis_data)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/experiment/<path:experiment_path>/latent-videos')
    def get_experiment_latent_videos(experiment_path):
        """Get latent videos and attention videos data for a specific experiment"""
        try:
            # Find the experiment directory
            full_experiment_path = Path(app.config['VIDEO_OUTPUTS_DIR']) / experiment_path
            
            if not full_experiment_path.exists():
                return jsonify({'error': 'Experiment not found'}), 404
            
            # Load latent videos data
            latent_videos_data = analyzer._load_latent_videos(full_experiment_path)
            
            # Load attention videos data
            attention_videos_data = analyzer._load_attention_videos(full_experiment_path)
            
            # Combine the data
            combined_data = {
                **latent_videos_data,
                **attention_videos_data
            }
            
            return jsonify(combined_data)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/experiment/<path:experiment_path>/attention-bending')
    def get_experiment_attention_bending(experiment_path):
        """Get structured attention bending data with filter options for visualization UI"""
        try:
            # Find the experiment directory
            full_experiment_path = Path(app.config['VIDEO_OUTPUTS_DIR']) / experiment_path
            
            if not full_experiment_path.exists():
                return jsonify({'error': 'Experiment not found'}), 404
            
            # Load attention bending data
            bending_data = analyzer._load_attention_bending_data(full_experiment_path)
            
            return jsonify(bending_data)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    
    @app.route('/api/scan')
    def rescan_experiments():
        """Trigger a rescan of the outputs directory"""
        try:
            tree = analyzer.scan_outputs()
            experiment_count = analyzer._count_experiments(tree)
            return jsonify({'message': f'Scanned {experiment_count} experiments'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/analysis-schema')
    def get_analysis_schema():
        """Get the VLM analysis schema for the frontend"""
        try:
            # Path to the schema file
            schema_path = Path(__file__).parent.parent.parent / 'src' / 'analysis' / 'vlm_analysis' / 'vlm_analysis_schema_new.json'
            
            if not schema_path.exists():
                return jsonify({'error': 'Analysis schema file not found'}), 404
            
            with open(schema_path, 'r') as f:
                schema_data = json.load(f)
            
            return jsonify({
                'vlm_analysis_schema': schema_data
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/video/<path:video_path>')
    def serve_video(video_path):
        """Serve video files from the outputs directory"""
        try:
            # Construct full path to video file
            full_path = Path(app.config['VIDEO_OUTPUTS_DIR']) / video_path
            
            # Security check - ensure the path is within outputs directory
            if not str(full_path.resolve()).startswith(str(Path(app.config['VIDEO_OUTPUTS_DIR']).resolve())):
                return jsonify({'error': 'Invalid video path'}), 403
            
            if not full_path.exists():
                return jsonify({'error': 'Video not found'}), 404
            
            # Serve the video file
            directory = str(full_path.parent)
            filename = full_path.name
            
            return send_from_directory(directory, filename, mimetype='video/mp4')
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    # Explicit static file route for React assets
    @app.route('/assets/<path:filename>')
    def serve_assets(filename):
        """Serve React build assets (JS, CSS, etc.)"""
        try:
            return send_from_directory(os.path.join(app.static_folder, 'assets'), filename)
        except FileNotFoundError:
            return jsonify({'error': 'Asset not found'}), 404
    
    # Catch-all route for React Router (SPA)
    @app.route('/<path:path>')
    def catch_all(path):
        """Serve React app for all non-API routes (SPA routing)"""
        # Don't serve React for API routes
        if path.startswith('api/'):
            return jsonify({'error': 'API endpoint not found'}), 404
        
        # Don't interfere with assets route
        if path.startswith('assets/'):
            return jsonify({'error': 'Asset not found'}), 404
        
        # Try to serve static files first (CSS, JS, images, etc.)
        try:
            return send_from_directory(app.static_folder, path)
        except FileNotFoundError:
            # If it's not a static file, serve the React app
            try:
                return send_from_directory(app.template_folder, 'index.html')
            except FileNotFoundError:
                return '''
                <h1>WAN Video Viewer</h1>
                <p>React build not found. Run <code>npm run build</code> to build the frontend.</p>
                <p>For development, use <code>npm run dev</code> to run the Vite dev server.</p>
                ''', 404
    
    return app


if __name__ == '__main__':
    app = create_app()
    
    print("🎬 WAN Video Matrix Viewer - Backend API")
    print("="*50)
    print(f"Outputs directory: {app.config['VIDEO_OUTPUTS_DIR']}")
    print("Starting Flask development server...")
    print("Open http://localhost:5000 in your browser")
    print("="*50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
