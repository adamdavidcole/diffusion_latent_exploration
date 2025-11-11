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
        
    def scan_outputs(self):
        """Scan the outputs directory and build hierarchical experiment tree"""
        print(f"Scanning outputs directory: {self.outputs_dir}")
        
        if not self.outputs_dir.exists():
            print(f"Outputs directory not found: {self.outputs_dir}")
            return {"type": "folder", "name": "outputs", "path": "", "children": []}
        
        # Build the tree structure
        tree = self._build_tree(self.outputs_dir, "")
        print(f"Built experiment tree with {self._count_experiments(tree)} experiments")
        return tree
    
    def _build_tree(self, directory, relative_path):
        """Recursively build tree structure from directory"""
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
                
                # Try to analyze as experiment first
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
                    folder_node = self._build_tree(item, item_relative_path)
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
    
    def _analyze_experiment(self, exp_dir):
        """Analyze a single experiment directory"""
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
            
            # Find all videos
            videos_dir = exp_dir / 'videos'
            videos = []
            
            if videos_dir.exists():
                for prompt_dir in videos_dir.iterdir():
                    if prompt_dir.is_dir() and prompt_dir.name.startswith('prompt_'):
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
                        for video_file in prompt_dir.glob('video_*.mp4'):  # Changed to match video_001.mp4 pattern
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
            
            if not videos:
                return None
                
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
                'videos': videos,
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
            
            return result
            
        except Exception as e:
            print(f"Error analyzing experiment {exp_dir.name}: {e}")
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
        # Group by variation_num (unique) rather than variation text (potentially duplicate)
        variations = {}
        for video in videos:
            var_num = video['variation_num']  # Use unique variation number as key
            if var_num not in variations:
                variations[var_num] = {
                    'videos': [],
                    'variation_text': video['variation'],  # Store the display text
                    'variation_num': var_num  # Keep the numeric order
                }
            variations[var_num]['videos'].append(video)
        
        # Create grid structure, sorted by variation number (original generation order)
        grid = []
        for var_num in sorted(variations.keys()):
            row = {
                'variation': variations[var_num]['variation_text'],  # Use the display text
                'variation_num': var_num,  # Include variation number for reference
                'videos': sorted(variations[var_num]['videos'], key=lambda x: x['seed'])
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
                    if not video_dir.is_dir() or not video_dir.name.startswith('vid'):
                        continue
                    
                    video_id = video_dir.name
                    video_num = int(video_id.replace('vid', '').lstrip('0') or '1')
                    
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
                'attention_videos': None
            }
        
        try:
            attention_videos_data = {}
            
            # Process each prompt directory
            for prompt_dir in sorted(attention_videos_dir.iterdir()):
                if not prompt_dir.is_dir() or not prompt_dir.name.startswith('prompt_'):
                    continue
                
                prompt_id = prompt_dir.name
                attention_videos_data[prompt_id] = {}
                
                # Process each video directory within this prompt
                for video_dir in sorted(prompt_dir.iterdir()):
                    if not video_dir.is_dir() or not video_dir.name.startswith('vid'):
                        continue
                    
                    video_id = video_dir.name
                    attention_videos_data[prompt_id][video_id] = {}
                    
                    # Process each token directory within this video
                    for token_dir in sorted(video_dir.iterdir()):
                        if not token_dir.is_dir() or not token_dir.name.startswith('token_'):
                            continue
                        
                        # Extract token name (remove 'token_' prefix)
                        token_name = token_dir.name.replace('token_', '')
                        attention_videos_data[prompt_id][video_id][token_name] = {}
                        
                        # Process each step file within this token
                        step_files = sorted(token_dir.glob('step_*.mp4'))
                        for mp4_file in step_files:
                            step_name = mp4_file.stem  # Gets "step_000" from "step_000.mp4"
                            
                            # Create relative path from outputs directory for video serving
                            rel_video_path = mp4_file.relative_to(self.outputs_dir)
                            
                            # Check for corresponding image file
                            jpg_file = mp4_file.with_suffix('.jpg')
                            rel_image_path = None
                            if jpg_file.exists():
                                rel_image_path = jpg_file.relative_to(self.outputs_dir)
                            
                            attention_videos_data[prompt_id][video_id][token_name][step_name] = {
                                'video_path': str(rel_video_path),
                                'image_path': str(rel_image_path) if rel_image_path else None
                            }
            
            return {
                'has_attention_videos': bool(attention_videos_data),
                'attention_videos': attention_videos_data if attention_videos_data else None
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
                'latent_videos': None
            }
        
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
                'latent_videos': latent_videos_data if latent_videos_data else None
            }
            
        except Exception as e:
            print(f"Error loading latent videos for {exp_dir.name}: {e}")
            return {
                'has_latent_videos': False,
                'latent_videos': None
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
        """Get hierarchical experiment tree"""
        try:
            tree = analyzer.scan_outputs()
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
