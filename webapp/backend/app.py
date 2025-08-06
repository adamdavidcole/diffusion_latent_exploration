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
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    if not prompt_template_path.exists():
                        base_prompt = config.get('base_prompt', 'Unknown prompt')
                    model_id = config.get('model_settings', {}).get('model_id', 'Unknown model')
                    # Extract duration from config
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
                        # Use the shorter variation text from variables.var_0 for easier skimming
                        variation_text = variation_info.get('variables', {}).get('var_0', f"Variation {variation_num}")
                        variation_id = variation_info.get('id', f"variation_{variation_num}")
                        
                        # Find video files in this prompt directory
                        for video_file in prompt_dir.glob('video_*.mp4'):  # Changed to match video_001.mp4 pattern
                            video_info = self._extract_video_metadata(video_file, variation_num, variation_text, variation_id)
                            if video_info:
                                videos.append(video_info)
            
            if not videos:
                return None
                
            # Organize videos by variation and seed
            video_grid = self._organize_videos(videos)
            
            # Extract unique seeds and variations
            seeds = sorted(list(set(v['seed'] for v in videos)))
            variations = sorted(list(set(v['variation'] for v in videos)))
            
            # Scan for attention videos
            attention_videos = self._scan_attention_videos(exp_dir)
            
            return {
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
                'path': str(exp_dir),
                'created_at': creation_datetime.isoformat(),
                'created_timestamp': creation_time,
                'attention_videos': attention_videos
            }
            
        except Exception as e:
            print(f"Error analyzing experiment {exp_dir.name}: {e}")
            return None
    
    def _extract_video_metadata(self, video_path, variation_num, variation_text, variation_id):
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
            
            metadata = {
                'video_path': str(video_path.relative_to(self.outputs_dir)),
                'variation': variation_text,  # Use actual variation text instead of generic "Variation X"
                'variation_id': variation_id,  # Add variation ID for reference
                'variation_num': variation_num,  # Keep the numeric identifier
                'filename': video_path.name,
                'video_number': video_number,
                'seed': video_number,  # Use video number as seed for now
                'steps': 20,  # Default from config
                'cfg_scale': 6.5,  # Default from config
                'width': 1024,
                'height': 576,
                'num_frames': 25
            }
            
            return metadata
            
        except Exception as e:
            print(f"Error extracting metadata from {video_path}: {e}")
            return None
    
    def _organize_videos(self, videos):
        """Organize videos into a grid structure by variation and seed"""
        # Group by variation text, but keep track of variation number for sorting
        variations = {}
        for video in videos:
            var = video['variation']
            if var not in variations:
                variations[var] = {
                    'videos': [],
                    'variation_num': video['variation_num']  # Keep the numeric order
                }
            variations[var]['videos'].append(video)
        
        # Create grid structure, sorted by variation number (original generation order)
        grid = []
        for variation in sorted(variations.keys(), key=lambda x: variations[x]['variation_num']):
            row = {
                'variation': variation,
                'videos': sorted(variations[variation]['videos'], key=lambda x: x['seed'])
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
    
    
    @app.route('/api/scan')
    def rescan_experiments():
        """Trigger a rescan of the outputs directory"""
        try:
            tree = analyzer.scan_outputs()
            experiment_count = analyzer._count_experiments(tree)
            return jsonify({'message': f'Scanned {experiment_count} experiments'})
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
