import os
import json
import re
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from flask import Flask, render_template, jsonify, send_file, request

app = Flask(__name__)

@dataclass
class VideoMetadata:
    """Metadata for a generated video."""
    video_path: str
    prompt: str
    prompt_variation: str  # The specific variation (e.g., "two people", "two men")
    seed: int
    width: int
    height: int
    num_frames: int
    steps: int
    cfg_scale: float
    generation_time: float
    model_id: str
    batch_name: str
    experiment_name: str
    base_prompt: str  # The original prompt template with brackets

@dataclass
class ExperimentBatch:
    """Represents a complete experiment batch with prompt variations."""
    name: str
    base_prompt: str
    variations: List[str]  # List of prompt variations
    seeds: List[int]       # List of seeds used
    videos: Dict[Tuple[str, int], VideoMetadata]  # (variation, seed) -> video
    generation_config: Dict[str, Any]

class VideoAnalyzer:
    """Analyzes and organizes generated videos from the outputs directory."""
    
    def __init__(self, outputs_dir: str = "outputs"):
        self.outputs_dir = Path(outputs_dir)
        self.experiments: Dict[str, ExperimentBatch] = {}
        
    def scan_outputs(self) -> None:
        """Scan the outputs directory for videos and extract metadata."""
        self.experiments = {}
        
        if not self.outputs_dir.exists():
            print(f"Outputs directory {self.outputs_dir} not found")
            return
            
        # Look for experiment directories
        for experiment_dir in self.outputs_dir.iterdir():
            if not experiment_dir.is_dir() or experiment_dir.name.startswith('.'):
                continue
                
            print(f"Scanning experiment: {experiment_dir.name}")
            experiment = self._scan_experiment(experiment_dir)
            if experiment:
                self.experiments[experiment.name] = experiment
        
        print(f"Found {len(self.experiments)} experiments")
        for name, exp in self.experiments.items():
            print(f"  {name}: {len(exp.variations)} variations × {len(exp.seeds)} seeds = {len(exp.videos)} videos")
    
    def _scan_experiment(self, experiment_dir: Path) -> Optional[ExperimentBatch]:
        """Scan a single experiment directory."""
        videos_dir = experiment_dir / "videos"
        if not videos_dir.exists():
            return None
        
        # Load generation config
        generation_config = self._load_generation_config(experiment_dir)
        base_prompt = self._extract_base_prompt(experiment_dir, generation_config)
        
        videos = {}
        variations = set()
        seeds = set()
        
        # Look for prompt directories (these represent variations)
        for prompt_dir in videos_dir.iterdir():
            if not prompt_dir.is_dir():
                continue
                    
            # Extract videos from this variation
            variation_videos = self._scan_prompt_directory(prompt_dir, experiment_dir.name, base_prompt)
            
            for video in variation_videos:
                key = (video.prompt_variation, video.seed)
                videos[key] = video
                variations.add(video.prompt_variation)
                seeds.add(video.seed)
        
        if not videos:
            return None
            
        return ExperimentBatch(
            name=experiment_dir.name,
            base_prompt=base_prompt,
            variations=sorted(list(variations)),
            seeds=sorted(list(seeds)),
            videos=videos,
            generation_config=generation_config
        )
    
    def _scan_prompt_directory(self, prompt_dir: Path, experiment_name: str, base_prompt: str) -> List[VideoMetadata]:
        """Scan a prompt directory for video files."""
        videos = []
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
        
        # Read the actual prompt for this variation
        prompt_file = prompt_dir / "prompt.txt"
        variation_prompt = "Unknown"
        if prompt_file.exists():
            with open(prompt_file, 'r', encoding='utf-8') as f:
                variation_prompt = f.read().strip()
        
        # Extract the variation name from the prompt or directory
        variation_name = self._extract_variation_name(variation_prompt, prompt_dir.name)
        
        for video_file in prompt_dir.iterdir():
            if video_file.suffix.lower() in video_extensions:
                metadata = self._extract_video_metadata(
                    video_file, prompt_dir, experiment_name, 
                    variation_prompt, variation_name, base_prompt
                )
                if metadata:
                    videos.append(metadata)
        
        return videos
    
    def _extract_video_metadata(self, video_file: Path, prompt_dir: Path, experiment_name: str,
                               variation_prompt: str, variation_name: str, base_prompt: str) -> Optional[VideoMetadata]:
        """Extract metadata from video file and associated files."""
        try:    
            # Try to extract seed from filename
            seed = self._extract_seed_from_filename(video_file.name)
            
            # Try to load generation config
            config_info = self._load_generation_config(prompt_dir.parent.parent)
            
            # Get video info (try to load from metadata files or use defaults)
            video_settings = config_info.get('video_settings', {})
            model_settings = config_info.get('model_settings', {})
            
            width = video_settings.get('width', 480)
            height = video_settings.get('height', 360)
            num_frames = video_settings.get('frames', 24)
            steps = model_settings.get('steps', 20)
            cfg_scale = model_settings.get('cfg_scale', 7.0)
            generation_time = config_info.get('generation_time', 0.0)
            model_id = model_settings.get('model_id', 'unknown')
            
            # Create relative path for serving
            relative_path = video_file.relative_to(self.outputs_dir)
            
            return VideoMetadata(
                video_path=str(relative_path),
                prompt=variation_prompt,
                prompt_variation=variation_name,
                seed=seed,
                width=width,
                height=height,
                num_frames=num_frames,
                steps=steps,
                cfg_scale=cfg_scale,
                generation_time=generation_time,
                model_id=model_id,
                batch_name=prompt_dir.name,
                experiment_name=experiment_name,
                base_prompt=base_prompt
            )
            
        except Exception as e:
            print(f"Error extracting metadata from {video_file}: {e}")
            return None
    
    def _extract_variation_name(self, prompt: str, dir_name: str) -> str:
        """Extract variation name from prompt or directory."""
        # Try to extract meaningful variation from prompt
        # Look for common patterns like "two people", "two men", etc.
        variations = ["two people", "two men", "two women", "a man and woman", 
                     "a couple", "lovers", "partners", "man", "woman", "person"]
        
        prompt_lower = prompt.lower()
        for variation in variations:
            if variation in prompt_lower:
                return variation
        
        # Fallback to directory name
        return dir_name.replace("prompt_", "").replace("_", " ")
    
    def _extract_seed_from_filename(self, filename: str) -> int:
        """Extract seed from filename."""
        # Look for patterns like "001", "seed_123", etc.
        seed_patterns = [
            r'(\d+)\.mp4$',  # number at end before extension
            r'seed_(\d+)',   # seed_number pattern
            r'_(\d+)_',      # number between underscores
            r'(\d+)'         # any number
        ]
        
        for pattern in seed_patterns:
            match = re.search(pattern, filename)
            if match:
                return int(match.group(1))
        
        return 0  # Default seed
    
    def _extract_base_prompt(self, experiment_dir: Path, config: Dict[str, Any]) -> str:
        """Extract the base prompt template from experiment."""
        # Try to find prompt template file
        template_file = experiment_dir / "configs" / "prompt_template.txt"
        if template_file.exists():
            with open(template_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        
        # Try from config
        if 'base_prompt' in config:
            return config['base_prompt']
        
        # Try from prompt variations file
        variations_file = experiment_dir / "configs" / "prompt_variations.json"
        if variations_file.exists():
            try:
                with open(variations_file, 'r') as f:
                    data = json.load(f)
                    if 'base_prompt' in data:
                        return data['base_prompt']
            except Exception:
                pass
        
        return f"Base prompt for {experiment_dir.name}"
    
    def _load_generation_config(self, experiment_dir: Path) -> Dict[str, Any]:
        """Load generation configuration from experiment directory."""
        config_file = experiment_dir / "configs" / "generation_config.yaml"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Error loading config from {config_file}: {e}")
        
        return {}
    
    def get_experiment_list(self) -> List[Dict[str, Any]]:
        """Get list of available experiments."""
        return [
            {
                "name": exp.name,
                "base_prompt": exp.base_prompt,
                "variations_count": len(exp.variations),
                "seeds_count": len(exp.seeds),
                "videos_count": len(exp.videos),
                "config": exp.generation_config
            }
            for exp in self.experiments.values()
        ]
    
    def get_experiment(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed data for a specific experiment."""
        if experiment_name not in self.experiments:
            return None
        
        exp = self.experiments[experiment_name]
        
        # Organize videos in grid format
        video_grid = []
        for variation in exp.variations:
            row = {
                "variation": variation,
                "videos": []
            }
            for seed in exp.seeds:
                key = (variation, seed)
                if key in exp.videos:
                    video = exp.videos[key]
                    row["videos"].append(asdict(video))
                else:
                    # Missing video - placeholder
                    row["videos"].append(None)
            video_grid.append(row)
        
        return {
            "name": exp.name,
            "base_prompt": exp.base_prompt,
            "variations": exp.variations,
            "seeds": exp.seeds,
            "video_grid": video_grid,
            "config": exp.generation_config,
            "stats": {
                "total_videos": len(exp.videos),
                "variations_count": len(exp.variations),
                "seeds_count": len(exp.seeds)
            }
        }

# Global analyzer instance
analyzer = VideoAnalyzer()

@app.route('/')
def index():
    """Serve the main video viewer interface."""
    return render_template('index.html')

@app.route('/api/scan')
def scan():
    """Trigger a rescan of the outputs directory."""
    analyzer.scan_outputs()
    return jsonify({"status": "success", "message": "Scan completed"})

@app.route('/api/experiments')
def experiments():
    """Get list of available experiments."""
    return jsonify(analyzer.get_experiment_list())

@app.route('/api/experiment/<experiment_name>')
def experiment_detail(experiment_name):
    """Get detailed data for a specific experiment."""
    experiment_data = analyzer.get_experiment(experiment_name)
    if not experiment_data:
        return jsonify({"error": "Experiment not found"}), 404
    
    return jsonify(experiment_data)

@app.route('/api/video/<path:video_path>')
def serve_video(video_path):
    """Serve a video file."""
    full_path = analyzer.outputs_dir / video_path
    if not full_path.exists():
        return "Video not found", 404
    
    return send_file(full_path, mimetype='video/mp4')

if __name__ == '__main__':
    # Initial scan
    analyzer.scan_outputs()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
