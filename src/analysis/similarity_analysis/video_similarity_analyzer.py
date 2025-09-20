"""
Video Similarity Analyzer

Main class for analyzing similarity between video prompts in diffusion experiments.
Designed for bias detection analysis with comprehensive metrics and caching.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from datetime import datetime
from tqdm import tqdm
import scipy.stats as stats

from .frame_extractor import FrameExtractor
from .similarity_metrics import SimilarityMetrics

logger = logging.getLogger(__name__)


class VideoSimilarityAnalyzer:
    """Analyzes visual similarity between video prompts for bias detection."""
    
    def __init__(self,
                 fps_sampling: float = 2.0,
                 enable_drift_correction: bool = True,
                 drift_search_frames: int = 1,
                 cache_dir: Optional[Path] = None,
                 device: str = "cuda",
                 metrics: List[str] = None,
                 weights: Dict[str, float] = None):
        """
        Initialize the video similarity analyzer.
        
        Args:
            fps_sampling: Frames per second to extract from videos
            enable_drift_correction: Whether to search for best frame alignment
            drift_search_frames: Number of frames to search ±1 for alignment
            cache_dir: Directory for caching embeddings and features
            device: Device for computation ('cuda' or 'cpu')
            metrics: List of metrics to compute ['clip', 'lpips', 'ssim', 'mse', 'phash']
            weights: Weights for combining metrics in final score
        """
        
        # Configuration
        self.fps_sampling = fps_sampling
        self.enable_drift_correction = enable_drift_correction
        self.drift_search_frames = drift_search_frames
        self.device = device
        self.cache_dir = cache_dir
        
        # Metrics configuration
        self.metrics = metrics or ['clip', 'lpips', 'ssim']
        self.weights = weights or {
            'clip': 0.5,
            'lpips': 0.3, 
            'ssim': 0.1,
            'mse': 0.05,
            'phash': 0.05
        }
        
        # Initialize components
        self.frame_extractor = FrameExtractor(
            fps_sampling=fps_sampling,
            enable_drift_correction=enable_drift_correction,
            drift_search_frames=drift_search_frames
        )
        
        self.similarity_metrics = SimilarityMetrics(
            device=device,
            cache_dir=cache_dir
        )
        
        # Analysis state
        self.baseline_prompt: Optional[str] = None
        self.baseline_cache: Dict[str, Any] = {}
        
    def detect_baseline_prompt(self, videos_dir: Path, baseline_prompt: Optional[str] = None) -> str:
        """
        Detect or validate the baseline prompt directory.
        
        Args:
            videos_dir: Directory containing prompt subdirectories
            baseline_prompt: Optional specific baseline prompt name
            
        Returns:
            Name of the baseline prompt directory
        """
        if not videos_dir.exists():
            raise FileNotFoundError(f"Videos directory not found: {videos_dir}")
            
        # Get all prompt directories
        prompt_dirs = [d for d in videos_dir.iterdir() 
                      if d.is_dir() and d.name.startswith('prompt_')]
        
        if not prompt_dirs:
            raise ValueError(f"No prompt directories found in {videos_dir}")
            
        # Sort alphabetically to get consistent baseline
        prompt_dirs.sort(key=lambda x: x.name)
        
        if baseline_prompt:
            baseline_path = videos_dir / baseline_prompt
            if not baseline_path.exists():
                raise ValueError(f"Specified baseline prompt not found: {baseline_prompt}")
            if baseline_prompt not in [d.name for d in prompt_dirs]:
                raise ValueError(f"Baseline prompt {baseline_prompt} is not a valid prompt directory")
            return baseline_prompt
        else:
            # Use first alphabetically
            baseline = prompt_dirs[0].name
            logger.info(f"Auto-detected baseline prompt: {baseline}")
            return baseline
            
    def get_prompt_groups(self, videos_dir: Path) -> Dict[str, List[Path]]:
        """
        Get all prompt groups and their video files.
        
        Args:
            videos_dir: Directory containing prompt subdirectories
            
        Returns:
            Dictionary mapping prompt names to lists of video paths
        """
        prompt_groups = {}
        
        for prompt_dir in videos_dir.iterdir():
            if not (prompt_dir.is_dir() and prompt_dir.name.startswith('prompt_')):
                continue
                
            video_files = list(prompt_dir.glob('*.mp4'))
            if video_files:
                # Sort by video filename for consistent ordering
                video_files.sort(key=lambda x: x.name)
                prompt_groups[prompt_dir.name] = video_files
                
        return prompt_groups
        
    def precompute_baseline_features(self, baseline_videos: List[Path]) -> Dict[str, Any]:
        """
        Precompute and cache features for all baseline videos.
        
        Args:
            baseline_videos: List of baseline video paths
            
        Returns:
            Dictionary with cached baseline features
        """
        logger.info(f"Precomputing baseline features for {len(baseline_videos)} videos...")
        
        baseline_cache = {}
        
        for i, video_path in enumerate(tqdm(baseline_videos, desc="Processing baseline videos")):
            try:
                # Extract frames
                frame_data = self.frame_extractor.extract_frames_with_metadata(video_path)
                frames = frame_data['frames']
                
                # Precompute embeddings for all metrics that benefit from caching
                frame_features = []
                for frame in frames:
                    frame_feature = {}
                    
                    # CLIP embeddings (expensive, good for caching)
                    if 'clip' in self.metrics:
                        clip_embedding = self.similarity_metrics.get_clip_embedding(frame, use_cache=True)
                        frame_feature['clip_embedding'] = clip_embedding.cpu()
                    
                    frame_features.append(frame_feature)
                
                baseline_cache[video_path.name] = {
                    'frames': frames,
                    'frame_features': frame_features,
                    'metadata': frame_data['metadata']
                }
                
            except Exception as e:
                logger.error(f"Error processing baseline video {video_path.name}: {e}")
                continue
                
        logger.info(f"✅ Cached baseline features for {len(baseline_cache)} videos")
        return baseline_cache
        
    def calculate_video_similarity(self, 
                                 baseline_video_data: Dict[str, Any],
                                 comparison_video: Path) -> Dict[str, float]:
        """
        Calculate similarity metrics between a baseline video and comparison video.
        
        Args:
            baseline_video_data: Cached baseline video data
            comparison_video: Path to comparison video
            
        Returns:
            Dictionary of similarity metrics
        """
        # Extract frames from comparison video
        comparison_data = self.frame_extractor.extract_frames_with_metadata(comparison_video)
        comparison_frames = comparison_data['frames']
        
        baseline_frames = baseline_video_data['frames']
        baseline_features = baseline_video_data['frame_features']
        
        # Align frames (simple pairing for now, can be enhanced with drift correction)
        min_frames = min(len(baseline_frames), len(comparison_frames))
        
        frame_similarities = []
        
        for i in range(min_frames):
            base_frame = baseline_frames[i]
            comp_frame = comparison_frames[i]
            base_features = baseline_features[i]
            
            frame_sim = {}
            
            # CLIP similarity (use cached embedding)
            if 'clip' in self.metrics:
                if 'clip_embedding' in base_features:
                    base_embed = base_features['clip_embedding'].to(self.device)
                    comp_embed = self.similarity_metrics.get_clip_embedding(comp_frame, use_cache=True)
                    
                    import torch.nn.functional as F
                    clip_sim = F.cosine_similarity(base_embed, comp_embed, dim=-1)
                    frame_sim['clip_similarity'] = float(clip_sim.cpu().item())
                    frame_sim['clip_distance'] = 1.0 - frame_sim['clip_similarity']
                else:
                    frame_sim['clip_distance'] = self.similarity_metrics.clip_distance(base_frame, comp_frame)
                    frame_sim['clip_similarity'] = 1.0 - frame_sim['clip_distance']
            
            # LPIPS distance
            if 'lpips' in self.metrics:
                frame_sim['lpips_distance'] = self.similarity_metrics.lpips_distance(base_frame, comp_frame)
                
            # SSIM similarity/distance
            if 'ssim' in self.metrics:
                frame_sim['ssim_similarity'] = self.similarity_metrics.ssim_similarity(base_frame, comp_frame)
                frame_sim['ssim_distance'] = 1.0 - frame_sim['ssim_similarity']
                
            # MSE distance
            if 'mse' in self.metrics:
                frame_sim['mse_distance'] = self.similarity_metrics.mse_distance(base_frame, comp_frame)
                
            # Perceptual hash distance
            if 'phash' in self.metrics:
                frame_sim['phash_distance'] = self.similarity_metrics.perceptual_hash_distance(base_frame, comp_frame)
                
            frame_similarities.append(frame_sim)
        
        # Aggregate frame similarities into video-level similarities
        video_similarities = {}
        for metric in frame_similarities[0].keys():
            scores = [frame_sim[metric] for frame_sim in frame_similarities if metric in frame_sim]
            if scores:
                video_similarities[metric] = float(np.mean(scores))
                
        return video_similarities
        
    def calculate_prompt_similarity(self, 
                                  baseline_videos: List[Path],
                                  comparison_videos: List[Path]) -> Dict[str, Any]:
        """
        Calculate similarity between two prompt groups.
        
        Args:
            baseline_videos: List of baseline video paths
            comparison_videos: List of comparison video paths
            
        Returns:
            Dictionary with per-video and aggregated similarities
        """
        
        # Ensure we have baseline cache
        if not self.baseline_cache:
            self.baseline_cache = self.precompute_baseline_features(baseline_videos)
            
        per_video_similarities = []
        
        # Match videos by index (same seed alignment)
        min_videos = min(len(baseline_videos), len(comparison_videos))
        
        for i in range(min_videos):
            baseline_path = baseline_videos[i]
            comparison_path = comparison_videos[i]
            
            # Get cached baseline data
            baseline_data = self.baseline_cache.get(baseline_path.name)
            if baseline_data is None:
                logger.warning(f"No cached data for baseline video: {baseline_path.name}")
                continue
                
            try:
                video_sim = self.calculate_video_similarity(baseline_data, comparison_path)
                per_video_similarities.append({
                    'baseline_video': baseline_path.name,
                    'comparison_video': comparison_path.name,
                    'similarities': video_sim
                })
                
            except Exception as e:
                logger.error(f"Error comparing {baseline_path.name} vs {comparison_path.name}: {e}")
                continue
        
        # Aggregate similarities across videos in the prompt group
        aggregated_similarities = {}
        if per_video_similarities:
            # Get all metric names from first video
            all_metrics = per_video_similarities[0]['similarities'].keys()
            
            for metric in all_metrics:
                scores = []
                for video_sim in per_video_similarities:
                    if metric in video_sim['similarities']:
                        scores.append(video_sim['similarities'][metric])
                        
                if scores:
                    aggregated_similarities[metric] = {
                        'mean': float(np.mean(scores)),
                        'std': float(np.std(scores)),
                        'min': float(np.min(scores)),
                        'max': float(np.max(scores))
                    }
        
        return {
            'per_video_similarities': per_video_similarities,
            'aggregated_similarities': aggregated_similarities,
            'video_count': min_videos
        }
        
    def normalize_and_rank_prompts(self, all_prompt_similarities: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Apply z-score normalization and calculate weighted rankings.
        
        Args:
            all_prompt_similarities: Dictionary mapping prompt names to similarity data
            
        Returns:
            Dictionary with normalized scores and rankings
        """
        
        # Extract mean values for each metric across all prompts
        prompt_names = list(all_prompt_similarities.keys())
        metric_names = set()
        
        for prompt_data in all_prompt_similarities.values():
            if 'aggregated_similarities' in prompt_data:
                metric_names.update(prompt_data['aggregated_similarities'].keys())
        
        metric_names = list(metric_names)
        
        # Build matrix of metric scores (prompts x metrics)
        score_matrix = {}
        for metric in metric_names:
            scores = []
            valid_prompts = []
            
            for prompt_name in prompt_names:
                prompt_data = all_prompt_similarities[prompt_name]
                if ('aggregated_similarities' in prompt_data and 
                    metric in prompt_data['aggregated_similarities']):
                    score = prompt_data['aggregated_similarities'][metric]['mean']
                    scores.append(score)
                    valid_prompts.append(prompt_name)
                    
            if len(scores) > 1:  # Need at least 2 data points for z-score
                z_scores = stats.zscore(scores)
                score_matrix[metric] = dict(zip(valid_prompts, z_scores))
            else:
                # If only one prompt or no valid scores, use raw scores
                score_matrix[metric] = dict(zip(valid_prompts, scores))
        
        # Calculate weighted final scores
        final_scores = {}
        for prompt_name in prompt_names:
            if prompt_name == self.baseline_prompt:
                continue  # Skip baseline in ranking
                
            individual_scores = {}
            weighted_sum = 0.0
            total_weight = 0.0
            
            for metric in metric_names:
                if (metric in score_matrix and 
                    prompt_name in score_matrix[metric]):
                    
                    score = score_matrix[metric][prompt_name]
                    individual_scores[metric] = float(score)
                    
                    # Map metric names to weights (handle _distance/_similarity suffixes)
                    weight_key = metric
                    if metric.endswith('_distance'):
                        weight_key = metric[:-9]  # Remove '_distance'
                    elif metric.endswith('_similarity'):
                        weight_key = metric[:-11]  # Remove '_similarity'
                    
                    if weight_key in self.weights:
                        weight = self.weights[weight_key]
                        weighted_sum += score * weight
                        total_weight += weight
            
            if total_weight > 0:
                final_scores[prompt_name] = {
                    'individual_z_scores': individual_scores,
                    'weighted_similarity_distance': float(weighted_sum / total_weight),
                    'total_weight': float(total_weight)
                }
        
        # Rank by weighted distance (higher = more different from baseline)
        ranked_prompts = sorted(
            final_scores.items(),
            key=lambda x: x[1]['weighted_similarity_distance'],
            reverse=True
        )
        
        return {
            'final_scores': final_scores,
            'normalization_matrix': score_matrix,
            'baseline_prompt': self.baseline_prompt,
            'ranking_order': [prompt for prompt, _ in ranked_prompts]  # Just the order for convenience
        }
        
    def analyze_experiment(self, experiment_path: Path, baseline_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Main analysis pipeline for an experiment.
        
        Args:
            experiment_path: Path to experiment directory
            baseline_prompt: Optional specific baseline prompt name
            
        Returns:
            Complete analysis results
        """
        
        logger.info(f"🎬 Starting video similarity analysis for: {experiment_path.name}")
        start_time = time.time()
        
        videos_dir = experiment_path / "videos"
        if not videos_dir.exists():
            raise FileNotFoundError(f"Videos directory not found: {videos_dir}")
            
        # 1. Detect baseline prompt
        self.baseline_prompt = self.detect_baseline_prompt(videos_dir, baseline_prompt)
        logger.info(f"📊 Using baseline prompt: {self.baseline_prompt}")
        
        # 2. Get all prompt groups
        prompt_groups = self.get_prompt_groups(videos_dir)
        logger.info(f"📁 Found {len(prompt_groups)} prompt groups")
        
        if self.baseline_prompt not in prompt_groups:
            raise ValueError(f"Baseline prompt {self.baseline_prompt} not found in prompt groups")
            
        baseline_videos = prompt_groups[self.baseline_prompt]
        
        # 3. Precompute baseline features for caching
        self.baseline_cache = self.precompute_baseline_features(baseline_videos)
        
        # 4. Calculate similarities for all prompt groups
        all_prompt_similarities = {}
        
        prompt_names = [name for name in prompt_groups.keys() if name != self.baseline_prompt]
        
        for prompt_name in tqdm(prompt_names, desc="Analyzing prompt groups"):
            try:
                comparison_videos = prompt_groups[prompt_name]
                
                similarities = self.calculate_prompt_similarity(
                    baseline_videos, comparison_videos
                )
                
                all_prompt_similarities[prompt_name] = similarities
                
                logger.debug(f"✅ Completed analysis for {prompt_name}")
                
            except Exception as e:
                logger.error(f"❌ Error analyzing {prompt_name}: {e}")
                continue
        
        # 5. Normalize and rank results
        rankings = self.normalize_and_rank_prompts(all_prompt_similarities)
        
        # 6. Compile final results
        elapsed_time = time.time() - start_time
        
        results = {
            'experiment_path': str(experiment_path),
            'baseline_prompt': self.baseline_prompt,
            'analysis_timestamp': datetime.utcnow().isoformat() + 'Z',
            'analysis_config': {
                'fps_sampling': self.fps_sampling,
                'enable_drift_correction': self.enable_drift_correction,
                'drift_search_frames': self.drift_search_frames,
                'metrics': self.metrics,
                'weights': self.weights,
                'device': self.device
            },
            'prompt_groups_analyzed': len(all_prompt_similarities),
            'total_videos_processed': sum(
                data['video_count'] for data in all_prompt_similarities.values()
            ),
            'processing_time_seconds': elapsed_time,
            'rankings': rankings,
            'detailed_similarities': all_prompt_similarities
        }
        
        logger.info(f"✅ Analysis completed in {elapsed_time:.1f}s")
        logger.info(f"📊 Analyzed {len(all_prompt_similarities)} prompt groups")
        
        return results
        
    def save_results(self, results: Dict[str, Any], output_path: Path):
        """Save analysis results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"💾 Results saved to: {output_path}")
        
    def cleanup(self):
        """Clean up resources and caches."""
        self.similarity_metrics.cleanup()
        self.baseline_cache.clear()
        logger.info("🧹 Cleaned up video similarity analyzer")