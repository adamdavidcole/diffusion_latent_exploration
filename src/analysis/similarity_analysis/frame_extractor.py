"""
Frame Extractor for Video Similarity Analysis

Handles video frame extraction with configurable sampling rates and timing drift correction.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import time

logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extracts frames from videos with intelligent sampling and drift correction."""
    
    def __init__(self, 
                 fps_sampling: float = 2.0,
                 enable_drift_correction: bool = True,
                 drift_search_frames: int = 1):
        """
        Initialize the frame extractor.
        
        Args:
            fps_sampling: Frames per second to extract
            enable_drift_correction: Whether to search for best frame alignment
            drift_search_frames: Number of frames to search ±1 for best alignment
        """
        self.fps_sampling = fps_sampling
        self.enable_drift_correction = enable_drift_correction
        self.drift_search_frames = drift_search_frames
        
    def extract_frames(self, video_path: Path) -> Tuple[List[np.ndarray], List[float]]:
        """
        Extract frames from a video at the specified sampling rate.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (frames, timestamps) where frames are BGR numpy arrays
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
            
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / video_fps
            
            # Calculate frame interval for sampling
            frame_interval = video_fps / self.fps_sampling
            
            frames = []
            timestamps = []
            
            frame_count = 0
            target_frame = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Check if this is a target frame
                if frame_count >= target_frame:
                    timestamp = frame_count / video_fps
                    frames.append(frame.copy())
                    timestamps.append(timestamp)
                    
                    # Calculate next target frame
                    target_frame += frame_interval
                    
                frame_count += 1
                
            logger.debug(f"Extracted {len(frames)} frames from {video_path.name} "
                        f"(duration: {duration:.2f}s, original fps: {video_fps:.2f})")
                        
            return frames, timestamps
            
        finally:
            cap.release()
            
    def extract_frames_with_metadata(self, video_path: Path) -> Dict:
        """
        Extract frames with additional metadata for caching and analysis.
        
        Returns:
            Dictionary with frames, timestamps, video metadata
        """
        frames, timestamps = self.extract_frames(video_path)
        
        # Get video metadata
        cap = cv2.VideoCapture(str(video_path))
        metadata = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        cap.release()
        
        return {
            'frames': frames,
            'timestamps': timestamps,
            'metadata': metadata,
            'extraction_config': {
                'fps_sampling': self.fps_sampling,
                'enable_drift_correction': self.enable_drift_correction,
                'drift_search_frames': self.drift_search_frames
            }
        }
        
    def find_best_aligned_frame(self, 
                               reference_frame: np.ndarray,
                               candidate_frames: List[np.ndarray],
                               metric_func) -> Tuple[int, float]:
        """
        Find the best aligned frame from candidates using the given metric.
        
        Args:
            reference_frame: Reference frame to align to
            candidate_frames: List of candidate frames to search
            metric_func: Function to calculate similarity (higher = better)
            
        Returns:
            Tuple of (best_index, best_score)
        """
        if not candidate_frames:
            return -1, 0.0
            
        best_score = -float('inf')
        best_index = 0
        
        for i, frame in enumerate(candidate_frames):
            try:
                score = metric_func(reference_frame, frame)
                if score > best_score:
                    best_score = score
                    best_index = i
            except Exception as e:
                logger.warning(f"Error computing alignment metric for frame {i}: {e}")
                continue
                
        return best_index, best_score
        
    def extract_aligned_frame_pairs(self, 
                                   baseline_video: Path,
                                   comparison_video: Path,
                                   alignment_metric_func=None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Extract frame pairs with drift correction alignment.
        
        Args:
            baseline_video: Path to baseline video
            comparison_video: Path to comparison video 
            alignment_metric_func: Function to compute frame similarity for alignment
            
        Returns:
            List of aligned frame pairs [(baseline_frame, comparison_frame), ...]
        """
        baseline_data = self.extract_frames_with_metadata(baseline_video)
        comparison_data = self.extract_frames_with_metadata(comparison_video)
        
        baseline_frames = baseline_data['frames']
        comparison_frames = comparison_data['frames']
        
        if not self.enable_drift_correction or alignment_metric_func is None:
            # Simple timestamp-based pairing
            min_frames = min(len(baseline_frames), len(comparison_frames))
            return [(baseline_frames[i], comparison_frames[i]) for i in range(min_frames)]
        
        # Drift-corrected alignment
        aligned_pairs = []
        
        for i, baseline_frame in enumerate(baseline_frames):
            # Define search window around expected index
            search_start = max(0, i - self.drift_search_frames)
            search_end = min(len(comparison_frames), i + self.drift_search_frames + 1)
            
            if search_start >= search_end:
                continue
                
            candidate_frames = comparison_frames[search_start:search_end]
            
            # Find best aligned frame
            best_local_idx, best_score = self.find_best_aligned_frame(
                baseline_frame, candidate_frames, alignment_metric_func
            )
            
            if best_local_idx >= 0:
                actual_idx = search_start + best_local_idx
                aligned_pairs.append((baseline_frame, comparison_frames[actual_idx]))
                
                logger.debug(f"Frame {i}: aligned to frame {actual_idx} "
                           f"(drift: {actual_idx - i}, score: {best_score:.3f})")
        
        return aligned_pairs