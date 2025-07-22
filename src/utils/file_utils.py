"""
Utility functions for the video generation project.
Includes file management, progress tracking, and logging utilities.
"""
import os
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json


class ProgressTracker:
    """Track and display progress of batch operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
    
    def update(self, current: int, status: str = ""):
        """Update progress."""
        self.current = current
        progress = (current / self.total) * 100 if self.total > 0 else 0
        
        elapsed = datetime.now() - self.start_time
        if current > 0:
            eta = elapsed * (self.total - current) / current
            eta_str = str(eta).split('.')[0]  # Remove microseconds
        else:
            eta_str = "Unknown"
        
        print(f"\r{self.description}: {current}/{self.total} ({progress:.1f}%) - ETA: {eta_str} - {status}", end="", flush=True)
        
        if current >= self.total:
            print()  # New line when complete
    
    def finish(self, message: str = "Complete"):
        """Mark as finished."""
        elapsed = datetime.now() - self.start_time
        print(f"\n{message} in {str(elapsed).split('.')[0]}")


class FileManager:
    """Manage file operations and organization."""
    
    @staticmethod
    def create_timestamped_dir(base_dir: str, prefix: str = "") -> Path:
        """Create a directory with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{prefix}{timestamp}" if prefix else timestamp
        
        output_dir = Path(base_dir) / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return output_dir
    
    @staticmethod
    def create_batch_structure(base_dir: str, 
                              batch_name: Optional[str] = None,
                              use_timestamp: bool = True) -> Dict[str, Path]:
        """Create organized directory structure for a batch."""
        base_path = Path(base_dir)
        
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if batch_name:
                batch_dir = base_path / f"{batch_name}_{timestamp}"
            else:
                batch_dir = base_path / f"batch_{timestamp}"
        else:
            batch_dir = base_path / (batch_name or "batch")
        
        # Create directory structure
        dirs = {
            "root": batch_dir,
            "videos": batch_dir / "videos",
            "logs": batch_dir / "logs", 
            "configs": batch_dir / "configs",
            "reports": batch_dir / "reports"
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        return dirs
    
    @staticmethod
    def safe_filename(filename: str, max_length: int = 100) -> str:
        """Create a safe filename from input string."""
        # Remove or replace problematic characters
        import re
        safe = re.sub(r'[<>:"/\\|?*]', '_', filename)
        safe = re.sub(r'\s+', '_', safe)
        safe = safe.strip('_.')
        
        # Limit length
        if len(safe) > max_length:
            safe = safe[:max_length].rstrip('_')
        
        return safe or "unnamed"
    
    @staticmethod
    def get_file_size(file_path: str) -> str:
        """Get human readable file size."""
        try:
            size = os.path.getsize(file_path)
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024.0:
                    return f"{size:.1f} {unit}"
                size /= 1024.0
            return f"{size:.1f} TB"
        except OSError:
            return "Unknown"
    
    @staticmethod
    def cleanup_empty_dirs(base_dir: str):
        """Remove empty directories recursively."""
        for root, dirs, files in os.walk(base_dir, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    if not os.listdir(dir_path):  # Directory is empty
                        os.rmdir(dir_path)
                except OSError:
                    pass  # Directory might not be empty or other issues


class LogManager:
    """Manage logging configuration and operations."""
    
    @staticmethod
    def setup_logging(log_dir: str, 
                     log_level: str = "INFO",
                     log_to_console: bool = True) -> logging.Logger:
        """Setup logging configuration."""
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"generation_{timestamp}.log"
        
        # Configure logging
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    @staticmethod
    def log_configuration(config: Dict[str, Any], logger: logging.Logger):
        """Log configuration details."""
        logger.info("=== Generation Configuration ===")
        for key, value in config.items():
            logger.info(f"{key}: {value}")
        logger.info("================================")


class MetadataManager:
    """Manage metadata for generated content."""
    
    @staticmethod
    def save_generation_metadata(output_dir: str, 
                                prompt: str,
                                config: Dict[str, Any],
                                results: list):
        """Save metadata for a generation batch."""
        metadata = {
            "generation_info": {
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "configuration": config,
                "total_videos": len(results),
                "successful_videos": sum(1 for r in results if r.success),
                "failed_videos": sum(1 for r in results if not r.success)
            },
            "results": [
                {
                    "success": r.success,
                    "video_path": r.video_path,
                    "error_message": r.error_message,
                    "generation_time": r.generation_time,
                    "metadata": r.metadata
                }
                for r in results
            ]
        }
        
        metadata_file = Path(output_dir) / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return metadata_file
    
    @staticmethod
    def load_generation_metadata(metadata_file: str) -> Dict[str, Any]:
        """Load generation metadata from file."""
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)


# Utility functions for common operations
def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{int(minutes)}m {remaining_seconds:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


def estimate_disk_space(num_videos: int, 
                       video_duration: float,
                       resolution: tuple = (512, 512),
                       fps: int = 24,
                       quality_factor: float = 1.0) -> str:
    """Estimate disk space requirements for video generation."""
    # Rough estimation based on typical video sizes
    # This is a very rough approximation
    pixels_per_frame = resolution[0] * resolution[1]
    frames_per_video = video_duration * fps
    
    # Assume roughly 0.1 bytes per pixel after compression (very rough)
    bytes_per_video = pixels_per_frame * frames_per_video * 0.1 * quality_factor
    total_bytes = bytes_per_video * num_videos
    
    # Convert to human readable
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if total_bytes < 1024.0:
            return f"{total_bytes:.1f} {unit}"
        total_bytes /= 1024.0
    
    return f"{total_bytes:.1f} PB"
