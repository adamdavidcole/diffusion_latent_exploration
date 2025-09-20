"""
Similarity Metrics for Video Analysis

Implements various similarity metrics for comparing video frames and content.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import hashlib
import pickle
from PIL import Image
import cv2

# Optional imports with fallbacks
try:
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    open_clip = None

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    lpips = None

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import mean_squared_error as mse
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    ssim = None
    mse = None

logger = logging.getLogger(__name__)


class SimilarityMetrics:
    """Computes various similarity metrics between video frames."""
    
    def __init__(self, 
                 device: str = "cuda",
                 cache_dir: Optional[Path] = None,
                 clip_model: str = "ViT-H-14",
                 clip_pretrained: str = "laion2b_s32b_b79k"):
        """
        Initialize similarity metrics with model loading and caching.
        
        Args:
            device: Device for computation ('cuda' or 'cpu')
            cache_dir: Directory for caching embeddings and features
            clip_model: CLIP model architecture
            clip_pretrained: CLIP pretrained weights
        """
        self.device = device
        self.cache_dir = cache_dir
        
        # Model configurations
        self.clip_model_name = clip_model
        self.clip_pretrained = clip_pretrained
        
        # Loaded models (lazy initialization)
        self._clip_model = None
        self._clip_preprocess = None
        self._lpips_model = None
        
        # Caches for embeddings and features
        self._embedding_cache: Dict[str, torch.Tensor] = {}
        self._feature_cache: Dict[str, Any] = {}
        
        if cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
    def _get_cache_key(self, frame: np.ndarray, model_type: str) -> str:
        """Generate a cache key for a frame and model type."""
        frame_hash = hashlib.md5(frame.tobytes()).hexdigest()[:16]
        return f"{model_type}_{frame_hash}"
        
    def _load_clip_model(self):
        """Lazy load CLIP model."""
        if self._clip_model is not None:
            return
            
        if not CLIP_AVAILABLE:
            raise ImportError("open_clip not available. Install with: pip install open-clip-torch")
            
        logger.info(f"Loading CLIP model: {self.clip_model_name}")
        self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
            self.clip_model_name, 
            pretrained=self.clip_pretrained,
            device=self.device
        )
        self._clip_model.eval()
        logger.info("✅ CLIP model loaded successfully")
        
    def _load_lpips_model(self):
        """Lazy load LPIPS model."""
        if self._lpips_model is not None:
            return
            
        if not LPIPS_AVAILABLE:
            raise ImportError("lpips not available. Install with: pip install lpips")
            
        logger.info("Loading LPIPS model")
        self._lpips_model = lpips.LPIPS(net='alex').to(self.device)
        self._lpips_model.eval()
        logger.info("✅ LPIPS model loaded successfully")
        
    def preprocess_frame_for_clip(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess a frame for CLIP analysis."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Apply CLIP preprocessing
        return self._clip_preprocess(pil_image).unsqueeze(0).to(self.device)
        
    def preprocess_frame_for_lpips(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess a frame for LPIPS analysis."""
        # Convert BGR to RGB and normalize to [-1, 1]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0 * 2.0 - 1.0
        
        # Rearrange to CHW format and add batch dimension
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        return frame_tensor
        
    def get_clip_embedding(self, frame: np.ndarray, use_cache: bool = True) -> torch.Tensor:
        """Get CLIP embedding for a frame with caching."""
        cache_key = self._get_cache_key(frame, "clip") if use_cache else None
        
        if use_cache and cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
            
        self._load_clip_model()
        
        with torch.no_grad():
            preprocessed = self.preprocess_frame_for_clip(frame)
            embedding = self._clip_model.encode_image(preprocessed)
            embedding = F.normalize(embedding, dim=-1)
            
        if use_cache and cache_key:
            self._embedding_cache[cache_key] = embedding.cpu()
            
        return embedding
        
    def clip_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate CLIP cosine similarity between two frames."""
        embed1 = self.get_clip_embedding(frame1)
        embed2 = self.get_clip_embedding(frame2)
        
        # Cosine similarity
        similarity = F.cosine_similarity(embed1, embed2, dim=-1)
        return float(similarity.cpu().item())
        
    def clip_distance(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate CLIP distance (1 - cosine similarity)."""
        return 1.0 - self.clip_similarity(frame1, frame2)
        
    def lpips_distance(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate LPIPS perceptual distance between two frames."""
        self._load_lpips_model()
        
        with torch.no_grad():
            tensor1 = self.preprocess_frame_for_lpips(frame1)
            tensor2 = self.preprocess_frame_for_lpips(frame2)
            
            distance = self._lpips_model(tensor1, tensor2)
            
        return float(distance.cpu().item())
        
    def ssim_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate SSIM structural similarity between two frames."""
        if not SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image not available. Install with: pip install scikit-image")
            
        # Convert to grayscale for SSIM
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate SSIM
        similarity = ssim(gray1, gray2, data_range=255)
        return float(similarity)
        
    def ssim_distance(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate SSIM distance (1 - SSIM similarity)."""
        return 1.0 - self.ssim_similarity(frame1, frame2)
        
    def mse_distance(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate Mean Squared Error distance between frames."""
        # Normalize frames to [0, 1] range
        f1_norm = frame1.astype(np.float32) / 255.0
        f2_norm = frame2.astype(np.float32) / 255.0
        
        mse_value = np.mean((f1_norm - f2_norm) ** 2)
        return float(mse_value)
        
    def perceptual_hash_distance(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate perceptual hash distance between frames."""
        def compute_phash(frame):
            # Resize to 8x8
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
            
            # Compute DCT
            dct = cv2.dct(np.float32(resized))
            
            # Take upper-left 8x8 of DCT coefficients
            dct_low = dct[:8, :8]
            
            # Compute hash as binary string based on median
            median = np.median(dct_low)
            hash_bits = dct_low > median
            
            return hash_bits.flatten()
            
        hash1 = compute_phash(frame1)
        hash2 = compute_phash(frame2)
        
        # Hamming distance
        hamming_dist = np.sum(hash1 != hash2) / len(hash1)
        return float(hamming_dist)
        
    def compute_all_metrics(self, frame1: np.ndarray, frame2: np.ndarray) -> Dict[str, float]:
        """Compute all available similarity metrics between two frames."""
        metrics = {}
        
        try:
            if CLIP_AVAILABLE:
                metrics['clip_similarity'] = self.clip_similarity(frame1, frame2)
                metrics['clip_distance'] = self.clip_distance(frame1, frame2)
        except Exception as e:
            logger.warning(f"Error computing CLIP metrics: {e}")
            
        try:
            if LPIPS_AVAILABLE:
                metrics['lpips_distance'] = self.lpips_distance(frame1, frame2)
        except Exception as e:
            logger.warning(f"Error computing LPIPS metrics: {e}")
            
        try:
            if SKIMAGE_AVAILABLE:
                metrics['ssim_similarity'] = self.ssim_similarity(frame1, frame2)
                metrics['ssim_distance'] = self.ssim_distance(frame1, frame2)
        except Exception as e:
            logger.warning(f"Error computing SSIM metrics: {e}")
            
        try:
            metrics['mse_distance'] = self.mse_distance(frame1, frame2)
            metrics['phash_distance'] = self.perceptual_hash_distance(frame1, frame2)
        except Exception as e:
            logger.warning(f"Error computing basic metrics: {e}")
            
        return metrics
        
    def clear_cache(self):
        """Clear all cached embeddings and features."""
        self._embedding_cache.clear()
        self._feature_cache.clear()
        logger.info("Cleared similarity metrics cache")
        
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'embedding_cache_size': len(self._embedding_cache),
            'feature_cache_size': len(self._feature_cache)
        }
        
    def save_cache(self, cache_file: Path):
        """Save cache to disk."""
        if not self.cache_dir:
            return
            
        cache_data = {
            'embedding_cache': self._embedding_cache,
            'feature_cache': self._feature_cache
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
            
        logger.info(f"Saved cache to {cache_file}")
        
    def load_cache(self, cache_file: Path):
        """Load cache from disk."""
        if not cache_file.exists():
            return
            
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                
            self._embedding_cache = cache_data.get('embedding_cache', {})
            self._feature_cache = cache_data.get('feature_cache', {})
            
            logger.info(f"Loaded cache from {cache_file}")
            
        except Exception as e:
            logger.warning(f"Failed to load cache from {cache_file}: {e}")
            
    def cleanup(self):
        """Clean up GPU memory and models."""
        if self._clip_model is not None:
            del self._clip_model
            self._clip_model = None
            
        if self._lpips_model is not None:
            del self._lpips_model
            self._lpips_model = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Cleaned up similarity metrics models")