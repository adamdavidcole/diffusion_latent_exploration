"""
Video generation engine using WAN 1.3B model.
Handles the actual video generation process with proper error handling and progress tracking.
"""
import os
import time
import logging
import gc
from typing import List, Optional, Dict, Any, Callable, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
import json
import re

# WAN model imports
try:
    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video
    import numpy as np
    import cv2
    WAN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"WAN model dependencies not available: {e}. Using mock implementation.")
    WAN_AVAILABLE = False

# Import our prompt weighting module
from src.prompts.prompt_weighting import (
    PromptWeightingProcessor, 
    create_clean_processor,
    create_repetition_processor,
    create_enhanced_language_processor
)
from src.prompts.wan_weighted_embeddings_fixed import create_wan_weighted_embeddings


def clear_gpu_memory():
    """Clear GPU memory cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def get_gpu_memory_info(device_id=None):
    """Get current GPU memory usage information for specified device."""
    if torch.cuda.is_available():
        if device_id is None:
            device_id = torch.cuda.current_device()
        elif isinstance(device_id, str) and device_id.startswith('cuda:'):
            device_id = int(device_id.split(':')[1])
        
        allocated = torch.cuda.memory_allocated(device_id) / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved(device_id) / (1024**3)   # GB
        total = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)  # GB
        free = total - allocated
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "free_gb": free,
            "device_id": device_id
        }
    return None


def setup_memory_optimization():
    """Set up PyTorch memory optimization for large models."""
    # Set environment variable for expandable segments
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Try to configure the memory allocator directly if possible
    if torch.cuda.is_available():
        try:
            # Set memory allocator settings for better fragmentation handling
            torch.cuda.memory._set_allocator_settings('expandable_segments:True')
        except Exception as e:
            # This may not be available in all PyTorch versions
            logging.debug(f"Could not set allocator settings directly: {e}")
    
    logging.info("Memory optimization settings configured")


def parse_weighted_prompt(prompt: str) -> List[Tuple[str, float]]:
    """
    Parse a prompt with weight syntax like (text:weight) into segments.
    
    Examples:
        "a beautiful (landscape:1.2)" -> [("a beautiful ", 1.0), ("landscape", 1.2)]
        "(romantic:1.5) kiss between (two people:0.8)" -> [("romantic", 1.5), (" kiss between ", 1.0), ("two people", 0.8)]
    
    Returns:
        List of (text, weight) tuples
    """
    # Pattern to match (text:weight) syntax
    pattern = r'\(([^:)]+):([0-9]*\.?[0-9]+)\)'
    
    segments = []
    last_end = 0
    
    for match in re.finditer(pattern, prompt):
        # Add text before this match with weight 1.0
        if match.start() > last_end:
            text_before = prompt[last_end:match.start()]
            if text_before.strip():
                segments.append((text_before, 1.0))
        
        # Add the weighted segment
        text = match.group(1)
        weight = float(match.group(2))
        segments.append((text, weight))
        
        last_end = match.end()
    
    # Add remaining text with weight 1.0
    if last_end < len(prompt):
        remaining_text = prompt[last_end:]
        if remaining_text.strip():
            segments.append((remaining_text, 1.0))
    
    return segments


def create_weighted_embeddings(pipe, prompt: str, device: str) -> torch.Tensor:
    """
    Create weighted text embeddings from a prompt with weight syntax.
    This implements attention re-weighting rather than direct embedding scaling.
    
    Args:
        pipe: The WAN pipeline with text encoder
        prompt: Prompt string that may contain (text:weight) syntax
        device: Device to create tensors on
        
    Returns:
        Weighted text embeddings tensor
    """
    if not hasattr(pipe, 'text_encoder') or not hasattr(pipe, 'tokenizer'):
        raise ValueError("Pipeline must have text_encoder and tokenizer for weighted prompts")
    
    # Parse the weighted prompt
    segments = parse_weighted_prompt(prompt)
    
    if len(segments) == 1 and segments[0][1] == 1.0:
        # No weighting needed, use regular embedding
        try:
            # Use a safe max_length to avoid tokenizer issues
            max_length = min(77, getattr(pipe.tokenizer, 'model_max_length', 77))
            
            text_inputs = pipe.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt"
            )
            text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(device))[0]
            return text_embeddings
        except Exception as e:
            logging.error(f"Error creating regular embeddings: {e}")
            raise
    
    # For weighted prompts, use token-level attention weighting
    logging.info(f"Creating weighted embeddings for {len(segments)} segments")
    
    try:
        # Create the base prompt without weight syntax
        clean_prompt = "".join(text for text, _ in segments)
        
        # Use safe tokenization parameters
        max_length = min(77, getattr(pipe.tokenizer, 'model_max_length', 77))
        
        # Tokenize the clean prompt
        text_inputs = pipe.tokenizer(
            clean_prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Get base embeddings
        input_ids = text_inputs.input_ids.to(device)
        base_embeddings = pipe.text_encoder(input_ids)[0]
        
        # Create attention mask for weighted tokens
        attention_weights = torch.ones_like(text_inputs.attention_mask, dtype=torch.float32)
        
        # Find tokens that need weighting
        for text, weight in segments:
            if weight != 1.0 and text.strip():
                # Tokenize this segment to find its token positions
                segment_tokens = pipe.tokenizer.encode(text.strip(), add_special_tokens=False)
                
                # Find positions of these tokens in the full tokenization
                full_tokens = text_inputs.input_ids[0].cpu().tolist()
                
                # Simple approach: apply weight to any matching tokens
                for token_id in segment_tokens:
                    for i, full_token in enumerate(full_tokens):
                        if full_token == token_id:
                            attention_weights[0, i] = weight
        
        # Apply attention weighting by interpolating between original and weighted
        # This is much more stable than direct scaling
        weight_factor = attention_weights.mean().item()
        if weight_factor != 1.0:
            # Use a gentler approach - interpolate between original and emphasized
            emphasis_strength = min((weight_factor - 1.0) * 0.3, 0.3)  # Cap at 30% adjustment
            
            # Create a slightly emphasized version by adjusting the embedding norm
            embedding_norm = torch.norm(base_embeddings, dim=-1, keepdim=True)
            normalized_embeddings = base_embeddings / (embedding_norm + 1e-8)
            
            # Apply gentle emphasis
            emphasized_embeddings = normalized_embeddings * (1.0 + emphasis_strength)
            weighted_embeddings = emphasized_embeddings * embedding_norm
            
            logging.info(f"Applied gentle emphasis: {emphasis_strength:.3f} (from weight factor {weight_factor:.2f})")
        else:
            weighted_embeddings = base_embeddings
        
        logging.info(f"Created weighted embeddings with shape {weighted_embeddings.shape}")
        return weighted_embeddings
        
    except Exception as e:
        logging.error(f"Error in weighted embedding creation: {e}")
        logging.info("Falling back to regular prompt processing")
        
        # Fallback: use regular tokenization without weights
        clean_prompt = "".join(text for text, _ in segments)
        max_length = min(77, getattr(pipe.tokenizer, 'model_max_length', 77))
        
        text_inputs = pipe.tokenizer(
            clean_prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(device))[0]
        return text_embeddings


@dataclass
class GenerationResult:
    """Result of a single video generation."""
    success: bool
    video_path: Optional[str] = None
    error_message: Optional[str] = None
    generation_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class WanVideoGenerator:
    """Real WAN model implementation for video generation with memory optimization."""
    
    def __init__(self, config):
        self.config = config
        self.model_id = config.model_settings.model_id  # Use model_id from config
        self.pipe = None
        
        # Initialize prompt weighting processor based on config
        weighting_method = getattr(config.prompt_settings, 'weighting_method', 'clean')
        if weighting_method == 'repetition':
            self.prompt_processor = create_repetition_processor(max_repetitions=3)
            logging.info("Using repetition-based prompt weighting")
        elif weighting_method == 'enhanced_language':
            self.prompt_processor = create_enhanced_language_processor()
            logging.info("Using enhanced language prompt weighting")
        else:
            self.prompt_processor = create_clean_processor()
            logging.info("Using clean prompt processing (weights removed)")
        
        # Set device from config with fallback logic
        config_device = getattr(config.model_settings, 'device', 'auto')
        logging.info(f"Device configuration: {config_device}")
        
        if config_device == 'auto':
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            logging.info(f"Auto-selected device: {self.device}")
        else:
            # Validate the specified device
            if config_device.startswith('cuda:') and torch.cuda.is_available():
                device_id = int(config_device.split(':')[1])
                if device_id < torch.cuda.device_count():
                    self.device = config_device
                    logging.info(f"Using specified device: {self.device}")
                else:
                    logging.warning(f"GPU {config_device} not available (only {torch.cuda.device_count()} GPUs found), falling back to cuda:0")
                    self.device = "cuda:0"
            elif config_device == 'cpu':
                self.device = "cpu"
                logging.info(f"Using CPU as specified: {self.device}")
            elif config_device.startswith('cuda:') and not torch.cuda.is_available():
                logging.warning(f"CUDA not available, falling back to CPU despite config specifying {config_device}")
                self.device = "cpu"
            else:
                logging.warning(f"Invalid device '{config_device}', falling back to auto selection")
                self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        logging.info(f"Final device selection: {self.device}")
        
        # Set the current CUDA device for this thread
        if self.device.startswith('cuda:'):
            device_id = int(self.device.split(':')[1])
            torch.cuda.set_device(device_id)
            logging.info(f"Set current CUDA device to: {device_id}")
            logging.info(f"PyTorch current device: {torch.cuda.current_device()}")
            logging.info(f"Available CUDA devices: {torch.cuda.device_count()}")
        
        # Get memory optimization settings from config
        self.memory_optimization = getattr(config, 'memory_settings', None) is not None
        if self.memory_optimization:
            self.memory_settings = config.memory_settings
        else:
            # Fallback defaults for backward compatibility
            from src.config.config_manager import MemorySettings
            self.memory_settings = MemorySettings()
        
        # Set up memory optimization early for large models
        if self._is_large_model() and self.memory_settings.enable_memory_optimization:
            setup_memory_optimization()
        
    def _is_large_model(self):
        """Check if this is a large model that needs memory optimization."""
        return "14B" in self.model_id or "13B" in self.model_id
    
    def create_variation_weighted_prompt(self, base_prompt: str, variation_text: str, variation_weight: float = 1.5) -> str:
        """
        Create a prompt with weighted variation text for emphasizing variations.
        
        Args:
            base_prompt: The base prompt text
            variation_text: The variation text to emphasize (e.g., "two men", "two women")
            variation_weight: Weight for the variation text (default 1.5 for stronger emphasis)
            
        Returns:
            Formatted prompt with weight syntax
            
        Example:
            create_variation_weighted_prompt(
                "a romantic kiss between two people", 
                "two men", 
                1.8
            ) -> "a romantic kiss between (two men:1.8)"
        """
        # Replace the variation text with weighted version
        if variation_text in base_prompt:
            weighted_variation = f"({variation_text}:{variation_weight})"
            weighted_prompt = base_prompt.replace(variation_text, weighted_variation)
            logging.debug(f"Created weighted prompt: {weighted_prompt}")
            return weighted_prompt
        else:
            # If variation text not found exactly, append it with weight
            weighted_prompt = f"{base_prompt} ({variation_text}:{variation_weight})"
            logging.debug(f"Appended weighted variation: {weighted_prompt}")
            return weighted_prompt
        
    def _unload_model(self):
        """Unload the model to free GPU memory."""
        if self.pipe is not None:
            # Move WAN components to CPU to free GPU memory
            if hasattr(self.pipe, 'transformer'):
                self.pipe.transformer.to('cpu')
                logging.info("Moved transformer to CPU")
            if hasattr(self.pipe, 'vae'):
                self.pipe.vae.to('cpu')
                logging.info("Moved VAE to CPU")
            if hasattr(self.pipe, 'text_encoder'):
                self.pipe.text_encoder.to('cpu')
                logging.info("Moved text_encoder to CPU")
            
            # Delete the pipeline
            del self.pipe
            self.pipe = None
            
            # Clear GPU cache
            clear_gpu_memory()
            logging.info("Model unloaded and GPU memory cleared")
        
    def _load_model(self):
        """Load the WAN model components with memory optimization."""
        if not WAN_AVAILABLE:
            raise RuntimeError("WAN model dependencies not available")
            
        # Clear any existing memory
        clear_gpu_memory()
        
        # Log memory before loading
        mem_info = get_gpu_memory_info(self.device)
        if mem_info:
            logging.info(f"GPU memory before loading: {mem_info['free_gb']:.1f}GB free of {mem_info['total_gb']:.1f}GB total on {self.device}")
            
        logging.info(f"Loading WAN model: {self.model_id}")
        logging.info(f"Using device: {self.device}")
        
        try:
            # Load VAE with memory optimization
            logging.info("Loading VAE...")
            vae = AutoencoderKLWan.from_pretrained(
                self.model_id, 
                subfolder="vae", 
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True  # Enable low CPU memory usage
            )
            
            # Configure scheduler based on resolution
            # 5.0 for 720P, 3.0 for 480P
            flow_shift = 5.0 if max(self.config.video_settings.width, 
                                   self.config.video_settings.height) >= 720 else 3.0
            
            scheduler = UniPCMultistepScheduler(
                prediction_type='flow_prediction',
                use_flow_sigmas=True,
                num_train_timesteps=1000,
                flow_shift=flow_shift
            )
            
            # Load pipeline with memory optimization
            logging.info("Loading pipeline...")
            self.pipe = WanPipeline.from_pretrained(
                self.model_id,
                vae=vae,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,  # Enable low CPU memory usage
                use_safetensors=True     # Use safetensors for better memory handling
            )
            self.pipe.scheduler = scheduler
            
            # Enable memory efficient attention if available and configured
            # WAN transformer uses different attention mechanism than UNet
            if self.memory_settings.enable_memory_efficient_attention:
                try:
                    # Check if XFormers is available
                    import xformers
                    import xformers.ops
                    
                    # Try to enable on the pipeline first
                    if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
                        self.pipe.enable_xformers_memory_efficient_attention()
                        logging.info("Enabled xformers memory efficient attention on pipeline")
                    
                    # Also try to enable on individual components for WAN transformer architecture
                    elif hasattr(self.pipe, 'transformer') and hasattr(self.pipe.transformer, 'enable_xformers_memory_efficient_attention'):
                        self.pipe.transformer.enable_xformers_memory_efficient_attention()
                        logging.info("Enabled xformers memory efficient attention on transformer")
                    
                    # For VAE if it has the method
                    if hasattr(self.pipe, 'vae') and hasattr(self.pipe.vae, 'enable_xformers_memory_efficient_attention'):
                        self.pipe.vae.enable_xformers_memory_efficient_attention()
                        logging.info("Enabled xformers memory efficient attention on VAE")
                        
                except ImportError:
                    logging.warning("XFormers not available - cannot enable memory efficient attention")
                except Exception as e:
                    logging.warning(f"Could not enable xformers memory efficient attention: {e}")
                    logging.info("This is normal for WAN transformer models - they may use different attention mechanisms")
            
            # Enable gradient checkpointing for large models if configured
            # WAN transformer supports gradient checkpointing
            if (self.memory_settings.use_gradient_checkpointing and 
                self._is_large_model() and 
                hasattr(self.pipe.transformer, 'enable_gradient_checkpointing')):
                self.pipe.transformer.enable_gradient_checkpointing()
                logging.info("Enabled gradient checkpointing for transformer")
            
            # Move to device and verify
            self.pipe.to(self.device)
            
            # Verify the model is actually on the correct device
            if hasattr(self.pipe, 'transformer') and hasattr(self.pipe.transformer, 'device'):
                actual_device = str(self.pipe.transformer.device)
                logging.info(f"Model transformer is on device: {actual_device}")
            if hasattr(self.pipe, 'vae') and hasattr(self.pipe.vae, 'device'):
                actual_device = str(next(self.pipe.vae.parameters()).device)
                logging.info(f"Model VAE is on device: {actual_device}")
            
            logging.info(f"PyTorch current device: {torch.cuda.current_device()}")
            
            # Log memory after loading
            mem_info = get_gpu_memory_info(self.device)
            if mem_info:
                logging.info(f"GPU memory after loading: {mem_info['free_gb']:.1f}GB free, {mem_info['allocated_gb']:.1f}GB allocated on {self.device}")
            
            logging.info("WAN model loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to load WAN model: {e}")
            # Try to clean up on failure
            clear_gpu_memory()
            raise
    
    def generate(self, prompt: str, output_path: str, **kwargs) -> GenerationResult:
        """Generate a video using the WAN model with memory optimization."""
        start_time = time.time()
        
        try:
            if self.pipe is None:
                self._load_model()
            
            # Extract generation parameters
            width = kwargs.get('width', self.config.video_settings.width)
            height = kwargs.get('height', self.config.video_settings.height)
            num_frames = kwargs.get('frames', self.config.video_settings.frames)
            num_inference_steps = kwargs.get('steps', self.config.model_settings.steps)
            guidance_scale = kwargs.get('cfg_scale', self.config.model_settings.cfg_scale)
            generator_seed = kwargs.get('seed', self.config.model_settings.seed)
            
            # Create generator for reproducibility
            generator = torch.Generator(device=self.device).manual_seed(generator_seed)
            
            logging.info(f"Generating video: {width}x{height}, {num_frames} frames")
            logging.info(f"Prompt: {prompt}")
            logging.info(f"Seed: {generator_seed}")
            
            # Log memory before generation
            mem_info = get_gpu_memory_info(self.device)
            if mem_info:
                logging.info(f"GPU memory before generation: {mem_info['free_gb']:.1f}GB free on {self.device}")
            
            # Clear cache before generation if configured
            if self.memory_settings.clear_cache_between_videos:
                clear_gpu_memory()
            
            # Process prompt using the configured weighting strategy
            processed_prompt = self.prompt_processor.process_prompt(prompt)
            
            # Check if we should use WAN weighted embeddings
            use_weighted_embeddings = (
                getattr(self.config.prompt_settings, 'use_weighted_embeddings', False) and
                self.prompt_processor.has_weights(prompt)
            )
            
            # Log prompt processing if weights were detected
            if self.prompt_processor.has_weights(prompt):
                weight_summary = self.prompt_processor.get_weight_summary(prompt)
                logging.info(f"Prompt weights detected: {weight_summary}")
                logging.info(f"Original prompt: {prompt}")
                logging.info(f"Processed prompt: {processed_prompt}")
                
                if use_weighted_embeddings:
                    logging.info("Using WAN weighted embeddings for generation")
                else:
                    logging.info("Using prompt processing (no weighted embeddings)")
            
            # Generate video with memory optimization
            with torch.no_grad():
                if use_weighted_embeddings:
                    try:
                        # Create weighted embeddings using configured method
                        embedding_method = getattr(self.config.prompt_settings, 'embedding_method', 'multiply')
                        prompt_embeds = create_wan_weighted_embeddings(
                            pipe=self.pipe,
                            prompt=prompt,  # Use original prompt with weight syntax
                            max_sequence_length=512,
                            weighting_method=embedding_method
                        )
                        
                        # Generate using prompt_embeds
                        video_frames = self.pipe(
                            prompt_embeds=prompt_embeds,
                            width=width,
                            height=height,
                            num_frames=num_frames,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator
                        ).frames[0]
                        
                        logging.info("Successfully generated video using weighted embeddings")
                        
                    except Exception as e:
                        logging.error(f"Weighted embeddings generation failed: {e}")
                        logging.info("Falling back to processed prompt generation")
                        
                        # Fallback to processed prompt
                        video_frames = self.pipe(
                            prompt=processed_prompt,
                            width=width,
                            height=height,
                            num_frames=num_frames,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator
                        ).frames[0]
                else:
                    # Use processed prompt (repetition, enhanced language, or clean)
                    video_frames = self.pipe(
                        prompt=processed_prompt,
                        width=width,
                        height=height,
                        num_frames=num_frames,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator
                    ).frames[0]
            
            # Debug video frames structure
            logging.info(f"Video frames type: {type(video_frames)}")
            if hasattr(video_frames, '__len__'):
                logging.info(f"Video frames length: {len(video_frames)}")
                if len(video_frames) > 0:
                    logging.info(f"First frame type: {type(video_frames[0])}")
                    if torch.is_tensor(video_frames[0]):
                        logging.info(f"First frame device: {video_frames[0].device}")
                        logging.info(f"First frame dtype: {video_frames[0].dtype}")
            
            # Clear memory after generation but before video export if configured
            if self._is_large_model() and self.memory_settings.clear_cache_between_videos:
                clear_gpu_memory()
                logging.info("Cleared GPU memory after video generation")
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Export video to file
            video_path = output_path
            if not video_path.endswith(('.mp4', '.avi', '.mov')):
                video_path += '.mp4'
            
            # Move video frames to CPU before export to free GPU memory
            # Handle different possible structures of video_frames
            def move_to_cpu(obj):
                """Recursively move tensors to CPU."""
                if torch.is_tensor(obj):
                    return obj.detach().cpu()
                elif isinstance(obj, (list, tuple)):
                    return type(obj)(move_to_cpu(item) for item in obj)
                elif isinstance(obj, dict):
                    return {key: move_to_cpu(value) for key, value in obj.items()}
                else:
                    return obj
            
            # Convert all video frames to CPU
            video_frames = move_to_cpu(video_frames)
            logging.info("Moved all video frames to CPU for export")
            
            # Clear GPU memory before export step for large models
            if self._is_large_model():
                clear_gpu_memory()
                logging.info("Cleared GPU memory before video export")
            
            # Export video frames - this can be memory intensive
            try:
                logging.info(f"Exporting video frames to: {video_path}")
                export_to_video(video_frames, video_path, fps=self.config.video_settings.fps)
                logging.info("Video export completed successfully")
            except Exception as e:
                # Check if it's a tensor conversion error
                if "cuda" in str(e).lower() and "numpy" in str(e).lower():
                    logging.error(f"Tensor conversion error during export: {e}")
                    logging.info("Attempting to force CPU conversion of remaining tensors...")
                    
                    # Try more aggressive CPU conversion
                    def force_cpu_conversion(obj):
                        if torch.is_tensor(obj):
                            return obj.detach().cpu().numpy()
                        elif hasattr(obj, 'cpu') and hasattr(obj, 'numpy'):
                            return obj.cpu().numpy()
                        elif isinstance(obj, (list, tuple)):
                            return type(obj)(force_cpu_conversion(item) for item in obj)
                        elif isinstance(obj, dict):
                            return {key: force_cpu_conversion(value) for key, value in obj.items()}
                        else:
                            return obj
                    
                    video_frames = force_cpu_conversion(video_frames)
                    logging.info("Forced CPU conversion completed, retrying export...")
                    export_to_video(video_frames, video_path, fps=self.config.video_settings.fps)
                    
                # If export fails due to memory, try with more aggressive cleanup
                elif "out of memory" in str(e).lower():
                    logging.warning(f"Video export failed with memory error, trying with cleanup: {e}")
                    clear_gpu_memory()
                    gc.collect()  # Extra garbage collection
                    time.sleep(1)  # Give system time to clean up
                    export_to_video(video_frames, video_path, fps=self.config.video_settings.fps)
                else:
                    logging.error(f"Unexpected error during video export: {e}")
                    raise
            
            # Final memory cleanup if configured
            if self._is_large_model() and self.memory_settings.clear_cache_between_videos:
                clear_gpu_memory()
            
            generation_time = time.time() - start_time
            
            # Log memory after export
            mem_info = get_gpu_memory_info(self.device)
            if mem_info:
                logging.info(f"GPU memory after export: {mem_info['free_gb']:.1f}GB free on device {mem_info['device_id']}")
            
            return GenerationResult(
                success=True,
                video_path=video_path,
                generation_time=generation_time,
                metadata={
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "num_frames": num_frames,
                    "steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": generator_seed,
                    "model_id": self.model_id,
                    "device": self.device
                }
            )
            
        except Exception as e:
            generation_time = time.time() - start_time
            logging.error(f"WAN video generation failed: {e}")
            
            # Try to clean up memory on error if configured
            if self._is_large_model() and self.memory_settings.clear_cache_between_videos:
                clear_gpu_memory()
                logging.info("Cleaned up GPU memory after error")
            
            return GenerationResult(
                success=False,
                error_message=str(e),
                generation_time=generation_time
            )


class MockVideoGenerator:
    """Mock implementation for development/testing when WAN is not available."""
    
    def __init__(self, config):
        self.config = config
        
    def generate(self, prompt: str, output_path: str, **kwargs) -> GenerationResult:
        """Mock video generation that creates a placeholder file."""
        import random
        
        # Simulate generation time
        generation_time = random.uniform(10, 30)
        time.sleep(min(generation_time / 10, 3))  # Shortened for testing
        
        try:
            # Create a placeholder file
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Create a simple text file as placeholder
            placeholder_content = f"""Mock Video Generated
Prompt: {prompt}
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
Settings: {json.dumps(kwargs, indent=2)}
"""
            
            with open(output_path + ".txt", "w") as f:
                f.write(placeholder_content)
            
            return GenerationResult(
                success=True,
                video_path=output_path + ".txt",
                generation_time=generation_time,
                metadata={
                    "prompt": prompt,
                    "settings": kwargs,
                    "mock": True
                }
            )
            
        except Exception as e:
            return GenerationResult(
                success=False,
                error_message=str(e),
                generation_time=generation_time
            )


class WAN13BVideoGenerator:
    """
    WAN Video Generator wrapper.
    Automatically chooses between real WAN implementation and mock based on availability.
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the WAN model or fallback to mock."""
        try:
            if WAN_AVAILABLE and torch.cuda.is_available():
                logging.info("Initializing real WAN model")
                self.model = WanVideoGenerator(self.config)
            else:
                if not WAN_AVAILABLE:
                    logging.warning("WAN dependencies not available, using mock implementation")
                else:
                    logging.warning("CUDA not available, using mock implementation")
                self.model = MockVideoGenerator(self.config)
                
        except Exception as e:
            logging.error(f"Failed to initialize WAN model: {e}")
            logging.info("Falling back to mock implementation")
            self.model = MockVideoGenerator(self.config)
    
    def generate(self, prompt: str, output_path: str, **kwargs) -> GenerationResult:
        """Generate a single video."""
        return self.model.generate(prompt, output_path, **kwargs)
    
    def is_available(self) -> bool:
        """Check if the model is properly loaded and available."""
        return self.model is not None
    
    def is_real_model(self) -> bool:
        """Check if using real WAN model (not mock)."""
        return isinstance(self.model, WanVideoGenerator)


class BatchVideoGenerator:
    """Handles batch generation of videos with progress tracking and memory optimization."""
    
    def __init__(self, generator: WAN13BVideoGenerator):
        self.generator = generator
        self.progress_callback: Optional[Callable] = None
        self.logger = logging.getLogger(__name__)
        # Enable model reloading for large models to prevent memory accumulation
        self.reload_model_between_videos = self._should_reload_model()
    
    def _should_reload_model(self) -> bool:
        """Determine if we should reload the model between videos for memory management."""
        if hasattr(self.generator, 'model') and hasattr(self.generator.model, 'model_id'):
            model_id = self.generator.model.model_id
            
            # Check current GPU memory availability
            device = getattr(self.generator.model, 'device', None)
            mem_info = get_gpu_memory_info(device)
            if mem_info and mem_info['free_gb'] > 5.0:  # If we have >5GB free, no need to reload
                self.logger.info(f"Sufficient GPU memory available ({mem_info['free_gb']:.1f}GB free on device {mem_info['device_id']}), disabling model reloading")
                return False
            
            # Check if model reloading is enabled in config
            if (hasattr(self.generator.model, 'memory_settings') and 
                hasattr(self.generator.model.memory_settings, 'reload_model_for_large_models')):
                should_reload = (self.generator.model.memory_settings.reload_model_for_large_models and 
                        "14B" in model_id)
                if should_reload and mem_info:
                    self.logger.info(f"Model reloading enabled for 14B model with {mem_info['free_gb']:.1f}GB free on device {mem_info['device_id']}")
                return should_reload
            
            # Fallback: only reload for very large models (14B+) when memory is tight
            if "14B" in model_id and mem_info and mem_info['free_gb'] < 2.0:
                self.logger.info(f"Enabling model reloading due to low memory ({mem_info['free_gb']:.1f}GB free on device {mem_info['device_id']})")
                return True
                
        return False
    
    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """Set callback function for progress updates."""
        self.progress_callback = callback
    
    def generate_batch(self, 
                      prompts: List[str],
                      output_dir: str,
                      videos_per_prompt: int = 1,
                      filename_template: str = "{prompt_id}_{video_num:03d}",
                      **generation_kwargs) -> Dict[str, List[GenerationResult]]:
        """
        Generate multiple videos for each prompt.
        
        Args:
            prompts: List of prompt strings
            output_dir: Base output directory
            videos_per_prompt: Number of videos to generate per prompt
            filename_template: Template for video filenames
            **generation_kwargs: Additional generation parameters
        
        Returns:
            Dictionary mapping prompts to their generation results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        total_videos = len(prompts) * videos_per_prompt
        current_video = 0
        
        for prompt_idx, prompt in enumerate(prompts):
            prompt_results = []
            prompt_id = f"prompt_{prompt_idx:03d}"
            
            # Create subfolder for this prompt
            prompt_dir = output_dir / prompt_id
            prompt_dir.mkdir(exist_ok=True)
            
            # Save prompt to file
            with open(prompt_dir / "prompt.txt", "w", encoding="utf-8") as f:
                f.write(prompt)
            
            for video_num in range(videos_per_prompt):
                current_video += 1
                
                # Update progress
                if self.progress_callback:
                    self.progress_callback(current_video, total_videos, prompt)
                
                # Generate filename
                filename = filename_template.format(
                    prompt_id=prompt_id,
                    video_num=video_num + 1,
                    prompt_idx=prompt_idx
                )
                
                # Add seed variation for multiple videos
                # For each prompt, restart seed sequence from base + video_num
                current_kwargs = generation_kwargs.copy()
                
                # Always set seed with variation, regardless of whether it was passed in
                base_seed = generation_kwargs.get('seed', self.generator.config.model_settings.seed)
                current_kwargs['seed'] = base_seed + video_num
                
                output_path = prompt_dir / filename
                
                self.logger.info(f"Generating video {current_video}/{total_videos}: {filename}")
                self.logger.info(f"Using seed: {current_kwargs['seed']} (base: {base_seed} + video_num: {video_num})")
                
                # For large models, reload the model between videos to prevent memory accumulation
                if self.reload_model_between_videos and current_video > 1:
                    if hasattr(self.generator.model, '_unload_model'):
                        self.logger.info("Unloading model to free memory...")
                        self.generator.model._unload_model()
                
                # Generate video
                result = self.generator.generate(
                    prompt=prompt,
                    output_path=str(output_path),
                    **current_kwargs
                )
                
                # Log result
                if result.success:
                    self.logger.info(f"Successfully generated: {result.video_path}")
                else:
                    self.logger.error(f"Generation failed: {result.error_message}")
                
                prompt_results.append(result)
            
            results[prompt] = prompt_results
        
        return results
    
    def generate_summary_report(self, 
                               results: Dict[str, List[GenerationResult]], 
                               output_path: str):
        """Generate a summary report of the batch generation."""
        total_videos = sum(len(prompt_results) for prompt_results in results.values())
        successful_videos = sum(
            sum(1 for result in prompt_results if result.success)
            for prompt_results in results.values()
        )
        
        total_time = sum(
            sum(result.generation_time for result in prompt_results)
            for prompt_results in results.values()
        )
        
        report = {
            "summary": {
                "total_prompts": len(results),
                "total_videos": total_videos,
                "successful_videos": successful_videos,
                "failed_videos": total_videos - successful_videos,
                "success_rate": successful_videos / total_videos if total_videos > 0 else 0,
                "total_generation_time": total_time,
                "average_time_per_video": total_time / total_videos if total_videos > 0 else 0
            },
            "prompt_details": {}
        }
        
        for prompt, prompt_results in results.items():
            successful = sum(1 for result in prompt_results if result.success)
            report["prompt_details"][prompt] = {
                "total_videos": len(prompt_results),
                "successful_videos": successful,
                "failed_videos": len(prompt_results) - successful,
                "generation_times": [result.generation_time for result in prompt_results],
                "errors": [result.error_message for result in prompt_results if not result.success]
            }
        
        # Save report
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report
