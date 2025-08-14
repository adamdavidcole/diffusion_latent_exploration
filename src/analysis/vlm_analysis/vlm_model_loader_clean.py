"""
Clean VLM Model Loader for Qwen2.5-VL models.
Follows the official Hugging Face example closely.
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Configuration
DEFAULT_FPS = 1.0
DEFAULT_MAX_PIXELS = 360 * 420  # Balance between quality and memory

logger = logging.getLogger(__name__)

class VLMModelLoader:
    """Simple VLM model loader following the official Hugging Face example."""
    
    def __init__(
        self, 
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype: str = "auto",
        device_map: str = "auto",
        max_pixels: int = DEFAULT_MAX_PIXELS,
        use_flash_attention: bool = True
    ):
        self.model_id = model_id
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.max_pixels = max_pixels
        self.use_flash_attention = use_flash_attention
        
        self.model = None
        self.processor = None
        
    def load_model(self) -> None:
        """Load the VLM model and processor following the official example."""
        logger.info(f"Loading Qwen2.5-VL model: {self.model_id}")
        
        try:
            # Load model following the official example
            if self.use_flash_attention:
                logger.info("Loading with flash_attention_2 for better performance")
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=self.torch_dtype,
                    attn_implementation="flash_attention_2",
                    device_map=self.device_map,
                    trust_remote_code=True
                )
            else:
                logger.info("Loading with default attention")
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=self.torch_dtype,
                    device_map=self.device_map,
                    trust_remote_code=True
                )
            
            # Load processor with pixel settings
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                max_pixels=self.max_pixels
            )
            
            logger.info("✅ Model and processor loaded successfully")
            
            # Log device information
            if hasattr(self.model, 'hf_device_map'):
                devices = set(self.model.hf_device_map.values())
                gpu_devices = [d for d in devices if isinstance(d, int)]
                logger.info(f"Model loaded on {len(gpu_devices)} GPU(s): {sorted(gpu_devices)}")
            else:
                device = next(self.model.parameters()).device
                logger.info(f"Model loaded on device: {device}")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
    def is_loaded(self) -> bool:
        """Check if model and processor are loaded."""
        return self.model is not None and self.processor is not None
        
    def analyze_video(
        self, 
        video_path: str, 
        text_prompt: str,
        max_new_tokens: int = 128,
        fps: float = DEFAULT_FPS,
        max_pixels: int = None,
        **generation_kwargs
    ) -> str:
        """
        Analyze video with text prompt following the official example.
        
        Args:
            video_path: Path to video file
            text_prompt: Text prompt for analysis
            max_new_tokens: Maximum tokens to generate
            fps: Video frames per second for processing
            max_pixels: Override max pixels for this video
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        logger.info(f"Analyzing video: {video_path}")
        logger.info(f"Prompt: {text_prompt}")
        
        # Use provided max_pixels or default
        video_max_pixels = max_pixels or self.max_pixels
        
        # Prepare messages following the official example
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": video_max_pixels,
                        "fps": fps,
                    },
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        
        # Preparation for inference (official example)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        
        # Move inputs to model device
        if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
            # Multi-GPU case - move to first device
            first_device = min([d for d in self.model.hf_device_map.values() if isinstance(d, int)])
            inputs = inputs.to(f"cuda:{first_device}")
        else:
            # Single device case
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
        
        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            **generation_kwargs
        }
        
        logger.info(f"Starting generation with {max_new_tokens} max tokens...")
        
        # Inference (official example)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
            
        # Decode following official example
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        # Return first result (batch_decode returns list)
        result = output_text[0] if output_text else ""
        
        logger.info(f"Generated {len(result)} characters")
        logger.info(f"Response: {result}")
        
        return result
        
    def analyze_image(
        self, 
        image_path: str, 
        text_prompt: str,
        max_new_tokens: int = 128,
        **generation_kwargs
    ) -> str:
        """
        Analyze image with text prompt.
        
        Args:
            image_path: Path to image file
            text_prompt: Text prompt for analysis
            max_new_tokens: Maximum tokens to generate
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        logger.info(f"Analyzing image: {image_path}")
        logger.info(f"Prompt: {text_prompt}")
        
        # Prepare messages for image analysis
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                        "max_pixels": self.max_pixels,
                    },
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        
        # Same processing as video but for images
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        
        # Move inputs to model device
        if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
            first_device = min([d for d in self.model.hf_device_map.values() if isinstance(d, int)])
            inputs = inputs.to(f"cuda:{first_device}")
        else:
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
        
        # Generation
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            **generation_kwargs
        }
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        result = output_text[0] if output_text else ""
        
        logger.info(f"Generated {len(result)} characters")
        logger.info(f"Response: {result}")
        
        return result
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_loaded():
            return {"status": "not_loaded"}
            
        info = {
            "status": "loaded",
            "model_id": self.model_id,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "torch_dtype": str(next(self.model.parameters()).dtype),
        }
        
        # Device information
        if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
            devices = set(self.model.hf_device_map.values())
            gpu_devices = [d for d in devices if isinstance(d, int)]
            info["devices"] = sorted(gpu_devices)
            info["multi_gpu"] = len(gpu_devices) > 1
        else:
            device = next(self.model.parameters()).device
            info["device"] = str(device)
            info["multi_gpu"] = False
            
        return info
        
    def cleanup(self):
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor  
            self.processor = None
        torch.cuda.empty_cache()
        logger.info("Model resources cleaned up")
