"""
VLM Model Loader for Qwen2.5-VL-32B-Instruct
Handles model loading and basic inference setup.
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Configuration
DEFAULT_FPS = 12.0  # Wan videos are 12fps
DEFAULT_MAX_PIXELS = 360 * 420  # Balance between quality and memory

logger = logging.getLogger(__name__)

class VLMModelLoader:
    """Manages loading and configuration of the Qwen2.5-VL model."""
    
    def __init__(
        self, 
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        cache_dir: Optional[str] = None,
        torch_dtype: str = "auto",
        device_map: str = "auto"
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        
        self.model = None
        self.processor = None
        
    def load_model(self) -> None:
        """Load the VLM model and processor."""
        logger.info(f"Loading Qwen2.5-VL model: {self.model_id}")
        
        try:

      
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map
            )
                
            # Load processor
            processor_kwargs = {}
                
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                max_pixels=DEFAULT_MAX_PIXELS
            )
            
            logger.info("✅ Model loaded successfully")
            logger.info(f"Model device: {next(self.model.parameters()).device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
    def is_loaded(self) -> bool:
        """Check if model and processor are loaded."""
        return self.model is not None and self.processor is not None
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_loaded():
            return {"status": "not_loaded"}
            
        return {
            "status": "loaded",
            "model_id": self.model_id,
            "device": str(next(self.model.parameters()).device),
            "dtype": str(next(self.model.parameters()).dtype),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
        }
        
    def prepare_video_input(
        self, 
        video_path: str, 
        text_prompt: str,
        fps: float = DEFAULT_FPS,
        max_pixels: int = DEFAULT_MAX_PIXELS
    ) -> Dict[str, Any]:
        """Prepare video input for the model using proven working approach."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        logger.info(f"Preparing video input: {video_path}")
        logger.debug(f"Prompt length: {len(text_prompt)} characters")
            
        # Use the proven working message format with explicit video parameters
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video", 
                        "video": video_path,
                        "max_pixels": max_pixels,
                        "fps": fps
                    },
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        logger.info(f"Video processing params: fps={fps}, max_pixels={max_pixels}")
        logger.info(f"Messages prepared: {messages}")

        logger.info("Applying chat template...")
        # Prepare for inference using the proven working workflow
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        logger.info(f"Text prompt prepared: {text[:100]}...")  # Show first 100 characters
        
        logger.info("Processing vision info...")
        # Process vision info with explicit max_pixels control
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
       
        logger.info("Running processor tokenization...")
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            # fps=fps,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self.model.device)

        
        logger.info(f"Input preparation complete. Keys: {list(inputs.keys())}")
        logger.info(f"Input IDs shape: {inputs['input_ids'].shape}")
        logger.info(f"Input IDs sample: {inputs['input_ids'][0][:10]}...")  # First 10 tokens
        return inputs
    
    def generate_response(
        self, 
        inputs: Dict[str, Any], 
        max_new_tokens: int = 128,
        do_sample: bool = False,
        **kwargs
    ) -> str:
        """Generate response from prepared inputs using proven working approach."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        try:
            logger.info(f"Starting generation with {max_new_tokens} max tokens...")
            
            # Use the proven working generation approach with anti-hang settings
            # gen_kwargs = {
            #     "max_new_tokens": max_new_tokens, 
            #     "do_sample": do_sample,
            #     "pad_token_id": self.processor.tokenizer.eos_token_id,
            #     "eos_token_id": self.processor.tokenizer.eos_token_id,
            #     "use_cache": True,
            # }
            
            # Add sampling parameters if do_sample is True
            # if do_sample:
                # gen_kwargs.update({
                #     "temperature": 0.7,
                #     "top_p": 0.9,
                #     "top_k": 50
                # })
            
            with torch.no_grad():
                # logger.info(f"Generation kwargs: {gen_kwargs}")
                # logger.info(f"Input device: {inputs.input_ids.device}")
                # logger.info(f"Model device: {next(self.model.parameters()).device}")
                
                generated_ids = self.model.generate(**inputs, max_new_tokens=128)
                
                logger.info(f"Generated IDs shape: {generated_ids.shape}")
                logger.info(f"Input IDs shape: {inputs.input_ids.shape}")

                # Use the proven working decoding method
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]

                
                logger.info(f"Trimmed IDs length: {[len(ids) for ids in generated_ids_trimmed]}")
                
                # Debug: show what token was generated
                if len(generated_ids_trimmed[0]) > 0:
                    logger.info(f"Generated token IDs: {generated_ids_trimmed[0][:5]}")
                    debug_decode = self.processor.batch_decode(generated_ids_trimmed[0:1], skip_special_tokens=False)
                    logger.info(f"Raw decode (with special tokens): {debug_decode}")
                
                # We take the first result from the batch
                response = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                logger.info(f"response: {response}")

            
            logger.info(f"Generated {len(response)} characters")
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
            
    def analyze_video(
        self, 
        video_path: str, 
        text_prompt: str,
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        **kwargs
    ) -> str:
        """Complete video analysis pipeline using proven working approach."""
        inputs = self.prepare_video_input(video_path, text_prompt)
        return self.generate_response(inputs, max_new_tokens, do_sample)
        
    def extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from VLM response using the proven working method."""
        # Look for JSON block
        json_str_start = response.find('```json')
        if json_str_start != -1:
            json_str_start += len('```json')
            json_str_end = response.find('```', json_str_start)
            json_str = response[json_str_start:json_str_end].strip()

            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                return {"error": "JSON Decode Error", "response": response}
        else:
            # Try to parse the entire response as JSON
            try:
                return json.loads(response.strip())
            except json.JSONDecodeError:
                return {"error": "No valid JSON found in response", "response": response}
        
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
