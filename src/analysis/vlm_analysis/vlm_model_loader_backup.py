"""
VLM Model Loader for Qwen2.5-VL-32B-Instru            if self.device_map and self.device_map != "cuda:0":
                logger.info("Using explicit device mapping for multi-GPU")
                load_kwargs = {
                    "torch_dtype": self.torch_dtype,
                    "device_map": self.device_map,
                    "attn_implementation": "eager",  # Use eager attention for multi-GPU stability
                    "trust_remote_code": True,
                }
                self.is_multi_gpu = True
            else:
                # For debugging multi-GPU issues, let's force single GPU for now
                logger.info("Using single GPU configuration")
                load_kwargs = {
                    "torch_dtype": self.torch_dtype,
                    "device_map": "cuda:0",  # Force single GPU to isolate multi-GPU issues
                    "attn_implementation": "flash_attention_2",
                    "trust_remote_code": True,
                }
                self.is_multi_gpu = Falseloading and basic inference setup.
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
        device_map: str = "auto",
        max_pixels: int = DEFAULT_MAX_PIXELS
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.max_pixels = max_pixels
        
        self.model = None
        self.processor = None
        
    def _create_smart_device_map(self) -> Dict[str, int]:
        """Create an optimized device map that minimizes cross-GPU communication."""
        device_map = {}
        
        # HYBRID STRATEGY: Balance performance and correctness
        # Keep tied weights together BUT optimize the pipeline flow
        # 
        # GPU 0: Vision processing + Embeddings + lm_head (tied weights together)
        # GPU 1: Most transformer layers (bulk computation)
        # This minimizes communication while keeping tied weights on same device
        
        if "32B" in self.model_id:
            # 32B model - Hybrid approach for optimal balance
            # Total 64 layers (0-63)
            
            # GPU 0: Vision + Embeddings + Output (tied weights) + Some layers
            gpu0_components = [
                'model.visual',                    # Vision encoder
                'model.language_model.embed_tokens', # Input embeddings  
                'lm_head',                        # Output head (tied with embeddings)
                'model.language_model.rotary_emb', # Positional embeddings
                'model.language_model.norm',       # Final norm
            ]
            
            # Put first 20 layers on GPU 0 (0-19) - lighter load
            for i in range(20):
                gpu0_components.append(f'model.language_model.layers.{i}')
            
            # GPU 1: Remaining layers (bulk computation, 44 layers)
            gpu1_components = []
            
            # Last 44 layers on GPU 1 (20-63) - heavy computation
            for i in range(20, 64):
                gpu1_components.append(f'model.language_model.layers.{i}')
                
        elif "3B" in self.model_id:
            # 3B model - Hybrid approach (though single GPU is better)
            # Keep tied weights together
            
            # GPU 0: Vision + Embeddings + Output (tied weights) + Most layers
            gpu0_components = [
                'model.visual',
                'model.language_model.embed_tokens',
                'lm_head',                        # Keep with embeddings (tied)
                'model.language_model.rotary_emb',
                'model.language_model.norm',
            ]
            
            # Most layers on GPU 0 (0-24)
            for i in range(25):
                gpu0_components.append(f'model.language_model.layers.{i}')
            
            # GPU 1: Final layers (25-35)
            gpu1_components = []
            
            for i in range(25, 36):
                gpu1_components.append(f'model.language_model.layers.{i}')
        else:
            # Default hybrid split
            logger.warning(f"Unknown model size for {self.model_id}, using hybrid split")
            gpu0_components = [
                'model.visual',
                'model.language_model.embed_tokens',
                'lm_head',                        # Keep tied weights together
                'model.language_model.rotary_emb',
                'model.language_model.norm',
            ]
            gpu1_components = []
        
        # Assign components to devices
        for component in gpu0_components:
            device_map[component] = 0
        for component in gpu1_components:
            device_map[component] = 1
            
        logger.info(f"Smart device map: {len(gpu0_components)} components on GPU 0, {len(gpu1_components)} on GPU 1")
        logger.info(f"GPU 0 gets: vision + embeddings + lm_head (tied weights together) + some layers")
        logger.info(f"GPU 1 gets: remaining layers (bulk computation)")
        logger.info(f"Strategy: Keep tied weights together for correctness, bulk layers on GPU 1")
        
        return device_map
        
    def load_model(self) -> None:
        """Load the VLM model and processor."""
        logger.info(f"Loading Qwen2.5-VL model: {self.model_id}")
        
        try:
            # Configure based on device_map type
            if isinstance(self.device_map, dict):
                # Explicit device mapping provided
                logger.info("Using explicit device mapping for multi-GPU")
                load_kwargs = {
                    "torch_dtype": self.torch_dtype,
                    "device_map": self.device_map,
                    "attn_implementation": "eager",  # Use eager attention for multi-GPU stability
                    "trust_remote_code": True,
                }
                self.is_multi_gpu = True
            elif self.device_map == "auto":
                # Auto device mapping - let transformers decide
                logger.info("Using auto device mapping")
                load_kwargs = {
                    "torch_dtype": self.torch_dtype,
                    "device_map": "auto",
                    "attn_implementation": "flash_attention_2",
                    "trust_remote_code": True,
                }
                self.is_multi_gpu = False  # Will be determined after loading
            elif self.device_map == "smart_split":
                # Smart device mapping - minimize cross-GPU communication
                logger.info("Using smart device mapping for optimal multi-GPU performance")
                device_map = self._create_smart_device_map()
                load_kwargs = {
                    "torch_dtype": self.torch_dtype,
                    "device_map": device_map,
                    "attn_implementation": "flash_attention_2",
                    "trust_remote_code": True,
                }
                self.is_multi_gpu = True
            else:
                # Single GPU or specific device
                logger.info("Using single GPU configuration")
                load_kwargs = {
                    "torch_dtype": self.torch_dtype,
                    "device_map": self.device_map,
                    "attn_implementation": "flash_attention_2",
                    "trust_remote_code": True,
                }
                self.is_multi_gpu = False
            
            # For multi-GPU inference, add specific configurations
            if self.device_map == "auto" or isinstance(self.device_map, dict):
                # load_kwargs.update({
                #     "low_cpu_mem_usage": True,
                #     "max_memory": {0: "96GB", 1: "96GB"},  # Use almost all VRAM on RTX A6000
                #     "offload_folder": None,  # Prevent disk offloading
                # })
                logger.info("Configuring for multi-GPU inference with aggressive memory allocation")
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                **load_kwargs
            )
            
            # Determine if we actually have multi-GPU after loading
            if hasattr(self.model, 'hf_device_map'):
                unique_devices = set(self.model.hf_device_map.values())
                gpu_devices = [d for d in unique_devices if isinstance(d, int)]
                self.is_multi_gpu = len(gpu_devices) > 1
                logger.info(f"Detected {len(gpu_devices)} GPU devices in use: {gpu_devices}")
                if self.is_multi_gpu:
                    logger.info("Model is using multi-GPU configuration")
                else:
                    logger.info("Model is using single GPU configuration")
            else:
                self.is_multi_gpu = False
                
            # # Load processor
            # processor_kwargs = {
            #     "max_memory": {0: "96GB", 1: "96GB"},  # Use almost all VRAM on RTX A6000

            # }
                
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                max_pixels=self.max_pixels,
                # **processor_kwargs
            )
            
            logger.info("✅ Model loaded successfully")
            
            # Detect if actually using multi-GPU after loading
            if hasattr(self.model, 'hf_device_map'):
                devices_used = set(self.model.hf_device_map.values())
                gpu_devices = [d for d in devices_used if isinstance(d, int)]
                
                logger.info(f"Detected {len(gpu_devices)} GPU devices in use: {sorted(gpu_devices)}")
                
                if len(gpu_devices) > 1:
                    logger.info("Model is using multi-GPU configuration")
                    self.is_multi_gpu = True
                else:
                    logger.info("Model is using single GPU configuration")
                    self.is_multi_gpu = False
            
            # CRITICAL DEBUG: Check for vocab size mismatch (common cause of multi-GPU index errors)
            tokenizer_vocab_size = self.processor.tokenizer.vocab_size
            model_vocab_size = self.model.config.vocab_size
            
            logger.info(f"Tokenizer vocabulary size: {tokenizer_vocab_size}")
            logger.info(f"Model vocabulary size: {model_vocab_size}")
            
            if tokenizer_vocab_size != model_vocab_size:
                logger.error("🚨 CRITICAL: Vocab size mismatch! This is likely the cause of the multi-GPU error.")
                logger.error(f"Tokenizer: {tokenizer_vocab_size}, Model: {model_vocab_size}")
                # Auto-fix by resizing model embeddings
                logger.info("Attempting to resize model token embeddings...")
                self.model.resize_token_embeddings(len(self.processor.tokenizer))
                
                # CRITICAL: Update config vocab size after resize for multi-GPU coordination
                self.model.config.vocab_size = len(self.processor.tokenizer)
                logger.info(f"✅ Model embeddings resized to match tokenizer: {self.model.config.vocab_size}")
                
                # Force re-tie weights after resize to prevent multi-GPU coordination issues
                if hasattr(self.model, 'tie_weights'):
                    self.model.tie_weights()
                    logger.info("✅ Model weights re-tied after resize")
            else:
                logger.info("✅ Vocab sizes match - not the cause of the error")
            
            # Log multi-GPU device distribution
            if hasattr(self.model, 'hf_device_map'):
                logger.info(f"Model device map: {self.model.hf_device_map}")
                
                # Check for potential tied weight issues
                embed_device = self.model.hf_device_map.get('model.language_model.embed_tokens', 'unknown')
                lm_head_device = self.model.hf_device_map.get('lm_head', 'unknown')
                
                if embed_device != lm_head_device:
                    logger.warning(f"⚠️  Tied weights on different devices: embed_tokens on {embed_device}, lm_head on {lm_head_device}")
                    logger.warning("This can cause tensor coordination issues in multi-GPU setups")
                else:
                    logger.info(f"✅ Tied weights on same device: {embed_device}")
                    
                # Validate pad token ID
                pad_token_id = self.processor.tokenizer.pad_token_id
                eos_token_id = self.processor.tokenizer.eos_token_id
                logger.info(f"Pad token ID: {pad_token_id}, EOS token ID: {eos_token_id}")
                
                if pad_token_id >= model_vocab_size:
                    logger.error(f"🚨 CRITICAL: pad_token_id {pad_token_id} >= vocab_size {model_vocab_size}")
                if eos_token_id >= model_vocab_size:
                    logger.error(f"🚨 CRITICAL: eos_token_id {eos_token_id} >= vocab_size {model_vocab_size}")
            
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
        logger.info(f"Prompt length: {len(text_prompt)} characters")
            
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
       
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        
        # CRITICAL: For multi-GPU, move ALL inputs to the same device as the first model layer
        # This prevents index out of bounds errors in scatter/gather operations
        if hasattr(self.model, 'hf_device_map') and len(self.model.hf_device_map) > 1:
            # Find the device of the embeddings layer (where input_ids will be processed first)
            embed_device = self.model.hf_device_map.get('model.language_model.embed_tokens', 'cuda:0')
            
            logger.info(f"Moving inputs to embeddings device: {embed_device}")
            inputs = inputs.to("cuda:0")
            
            # CRITICAL DEBUG: Validate input_ids are within vocab bounds
            max_input_id = inputs['input_ids'].max().item()
            model_vocab_size = self.model.config.vocab_size
            
            logger.info(f"Max input ID: {max_input_id}, Model vocab size: {model_vocab_size}")
            
            if max_input_id >= model_vocab_size:
                logger.error(f"🚨 CRITICAL: input_ids contains {max_input_id} >= vocab_size {model_vocab_size}")
                logger.error("This will cause 'index out of bounds' errors in embedding lookup!")
                # Clip invalid tokens as emergency fix
                logger.warning("Clipping invalid tokens to prevent crash...")
                inputs['input_ids'] = torch.clamp(inputs['input_ids'], 0, model_vocab_size - 1)
                
        else:
            # Single GPU case
            inputs = inputs.to("cuda")
            logger.info("Single GPU: inputs moved to cuda")

                
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
                # Optimized generation parameters for multi-GPU performance
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "pad_token_id": self.processor.tokenizer.eos_token_id,
                    "use_cache": True,  # Re-enable cache for better performance
                    "output_scores": False,  
                    "return_dict_in_generate": False,  
                }
                
                # For multi-GPU, optimize for performance while maintaining stability
                if hasattr(self.model, 'hf_device_map') and len(self.model.hf_device_map) > 1:
                    logger.info("Using optimized multi-GPU generation parameters")
                    
                    # Check memory distribution
                    disk_layers = [k for k, v in self.model.hf_device_map.items() if v == 'disk']
                    if disk_layers:
                        logger.warning(f"Warning: {len(disk_layers)} layers offloaded to disk. Performance will be degraded.")
                    else:
                        logger.info("✅ No disk offloading - model fits in GPU memory")
                    
                    # Enhanced generation settings for multi-GPU
                    gen_kwargs.update({
                        "eos_token_id": self.processor.tokenizer.eos_token_id,
                        "early_stopping": True,  # Stop at EOS for efficiency
                    })
                
                if do_sample:
                    gen_kwargs.update({
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 50
                    })
                
                logger.info(f"Generation kwargs: {gen_kwargs}")
                
                generated_ids = self.model.generate(**inputs, **gen_kwargs)
                
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
                )[0]  # Take first element since batch_decode returns a list

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
        fps: float = 1.0,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        **kwargs
    ) -> str:
        """Complete video analysis pipeline using proven working approach."""
        inputs = self.prepare_video_input(video_path, text_prompt, fps=fps, max_pixels=max_pixels)
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
