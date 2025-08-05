"""
Utilities for storing and managing attention maps during diffusion generation.
"""
import os
import re
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Tuple
import logging
import json
from dataclasses import dataclass, asdict
import gzip
from transformers import AutoTokenizer


@dataclass
class AttentionMetadata:
    """Metadata for stored attention maps."""
    video_id: str
    step: int
    timestep: float
    total_steps: int
    token_word: str  # The word in parentheses from prompt
    token_ids: List[int]  # Token IDs corresponding to the word
    token_texts: List[str]  # Actual token texts
    aggregation_method: str  # "averaged", "per_head", "per_block", "individual"
    attention_shape: tuple  # Shape of the attention map
    spatial_resolution: tuple  # (height, width) of spatial attention
    num_blocks: int
    num_heads: int
    threshold_applied: Optional[float] = None
    block_idx: Optional[int] = None  # If storing per-block
    head_idx: Optional[int] = None   # If storing per-head
    dtype: str = "float32"
    prompt: str = ""
    seed: Optional[int] = None
    cfg_scale: Optional[float] = None
    # Video dimensions for reconstructing spatial context
    video_width: Optional[int] = None
    video_height: Optional[int] = None
    video_frames: Optional[int] = None


class WanAttentionWrapper(torch.nn.Module):
    """Wrapper for WAN attention modules to capture attention maps."""
    
    def __init__(self, original_module, attention_storage, block_index, layer_name):
        super().__init__()
        self.original_module = original_module
        self.attention_storage = attention_storage
        self.block_index = block_index
        self.layer_name = layer_name
        self.has_done_attention_at_step = {}  # Track which steps we've processed
        
    def forward(self, *args, **kwargs):
        """Forward pass that captures attention maps."""
        try:
            # Get current diffusion step from scheduler
            scheduler = getattr(self.attention_storage, '_current_scheduler', None)
            if scheduler and hasattr(scheduler, 'step_index') and scheduler.step_index is not None:
                diffusion_step = scheduler.step_index
            else:
                # Fallback: estimate step from call pattern
                diffusion_step = getattr(self, '_estimated_step', 0)
            
            # Only capture attention once per diffusion step (like your working code)
            if self.has_done_attention_at_step.get(diffusion_step, False):
                return self.original_module(*args, **kwargs)
            else:
                self.has_done_attention_at_step[diffusion_step] = True
            
            # Check if we have the right inputs for cross-attention computation
            hidden_states = kwargs.get("hidden_states", None)
            encoder_hidden_states = kwargs.get("encoder_hidden_states", None)
            
            if hidden_states is not None and encoder_hidden_states is not None:
                # Manual attention computation based on your working code
                self.attention_storage.logger.debug(f"Computing attention for {self.layer_name}, step {diffusion_step}")
                
                # Ensure tensors are copied to avoid modifying the original
                hidden_states_copy = torch.empty_like(hidden_states).copy_(hidden_states)
                encoder_hidden_states_copy = torch.empty_like(encoder_hidden_states).copy_(encoder_hidden_states)
                
                # Prepare query, key, and value tensors
                query = self.original_module.to_q(hidden_states_copy)
                key = self.original_module.to_k(encoder_hidden_states_copy)
                
                # Reshape query, key tensors
                batch_size, seq_len_q, embed_dim = query.size()
                seq_len_k = key.size(1)
                num_heads = self.original_module.heads
                head_dim = embed_dim // num_heads
                
                query = query.view(batch_size, seq_len_q, num_heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, seq_len_k, num_heads, head_dim).transpose(1, 2)
                
                # Compute scaled dot-product attention
                key_transposed = key.transpose(-2, -1)
                attention_scores = torch.matmul(query, key_transposed) / (head_dim ** 0.5)
                attention_maps = torch.softmax(attention_scores, dim=-1)
                
                # Store attention maps for this block (following your working code pattern)
                # Shape: [batch, heads, seq_len_spatial, seq_len_text]
                self.attention_storage.current_attention_maps[self.block_index] = attention_maps.detach().cpu()
                
                self.attention_storage.logger.debug(
                    f"Captured attention for block {self.block_index}, step {diffusion_step}: "
                    f"shape {attention_maps.shape} "
                    f"(batch={attention_maps.shape[0]}, heads={attention_maps.shape[1]}, "
                    f"spatial={attention_maps.shape[2]}, text={attention_maps.shape[3]})"
                )
                
            else:
                self.attention_storage.logger.warning(f"Missing inputs for attention computation in {self.layer_name}")
        
        except Exception as e:
            self.attention_storage.logger.error(f"Error in attention wrapper for {self.layer_name}: {e}")
            import traceback
            traceback.print_exc()
        
        # Always call the original module
        return self.original_module(*args, **kwargs)


class AttentionStorage:
    """Manages storage and retrieval of attention maps during diffusion."""
    
    def __init__(self, 
                 storage_dir: Union[str, Path],
                 tokenizer_name: str = "google/umt5-xxl",  # Default T5 tokenizer used by WAN
                 storage_format: str = "numpy",
                 compress: bool = True,
                 storage_interval: int = 1,
                 storage_dtype: str = "float32",
                 # Attention-specific config
                 store_per_head: bool = False,
                 store_per_block: bool = False,
                 store_individual_tokens: bool = False,
                 attention_threshold: Optional[float] = None,
                 spatial_downsample_factor: int = 1,
                 # NEW: Consistent storage options
                 store_full_per_step: bool = True):     # Store detailed per-step tensors (legacy compatibility)
        """
        Initialize attention storage manager.
        
        Args:
            storage_dir: Directory to store attention files
            tokenizer_name: Hugging Face tokenizer for token parsing
            storage_format: Format for storage ("numpy" or "torch") 
            compress: Whether to compress stored attention maps
            storage_interval: Store every N steps (1 = all steps)
            storage_dtype: Data type for storage ("float32" or "float16")
            store_per_head: Whether to preserve individual attention heads (False = average across heads)
            store_per_block: Whether to preserve individual transformer blocks (False = average across blocks)
            store_individual_tokens: Whether to store individual token attention
            attention_threshold: Threshold for filtering attention values
            spatial_downsample_factor: Factor to downsample spatial dimensions
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.storage_format = storage_format
        self.compress = compress
        self.storage_interval = storage_interval
        self.storage_dtype = storage_dtype
        
        # Attention-specific configuration
        self.store_per_head = store_per_head
        self.store_per_block = store_per_block
        self.store_individual_tokens = store_individual_tokens
        self.attention_threshold = attention_threshold
        self.spatial_downsample_factor = spatial_downsample_factor
        
        # NEW: Consistent storage configuration
        self.store_full_per_step = store_full_per_step
        
        # Use storage_dir directly as the attention directory
        self.attention_dir = self.storage_dir
        
        # Initialize tokenizer for parsing parenthetical tokens
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            logging.warning(f"Could not load tokenizer {tokenizer_name}: {e}")
            self.tokenizer = None
        
        self.logger = logging.getLogger(__name__)
        
        # Storage tracking
        self.current_video_id = None
        self.current_video_dir = None
        self.current_prompt = None
        self.current_generation_params = {}
        self.stored_steps = []
        
        # Attention extraction state
        self.target_tokens = {}  # word -> token_ids mapping
        self.original_modules = {}  # Store original attn2 modules for restoration
        self.current_attention_maps = {}  # Store current step's attention
        self.steps_processed = set()  # Track which steps we've already processed to avoid duplicates
        
    def parse_parenthetical_tokens(self, prompt: str) -> Dict[str, List[int]]:
        """Parse prompt to find words in parentheses and get their token IDs."""
        if not self.tokenizer:
            self.logger.warning("No tokenizer available for token parsing")
            return {}
        
        # Find all words in parentheses
        parenthetical_pattern = r'\(([^)]+)\)'
        matches = re.findall(parenthetical_pattern, prompt)
        
        token_mapping = {}
        
        # Tokenize the full prompt to get the actual context-sensitive tokens
        full_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        full_text_tokens = [self.tokenizer.decode([tid]) for tid in full_tokens]
        
        for word in matches:
            word = word.strip()
            
            # Find the token positions for this word in the context
            word_token_ids = []
            word_positions = []
            
            # Look for the word in the tokenized sequence
            for i, token_text in enumerate(full_text_tokens):
                if word.lower() in token_text.lower() or token_text.lower() in word.lower():
                    word_token_ids.append(full_tokens[i])
                    word_positions.append(i)
            
            # If we didn't find it by text matching, try a different approach
            if not word_token_ids:
                # Tokenize just the word to see what we get
                individual_tokens = self.tokenizer.encode(word, add_special_tokens=False)
                
                # Try to find these tokens in the full sequence
                for token_id in individual_tokens:
                    if token_id in full_tokens:
                        idx = full_tokens.index(token_id)
                        word_token_ids.append(token_id)
                        word_positions.append(idx)
            
            # Still no match? Log what we found and use individual tokenization as fallback
            if not word_token_ids:
                individual_tokens = self.tokenizer.encode(word, add_special_tokens=False)
                word_token_ids = individual_tokens
                self.logger.warning(f"Could not find '{word}' in context, using individual tokenization: {individual_tokens}")
            
            token_mapping[word] = word_token_ids
            
            self.logger.info(f"Found parenthetical word '{word}' -> tokens {word_token_ids} at positions {word_positions}")
            self.logger.debug(f"Full prompt tokens: {list(zip(full_tokens, full_text_tokens))}")
        
        return token_mapping
    
    def _map_target_words_to_tokens(self, target_words: List[str], prompt: str) -> Dict[str, List[int]]:
        """
        Map target words to their token positions in the processed prompt.
        
        Args:
            target_words: List of words/phrases to track (e.g., ["flower", "tree"])
            prompt: Final processed prompt that will be sent to the model
            
        Returns:
            Mapped tokens: word -> token_ids
        """
        if not self.tokenizer or not target_words:
            return {}
        
        mapped_tokens = {}
        
        # Tokenize the prompt to get the actual context
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_text_tokens = [self.tokenizer.decode([tid]) for tid in prompt_tokens]
        
        self.logger.debug(f"Mapping target words to prompt tokens:")
        self.logger.debug(f"  Prompt: '{prompt}'")
        self.logger.debug(f"  Target words: {target_words}")
        
        for target_word in target_words:
            # Clean the target word (remove weight syntax if any)
            clean_word = target_word.split(':')[0].strip()
            
            # Find the word in the prompt tokens
            word_token_ids = []
            word_positions = []
            
            # Method 1: Look for the word in the tokenized sequence
            for i, token_text in enumerate(prompt_text_tokens):
                if clean_word.lower() in token_text.lower() or token_text.lower() in clean_word.lower():
                    word_token_ids.append(prompt_tokens[i])
                    word_positions.append(i)
            
            # Method 2: If not found, try tokenizing the word and finding matches
            if not word_token_ids:
                individual_tokens = self.tokenizer.encode(clean_word, add_special_tokens=False)
                for token_id in individual_tokens:
                    if token_id in prompt_tokens:
                        idx = prompt_tokens.index(token_id)
                        word_token_ids.append(token_id)
                        word_positions.append(idx)
            
            if word_token_ids:
                # Use original target_word as key to preserve any weight info for metadata
                mapped_tokens[target_word] = word_token_ids
                self.logger.info(f"Mapped target word '{target_word}' -> '{clean_word}' -> tokens {word_token_ids} at positions {word_positions}")
            else:
                self.logger.warning(f"Could not find target word '{target_word}' in prompt '{prompt}'")
        
        return mapped_tokens
    
    def _sanitize_video_id(self, video_id: str) -> str:
        """Sanitize video ID for filesystem use."""
        # Replace problematic characters
        sanitized = video_id.replace(':', '_').replace('/', '_').replace('\\', '_')
        return sanitized
    
    def start_video_storage(self, video_id: str, prompt: str, target_words: List[str] = None, **generation_params):
        """Start storage for a new video.
        
        Args:
            video_id: Unique identifier for this video
            prompt: Final processed prompt that will be sent to the model
            target_words: List of specific words/phrases to track attention for (e.g., ["flower", "tree"])
            **generation_params: Additional generation parameters
        """
        self.current_video_id = video_id
        self.current_prompt = prompt
        self.current_generation_params = generation_params
        self.stored_steps = []
        
        # Clear previous state
        self.target_tokens = {}
        self.current_attention_maps = {}
        self.steps_processed = set()
        
        # Create video directory
        self.current_video_dir = self.attention_dir / self._sanitize_video_id(video_id)
        self.current_video_dir.mkdir(parents=True, exist_ok=True)
        
        # Map target words to tokens in the processed prompt
        if target_words:
            self.target_tokens = self._map_target_words_to_tokens(target_words, prompt)
        else:
            # Fallback: parse parenthetical tokens from the prompt itself
            self.target_tokens = self.parse_parenthetical_tokens(prompt)
        
        self.logger.info(f"Started attention storage for video: {video_id}")
        self.logger.info(f"Prompt: {prompt}")
        if target_words:
            self.logger.info(f"Target words specified: {target_words}")
        
        if self.target_tokens:
            self.logger.info(f"Found {len(self.target_tokens)} target tokens for attention tracking:")
            for word, token_ids in self.target_tokens.items():
                self.logger.info(f"  '{word}' -> tokens {token_ids}")
        else:
            self.logger.info("No target words or parenthetical tokens found - no attention maps will be stored")
    
    def set_scheduler(self, scheduler):
        """Set scheduler reference for step tracking in attention wrappers."""
        self._current_scheduler = scheduler
        self.logger.debug(f"Set scheduler reference: {type(scheduler).__name__}")
    
    def register_attention_hooks(self, model):
        """Replace attn2 modules with WanAttentionWrapper to capture attention maps."""
        self.original_modules = {}  # Store original modules for later restoration
        
        # First, let's inspect the model structure to understand the architecture
        # self.logger.info("Inspecting WAN model structure...")
        # self._inspect_model_structure(model)
        
        # Replace attn2 modules with our wrapper
        wrap_count = 0
        
        # Find all attn2 modules and replace them with our wrapper
        for name, module in model.named_modules():
            # Only target the actual attn2 modules, not their sub-modules
            if name.endswith('.attn2'):
                # Get the parent module and the module name within the parent
                name_parts = name.split('.')
                parent_module = model
                
                # Navigate to the parent module
                for part in name_parts[:-1]:
                    parent_module = getattr(parent_module, part)
                
                module_name = name_parts[-1]  # The actual module name within the parent
                current_module = getattr(parent_module, module_name)
                
                # Check if this module is already wrapped
                if isinstance(current_module, WanAttentionWrapper):
                    self.logger.debug(f"Module {name} is already wrapped, reusing existing wrapper")
                    # Update our tracking with the existing wrapper's original module
                    self.original_modules[name] = current_module.original_module
                    # Update the wrapper's attention_storage reference to this instance
                    current_module.attention_storage = self
                else:
                    # Store the original module for later restoration
                    self.original_modules[name] = current_module
                    
                    # Create wrapper with reference to this attention storage instance
                    wrapper = WanAttentionWrapper(current_module, self, wrap_count, name)
                    
                    # Replace the module with our wrapper
                    setattr(parent_module, module_name, wrapper)
                    
                    # self.logger.info(f"Wrapped attn2 module {name} with WanAttentionWrapper (block {wrap_count})")
                
                wrap_count += 1
        
        if wrap_count == 0:
            self.logger.warning("❌ No attn2 modules found to wrap!")
            return False
        
        self.logger.info(f"✅ Successfully wrapped {wrap_count} attn2 modules with WanAttentionWrapper")
        return True
    
    def _inspect_model_structure(self, model, max_depth=3):
        """Inspect and log the model structure to understand architecture."""
        self.logger.info("WAN Model Structure:")
        
        def log_structure(module, name="", depth=0, max_depth=3):
            if depth > max_depth:
                return
            
            indent = "  " * depth
            module_type = type(module).__name__
            
            # Log module info
            if hasattr(module, 'shape'):
                self.logger.info(f"{indent}{name}: {module_type} {getattr(module, 'shape', '')}")
            else:
                self.logger.info(f"{indent}{name}: {module_type}")
            
            # Log children
            if depth < max_depth:
                for child_name, child_module in module.named_children():
                    full_name = f"{name}.{child_name}" if name else child_name
                    log_structure(child_module, full_name, depth + 1, max_depth)
        
        log_structure(model, "model", 0, max_depth)
    
    def remove_attention_hooks(self, model=None):
        """Restore original attn2 modules from wrapped versions."""
        if not hasattr(self, 'original_modules') or not self.original_modules:
            self.logger.debug("No wrapped modules to restore")
            return
            
        if model is None:
            self.logger.warning("Model not provided for module restoration")
            return
        
        # Restore original modules
        restored_count = 0
        for name, original_module in self.original_modules.items():
            try:
                # Get the parent module and the module name within the parent
                name_parts = name.split('.')
                parent_module = model
                
                # Navigate to the parent module
                for part in name_parts[:-1]:
                    parent_module = getattr(parent_module, part)
                
                module_name = name_parts[-1]  # The actual module name within the parent
                
                # Restore the original module
                setattr(parent_module, module_name, original_module)
                restored_count += 1
                self.logger.debug(f"Restored original module: {name}")
                
            except Exception as e:
                self.logger.error(f"Failed to restore module {name}: {e}")
        
        # Clear the stored modules
        self.original_modules = {}
        
        self.logger.info(f"Restored {restored_count} original attn2 modules")
    
    def store_attention_maps(self, step: int, timestep: float, total_steps: int) -> bool:
        """
        Store attention maps for the current step.
        
        Args:
            step: Current denoising step
            timestep: Current timestep value
            total_steps: Total number of steps
            
        Returns:
            bool: True if attention maps were stored, False if skipped
        """
        if self.current_video_id is None:
            self.logger.warning("No active video for attention storage")
            return False
        
        if not self.target_tokens:
            self.logger.debug(f"No target tokens for step {step}")
            return False
        
        # Check if we should store this step based on interval
        if step % self.storage_interval != 0:
            self.logger.debug(f"Skipping step {step} (not at storage interval {self.storage_interval})")
            return False
        
        # Debug: Log current state
        # self.logger.info(f"🎯 Attempting to store attention for step {step}")
        # self.logger.info(f"Current attention maps keys: {list(self.current_attention_maps.keys())}")
        # self.logger.info(f"Number of wrapped modules: {len(self.original_modules) if hasattr(self, 'original_modules') else 0}")
        
        if not self.current_attention_maps:
            self.logger.error(f"❌ No attention maps captured for step {step}")
            self.logger.error(f"Wrapped modules: {len(self.original_modules) if hasattr(self, 'original_modules') else 0}")
            self.logger.error(f"This suggests wrapped modules are not being called or not capturing data")
            
            return False
        
        # self.logger.info(f"Storing attention maps for step {step} with {len(self.current_attention_maps)} captured maps")
        
        try:
            # Process each target word
            for word, token_ids in self.target_tokens.items():
                word_dir = self.current_video_dir / f"token_{word}"
                word_dir.mkdir(exist_ok=True)
                
                # Process attention maps for this word
                self._process_word_attention(word, token_ids, step, timestep, total_steps, word_dir)
            
            self.stored_steps.append(step)
            self.current_attention_maps = {}  # Clear for next step
            
            # self.logger.info(f"Successfully stored attention maps for step {step}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing attention maps for step {step}: {e}")
            return False
    
    def _process_word_attention(self, word: str, token_ids: List[int], 
                               step: int, timestep: float, total_steps: int,
                               word_dir: Path):
        """Process and store attention maps for a specific word."""
        self.logger.debug(f"Processing attention for word '{word}' with token IDs {token_ids}")
        self.logger.debug(f"Available attention map blocks: {list(self.current_attention_maps.keys())}")
        
        filename_base = f"step_{step:03d}"
        
        # Collect attention maps from all blocks
        block_attentions = []
        for block_idx in sorted(self.current_attention_maps.keys()):
            attention = self.current_attention_maps[block_idx]  # Expected: [batch, heads, seq_len_spatial, seq_len_text]
            
            self.logger.debug(f"Block {block_idx} raw attention shape: {attention.shape}")
            
            # Extract attention for target tokens
            token_attention = self._extract_token_attention(attention, token_ids)
            self.logger.debug(f"Block {block_idx} token attention shape after extraction: {token_attention.shape}")
            
            block_attentions.append(token_attention)
        
        if not block_attentions:
            return
        
        self.logger.debug(f"Number of block attentions: {len(block_attentions)}")
        self.logger.debug(f"First block attention shape: {block_attentions[0].shape}")
        
        # Store the main attention tensor with shape determined by aggregation settings
        if block_attentions:
            # Stack to create [blocks, batch, heads, spatial, tokens]
            stacked_attention = torch.stack(block_attentions, dim=0)
            
            # Remove batch dimension to get [blocks, heads, spatial, tokens]
            full_step_attention = stacked_attention.squeeze(1)  # Remove batch dim
            
            self.logger.debug(f"Full step attention shape before aggregation: {full_step_attention.shape}")
            
            # Apply aggregation based on storage settings
            final_attention = full_step_attention
            aggregation_method = "full"
            
            # Aggregate across blocks if store_per_block is False
            if not self.store_per_block:
                final_attention = final_attention.mean(dim=0, keepdim=True)  # [1, heads, spatial, tokens]
                aggregation_method = "blocks_averaged"
                self.logger.debug(f"Aggregated across blocks: {final_attention.shape}")
            
            # Aggregate across heads if store_per_head is False
            if not self.store_per_head:
                final_attention = final_attention.mean(dim=1, keepdim=True)  # [blocks/1, 1, spatial, tokens]
                aggregation_method = "heads_averaged" if aggregation_method == "full" else "blocks_and_heads_averaged"
                self.logger.debug(f"Aggregated across heads: {final_attention.shape}")
            
            # self.logger.info(f"Final attention shape after aggregation ({aggregation_method}): {final_attention.shape}")
            
            # Store the aggregated tensor
            self._store_attention_tensor(
                final_attention, word_dir, filename_base,
                word, token_ids, step, timestep, total_steps,
                aggregation_method=aggregation_method
            )
    
    def _extract_token_attention(self, attention: torch.Tensor, token_ids: List[int]) -> torch.Tensor:
        """Extract attention weights for specific tokens."""
        # attention shape: [batch, heads, seq_len_spatial, seq_len_text]
        # We want cross-attention from spatial positions to specific text tokens
        
        if self.store_individual_tokens:
            # Store attention for each token separately
            # This would require a more complex storage structure
            pass
        
        # Find token positions in the current prompt's tokenized sequence
        token_positions = []
        if self.tokenizer and self.current_prompt:
            # Tokenize the current prompt to get the sequence
            tokens = self.tokenizer(
                self.current_prompt, 
                return_tensors="pt", 
                padding="max_length", 
                max_length=512, 
                truncation=True
            )
            token_sequence = tokens["input_ids"][0].tolist()
            
            # Debug: Log tokenization details
            self.logger.debug(f"Current prompt: '{self.current_prompt}'")
            self.logger.debug(f"Token sequence length: {len(token_sequence)}")
            self.logger.debug(f"First 20 tokens: {token_sequence[:20]}")
            self.logger.debug(f"Looking for token IDs: {token_ids}")
            
            # Also check without special tokens
            tokens_no_special = self.tokenizer(
                self.current_prompt, 
                return_tensors="pt", 
                add_special_tokens=False
            )
            token_sequence_no_special = tokens_no_special["input_ids"][0].tolist()
            self.logger.debug(f"Tokens without special tokens: {token_sequence_no_special}")
            
            # Find positions of our target token IDs in the sequence
            for token_id in token_ids:
                positions = [i for i, tid in enumerate(token_sequence) if tid == token_id]
                positions_no_special = [i for i, tid in enumerate(token_sequence_no_special) if tid == token_id]
                token_positions.extend(positions)
                
                self.logger.debug(f"Token ID {token_id} found at positions {positions} (with special) and {positions_no_special} (without special)")
            
            self.logger.debug(f"Token IDs {token_ids} found at positions {token_positions} in sequence")
        
        if not token_positions:
            self.logger.warning(f"Token IDs {token_ids} not found in tokenized sequence")
            # Return zeros with correct shape
            return torch.zeros(attention.shape[0], attention.shape[1], attention.shape[2], 1, 
                             dtype=attention.dtype, device=attention.device)
        
        # Extract attention for the found token positions
        token_attentions = []
        for pos in token_positions:
            if pos < attention.shape[-1]:  # Check if position is within sequence length
                token_attentions.append(attention[:, :, :, pos:pos+1])
        
        if token_attentions:
            if len(token_attentions) == 1:
                return token_attentions[0]
            else:
                # Average attention across multiple token positions
                return torch.cat(token_attentions, dim=-1).mean(dim=-1, keepdim=True)
        else:
            self.logger.error(f"No valid token positions found for token IDs {token_ids}")
            return torch.zeros(attention.shape[0], attention.shape[1], attention.shape[2], 1,
                             dtype=attention.dtype, device=attention.device)
    
    def _downsample_attention(self, attention: torch.Tensor) -> torch.Tensor:
        """Apply spatial downsampling to attention maps."""
        if self.spatial_downsample_factor <= 1:
            return attention
        
        # Simple average pooling downsampling
        # attention shape: [batch, seq_len, seq_len] 
        # For spatial attention, seq_len corresponds to spatial positions
        
        factor = self.spatial_downsample_factor
        b, h, w = attention.shape
        
        if h % factor == 0 and w % factor == 0:
            # Reshape and average pool
            attention = attention.view(b, h//factor, factor, w//factor, factor)
            attention = attention.mean(dim=[2, 4])
        
        return attention
    
    def _store_attention_tensor(self, attention: torch.Tensor, word_dir: Path, filename_base: str,
                               word: str, token_ids: List[int], step: int, timestep: float, total_steps: int,
                               block_idx: Optional[int] = None, head_idx: Optional[int] = None,
                               aggregation_method: str = "averaged"):
        """Store a single attention tensor with metadata."""
        
        # Apply dtype conversion if specified
        if self.storage_dtype == "float16":
            attention = attention.half()
        elif self.storage_dtype == "float32":
            attention = attention.float()
        
        # Create metadata
        token_texts = []
        if self.tokenizer:
            for token_id in token_ids:
                try:
                    token_text = self.tokenizer.decode([token_id])
                    token_texts.append(token_text)
                except:
                    token_texts.append(f"<token_{token_id}>")
        
        valid_params = {}
        if 'seed' in self.current_generation_params:
            valid_params['seed'] = self.current_generation_params['seed']
        if 'cfg_scale' in self.current_generation_params:
            valid_params['cfg_scale'] = self.current_generation_params['cfg_scale']
        
        # Extract video dimensions for spatial context reconstruction
        video_width = self.current_generation_params.get('width', None)
        video_height = self.current_generation_params.get('height', None)
        video_frames = self.current_generation_params.get('num_frames', None)
        
        metadata = AttentionMetadata(
            video_id=self.current_video_id,
            step=step,
            timestep=timestep,
            total_steps=total_steps,
            token_word=word,
            token_ids=token_ids,
            token_texts=token_texts,
            aggregation_method=aggregation_method,
            attention_shape=tuple(attention.shape),
            spatial_resolution=(attention.shape[-2], attention.shape[-1]),
            num_blocks=len(self.current_attention_maps),
            # For per_step_full: shape is [blocks, heads, spatial, tokens] 
            # For aggregated: shape is [batch, spatial, tokens] with heads=1
            num_heads=attention.shape[1] if aggregation_method == "per_step_full" else 1,
            threshold_applied=self.attention_threshold,
            block_idx=block_idx,
            head_idx=head_idx,
            dtype=str(attention.dtype),
            prompt=self.current_prompt,
            video_width=video_width,
            video_height=video_height,
            video_frames=video_frames,
            **valid_params
        )
        
        # Store attention tensor
        if self.storage_format == "numpy":
            attention_array = attention.numpy()
            attention_file = word_dir / f"{filename_base}.npy"
            
            if self.compress:
                attention_file = attention_file.with_suffix(".npy.gz")
                with gzip.open(attention_file, 'wb') as f:
                    np.save(f, attention_array)
            else:
                np.save(attention_file, attention_array)
                
        elif self.storage_format == "torch":
            attention_file = word_dir / f"{filename_base}.pt"
            
            if self.compress:
                attention_file = attention_file.with_suffix(".pt.gz")
                with gzip.open(attention_file, 'wb') as f:
                    torch.save(attention, f)
            else:
                torch.save(attention, attention_file)
        
        # Store metadata
        metadata_file = word_dir / f"{filename_base}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        self.logger.debug(f"Stored attention for {word} at step {step}: {attention_file}")
    
    def finish_video_storage(self) -> Dict[str, Any]:
        """Finish storing attention maps for the current video."""
        if self.current_video_id is None:
            return {}
        
        summary = {
            "video_id": self.current_video_id,
            "stored_steps": self.stored_steps.copy(),
            "total_stored": len(self.stored_steps),
            "target_tokens": self.target_tokens,
            "storage_dir": str(self.storage_dir),
            "storage_format": self.storage_format,
            "compressed": self.compress,
            "storage_type": "aggregated_per_step",  # NEW: Indicate storage approach
            "attention_shape_per_step": "Dynamic based on store_per_block/store_per_head settings",  # NEW: Document shape
            "config": {
                # NEW: Updated config that reflects actual storage behavior
                "store_full_per_step": self.store_full_per_step,  # Always stores per-step tensors
                "stores_all_blocks": self.store_per_block,   # Depends on setting
                "stores_all_heads": self.store_per_head,    # Depends on setting 
                "store_individual_tokens": self.store_individual_tokens,
                "attention_threshold": self.attention_threshold,
                "spatial_downsample_factor": self.spatial_downsample_factor,
                # Storage configuration (reflects actual settings)
                "store_per_head": self.store_per_head,
                "store_per_block": self.store_per_block,
            }
        }
        
        # Save summary to the individual video directory
        summary_file = self.current_video_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Finished storing attention maps for {self.current_video_id}: {len(self.stored_steps)} steps stored")
        
        # Reset for next video
        self.current_video_id = None
        self.current_video_dir = None
        self.current_prompt = None
        self.current_generation_params = {}
        self.stored_steps = []
        self.target_tokens = {}
        
        return summary
    
    def load_attention_map(self, video_id: str, token_word: str, step: int,
                          block_idx: Optional[int] = None, head_idx: Optional[int] = None) -> Optional[torch.Tensor]:
        """Load a stored attention map."""
        # Extract prompt and video parts from video_id
        if "_vid" in video_id:
            prompt_part, vid_num = video_id.split("_vid")
            vid_part = f"vid_{vid_num}"
        else:
            prompt_part = video_id
            vid_part = "vid_001"
        
        video_dir = self.attention_dir / prompt_part / vid_part
        word_dir = video_dir / f"token_{token_word}"
        
        # Build filename
        filename_base = f"step_{step:03d}"
        if block_idx is not None:
            filename_base += f"_block_{block_idx:02d}"
        if head_idx is not None:
            filename_base += f"_head_{head_idx:02d}"
        
        # Try different file extensions
        possible_files = [
            word_dir / f"{filename_base}.npy",
            word_dir / f"{filename_base}.npy.gz",
            word_dir / f"{filename_base}.pt",
            word_dir / f"{filename_base}.pt.gz"
        ]
        
        for attention_file in possible_files:
            if attention_file.exists():
                try:
                    if attention_file.suffix == '.gz':
                        with gzip.open(attention_file, 'rb') as f:
                            if '.npy' in attention_file.name:
                                attention_array = np.load(f)
                                return torch.from_numpy(attention_array)
                            elif '.pt' in attention_file.name:
                                return torch.load(f)
                    else:
                        if attention_file.suffix == '.npy':
                            attention_array = np.load(attention_file)
                            return torch.from_numpy(attention_array)
                        elif attention_file.suffix == '.pt':
                            return torch.load(attention_file)
                except Exception as e:
                    self.logger.error(f"Error loading attention from {attention_file}: {e}")
                    continue
        
        self.logger.warning(f"Attention map not found for video {video_id}, token {token_word}, step {step}")
        return None
    
    def load_attention_metadata(self, video_id: str, token_word: str, step: int,
                               block_idx: Optional[int] = None, head_idx: Optional[int] = None) -> Optional[AttentionMetadata]:
        """Load metadata for a stored attention map."""
        # Extract prompt and video parts from video_id
        if "_vid" in video_id:
            prompt_part, vid_num = video_id.split("_vid")
            vid_part = f"vid_{vid_num}"
        else:
            prompt_part = video_id
            vid_part = "vid_001"
        
        video_dir = self.attention_dir / prompt_part / vid_part
        word_dir = video_dir / f"token_{token_word}"
        
        # Build filename
        filename_base = f"step_{step:03d}"
        if block_idx is not None:
            filename_base += f"_block_{block_idx:02d}"
        if head_idx is not None:
            filename_base += f"_head_{head_idx:02d}"
        
        metadata_file = word_dir / f"{filename_base}_metadata.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                return AttentionMetadata(**metadata_dict)
            except Exception as e:
                self.logger.error(f"Error loading attention metadata from {metadata_file}: {e}")
        
        return None


def create_attention_callback(attention_storage: AttentionStorage):
    """
    Create a callback function for the diffusion pipeline to store attention maps.
    
    Args:
        attention_storage: AttentionStorage instance to use for storing
        
    Returns:
        Callback function compatible with diffusion pipelines
    """
    def callback(step: int, timestep: torch.Tensor, latents: torch.Tensor):
        """Callback function to store attention maps during denoising."""
        try:
            # Convert timestep to float if it's a tensor
            if torch.is_tensor(timestep):
                timestep_val = timestep.item()
            else:
                timestep_val = float(timestep)
            
            # Store attention maps for this step
            total_steps = 50  # Default assumption, should be configured
            attention_storage.store_attention_maps(
                step=step,
                timestep=timestep_val,
                total_steps=total_steps
            )
            
        except Exception as e:
            logging.error(f"Error in attention callback: {e}")
    
    return callback
