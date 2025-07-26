"""
WAN-specific weighted embeddings implementation.
Based on analysis of WAN pipeline and T5 encoder behavior.
"""

import torch
import logging
from typing import List, Tuple, Dict, Optional, Union
from .prompt_weighting import PromptWeightingProcessor, WeightedSegment

logger = logging.getLogger(__name__)


class WANWeightedEmbeddings:
    """
    WAN-specific implementation of weighted embeddings that follows the exact
    pattern used by the WAN pipeline for T5 text encoding.
    """
    
    def __init__(self, pipe):
        self.pipe = pipe
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.device = pipe.device
        self.processor = PromptWeightingProcessor()
    
    def _get_token_positions(self, full_tokens: List[int], target_tokens: List[int]) -> List[int]:
        """
        Find positions of target tokens within the full token sequence.
        This is more robust than simple matching.
        """
        positions = []
        target_idx = 0
        
        for i, token in enumerate(full_tokens):
            if target_idx < len(target_tokens) and token == target_tokens[target_idx]:
                positions.append(i)
                target_idx += 1
                
                # If we've found all target tokens in sequence, we're done
                if target_idx >= len(target_tokens):
                    break
        
        return positions
    
    def _tokenize_segment(self, text: str) -> Tuple[List[int], int]:
        """
        Tokenize a text segment and return tokens and their count.
        """
        if not text.strip():
            return [], 0
            
        # Tokenize without special tokens to get just the content tokens
        tokens = self.tokenizer.encode(text.strip(), add_special_tokens=False)
        return tokens, len(tokens)
    
    def _create_weighted_embeddings_t5(
        self, 
        prompt: str, 
        max_sequence_length: int = 512,
        num_videos_per_prompt: int = 1
    ) -> torch.Tensor:
        """
        Create weighted embeddings following WAN's T5 encoding pattern exactly.
        """
        logger.info(f"Creating T5 weighted embeddings for: {prompt}")
        
        # Parse the weighted prompt
        segments = self.processor.parse_weighted_prompt(prompt)
        
        # Check if we actually have weights
        has_weights = any(seg.weight != 1.0 for seg in segments)
        if not has_weights:
            logger.info("No weights found, using standard encoding")
            return self._get_standard_t5_embeddings(prompt, max_sequence_length, num_videos_per_prompt)
        
        # Build clean prompt and analyze token structure
        clean_prompt = "".join(seg.text for seg in segments)
        
        # Tokenize the full prompt to understand the structure
        text_inputs = self.tokenizer(
            [clean_prompt],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        
        input_ids = text_inputs.input_ids[0]  # [seq_len]
        attention_mask = text_inputs.attention_mask[0]  # [seq_len]
        seq_len = attention_mask.sum().item()
        
        logger.info(f"Full sequence length: {seq_len}, Max length: {max_sequence_length}")
        
        # Get base embeddings
        with torch.no_grad():
            base_embeddings = self.text_encoder(
                input_ids.unsqueeze(0).to(self.device), 
                attention_mask.unsqueeze(0).to(self.device)
            ).last_hidden_state  # [1, seq_len, hidden_dim]
        
        # Create weight map for each token position
        token_weights = torch.ones(seq_len, device=self.device, dtype=base_embeddings.dtype)
        
        # Map weights to token positions
        current_pos = 1  # Skip the start token
        
        for segment in segments:
            if not segment.text.strip():
                continue
                
            # Tokenize this segment
            segment_tokens, segment_len = self._tokenize_segment(segment.text)
            
            if segment_len > 0 and current_pos + segment_len <= seq_len:
                # Apply weight to this segment's tokens
                end_pos = current_pos + segment_len
                if segment.weight != 1.0:
                    token_weights[current_pos:end_pos] = segment.weight
                    logger.debug(f"Applied weight {segment.weight} to positions {current_pos}:{end_pos} for '{segment.text.strip()}'")
                
                current_pos = end_pos
        
        logger.info(f"Token weights applied: min={token_weights.min():.3f}, max={token_weights.max():.3f}, mean={token_weights.mean():.3f}")
        
        # Apply weights using different methods
        weighted_embeddings = self._apply_token_weights(
            base_embeddings, 
            token_weights, 
            method="interpolation"  # Start with interpolation method
        )
        
        # Follow WAN's exact embedding processing pattern
        weighted_embeddings = weighted_embeddings.to(dtype=self.text_encoder.dtype, device=self.device)
        
        # Trim to actual sequence length and pad like WAN does
        actual_seq_len = seq_len
        trimmed_embeddings = weighted_embeddings[0, :actual_seq_len]  # [actual_seq_len, hidden_dim]
        
        # Pad back to max_sequence_length following WAN pattern
        hidden_dim = trimmed_embeddings.size(-1)
        padded_embeddings = torch.zeros(max_sequence_length, hidden_dim, dtype=trimmed_embeddings.dtype, device=self.device)
        padded_embeddings[:actual_seq_len] = trimmed_embeddings
        
        # Duplicate for num_videos_per_prompt following WAN pattern
        final_embeddings = padded_embeddings.unsqueeze(0).repeat(num_videos_per_prompt, 1, 1)
        
        logger.info(f"Final weighted embeddings shape: {final_embeddings.shape}")
        return final_embeddings
    
    def _apply_token_weights(self, embeddings: torch.Tensor, weights: torch.Tensor, method: str = "interpolation") -> torch.Tensor:
        """
        Apply token weights to embeddings using different methods.
        
        Args:
            embeddings: [1, seq_len, hidden_dim]
            weights: [seq_len]
            method: "multiply", "interpolation", or "norm_preserving"
        """
        if method == "multiply":
            # Simple multiplication (what we tried before)
            weighted = embeddings * weights.unsqueeze(0).unsqueeze(-1)
            
        elif method == "interpolation":
            # Interpolation with EOS token (like SD_embed method 2)
            eos_embedding = embeddings[0, -1:, :]  # [1, hidden_dim] - last token
            
            # Expand EOS to match sequence length
            eos_expanded = eos_embedding.expand(1, embeddings.size(1), -1)  # [1, seq_len, hidden_dim]
            
            # Interpolate between original and EOS based on weight
            # weight=1.0 -> original, weight>1.0 -> move away from EOS, weight<1.0 -> move toward EOS
            weight_factors = weights.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
            weighted = eos_expanded + (embeddings - eos_expanded) * weight_factors
            
        elif method == "norm_preserving":
            # Preserve the norm of embeddings while applying weights
            original_norms = torch.norm(embeddings, dim=-1, keepdim=True)  # [1, seq_len, 1]
            
            # Apply weights
            weighted = embeddings * weights.unsqueeze(0).unsqueeze(-1)
            
            # Restore original norms
            weighted_norms = torch.norm(weighted, dim=-1, keepdim=True)
            weighted = weighted * (original_norms / (weighted_norms + 1e-8))
            
        else:
            raise ValueError(f"Unknown weighting method: {method}")
        
        return weighted
    
    def _get_standard_t5_embeddings(self, prompt: str, max_sequence_length: int, num_videos_per_prompt: int) -> torch.Tensor:
        """Get standard T5 embeddings using WAN's exact method."""
        return self.pipe._get_t5_prompt_embeds(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=self.device,
            dtype=self.text_encoder.dtype
        )
    
    def create_weighted_embeddings(
        self, 
        prompt: str, 
        max_sequence_length: int = 512,
        num_videos_per_prompt: int = 1,
        weighting_method: str = "interpolation"
    ) -> torch.Tensor:
        """
        Main interface for creating weighted embeddings.
        
        Args:
            prompt: Prompt with weight syntax like "(dancing:1.5)"
            max_sequence_length: Maximum sequence length for T5
            num_videos_per_prompt: Number of videos per prompt
            weighting_method: "multiply", "interpolation", or "norm_preserving"
        
        Returns:
            Weighted embeddings tensor ready for WAN pipeline
        """
        try:
            # Store the method for _apply_token_weights
            self._current_method = weighting_method
            
            # Create weighted embeddings
            embeddings = self._create_weighted_embeddings_t5(
                prompt, 
                max_sequence_length, 
                num_videos_per_prompt
            )
            
            # Validate embeddings
            if torch.isnan(embeddings).any():
                raise ValueError("Generated embeddings contain NaN values")
            if torch.isinf(embeddings).any():
                raise ValueError("Generated embeddings contain infinite values")
            
            # Check if embeddings are in reasonable range
            mean_val = embeddings.mean().item()
            std_val = embeddings.std().item()
            max_val = embeddings.max().item()
            min_val = embeddings.min().item()
            
            logger.info(f"Weighted embeddings stats: mean={mean_val:.6f}, std={std_val:.6f}, min={min_val:.6f}, max={max_val:.6f}")
            
            # Sanity check - compare with standard embeddings
            clean_prompt = "".join(seg.text for seg in self.processor.parse_weighted_prompt(prompt))
            standard_embeddings = self._get_standard_t5_embeddings(clean_prompt, max_sequence_length, num_videos_per_prompt)
            
            # Calculate difference
            diff_norm = torch.norm(embeddings - standard_embeddings).item()
            relative_diff = diff_norm / torch.norm(standard_embeddings).item()
            
            logger.info(f"Difference from standard: L2={diff_norm:.6f}, relative={relative_diff:.6f} ({relative_diff*100:.2f}%)")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to create weighted embeddings: {e}")
            logger.info("Falling back to standard embeddings")
            
            clean_prompt = "".join(seg.text for seg in self.processor.parse_weighted_prompt(prompt))
            return self._get_standard_t5_embeddings(clean_prompt, max_sequence_length, num_videos_per_prompt)
    
    # Update the _apply_token_weights call to use the stored method
    def _create_weighted_embeddings_t5(
        self, 
        prompt: str, 
        max_sequence_length: int = 512,
        num_videos_per_prompt: int = 1
    ) -> torch.Tensor:
        """
        Create weighted embeddings following WAN's T5 encoding pattern exactly.
        """
        logger.info(f"Creating T5 weighted embeddings for: {prompt}")
        
        # Parse the weighted prompt
        segments = self.processor.parse_weighted_prompt(prompt)
        
        # Check if we actually have weights
        has_weights = any(seg.weight != 1.0 for seg in segments)
        if not has_weights:
            logger.info("No weights found, using standard encoding")
            return self._get_standard_t5_embeddings(prompt, max_sequence_length, num_videos_per_prompt)
        
        # Build clean prompt and analyze token structure
        clean_prompt = "".join(seg.text for seg in segments)
        
        # Tokenize the full prompt to understand the structure
        text_inputs = self.tokenizer(
            [clean_prompt],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        
        input_ids = text_inputs.input_ids[0]  # [seq_len]
        attention_mask = text_inputs.attention_mask[0]  # [seq_len]
        seq_len = attention_mask.sum().item()
        
        logger.info(f"Full sequence length: {seq_len}, Max length: {max_sequence_length}")
        
        # Get base embeddings
        with torch.no_grad():
            base_embeddings = self.text_encoder(
                input_ids.unsqueeze(0).to(self.device), 
                attention_mask.unsqueeze(0).to(self.device)
            ).last_hidden_state  # [1, seq_len, hidden_dim]
        
        # Create weight map for each token position
        token_weights = torch.ones(seq_len, device=self.device, dtype=base_embeddings.dtype)
        
        # Map weights to token positions
        current_pos = 1  # Skip the start token
        
        for segment in segments:
            if not segment.text.strip():
                continue
                
            # Tokenize this segment
            segment_tokens, segment_len = self._tokenize_segment(segment.text)
            
            if segment_len > 0 and current_pos + segment_len <= seq_len:
                # Apply weight to this segment's tokens
                end_pos = current_pos + segment_len
                if segment.weight != 1.0:
                    token_weights[current_pos:end_pos] = segment.weight
                    logger.debug(f"Applied weight {segment.weight} to positions {current_pos}:{end_pos} for '{segment.text.strip()}'")
                
                current_pos = end_pos
        
        logger.info(f"Token weights applied: min={token_weights.min():.3f}, max={token_weights.max():.3f}, mean={token_weights.mean():.3f}")
        
        # Apply weights using the specified method
        weighted_embeddings = self._apply_token_weights(
            base_embeddings, 
            token_weights, 
            method=getattr(self, '_current_method', 'interpolation')
        )
        
        # Follow WAN's exact embedding processing pattern
        weighted_embeddings = weighted_embeddings.to(dtype=self.text_encoder.dtype, device=self.device)
        
        # Trim to actual sequence length and pad like WAN does
        actual_seq_len = seq_len
        trimmed_embeddings = weighted_embeddings[0, :actual_seq_len]  # [actual_seq_len, hidden_dim]
        
        # Pad back to max_sequence_length following WAN pattern
        hidden_dim = trimmed_embeddings.size(-1)
        padded_embeddings = torch.zeros(max_sequence_length, hidden_dim, dtype=trimmed_embeddings.dtype, device=self.device)
        padded_embeddings[:actual_seq_len] = trimmed_embeddings
        
        # Duplicate for num_videos_per_prompt following WAN pattern
        final_embeddings = padded_embeddings.unsqueeze(0).repeat(num_videos_per_prompt, 1, 1)
        
        logger.info(f"Final weighted embeddings shape: {final_embeddings.shape}")
        return final_embeddings


def create_wan_weighted_embeddings(pipe, prompt: str, max_sequence_length: int = 512, weighting_method: str = "interpolation") -> torch.Tensor:
    """
    Convenience function to create WAN weighted embeddings.
    
    Args:
        pipe: WAN pipeline instance
        prompt: Prompt with weight syntax
        max_sequence_length: Max sequence length for T5
        weighting_method: "multiply", "interpolation", or "norm_preserving"
    
    Returns:
        Weighted embeddings tensor
    """
    wan_embeddings = WANWeightedEmbeddings(pipe)
    return wan_embeddings.create_weighted_embeddings(
        prompt=prompt,
        max_sequence_length=max_sequence_length,
        weighting_method=weighting_method
    )
