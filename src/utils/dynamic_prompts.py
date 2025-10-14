"""
Dynamic Prompt Interpolation utilities for WAN video generation.
Allows smoothly transitioning between different prompts during the diffusion process.

This module enables "prompt bending" - morphing the semantic guidance during generation
to explore creative transitions and hybrid concepts.
"""

import logging
import traceback
import torch
import math
from typing import Dict, Optional, Callable, List, Tuple
from pathlib import Path
import json


class PromptScheduler:
    """Handles scheduling and interpolation of prompts across diffusion steps."""
    
    def __init__(self, 
                 schedule: Dict[int, str], 
                 interpolation: str = "slerp",
                 total_steps: Optional[int] = None):
        """
        Initialize prompt scheduler.
        
        Args:
            schedule: Dictionary mapping step numbers to prompt strings
                     e.g., {0: "a person smiling in a sunny park", 
                           20: "a person crying in a dark dungeon"}
            interpolation: Interpolation method - "slerp" (spherical), "lerp" (linear), or "step"
            total_steps: Total number of inference steps (for validation)
        """
        self.schedule = dict(sorted(schedule.items()))  # Sort by step number
        self.interpolation = interpolation.lower()
        self.total_steps = total_steps
        self.logger = logging.getLogger(__name__)
        
        # Cache for embeddings to avoid recomputation
        self._embedding_cache = {}
        
        # Validate schedule
        self._validate_schedule()
        
    def _validate_schedule(self):
        """Validate the prompt schedule."""
        if not self.schedule:
            raise ValueError("Prompt schedule cannot be empty")
            
        # Check that all keys are non-negative integers
        for step in self.schedule.keys():
            if not isinstance(step, int) or step < 0:
                raise ValueError(f"Schedule step must be non-negative integer, got: {step}")
                
        # Check that all values are non-empty strings
        for step, prompt in self.schedule.items():
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Prompt must be non-empty string, got: {prompt} at step {step}")
                
        # Warn if total_steps is known and schedule extends beyond it
        if self.total_steps is not None:
            max_step = max(self.schedule.keys())
            if max_step >= self.total_steps:
                self.logger.warning(f"Schedule step {max_step} >= total steps {self.total_steps}")
        
        self.logger.info(f"📝 Prompt schedule validated with {len(self.schedule)} keyframes")
        for step, prompt in self.schedule.items():
            self.logger.info(f"   Step {step}: '{prompt[:60]}{'...' if len(prompt) > 60 else ''}'")
    
    def get_surrounding_keyframes(self, step: int) -> Tuple[Optional[int], Optional[int]]:
        """
        Get the keyframe steps surrounding a given step.
        
        Args:
            step: Current inference step (0-indexed)
            
        Returns:
            Tuple of (before_step, after_step). Either can be None at boundaries.
        """
        before_step = None
        after_step = None
        
        for sched_step in self.schedule.keys():
            if sched_step <= step:
                before_step = sched_step
            elif sched_step > step and after_step is None:
                after_step = sched_step
                break
                
        return before_step, after_step
    
    def get_prompt_for_step(self, step: int) -> str:
        """
        Get the raw prompt string for a given step (no interpolation).
        Used for step interpolation mode or getting keyframe prompts.
        
        Args:
            step: Current inference step (0-indexed)
            
        Returns:
            Prompt string for this step
        """
        # If step is directly in schedule, return it
        if step in self.schedule:
            return self.schedule[step]
            
        # Find surrounding keyframes
        before_step, after_step = self.get_surrounding_keyframes(step)
        
        # Handle edge cases
        if before_step is None:
            # Step is before first scheduled point, use first value
            return list(self.schedule.values())[0]
            
        if after_step is None:
            # Step is after last scheduled point, use last value
            return self.schedule[before_step]
            
        # For step interpolation, use the "before" prompt
        if self.interpolation == "step":
            return self.schedule[before_step]
        
        # For smooth interpolation, we need embeddings (handled by get_interpolated_embeddings)
        # This shouldn't be called for lerp/slerp, but return before prompt as fallback
        return self.schedule[before_step]
    
    def get_interpolated_embeddings(self, 
                                   step: int,
                                   pipe,
                                   device: str) -> torch.Tensor:
        """
        Get interpolated prompt embeddings for a given step.
        
        Args:
            step: Current inference step (0-indexed)
            pipe: The WAN pipeline with text encoder
            device: Device to create tensors on
            
        Returns:
            Interpolated prompt embeddings tensor
        """
        # If step is directly in schedule, return its embedding
        if step in self.schedule:
            return self._get_or_compute_embedding(step, pipe, device)
            
        # Find surrounding keyframes
        before_step, after_step = self.get_surrounding_keyframes(step)
        
        # Handle edge cases
        if before_step is None:
            # Step is before first scheduled point, use first embedding
            first_step = min(self.schedule.keys())
            return self._get_or_compute_embedding(first_step, pipe, device)
            
        if after_step is None:
            # Step is after last scheduled point, use last embedding
            return self._get_or_compute_embedding(before_step, pipe, device)
            
        # Get embeddings for surrounding keyframes
        before_embeds = self._get_or_compute_embedding(before_step, pipe, device)
        after_embeds = self._get_or_compute_embedding(after_step, pipe, device)
        
        # Calculate interpolation progress
        progress = (step - before_step) / (after_step - before_step)
        
        # Interpolate based on method
        if self.interpolation == "step":
            # Step interpolation - use before embedding until next keyframe
            return before_embeds
            
        elif self.interpolation == "lerp":
            # Linear interpolation in embedding space
            return self._lerp_embeddings(before_embeds, after_embeds, progress)
            
        elif self.interpolation == "slerp":
            # Spherical linear interpolation (better for high-dimensional spaces)
            return self._slerp_embeddings(before_embeds, after_embeds, progress)
            
        else:
            raise ValueError(f"Unknown interpolation method: {self.interpolation}")
    
    def _get_or_compute_embedding(self, step: int, pipe, device: str) -> torch.Tensor:
        """Get cached embedding or compute it."""
        if step not in self._embedding_cache:
            prompt = self.schedule[step]
            self._embedding_cache[step] = self._compute_embedding(prompt, pipe, device)
            self.logger.debug(f"Computed and cached embedding for step {step}")
        return self._embedding_cache[step]
    
    def _compute_embedding(self, prompt: str, pipe, device: str) -> torch.Tensor:
        """Compute text embedding for a prompt using the pipeline's encode_prompt method."""
        try:
            # CRITICAL: Use the pipeline's encode_prompt method to ensure we get
            # the exact same embeddings as normal generation would produce.
            # This handles WAN-specific preprocessing like prompt_clean(), attention masks,
            # sequence length padding, etc.
            prompt_embeds, _ = pipe.encode_prompt(
                prompt=prompt,
                negative_prompt=None,  # We don't need negative here, just get positive
                do_classifier_free_guidance=False,  # Don't need CFG processing
                num_videos_per_prompt=1,
                device=device,
            )
            
            self.logger.debug(f"Computed embedding using pipe.encode_prompt: shape={prompt_embeds.shape}, dtype={prompt_embeds.dtype}")
            return prompt_embeds
            
        except Exception as e:
            self.logger.error(f"Error computing embedding for prompt '{prompt[:50]}...': {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def _lerp_embeddings(self, 
                        embed1: torch.Tensor, 
                        embed2: torch.Tensor, 
                        t: float) -> torch.Tensor:
        """
        Linear interpolation between two embeddings.
        
        Args:
            embed1: First embedding
            embed2: Second embedding
            t: Interpolation factor (0.0 = embed1, 1.0 = embed2)
            
        Returns:
            Interpolated embedding
        """
        return (1.0 - t) * embed1 + t * embed2
    
    def _slerp_embeddings(self, 
                         embed1: torch.Tensor, 
                         embed2: torch.Tensor, 
                         t: float) -> torch.Tensor:
        """
        Spherical linear interpolation between two embeddings.
        This is generally better for high-dimensional embedding spaces as it
        maintains the "magnitude" better and creates smoother semantic transitions.
        
        Args:
            embed1: First embedding
            embed2: Second embedding
            t: Interpolation factor (0.0 = embed1, 1.0 = embed2)
            
        Returns:
            Interpolated embedding
        """
        # Normalize embeddings
        embed1_norm = embed1 / (torch.norm(embed1, dim=-1, keepdim=True) + 1e-8)
        embed2_norm = embed2 / (torch.norm(embed2, dim=-1, keepdim=True) + 1e-8)
        
        # Calculate angle between embeddings
        dot_product = (embed1_norm * embed2_norm).sum(dim=-1, keepdim=True)
        # Clamp to avoid numerical issues with arccos
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        omega = torch.acos(dot_product)
        
        # Handle near-parallel vectors (fall back to lerp)
        sin_omega = torch.sin(omega)
        if (sin_omega.abs() < 1e-6).any():
            self.logger.debug("Embeddings nearly parallel, using lerp instead of slerp")
            return self._lerp_embeddings(embed1, embed2, t)
        
        # Compute slerp
        coef1 = torch.sin((1.0 - t) * omega) / sin_omega
        coef2 = torch.sin(t * omega) / sin_omega
        
        return coef1 * embed1 + coef2 * embed2
    
    def generate_full_schedule(self, total_steps: int) -> Dict[int, str]:
        """
        Generate the full prompt schedule for all timesteps.
        
        Args:
            total_steps: Total number of inference steps
            
        Returns:
            Dictionary mapping each step to its prompt string
        """
        full_schedule = {}
        for step in range(total_steps):
            full_schedule[step] = self.get_prompt_for_step(step)
        return full_schedule


class DynamicPromptEmbeddings:
    """
    Wrapper for prompt embeddings that can be updated dynamically.
    Unlike DynamicScale which is a simple number, this needs to handle tensor operations.
    """
    
    def __init__(self, initial_embeddings: torch.Tensor):
        """
        Initialize with starting embeddings.
        
        Args:
            initial_embeddings: Initial prompt embeddings tensor
        """
        self._embeddings = initial_embeddings.clone()
        self._device = initial_embeddings.device
        self._dtype = initial_embeddings.dtype
        self.logger = logging.getLogger(__name__)
    
    def update(self, new_embeddings: torch.Tensor):
        """Update the embeddings in-place."""
        if new_embeddings.shape != self._embeddings.shape:
            self.logger.warning(
                f"Embedding shape mismatch: current {self._embeddings.shape}, "
                f"new {new_embeddings.shape}. Skipping update."
            )
            return
        
        # Log statistics before update
        old_norm = torch.norm(self._embeddings).item()
        new_norm = torch.norm(new_embeddings).item()
        
        self._embeddings.copy_(new_embeddings)
        
        self.logger.debug(
            f"✅ Updated embeddings - Old norm: {old_norm:.4f}, New norm: {new_norm:.4f}, "
            f"Shape: {self._embeddings.shape}, Device: {self._embeddings.device}"
        )
    
    def get(self) -> torch.Tensor:
        """Get current embeddings."""
        return self._embeddings
    
    @property
    def device(self):
        return self._device
    
    @property
    def dtype(self):
        return self._dtype
    
    @property
    def shape(self):
        return self._embeddings.shape


class DynamicPromptCallback:
    """Callback that dynamically updates prompt embeddings during generation."""
    
    def __init__(self,
                 prompt_scheduler: PromptScheduler,
                 dynamic_embeddings: DynamicPromptEmbeddings,
                 pipe,
                 device: str,
                 verbose: bool = False):
        """
        Initialize dynamic prompt callback.
        
        Args:
            prompt_scheduler: PromptScheduler instance
            dynamic_embeddings: DynamicPromptEmbeddings instance to update
            pipe: The WAN pipeline with text encoder
            device: Device for tensor operations
            verbose: Whether to log prompt changes
        """
        self.prompt_scheduler = prompt_scheduler
        self.dynamic_embeddings = dynamic_embeddings
        self.pipe = pipe
        self.device = device
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        # Track previous step to avoid redundant updates
        self._last_step = -1
    
    def __call__(self, pipe, step: int, timestep: torch.Tensor, callback_kwargs: dict):
        """Update the prompt embeddings for the next step."""
        try:
            # Log what's in callback_kwargs
            if step == 0:
                self.logger.info(f"🔍 Callback kwargs keys at step {step}: {list(callback_kwargs.keys())}")
            
            # Only update if we've moved to a new step
            if step == self._last_step:
                return callback_kwargs
            
            self._last_step = step
            
            # Calculate embeddings for the NEXT step (since callback runs AFTER current step)
            next_step = step + 1
            
            # Log current state
            if step == 0 or self.verbose:
                current_emb_norm = torch.norm(self.dynamic_embeddings.get()).item()
                self.logger.info(
                    f"🎨 Prompt callback at step {step} → {next_step}, "
                    f"current embedding norm: {current_emb_norm:.4f}"
                )
            
            # Get interpolated embeddings for next step
            next_embeddings = self.prompt_scheduler.get_interpolated_embeddings(
                next_step, self.pipe, self.device
            )
            
            # CRITICAL: Ensure embeddings are in the correct dtype for the transformer
            # The WAN pipeline converts embeddings with .to(transformer_dtype) which creates a new tensor
            # We need to match that dtype in our callback to ensure compatibility
            if 'prompt_embeds' in callback_kwargs and callback_kwargs['prompt_embeds'] is not None:
                target_dtype = callback_kwargs['prompt_embeds'].dtype
                if next_embeddings.dtype != target_dtype:
                    self.logger.info(f"🔧 Converting embeddings from {next_embeddings.dtype} to {target_dtype}")
                    next_embeddings = next_embeddings.to(target_dtype)
            
            # Log what we got
            next_emb_norm = torch.norm(next_embeddings).item()
            if step == 0 or self.verbose:
                self.logger.info(
                    f"🔄 Generated embeddings for step {next_step}: "
                    f"shape={next_embeddings.shape}, norm={next_emb_norm:.4f}, "
                    f"device={next_embeddings.device}, dtype={next_embeddings.dtype}"
                )
            
            # Update the dynamic embeddings
            self.dynamic_embeddings.update(next_embeddings)
            
            # CRITICAL FIX: Only update callback_kwargs if embeddings have actually changed
            # This prevents unnecessary updates when using a single keyframe
            if 'prompt_embeds' in callback_kwargs:
                current_embeds = callback_kwargs['prompt_embeds']
                
                # Check if embeddings have changed (with tolerance for numerical precision)
                if torch.allclose(current_embeds, next_embeddings, rtol=1e-6, atol=1e-6):
                    if step == 0:
                        self.logger.info(f"✅ Embeddings unchanged - skipping update (single keyframe or identical prompts)")
                    # Don't update callback_kwargs - return as-is to maintain exact numerical state
                else:
                    old_norm = torch.norm(current_embeds).item()
                    callback_kwargs['prompt_embeds'] = next_embeddings
                    if step == 0 or self.verbose:
                        self.logger.info(
                            f"✅ Updated callback_kwargs['prompt_embeds']: "
                            f"old_norm={old_norm:.4f}, new_norm={next_emb_norm:.4f}, "
                            f"dtype={next_embeddings.dtype}"
                        )
            else:
                self.logger.warning(f"⚠️ 'prompt_embeds' not in callback_kwargs. Available keys: {list(callback_kwargs.keys())}")
            
            # Also update negative_prompt_embeds if present (for CFG)
            # Keep them unchanged but ensure they're in callback_kwargs for the pipeline to use
            if 'negative_prompt_embeds' in callback_kwargs and step == 0:
                self.logger.info(f"✅ negative_prompt_embeds present in callback_kwargs (unchanged)")
            
            # Log the change
            if self.verbose:
                # Get prompt text for display
                before_step, after_step = self.prompt_scheduler.get_surrounding_keyframes(next_step)
                
                if next_step in self.prompt_scheduler.schedule:
                    prompt_text = self.prompt_scheduler.schedule[next_step]
                    self.logger.info(
                        f"🎨 Step {step} → {next_step}: KEYFRAME prompt: "
                        f"'{prompt_text[:50]}{'...' if len(prompt_text) > 50 else ''}'"
                    )
                elif before_step is not None and after_step is not None:
                    progress = (next_step - before_step) / (after_step - before_step)
                    before_prompt = self.prompt_scheduler.schedule[before_step]
                    after_prompt = self.prompt_scheduler.schedule[after_step]
                    self.logger.info(
                        f"🎨 Step {step} → {next_step}: Interpolating {progress:.2%} between:\n"
                        f"     [{before_step}] '{before_prompt[:40]}...'\n"
                        f"     [{after_step}] '{after_prompt[:40]}...'"
                    )
                
        except Exception as e:
            self.logger.error(f"❌ Error in dynamic prompt callback: {e}")
            self.logger.error(traceback.format_exc())
            
        return callback_kwargs


def create_prompt_callback(schedule: Dict[int, str],
                          interpolation: str = "slerp",
                          total_steps: Optional[int] = None,
                          pipe = None,
                          device: str = "cuda",
                          verbose: bool = False) -> Tuple[DynamicPromptEmbeddings, DynamicPromptCallback]:
    """
    Factory function to create dynamic prompt embeddings and callback.
    
    Args:
        schedule: Dictionary mapping step numbers to prompt strings
        interpolation: Interpolation method - "slerp", "lerp", or "step"
        total_steps: Total number of inference steps
        pipe: The WAN pipeline with text encoder
        device: Device for tensor operations
        verbose: Whether to log prompt changes
        
    Returns:
        Tuple of (DynamicPromptEmbeddings, DynamicPromptCallback)
    """
    if pipe is None:
        raise ValueError("pipe must be provided to create prompt callback")
    
    logger = logging.getLogger(__name__)
    
    scheduler = PromptScheduler(schedule, interpolation, total_steps)
    
    # Get initial embeddings (for step 0)
    logger.info("🎬 Creating initial prompt embeddings for step 0...")
    initial_embeddings = scheduler.get_interpolated_embeddings(0, pipe, device)
    
    # Log initial embedding statistics
    logger.info(f"✨ Initial embedding stats:")
    logger.info(f"   Shape: {initial_embeddings.shape}")
    logger.info(f"   Device: {initial_embeddings.device}")
    logger.info(f"   Dtype: {initial_embeddings.dtype}")
    logger.info(f"   Norm: {torch.norm(initial_embeddings).item():.4f}")
    logger.info(f"   Mean: {initial_embeddings.mean().item():.4f}")
    logger.info(f"   Std: {initial_embeddings.std().item():.4f}")
    
    # Check for NaN/Inf in initial embeddings
    if torch.isnan(initial_embeddings).any():
        logger.error("❌ CRITICAL: Initial embeddings contain NaN values!")
    if torch.isinf(initial_embeddings).any():
        logger.error("❌ CRITICAL: Initial embeddings contain Inf values!")
    
    # Get the prompt text for step 0
    step_0_prompt = schedule.get(0, list(schedule.values())[0])
    logger.info(f"📝 Step 0 prompt: '{step_0_prompt[:80]}{'...' if len(step_0_prompt) > 80 else ''}'")
    
    # Create dynamic embeddings wrapper
    dynamic_embeddings = DynamicPromptEmbeddings(initial_embeddings)
    logger.info(f"✅ Created DynamicPromptEmbeddings wrapper")
    
    # Create callback that updates the embeddings
    callback = DynamicPromptCallback(
        prompt_scheduler=scheduler,
        dynamic_embeddings=dynamic_embeddings,
        pipe=pipe,
        device=device,
        verbose=verbose
    )
    logger.info(f"✅ Created DynamicPromptCallback")
    
    return dynamic_embeddings, callback


def save_prompt_schedule(schedule: Dict[int, str], 
                        interpolation: str,
                        total_steps: int,
                        output_path: str):
    """
    Save the prompt schedule configuration to a JSON file.
    
    Args:
        schedule: Dictionary mapping steps to prompts
        interpolation: Interpolation method used
        total_steps: Total number of steps
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    schedule_data = {
        "schedule": schedule,
        "interpolation": interpolation,
        "total_steps": total_steps,
        "keyframes": list(schedule.keys())
    }
    
    with open(output_path, 'w') as f:
        json.dump(schedule_data, f, indent=2)
    
    logging.info(f"Saved prompt schedule to {output_path}")
