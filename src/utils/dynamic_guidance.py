"""
Dynamic Guidance Scale utilities for WAN video generation.
Allows modulating classifier-free guidance scale during generation based on a schedule.
"""

import logging
import traceback
from typing import Dict, Union, Optional, Callable
import torch


class GuidanceScheduler:
    """Handles scheduling and interpolation of guidance scale values."""
    
    def __init__(self, schedule: Dict[int, float], interpolation: str = "linear", total_steps: Optional[int] = None):
        """
        Initialize guidance scheduler.
        
        Args:
            schedule: Dictionary mapping step numbers to guidance scale values
                     e.g., {0: 6.5, 10: 3.5, 19: 0.0}
            interpolation: Interpolation method - "linear", "step", or "cosine"
            total_steps: Total number of inference steps (for validation)
        """
        self.schedule = dict(sorted(schedule.items()))  # Sort by step number
        self.interpolation = interpolation.lower()
        self.total_steps = total_steps
        self.logger = logging.getLogger(__name__)
        
        # Validate schedule
        self._validate_schedule()
        
    def _validate_schedule(self):
        """Validate the guidance schedule."""
        if not self.schedule:
            raise ValueError("Guidance schedule cannot be empty")
            
        # Check that all keys are non-negative integers
        for step in self.schedule.keys():
            if not isinstance(step, int) or step < 0:
                raise ValueError(f"Schedule step must be non-negative integer, got: {step}")
                
        # Check that all values are non-negative floats
        for step, value in self.schedule.items():
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"Guidance scale must be non-negative number, got: {value} at step {step}")
                
        # Warn if total_steps is known and schedule extends beyond it
        if self.total_steps is not None:
            max_step = max(self.schedule.keys())
            if max_step >= self.total_steps:
                self.logger.warning(f"Schedule step {max_step} >= total steps {self.total_steps}")
    
    def get_guidance_scale(self, step: int) -> float:
        """
        Get the guidance scale for a given step.
        
        Args:
            step: Current inference step (0-indexed)
            
        Returns:
            Guidance scale value for this step
        """
        # If step is directly in schedule, return it
        if step in self.schedule:
            return self.schedule[step]
            
        # Find the surrounding keyframes
        before_step = None
        after_step = None
        
        for sched_step in self.schedule.keys():
            if sched_step <= step:
                before_step = sched_step
            elif sched_step > step and after_step is None:
                after_step = sched_step
                break
                
        # Handle edge cases
        if before_step is None:
            # Step is before first scheduled point, use first value
            return list(self.schedule.values())[0]
            
        if after_step is None:
            # Step is after last scheduled point, use last value
            return self.schedule[before_step]
            
        # Interpolate between keyframes
        before_value = self.schedule[before_step]
        after_value = self.schedule[after_step]
        
        if self.interpolation == "step":
            # Step interpolation - use the "before" value until we hit the next keyframe
            return before_value
            
        elif self.interpolation == "linear":
            # Linear interpolation
            progress = (step - before_step) / (after_step - before_step)
            return before_value + progress * (after_value - before_value)
            
        elif self.interpolation == "cosine":
            # Cosine interpolation (smoother)
            progress = (step - before_step) / (after_step - before_step)
            cosine_progress = 0.5 * (1 - torch.cos(torch.tensor(progress * 3.14159)).item())
            return before_value + cosine_progress * (after_value - before_value)
            
        else:
            raise ValueError(f"Unknown interpolation method: {self.interpolation}")

    def generate_full_schedule(self, total_steps: int) -> Dict[int, float]:
        """
        Generate the full guidance scale schedule for all timesteps.
        
        Args:
            total_steps: Total number of inference steps
            
        Returns:
            Dictionary mapping each step (0 to total_steps-1) to its guidance scale value
        """
        full_schedule = {}
        for step in range(total_steps):
            full_schedule[step] = self.get_guidance_scale(step)
        return full_schedule


class DynamicGuidanceCallback:
    """Callback that dynamically adjusts guidance scale during generation."""
    
    def __init__(self, 
                 guidance_scheduler: GuidanceScheduler,
                 apply_to_guidance_2: bool = True,
                 verbose: bool = False):
        """
        Initialize dynamic guidance callback.
        
        Args:
            guidance_scheduler: GuidanceScheduler instance
            apply_to_guidance_2: Whether to also apply schedule to guidance_scale_2 (for dual-model WAN)
            verbose: Whether to log guidance scale changes
        """
        self.guidance_scheduler = guidance_scheduler
        self.apply_to_guidance_2 = apply_to_guidance_2
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        # Track original values for restoration
        self.original_guidance_scale = None
        self.original_guidance_scale_2 = None


    def __call__(self, pipe, step: int, timestep: torch.Tensor, callback_kwargs: dict):
        """Apply the guidance schedule at this step."""
        try:
            # Store original values on first call
            if self.original_guidance_scale is None:
                if hasattr(pipe, '_guidance_scale'):
                    self.original_guidance_scale = pipe._guidance_scale
                    
                if hasattr(pipe, '_guidance_scale_2'):
                    self.original_guidance_scale_2 = pipe._guidance_scale_2
            
            # Calculate the guidance value for this step
            guidance_value = self.guidance_scheduler.get_guidance_scale(step)
            
            # Apply to WAN pipeline - it stores guidance in _guidance_scale and _guidance_scale_2
            if hasattr(pipe, '_guidance_scale'):
                pipe._guidance_scale = guidance_value
                logging.debug(f"Step {step}: Set _guidance_scale to {guidance_value}")
                
                # Also set _guidance_scale_2 if it exists (for dual-stage WAN models)
                if hasattr(pipe, '_guidance_scale_2'):
                    pipe._guidance_scale_2 = guidance_value
                    logging.debug(f"Step {step}: Set _guidance_scale_2 to {guidance_value}")
                    
            elif hasattr(pipe, 'guidance_scale'):
                pipe.guidance_scale = guidance_value
                logging.debug(f"Step {step}: Set guidance_scale to {guidance_value}")
            else:
                logging.warning(f"Pipeline {type(pipe)} doesn't have a known guidance_scale attribute")
                
        except Exception as e:
            logging.error(f"Error in dynamic guidance callback: {e}")
            logging.error(traceback.format_exc())
            
        return callback_kwargs
    
    def restore_original_guidance(self, pipe):
        """Restore original guidance scale values after generation."""
        if self.original_guidance_scale is not None:
            pipe._guidance_scale = self.original_guidance_scale
            
        if self.original_guidance_scale_2 is not None and hasattr(pipe, '_guidance_scale_2'):
            pipe._guidance_scale_2 = self.original_guidance_scale_2


def create_guidance_callback(schedule: Dict[int, float], 
                           interpolation: str = "linear",
                           total_steps: Optional[int] = None,
                           apply_to_guidance_2: bool = True,
                           verbose: bool = False) -> DynamicGuidanceCallback:
    """
    Factory function to create a dynamic guidance callback.
    
    Args:
        schedule: Dictionary mapping step numbers to guidance scale values
        interpolation: Interpolation method - "linear", "step", or "cosine"
        total_steps: Total number of inference steps
        apply_to_guidance_2: Whether to apply to guidance_scale_2 as well
        verbose: Whether to log guidance changes
        
    Returns:
        DynamicGuidanceCallback instance
    """
    scheduler = GuidanceScheduler(schedule, interpolation, total_steps)
    return DynamicGuidanceCallback(
        guidance_scheduler=scheduler, 
        apply_to_guidance_2=apply_to_guidance_2, 
        verbose=verbose
    )


def parse_guidance_schedule_config(config_dict: Dict) -> Optional[DynamicGuidanceCallback]:
    """
    Parse guidance schedule configuration from config dictionary.
    
    Args:
        config_dict: Configuration dictionary containing cfg_schedule_settings
        
    Returns:
        DynamicGuidanceCallback if schedule is configured, None otherwise
    """
    cfg_schedule = config_dict.get('cfg_schedule_settings')
    if not cfg_schedule:
        return None
        
    schedule = cfg_schedule.get('schedule')
    if not schedule:
        return None
        
    interpolation = cfg_schedule.get('interpolation', 'linear')
    apply_to_guidance_2 = cfg_schedule.get('apply_to_guidance_2', True)
    verbose = cfg_schedule.get('verbose', False)
    
    # Convert string keys to integers if needed
    if isinstance(schedule, dict):
        schedule = {int(k): float(v) for k, v in schedule.items()}
    
    return create_guidance_callback(
        schedule=schedule,
        interpolation=interpolation,
        apply_to_guidance_2=apply_to_guidance_2,
        verbose=verbose
    )
