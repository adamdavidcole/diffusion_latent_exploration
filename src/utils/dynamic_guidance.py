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


class DynamicScale:
    """Number-like object whose value you can update on the fly."""
    def __init__(self, get_value):
        self._get_value = get_value  # callable returning the current float

    # arithmetic: used in `current_guidance_scale * (tensor)`
    def __mul__(self, other):
        # left operand is self; right is a torch.Tensor
        return other * float(self)  # lets torch handle tensor * float

    def __rmul__(self, other):
        # handle float_or_tensor * self, just in case
        return other * float(self)

    # comparisons: used for `self._guidance_scale > 1.0`
    def __gt__(self, other):
        return float(self) > float(other)

    def __float__(self):
        return float(self._get_value())


class DynamicGuidanceCallback:
    """Callback that dynamically adjusts guidance scale during generation."""
    
    def __init__(self, 
                 guidance_scheduler: GuidanceScheduler,
                 schedule_state: Dict[str, float],
                 apply_to_guidance_2: bool = True,
                 verbose: bool = False):
        """
        Initialize dynamic guidance callback.
        
        Args:
            guidance_scheduler: GuidanceScheduler instance
            schedule_state: Mutable dict like {"value": 5.0} that gets updated each step
            apply_to_guidance_2: Whether to also apply schedule to guidance_scale_2 (for dual-model WAN)
            verbose: Whether to log guidance scale changes
        """
        self.guidance_scheduler = guidance_scheduler
        self.schedule_state = schedule_state
        self.apply_to_guidance_2 = apply_to_guidance_2
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)


    def __call__(self, pipe, step: int, timestep: torch.Tensor, callback_kwargs: dict):
        """Update the mutable schedule state for the next step."""
        try:
            # Calculate the guidance value for the NEXT step (since callback runs AFTER current step)
            next_step = step + 1
            next_guidance_scale = self.guidance_scheduler.get_guidance_scale(next_step)
            
            # Update the mutable schedule state for the next iteration
            old_value = self.schedule_state.get("value", 0.0)
            self.schedule_state["value"] = next_guidance_scale
            
            # Log the guidance scale change with step details
            if self.verbose or old_value != next_guidance_scale:
                logging.info(f"🎯 After step {step}: updated guidance_scale for next step ({next_step}): {old_value:.2f} → {next_guidance_scale:.2f}")
                
        except Exception as e:
            logging.error(f"❌ Error in dynamic guidance callback: {e}")
            logging.error(traceback.format_exc())
            
        return callback_kwargs


def create_guidance_callback(schedule: Dict[int, float], 
                           interpolation: str = "linear",
                           total_steps: Optional[int] = None,
                           apply_to_guidance_2: bool = True,
                           verbose: bool = False) -> tuple:
    """
    Factory function to create dynamic guidance scale and callback.
    
    Args:
        schedule: Dictionary mapping step numbers to guidance scale values
        interpolation: Interpolation method - "linear", "step", or "cosine"
        total_steps: Total number of inference steps
        apply_to_guidance_2: Whether to apply to guidance_scale_2 as well
        verbose: Whether to log guidance changes
        
    Returns:
        Tuple of (DynamicScale, DynamicGuidanceCallback)
    """
    scheduler = GuidanceScheduler(schedule, interpolation, total_steps)
    
    # Create mutable state for the guidance scale, starting with the value for step 0
    initial_value = scheduler.get_guidance_scale(0)
    schedule_state = {"value": initial_value}
    
    # Create DynamicScale that reads from the mutable state
    dynamic_scale = DynamicScale(lambda: schedule_state["value"])
    
    # Create callback that updates the mutable state
    callback = DynamicGuidanceCallback(
        guidance_scheduler=scheduler,
        schedule_state=schedule_state,
        apply_to_guidance_2=apply_to_guidance_2,
        verbose=verbose
    )
    
    return dynamic_scale, callback


def parse_guidance_schedule_config(config_dict: Dict) -> Optional[tuple]:
    """
    Parse guidance schedule configuration from config dictionary.
    
    Args:
        config_dict: Configuration dictionary containing cfg_schedule_settings
        
    Returns:
        Tuple of (DynamicScale, DynamicGuidanceCallback) if schedule is configured, None otherwise
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
