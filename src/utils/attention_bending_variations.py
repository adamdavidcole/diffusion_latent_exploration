"""
Attention Bending Variations Generator

Generates systematic variations of attention bending configurations to explore
parameter spaces across operations, timesteps, and layers.

Inspired by the PromptTemplate variation system, this creates a Cartesian product
of (parameter_values × timestep_configs × layer_configs) to enable comprehensive
attention bending experiments.

Example:
    5 scale values × 2 timestep ranges × 2 layer configs = 20 variations per seed
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging

from src.utils.attention_bending import BendingConfig, BendingMode

logger = logging.getLogger(__name__)


@dataclass
class BendingVariation:
    """
    Represents a single attention bending variation.
    
    Analogous to PromptVariation but for bending configurations.
    Each variation represents one unique combination of (parameter, timesteps, layers).
    """
    config: BendingConfig  # The complete BendingConfig for this variation
    variation_id: str  # Unique identifier (e.g., "scale_0.75_t0-10_lALL")
    display_name: str  # Human-readable name (e.g., "Scale: 0.75× | T: 0-10 | L: ALL")
    metadata: Dict[str, Any]  # Additional metadata for tracking
    
    # Core variation dimensions
    operation: str  # "scale", "rotate", "translate", etc.
    parameter_name: str  # "scale_factor", "angle", "translate_x", etc.
    parameter_value: float  # The actual value (0.75, 45, etc.)
    timestep_spec: Any  # "ALL", "0-10", [0, 5, 10], 5, or None
    layer_spec: Any  # "ALL", "14-20", [14, 15], 10, or None


@dataclass
class OperationSpec:
    """Specification for generating variations of a single operation.
    
    Behavior:
    - If apply_to_timesteps is a list: generate variations for each element
    - If apply_to_timesteps is a single value or None: use single value (defaults to "ALL")
    - Same logic for apply_to_layers
    """
    operation: str  # "scale", "rotate", "translate_x", "translate_y", "flip", etc.
    parameter_name: str  # Which parameter to vary
    
    # Parameter variations
    range: Optional[Tuple[float, float]] = None  # (min, max) for continuous params
    steps: int = 5  # Number of values in the range
    values: Optional[List[float]] = None  # Explicit list of values (overrides range)
    
    # Meta-parameter variations (orthogonal to parameter)
    # If list → generate variations, if single value/None → single application
    apply_to_timesteps: Union[str, int, List[Union[str, int]], None] = None
    apply_to_layers: Union[str, int, List[Union[str, int]], None] = None
    
    # Fixed operation parameters
    target_token: Union[str, List[str], None] = None  # Token(s) to apply bending to - supports list for multi-token variations
    strength: float = 1.0
    padding_mode: str = 'border'
    renormalize: bool = False
    
    # Additional operation-specific fixed parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)


# Default parameter ranges for common operations
DEFAULT_OPERATION_RANGES = {
    "scale": {"range": (0.75, 1.25), "steps": 5, "parameter": "scale_factor"},
    "rotate": {"range": (0, 180), "steps": 7, "parameter": "angle"},
    "translate_x": {"range": (-0.3, 0.3), "steps": 7, "parameter": "translate_x"},
    "translate_y": {"range": (-0.3, 0.3), "steps": 7, "parameter": "translate_y"},
    "amplify": {"range": (0.5, 2.0), "steps": 7, "parameter": "amplify_factor"},
    "blur": {"range": (1.0, 5.0), "steps": 5, "parameter": "sigma"},
}


class AttentionBendingVariationGenerator:
    """
    Generates systematic variations of attention bending configurations.
    
    Creates Cartesian products of:
    - Parameter values (e.g., scale_factor = [0.75, 0.875, 1.0, 1.125, 1.25])
    - Timestep configurations (e.g., ["0-10", "10-19"])
    - Layer configurations (e.g., ["ALL", [14, 15]])
    
    Example usage:
        generator = AttentionBendingVariationGenerator()
        spec = OperationSpec(
            operation="scale",
            parameter_name="scale_factor",
            range=(0.75, 1.25),
            steps=5,
            target_token="kiss",
            apply_to_timesteps=["0-10", "10-19"],  # List → 2 variations
            apply_to_layers=["ALL", [14, 15]]  # List → 2 variations
        )
        variations = generator.generate_variations(spec)
        # Returns: 5 × 2 × 2 = 20 BendingVariation objects
    """
    
    def __init__(self, prompt: Optional[str] = None):
        """Initialize the variation generator.
        
        Args:
            prompt: Optional prompt (kept for compatibility, not used for filtering)
        """
        self.default_ranges = DEFAULT_OPERATION_RANGES
        self.prompt = prompt
    
    def generate_variations(self, spec: OperationSpec) -> List[BendingVariation]:
        """
        Generate all variations for an operation specification.
        
        Creates Cartesian product: parameters × timesteps × layers
        
        Args:
            spec: OperationSpec defining the variations to generate
            
        Returns:
            List of BendingVariation objects, one per unique combination
        """
        # Step 1: Generate parameter values
        param_values = self._generate_parameter_values(spec)
        
        # Step 2: Generate token specifications (no filtering - uses target_token directly)
        token_specs = self._generate_token_specs(spec)
        
        # Step 3: Generate timestep configurations
        timestep_specs = self._generate_timestep_specs(spec)
        
        # Step 4: Generate layer configurations
        layer_specs = self._generate_layer_specs(spec)
        
        # Step 5: Create Cartesian product
        variations = []
        for param_val in param_values:
            for token_spec in token_specs:
                for timestep_spec in timestep_specs:
                    for layer_spec in layer_specs:
                        variation = self._create_variation(
                            spec, param_val, token_spec, timestep_spec, layer_spec
                        )
                        variations.append(variation)
        
        logger.info(
            f"Generated {len(variations)} variations for {spec.operation}: "
            f"{len(param_values)} params × {len(token_specs)} tokens × {len(timestep_specs)} timesteps × {len(layer_specs)} layers"
        )
        
        return variations
    
    def _generate_parameter_values(self, spec: OperationSpec) -> List[float]:
        """Generate parameter values from range or explicit list.
        
        Special handling for flip operations: if no values specified, default to [True]
        since defining a flip operation implies you want to apply it.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Special case: flip operations default to [True] if no values specified
        if spec.operation == "flip" and spec.values is None and spec.range is None:
            logger.debug(f"  Flip operation with no values/range, returning [True]")
            return [True]
        
        if spec.values is not None:
            logger.debug(f"  Using explicit values: {spec.values} (type: {type(spec.values)}, elem types: {[type(v) for v in spec.values] if isinstance(spec.values, list) else 'N/A'})")
            return spec.values
        
        if spec.range is not None:
            min_val, max_val = spec.range
            logger.debug(f"  Generating linspace from range ({min_val}, {max_val}) with {spec.steps} steps")
            logger.debug(f"    min_val type: {type(min_val)}, max_val type: {type(max_val)}, steps type: {type(spec.steps)}")
            result = np.linspace(min_val, max_val, spec.steps).tolist()
            logger.debug(f"    Result: {result[:3]}... (length: {len(result)}, types: {[type(v) for v in result[:3]]})")
            return result
        
        # Fallback to defaults
        if spec.operation in self.default_ranges:
            defaults = self.default_ranges[spec.operation]
            min_val, max_val = defaults["range"]
            logger.debug(f"  Using default range ({min_val}, {max_val}) with {spec.steps} steps")
            return np.linspace(min_val, max_val, spec.steps).tolist()
        
        raise ValueError(
            f"No parameter range specified for {spec.operation} and no defaults found"
        )
    
    def _generate_timestep_specs(self, spec: OperationSpec) -> List[Any]:
        """Generate list of timestep specifications.
        
        If apply_to_timesteps is a list: return list (generate variations)
        If single value or None: return single-element list (no variations)
        """
        if spec.apply_to_timesteps is None:
            return ["ALL"]
        
        if isinstance(spec.apply_to_timesteps, list):
            return spec.apply_to_timesteps
        else:
            return [spec.apply_to_timesteps]
    
    def _generate_layer_specs(self, spec: OperationSpec) -> List[Any]:
        """Generate list of layer specifications.
        
        If apply_to_layers is a list: return list (generate variations)
        If single value or None: return single-element list (no variations)
        """
        if spec.apply_to_layers is None:
            return ["ALL"]
        
        if isinstance(spec.apply_to_layers, list):
            return spec.apply_to_layers
        else:
            return [spec.apply_to_layers]
    
    def _generate_token_specs(self, spec: OperationSpec) -> List[str]:
        """Generate list of token specifications.
        
        Returns target_token as a list without filtering.
        Assumes tokens will be in the token map if bending is needed.
        
        Args:
            spec: OperationSpec with target_token field
            
        Returns:
            List of token strings to generate variations for
        """
        if spec.target_token is None:
            return ["ALL"]
        
        # If target_token is already a list, return it
        if isinstance(spec.target_token, list):
            return spec.target_token
        
        # If single string, wrap in list
        return [spec.target_token]
    
    def _create_variation(
        self,
        spec: OperationSpec,
        param_value: float,
        token_spec: str,
        timestep_spec: Any,
        layer_spec: Any
    ) -> BendingVariation:
        """Create a single BendingVariation from specifications."""
        # Create BendingConfig
        config = self._build_bending_config(
            spec, param_value, token_spec, timestep_spec, layer_spec
        )
        
        # Generate variation ID
        variation_id = format_variation_id(
            operation=spec.operation,
            parameter_value=param_value,
            timestep_spec=timestep_spec,
            layer_spec=layer_spec,
            parameter_name=spec.parameter_name,
            token_spec=token_spec
        )
        
        # Generate display name
        display_name = format_display_name(
            operation=spec.operation,
            parameter_value=param_value,
            timestep_spec=timestep_spec,
            layer_spec=layer_spec,
            parameter_name=spec.parameter_name,
            token_spec=token_spec
        )
        
        # Build metadata using new attention bending format
        # Extract transformation type and params from the config
        transformation_params = {}
        if spec.operation == "scale":
            transformation_params = {"scale_x": param_value, "scale_y": param_value}
        elif spec.operation == "rotate":
            transformation_params = {"angle": param_value}
        elif spec.operation == "translate_x":
            transformation_params = {"shift_x": param_value}
        elif spec.operation == "translate_y":
            transformation_params = {"shift_y": param_value}
        elif spec.operation == "blur":
            transformation_params = {"sigma": param_value}
        elif spec.operation == "flip_horizontal":
            transformation_params = {}
        elif spec.operation == "flip_vertical":
            transformation_params = {}
        elif spec.operation == "edge_enhance":
            transformation_params = {"alpha": param_value}
        elif spec.operation == "frequency_filter":
            transformation_params = {
                "cutoff_freq": param_value,
                "mode": spec.extra_params.get("mode", "lowpass")
            }
        else:
            # Generic fallback
            transformation_params = {spec.parameter_name: param_value}
        
        # Convert timestep_spec to range format [start, end]
        timestep_range = None
        if timestep_spec is not None and timestep_spec != "ALL":
            if isinstance(timestep_spec, str) and "-" in timestep_spec:
                start, end = map(int, timestep_spec.split("-"))
                timestep_range = [start, end]
            elif isinstance(timestep_spec, int):
                timestep_range = [timestep_spec, timestep_spec]
        
        # Convert layer_spec to indices list using the same parser as BendingConfig
        layer_indices = self._parse_layer_spec(layer_spec)
        
        # For flip operations, use parameter_name as transformation_type
        transformation_type = spec.parameter_name if spec.operation == "flip" else spec.operation
        
        metadata = {
            "display_name": display_name,  # Include display_name for UI
            "variation_id": variation_id,  # Include variation_id for reference
            "transformation_type": transformation_type,
            "transformation_params": transformation_params,
            "phase": spec.extra_params.get("phase"),  # Phase 1 or 2 (or None)
            "timestep_range": timestep_range,
            "layer_indices": layer_indices,
            "target_token": token_spec if token_spec else None,
        }
        
        return BendingVariation(
            config=config,
            variation_id=variation_id,
            display_name=display_name,
            metadata=metadata,
            operation=spec.operation,
            parameter_name=spec.parameter_name,
            parameter_value=param_value,
            timestep_spec=timestep_spec,
            layer_spec=layer_spec
        )
    
    def _build_bending_config(
        self,
        spec: OperationSpec,
        param_value: float,
        token_spec: str,
        timestep_spec: Any,
        layer_spec: Any
    ) -> BendingConfig:
        """Build a BendingConfig from variation specifications."""
        # Convert timestep spec to BendingConfig format
        apply_to_timesteps = self._parse_timestep_spec(timestep_spec)
        
        # Convert layer spec to BendingConfig format
        apply_to_layers = self._parse_layer_spec(layer_spec)
        
        # Determine mode from operation
        mode = self._operation_to_mode(spec.operation)
        
        # Build parameter dict
        params = {
            "token": token_spec,  # Use the specific token for this variation
            "mode": mode,
            "strength": spec.strength,
            "padding_mode": spec.padding_mode,
            "renormalize": spec.renormalize,
            "apply_to_timesteps": apply_to_timesteps,
            "apply_to_layers": apply_to_layers,
        }
        
        # Add operation-specific parameter
        if spec.parameter_name in ["scale_factor", "angle", "translate_x", "translate_y", 
                                     "amplify_factor", "sigma", "sharpen_amount",
                                     "flip_horizontal", "flip_vertical"]:
            params[spec.parameter_name] = param_value
        
        # Add extra parameters
        params.update(spec.extra_params)
        
        return BendingConfig(**params)
    
    def _parse_timestep_spec(self, spec: Any) -> Optional[Tuple[int, int]]:
        """
        Parse timestep specification to BendingConfig format.
        
        Converts various formats to (start, end) tuple:
        - "ALL" → None (all timesteps)
        - "0-10" → (0, 10)
        - 5 → (5, 5)
        - [0, 5, 10] → None (discrete list not supported in BendingConfig yet)
        """
        if spec == "ALL" or spec is None:
            return None
        
        if isinstance(spec, str) and "-" in spec:
            start, end = map(int, spec.split("-"))
            return (start, end)
        
        if isinstance(spec, int):
            return (spec, spec)
        
        if isinstance(spec, list):
            # For now, convert list to range (min, max)
            # Future: BendingConfig should support discrete lists
            return (min(spec), max(spec))
        
        return None
    
    def _parse_layer_spec(self, spec: Any) -> Optional[List[int]]:
        """
        Parse layer specification to BendingConfig format.
        
        Converts various formats to List[int] or None:
        - "ALL" → None (all layers)
        - "14-20" → [14, 15, 16, 17, 18, 19, 20]
        - 10 → [10]
        - [14, 15] → [14, 15]
        """
        if spec == "ALL" or spec is None:
            return None
        
        if isinstance(spec, str) and "-" in spec:
            start, end = map(int, spec.split("-"))
            return list(range(start, end + 1))
        
        if isinstance(spec, int):
            return [spec]
        
        if isinstance(spec, list):
            return spec
        
        return None
    
    def _operation_to_mode(self, operation: str) -> BendingMode:
        """Convert operation string to BendingMode enum."""
        mode_map = {
            "scale": BendingMode.SCALE,
            "rotate": BendingMode.ROTATE,
            "translate": BendingMode.TRANSLATE,
            "translate_x": BendingMode.TRANSLATE,
            "translate_y": BendingMode.TRANSLATE,
            "flip": BendingMode.FLIP,
            "amplify": BendingMode.AMPLIFY,
            "blur": BendingMode.BLUR,
            "sharpen": BendingMode.SHARPEN,
            "regional_mask": BendingMode.REGIONAL_MASK,
            "frequency_filter": BendingMode.FREQUENCY_FILTER,
        }
        
        if operation not in mode_map:
            raise ValueError(f"Unknown operation: {operation}")
        
        return mode_map[operation]


# ============================================================================
# Formatting Functions
# ============================================================================

def format_variation_id(
    operation: str,
    parameter_value: float,
    timestep_spec: Any = None,
    layer_spec: Any = None,
    prompt_id: Optional[str] = None,
    seed: Optional[int] = None,
    parameter_name: Optional[str] = None,
    token_spec: Optional[str] = None
) -> str:
    """
    Format unique variation identifier.
    
    Format: [prompt_id]_[seed]_operation_value[_token]_t<timesteps>_l<layers>
    
    Examples:
        format_variation_id("scale", 0.75, "0-10", "ALL")
        → "scale_0.75_t0-10_lALL"
        
        format_variation_id("scale", 0.75, "0-10", [14, 15], "prompt_000", 42, token_spec="rose")
        → "prompt_000_seed_42_scale_0.75_rose_t0-10_l14,15"
        
        format_variation_id("flip", 1.0, parameter_name="flip_horizontal")
        → "flip_horizontal_1_tALL_lALL"
    """
    parts = []
    
    # Optional prompt and seed prefix
    if prompt_id:
        parts.append(prompt_id)
    if seed is not None:
        parts.append(f"seed_{seed}")
    
    # Operation and value - use parameter_name if available for operations with sub-parameters
    op_name = parameter_name if parameter_name and operation in ["flip", "translate"] else operation
    parts.append(f"{op_name}_{parameter_value:.4g}")
    
    # Add token spec (only if not "ALL" - omit for brevity)
    if token_spec and token_spec != "ALL":
        parts.append(token_spec.lower())
    
    # Timestep suffix
    if timestep_spec is not None:
        t_suffix = _format_spec_suffix(timestep_spec, "t")
        if t_suffix:
            parts.append(t_suffix)
    
    # Layer suffix
    if layer_spec is not None:
        l_suffix = _format_spec_suffix(layer_spec, "l")
        if l_suffix:
            parts.append(l_suffix)
    
    return "_".join(parts)


def format_display_name(
    operation: str,
    parameter_value: float,
    timestep_spec: Any = None,
    layer_spec: Any = None,
    parameter_name: Optional[str] = None,
    token_spec: Optional[str] = None
) -> str:
    """
    Format human-readable display name.
    
    Format: Operation: value [unit] | Token: <token> | T: <timesteps> | L: <layers>
    
    Examples:
        format_display_name("scale", 0.75, "0-10", "ALL", token_spec="rose")
        → "Scale: 0.75× | Token: rose | T: 0-10 | L: ALL"
        
        format_display_name("rotate", 45, "ALL", [14, 15], token_spec="ALL")
        → "Rotate: 45° | T: ALL | L: 14,15"  (token omitted for "ALL")
        
        format_display_name("flip", 1.0, parameter_name="flip_horizontal")
        → "Flip H: True | T: ALL | L: ALL"
    """
    # Format operation name and value with unit
    op_display = {
        "scale": lambda v: f"Scale: {v:.3g}×",
        "rotate": lambda v: f"Rotate: {v:.0f}°",
        "translate_x": lambda v: f"Translate X: {v:+.2f}",
        "translate_y": lambda v: f"Translate Y: {v:+.2f}",
        "amplify": lambda v: f"Amplify: {v:.2f}×",
        "blur": lambda v: f"Blur: σ={v:.1f}",
        "sharpen": lambda v: f"Sharpen: {v:.2f}",
        "flip": lambda v: f"Flip: {v}",
        "flip_horizontal": lambda v: f"Flip H: {bool(v)}",
        "flip_vertical": lambda v: f"Flip V: {bool(v)}",
    }
    
    # Use parameter_name if available for operations with sub-parameters (flip, translate)
    # For flip operations, parameter_name will be "flip_horizontal" or "flip_vertical"
    op_key = parameter_name if parameter_name in op_display else operation
    formatter = op_display.get(op_key, lambda v: f"{operation.title()}: {v}")
    
    name_parts = []
    
    # Add token info FIRST (only if not "ALL" - omit for brevity)
    if token_spec is not None and token_spec != "ALL":
        name_parts.append(f"Token: {token_spec}")
    
    # Add operation and value
    name_parts.append(formatter(parameter_value))
    
    # Add timestep info
    if timestep_spec is not None:
        t_display = _format_spec_display(timestep_spec, "T")
        name_parts.append(t_display)
    
    # Add layer info
    if layer_spec is not None:
        l_display = _format_spec_display(layer_spec, "L")
        name_parts.append(l_display)
    
    return " | ".join(name_parts)


def _format_spec_suffix(spec: Any, prefix: str) -> str:
    """Format spec for use in variation ID (compact)."""
    if spec == "ALL":
        return f"{prefix}ALL"
    elif isinstance(spec, str):  # Range like "0-10"
        return f"{prefix}{spec}"
    elif isinstance(spec, int):  # Single value
        return f"{prefix}{spec}"
    elif isinstance(spec, list):  # List like [14, 15]
        return f"{prefix}{','.join(map(str, spec))}"
    return ""


def _format_spec_display(spec: Any, label: str) -> str:
    """Format spec for display name (readable)."""
    if spec == "ALL":
        return f"{label}: ALL"
    elif isinstance(spec, str):  # Range like "0-10"
        return f"{label}: {spec}"
    elif isinstance(spec, int):  # Single value
        return f"{label}: {spec}"
    elif isinstance(spec, list):  # List like [14, 15]
        return f"{label}: {','.join(map(str, spec))}"
    return ""


def format_filename(
    operation: str,
    parameter_value: float,
    timestep_spec: Any = None,
    layer_spec: Any = None,
    video_id: Optional[str] = None
) -> str:
    """
    Format video filename for bending variation.
    
    Format: [video_id_]operation_value_t<timesteps>_l<layers>.mp4
    
    Examples:
        format_filename("scale", 0.75, "0-10", "ALL", "vid_001")
        → "vid_001_scale_0.75_t0-10_lALL.mp4"
    """
    variation_id = format_variation_id(
        operation, parameter_value, timestep_spec, layer_spec
    )
    
    if video_id:
        return f"{video_id}_{variation_id}.mp4"
    else:
        return f"{variation_id}.mp4"
