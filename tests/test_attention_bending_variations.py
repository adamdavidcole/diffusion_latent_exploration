"""
Unit tests for attention bending variations generator.

Tests the Cartesian product generation, formatting functions, and configuration parsing.
"""

import pytest
from src.utils.attention_bending_variations import (
    AttentionBendingVariationGenerator,
    OperationSpec,
    BendingVariation,
    format_variation_id,
    format_display_name,
    format_filename,
)
from src.utils.attention_bending import BendingMode


class TestVariationIDFormatting:
    """Test variation ID formatting functions."""
    
    def test_basic_variation_id(self):
        """Test basic variation ID without timesteps/layers."""
        vid = format_variation_id("scale", 0.75)
        assert vid == "scale_0.75"
    
    def test_variation_id_with_timesteps(self):
        """Test variation ID with timestep specification."""
        vid = format_variation_id("scale", 0.75, "0-10", None)
        assert vid == "scale_0.75_t0-10"
    
    def test_variation_id_with_layers(self):
        """Test variation ID with layer specification."""
        vid = format_variation_id("scale", 0.75, None, "ALL")
        assert vid == "scale_0.75_lALL"
    
    def test_variation_id_with_both(self):
        """Test variation ID with both timesteps and layers."""
        vid = format_variation_id("scale", 0.75, "0-10", "ALL")
        assert vid == "scale_0.75_t0-10_lALL"
    
    def test_variation_id_with_list_layers(self):
        """Test variation ID with list of layers."""
        vid = format_variation_id("scale", 0.75, "0-10", [14, 15])
        assert vid == "scale_0.75_t0-10_l14,15"
    
    def test_variation_id_with_prompt_and_seed(self):
        """Test variation ID with prompt and seed prefix."""
        vid = format_variation_id("scale", 0.75, "0-10", "ALL", "prompt_000", 42)
        assert vid == "prompt_000_seed_42_scale_0.75_t0-10_lALL"
    
    def test_variation_id_float_formatting(self):
        """Test that float values are formatted compactly."""
        vid = format_variation_id("scale", 0.875, "0-10", "ALL")
        assert vid == "scale_0.875_t0-10_lALL"
        
        vid = format_variation_id("scale", 1.0, "0-10", "ALL")
        assert vid == "scale_1_t0-10_lALL"


class TestDisplayNameFormatting:
    """Test display name formatting functions."""
    
    def test_scale_display_name(self):
        """Test scale operation display name."""
        name = format_display_name("scale", 0.75, "0-10", "ALL")
        assert name == "Scale: 0.75× | T: 0-10 | L: ALL"
    
    def test_rotate_display_name(self):
        """Test rotate operation display name."""
        name = format_display_name("rotate", 45, "ALL", [14, 15])
        assert name == "Rotate: 45° | T: ALL | L: 14,15"
    
    def test_translate_display_name(self):
        """Test translate operation display name."""
        name = format_display_name("translate_x", -0.3, "5-15", "ALL")
        assert name == "Translate X: -0.30 | T: 5-15 | L: ALL"
    
    def test_amplify_display_name(self):
        """Test amplify operation display name."""
        name = format_display_name("amplify", 1.5, "ALL", "ALL")
        assert name == "Amplify: 1.50× | T: ALL | L: ALL"


class TestFilenameFormatting:
    """Test filename formatting functions."""
    
    def test_basic_filename(self):
        """Test basic filename without video ID."""
        fname = format_filename("scale", 0.75, "0-10", "ALL")
        assert fname == "scale_0.75_t0-10_lALL.mp4"
    
    def test_filename_with_video_id(self):
        """Test filename with video ID prefix."""
        fname = format_filename("scale", 0.75, "0-10", "ALL", "vid_001")
        assert fname == "vid_001_scale_0.75_t0-10_lALL.mp4"


class TestOperationSpec:
    """Test OperationSpec dataclass."""
    
    def test_basic_operation_spec(self):
        """Test creating basic operation spec."""
        spec = OperationSpec(
            operation="scale",
            parameter_name="scale_factor",
            range=(0.75, 1.25),
            steps=5,
            target_token="kiss"
        )
        assert spec.operation == "scale"
        assert spec.parameter_name == "scale_factor"
        assert spec.range == (0.75, 1.25)
        assert spec.steps == 5
        assert spec.target_token == "kiss"
        # No vary flags - behavior determined by list vs single value
    
    def test_operation_spec_with_variations(self):
        """Test operation spec with timestep/layer variations."""
        spec = OperationSpec(
            operation="scale",
            parameter_name="scale_factor",
            range=(0.75, 1.25),
            steps=5,
            target_token="kiss",
            apply_to_timesteps=["0-10", "10-19"],  # List = variations
            apply_to_layers=["ALL", [14, 15]]  # List = variations
        )
        # Behavior determined by list format
        assert spec.apply_to_timesteps == ["0-10", "10-19"]
        assert spec.apply_to_layers == ["ALL", [14, 15]]


class TestVariationGenerator:
    """Test AttentionBendingVariationGenerator."""
    
    def test_generator_initialization(self):
        """Test generator initializes with default ranges."""
        gen = AttentionBendingVariationGenerator()
        assert "scale" in gen.default_ranges
        assert "rotate" in gen.default_ranges
    
    def test_generate_parameter_values_from_range(self):
        """Test parameter value generation from range."""
        gen = AttentionBendingVariationGenerator()
        spec = OperationSpec(
            operation="scale",
            parameter_name="scale_factor",
            range=(0.75, 1.25),
            steps=5,
            target_token="kiss"
        )
        values = gen._generate_parameter_values(spec)
        assert len(values) == 5
        assert values[0] == pytest.approx(0.75)
        assert values[-1] == pytest.approx(1.25)
        assert values[2] == pytest.approx(1.0)  # Middle value
    
    def test_generate_parameter_values_from_explicit_list(self):
        """Test parameter value generation from explicit list."""
        gen = AttentionBendingVariationGenerator()
        spec = OperationSpec(
            operation="scale",
            parameter_name="scale_factor",
            values=[0.5, 1.0, 1.5],
            target_token="kiss"
        )
        values = gen._generate_parameter_values(spec)
        assert values == [0.5, 1.0, 1.5]
    
    def test_generate_timestep_specs_no_variation(self):
        """Test timestep spec generation when not varying."""
        gen = AttentionBendingVariationGenerator()
        spec = OperationSpec(
            operation="scale",
            parameter_name="scale_factor",
            range=(0.75, 1.25),
            steps=5,
            target_token="kiss",
            apply_to_timesteps="ALL"  # Single value = no variation
        )
        specs = gen._generate_timestep_specs(spec)
        assert len(specs) == 1
        assert specs[0] == "ALL"
    
    def test_generate_timestep_specs_with_variation(self):
        """Test timestep spec generation when varying."""
        gen = AttentionBendingVariationGenerator()
        spec = OperationSpec(
            operation="scale",
            parameter_name="scale_factor",
            range=(0.75, 1.25),
            steps=5,
            target_token="kiss",
            apply_to_timesteps=["0-10", "10-19"]  # List = variations
        )
        specs = gen._generate_timestep_specs(spec)
        assert len(specs) == 2
        assert specs == ["0-10", "10-19"]
    
    def test_generate_layer_specs_no_variation(self):
        """Test layer spec generation when not varying."""
        gen = AttentionBendingVariationGenerator()
        spec = OperationSpec(
            operation="scale",
            parameter_name="scale_factor",
            range=(0.75, 1.25),
            steps=5,
            target_token="kiss",
            apply_to_layers="ALL"  # Single value = no variation
        )
        specs = gen._generate_layer_specs(spec)
        assert len(specs) == 1
        assert specs[0] == "ALL"
    
    def test_generate_layer_specs_with_variation(self):
        """Test layer spec generation when varying."""
        gen = AttentionBendingVariationGenerator()
        spec = OperationSpec(
            operation="scale",
            parameter_name="scale_factor",
            range=(0.75, 1.25),
            steps=5,
            target_token="kiss",
            apply_to_layers=["ALL", [14, 15]]  # List = variations
        )
        specs = gen._generate_layer_specs(spec)
        assert len(specs) == 2
        assert specs[0] == "ALL"
        assert specs[1] == [14, 15]
    
    def test_parse_timestep_spec_all(self):
        """Test parsing 'ALL' timestep spec."""
        gen = AttentionBendingVariationGenerator()
        result = gen._parse_timestep_spec("ALL")
        assert result is None
    
    def test_parse_timestep_spec_range(self):
        """Test parsing range timestep spec."""
        gen = AttentionBendingVariationGenerator()
        result = gen._parse_timestep_spec("0-10")
        assert result == (0, 10)
    
    def test_parse_timestep_spec_single(self):
        """Test parsing single timestep spec."""
        gen = AttentionBendingVariationGenerator()
        result = gen._parse_timestep_spec(5)
        assert result == (5, 5)
    
    def test_parse_layer_spec_all(self):
        """Test parsing 'ALL' layer spec."""
        gen = AttentionBendingVariationGenerator()
        result = gen._parse_layer_spec("ALL")
        assert result is None
    
    def test_parse_layer_spec_range(self):
        """Test parsing range layer spec."""
        gen = AttentionBendingVariationGenerator()
        result = gen._parse_layer_spec("14-20")
        assert result == [14, 15, 16, 17, 18, 19, 20]
    
    def test_parse_layer_spec_single(self):
        """Test parsing single layer spec."""
        gen = AttentionBendingVariationGenerator()
        result = gen._parse_layer_spec(10)
        assert result == [10]
    
    def test_parse_layer_spec_list(self):
        """Test parsing list layer spec."""
        gen = AttentionBendingVariationGenerator()
        result = gen._parse_layer_spec([14, 15])
        assert result == [14, 15]
    
    def test_operation_to_mode(self):
        """Test operation string to BendingMode conversion."""
        gen = AttentionBendingVariationGenerator()
        assert gen._operation_to_mode("scale") == BendingMode.SCALE
        assert gen._operation_to_mode("rotate") == BendingMode.ROTATE
        assert gen._operation_to_mode("translate") == BendingMode.TRANSLATE
        assert gen._operation_to_mode("amplify") == BendingMode.AMPLIFY


class TestVariationGeneration:
    """Test complete variation generation."""
    
    def test_generate_single_parameter_variation(self):
        """Test generating variations with only parameter sweep."""
        gen = AttentionBendingVariationGenerator()
        spec = OperationSpec(
            operation="scale",
            parameter_name="scale_factor",
            range=(0.75, 1.25),
            steps=5,
            target_token="kiss",
            apply_to_timesteps="ALL",  # Single value
            apply_to_layers="ALL"  # Single value
        )
        variations = gen.generate_variations(spec)
        
        # Should generate 5 variations (5 param values × 1 timestep × 1 layer)
        assert len(variations) == 5
        
        # Check first variation
        var = variations[0]
        assert isinstance(var, BendingVariation)
        assert var.operation == "scale"
        assert var.parameter_name == "scale_factor"
        assert var.parameter_value == pytest.approx(0.75)
        assert var.timestep_spec == "ALL"
        assert var.layer_spec == "ALL"
        assert "scale_0.75" in var.variation_id
        assert "Scale:" in var.display_name
    
    def test_generate_multiplicative_variations(self):
        """Test generating multiplicative variations (param × timestep × layer)."""
        gen = AttentionBendingVariationGenerator()
        spec = OperationSpec(
            operation="scale",
            parameter_name="scale_factor",
            range=(0.75, 1.25),
            steps=5,
            target_token="kiss",
            apply_to_timesteps=["0-10", "10-19"],  # List = 2 variations
            apply_to_layers=["ALL", [14, 15]]  # List = 2 variations
        )
        variations = gen.generate_variations(spec)
        
        # Should generate 20 variations (5 params × 2 timesteps × 2 layers)
        assert len(variations) == 20
        
        # Check that we have all combinations
        param_values = set()
        timestep_specs = set()
        layer_specs = set()
        
        for var in variations:
            param_values.add(var.parameter_value)
            timestep_specs.add(str(var.timestep_spec))
            layer_specs.add(str(var.layer_spec))
        
        assert len(param_values) == 5
        assert len(timestep_specs) == 2
        # Layer specs: "ALL" and "[14, 15]" (as strings)
        assert len(layer_specs) == 2
    
    def test_generate_timestep_only_variations(self):
        """Test generating variations with only timestep sweep."""
        gen = AttentionBendingVariationGenerator()
        spec = OperationSpec(
            operation="scale",
            parameter_name="scale_factor",
            values=[1.0],  # Single parameter value
            target_token="kiss",
            apply_to_timesteps=["0-5", "5-10", "10-15", "15-19"],  # List = variations
            apply_to_layers="ALL"  # Single value
        )
        variations = gen.generate_variations(spec)
        
        # Should generate 4 variations (1 param × 4 timesteps × 1 layer)
        assert len(variations) == 4
        
        timestep_specs = [var.timestep_spec for var in variations]
        assert timestep_specs == ["0-5", "5-10", "10-15", "15-19"]
    
    def test_generate_layer_only_variations(self):
        """Test generating variations with only layer sweep."""
        gen = AttentionBendingVariationGenerator()
        spec = OperationSpec(
            operation="rotate",
            parameter_name="angle",
            values=[45],  # Single parameter value
            target_token="object",
            apply_to_timesteps="ALL",  # Single value
            apply_to_layers=["0-10", "10-20", "20-30", [14, 15]]  # List = variations
        )
        variations = gen.generate_variations(spec)
        
        # Should generate 4 variations (1 param × 1 timestep × 4 layers)
        assert len(variations) == 4
        
        layer_specs = [var.layer_spec for var in variations]
        assert len(layer_specs) == 4
    
    def test_variation_metadata(self):
        """Test that variation metadata is correctly populated."""
        gen = AttentionBendingVariationGenerator()
        spec = OperationSpec(
            operation="scale",
            parameter_name="scale_factor",
            values=[0.75],
            target_token="kiss",
            apply_to_timesteps=["0-10"],  # List with single element
            apply_to_layers="ALL"
        )
        variations = gen.generate_variations(spec)
        
        var = variations[0]
        # Check new metadata format
        assert var.metadata["transformation_type"] == "scale"
        assert var.metadata["transformation_params"]["scale_x"] == 0.75
        assert var.metadata["transformation_params"]["scale_y"] == 0.75
        assert var.metadata["timestep_range"] == [0, 10]
        assert var.metadata["layer_indices"] is None  # "ALL" means None
        assert var.metadata["target_token"] == "kiss"
        
        # Check legacy attributes still accessible
        assert var.operation == "scale"
        assert var.parameter_name == "scale_factor"
        assert var.parameter_value == 0.75
        assert var.timestep_spec == "0-10"
        assert var.layer_spec == "ALL"
    
    def test_bending_config_creation(self):
        """Test that BendingConfig is correctly created."""
        gen = AttentionBendingVariationGenerator()
        spec = OperationSpec(
            operation="scale",
            parameter_name="scale_factor",
            values=[0.75],
            target_token="kiss",
            strength=0.8,
            padding_mode="reflection",
            apply_to_timesteps="0-10",  # Single value
            apply_to_layers="14-15"  # Use range string for single spec
        )
        variations = gen.generate_variations(spec)
        
        # Should generate 1 variation
        assert len(variations) == 1
        
        config = variations[0].config
        assert config.token == "kiss"
        assert config.mode == BendingMode.SCALE
        assert config.scale_factor == 0.75
        assert config.strength == 0.8
        assert config.padding_mode == "reflection"
        assert config.apply_to_timesteps == (0, 10)
        # "14-15" parsed to range [14, 15]
        assert config.apply_to_layers == list(range(14, 16))


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_no_range_or_values_raises_error(self):
        """Test that missing range and values raises error."""
        gen = AttentionBendingVariationGenerator()
        spec = OperationSpec(
            operation="unknown_op",  # Not in defaults
            parameter_name="unknown_param",
            target_token="test"
        )
        with pytest.raises(ValueError):
            gen._generate_parameter_values(spec)
    
    def test_unknown_operation_raises_error(self):
        """Test that unknown operation raises error."""
        gen = AttentionBendingVariationGenerator()
        with pytest.raises(ValueError):
            gen._operation_to_mode("nonexistent_operation")
    
    def test_single_step_generates_one_value(self):
        """Test that steps=1 generates a single value."""
        gen = AttentionBendingVariationGenerator()
        spec = OperationSpec(
            operation="scale",
            parameter_name="scale_factor",
            range=(1.0, 1.0),
            steps=1,
            target_token="test"
        )
        values = gen._generate_parameter_values(spec)
        assert len(values) == 1
        assert values[0] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
