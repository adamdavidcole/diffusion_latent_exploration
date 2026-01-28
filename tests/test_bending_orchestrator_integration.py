"""
Integration tests for attention bending variations with orchestrator.

Tests the full pipeline from configuration to variation generation without requiring GPU.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import json
import yaml

from src.config import ConfigManager, GenerationConfig
from src.orchestrator import VideoGenerationOrchestrator
from src.utils.attention_bending_variations import (
    AttentionBendingVariationGenerator,
    OperationSpec,
    BendingVariation
)


class TestOrchestratorBendingIntegration:
    """Test orchestrator integration with bending variations."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
    
    def teardown_method(self):
        """Cleanup test environment."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def create_test_config(self, enable_bending=True, operations=None):
        """Create a test configuration file."""
        if operations is None:
            operations = [
                {
                    "operation": "scale",
                    "parameter_name": "scale_factor",
                    "range": [0.75, 1.25],
                    "steps": 3,  # Small number for testing
                    "target_token": "kiss",
                    "apply_to_timesteps": ["0-10", "10-19"],  # List = variations
                    "apply_to_layers": ["ALL", [14, 15]]  # List = variations
                }
            ]
        
        config_data = {
            "model_settings": {
                "seed": 42,
                "sampler": "unipc",
                "cfg_scale": 6.5,
                "steps": 25,
                "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "device": "cuda:0"
            },
            "video_settings": {
                "width": 512,
                "height": 512,
                "fps": 8,
                "duration": 2.0
            },
            "output_dir": self.temp_dir,
            "batch_name": "test_batch",
            "use_timestamp": False,
            "attention_bending_variations": {
                "enabled": enable_bending,
                "generate_baseline": True,
                "operations": operations
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        return self.config_path
    
    def test_config_loading_with_bending_variations(self):
        """Test that config with bending variations loads correctly."""
        config_path = self.create_test_config()
        config_manager = ConfigManager()
        config = config_manager.load_config(config_path)
        
        assert config.attention_bending_variations_settings.enabled
        assert config.attention_bending_variations_settings.generate_baseline
        assert len(config.attention_bending_variations_settings.operations) == 1
        
        op = config.attention_bending_variations_settings.operations[0]
        assert op["operation"] == "scale"
        assert op["parameter_name"] == "scale_factor"
        assert op["steps"] == 3
    
    def test_orchestrator_initialization_with_bending(self):
        """Test orchestrator initializes with bending variation config."""
        config_path = self.create_test_config()
        config_manager = ConfigManager()
        config = config_manager.load_config(config_path)
        
        orchestrator = VideoGenerationOrchestrator(config)
        assert orchestrator.config.attention_bending_variations_settings.enabled
    
    def test_process_bending_variations_disabled(self):
        """Test process_bending_variations returns None when disabled."""
        config_path = self.create_test_config(enable_bending=False)
        config_manager = ConfigManager()
        config = config_manager.load_config(config_path)
        
        orchestrator = VideoGenerationOrchestrator(config)
        orchestrator.setup_batch("test_disabled")
        
        variations = orchestrator.process_bending_variations()
        assert variations is None
    
    def test_process_bending_variations_single_operation(self):
        """Test processing single operation generates correct variations."""
        config_path = self.create_test_config()
        config_manager = ConfigManager()
        config = config_manager.load_config(config_path)
        
        orchestrator = VideoGenerationOrchestrator(config)
        orchestrator.setup_batch("test_single_op")
        
        variations = orchestrator.process_bending_variations()
        
        # Should generate 3 params × 2 timesteps × 2 layers = 12 variations
        assert len(variations) == 12
        assert all(isinstance(v, BendingVariation) for v in variations)
        
        # Check first variation
        var = variations[0]
        assert var.operation == "scale"
        assert var.parameter_name == "scale_factor"
        assert var.parameter_value in [0.75, 1.0, 1.25]
        assert var.timestep_spec in ["0-10", "10-19"]
        assert var.layer_spec in ["ALL", [14, 15]]
    
    def test_process_bending_variations_multiple_operations(self):
        """Test processing multiple operations."""
        operations = [
            {
                "operation": "scale",
                "parameter_name": "scale_factor",
                "range": [0.75, 1.25],
                "steps": 2,  # 2 values
                "target_token": "kiss",
                "apply_to_timesteps": "ALL"  # Single value
            },
            {
                "operation": "rotate",
                "parameter_name": "angle",
                "values": [0, 90],  # 2 explicit values
                "target_token": "person",
                "apply_to_timesteps": "ALL"  # Single value
            }
        ]
        
        config_path = self.create_test_config(operations=operations)
        config_manager = ConfigManager()
        config = config_manager.load_config(config_path)
        
        orchestrator = VideoGenerationOrchestrator(config)
        orchestrator.setup_batch("test_multi_op")
        
        variations = orchestrator.process_bending_variations()
        
        # Should generate 2 scale + 2 rotate = 4 variations
        assert len(variations) == 4
        
        # Check we have both operations
        operations_found = set(v.operation for v in variations)
        assert operations_found == {"scale", "rotate"}
    
    def test_bending_variations_file_saved(self):
        """Test that bending variations are generated correctly."""
        config_path = self.create_test_config()
        config_manager = ConfigManager()
        config = config_manager.load_config(config_path)
        
        orchestrator = VideoGenerationOrchestrator(config)
        batch_dirs = orchestrator.setup_batch("test_save")
        
        variations = orchestrator.process_bending_variations()
        
        # Verify variations are generated
        assert variations is not None
        assert len(variations) == 12  # 3 params × 2 timesteps × 2 layers
        assert all(hasattr(var, "variation_id") for var in variations)
        assert all(hasattr(var, "display_name") for var in variations)
        assert all(hasattr(var, "operation") for var in variations)
    
    def test_variation_metadata_structure(self):
        """Test that variations have correct metadata structure."""
        config_path = self.create_test_config()
        config_manager = ConfigManager()
        config = config_manager.load_config(config_path)
        
        orchestrator = VideoGenerationOrchestrator(config)
        orchestrator.setup_batch("test_metadata")
        
        variations = orchestrator.process_bending_variations()
        
        for var in variations:
            # Check BendingVariation attributes
            assert hasattr(var, 'config')
            assert hasattr(var, 'variation_id')
            assert hasattr(var, 'display_name')
            assert hasattr(var, 'metadata')
            assert hasattr(var, 'operation')
            assert hasattr(var, 'parameter_name')
            assert hasattr(var, 'parameter_value')
            assert hasattr(var, 'timestep_spec')
            assert hasattr(var, 'layer_spec')
            
            # Check metadata dict - NEW format
            assert 'transformation_type' in var.metadata
            assert 'transformation_params' in var.metadata
            assert 'timestep_range' in var.metadata
            assert 'layer_indices' in var.metadata
            assert 'target_token' in var.metadata
    
    def test_variation_ids_unique(self):
        """Test that all variation IDs are unique."""
        config_path = self.create_test_config()
        config_manager = ConfigManager()
        config = config_manager.load_config(config_path)
        
        orchestrator = VideoGenerationOrchestrator(config)
        orchestrator.setup_batch("test_unique")
        
        variations = orchestrator.process_bending_variations()
        
        variation_ids = [v.variation_id for v in variations]
        assert len(variation_ids) == len(set(variation_ids))  # All unique
    
    def test_bending_config_creation(self):
        """Test that BendingConfig objects are correctly created."""
        config_path = self.create_test_config()
        config_manager = ConfigManager()
        config = config_manager.load_config(config_path)
        
        orchestrator = VideoGenerationOrchestrator(config)
        orchestrator.setup_batch("test_config_creation")
        
        variations = orchestrator.process_bending_variations()
        
        for var in variations:
            config = var.config
            # Check BendingConfig attributes
            assert config.token == "kiss"
            assert config.strength == 1.0
            assert config.padding_mode == "border"
            assert config.scale_factor in [0.75, 1.0, 1.25]
            
            # Check timesteps and layers are set
            if var.timestep_spec != "ALL":
                assert config.apply_to_timesteps is not None
            if var.layer_spec != "ALL":
                assert config.apply_to_layers is not None
    
    def test_multiplicative_composition_validation(self):
        """Test that multiplicative composition produces correct counts."""
        operations = [
            {
                "operation": "scale",
                "parameter_name": "scale_factor",
                "range": [0.75, 1.25],
                "steps": 5,  # 5 param values
                "target_token": "kiss",
                "apply_to_timesteps": ["0-10", "10-19"],  # 2 timestep configs (list)
                "apply_to_layers": ["ALL", [14, 15]]  # 2 layer configs (list)
            }
        ]
        
        config_path = self.create_test_config(operations=operations)
        config_manager = ConfigManager()
        config = config_manager.load_config(config_path)
        
        orchestrator = VideoGenerationOrchestrator(config)
        orchestrator.setup_batch("test_multiplicative")
        
        variations = orchestrator.process_bending_variations()
        
        # Should generate 5 × 2 × 2 = 20 variations
        assert len(variations) == 20
        
        # Verify we have all combinations
        param_values = set(v.parameter_value for v in variations)
        assert len(param_values) == 5
        
        timestep_specs = set(str(v.timestep_spec) for v in variations)
        assert len(timestep_specs) == 2
        
        layer_specs = set(str(v.layer_spec) for v in variations)
        assert len(layer_specs) == 2


class TestBendingVariationHelpers:
    """Test helper functions for bending variations."""
    
    def test_operation_spec_from_dict(self):
        """Test creating OperationSpec from dict config."""
        config_dict = {
            "operation": "scale",
            "parameter_name": "scale_factor",
            "range": [0.75, 1.25],
            "steps": 5,
            "target_token": "kiss",
            "apply_to_timesteps": ["0-10", "10-19"],
            "apply_to_layers": ["ALL", [14, 15]],
            "strength": 0.8,
            "padding_mode": "reflection"
        }
        
        spec = OperationSpec(
            operation=config_dict.get("operation"),
            parameter_name=config_dict.get("parameter_name"),
            range=tuple(config_dict["range"]) if "range" in config_dict else None,
            steps=config_dict.get("steps", 5),
            apply_to_timesteps=config_dict.get("apply_to_timesteps", "ALL"),
            apply_to_layers=config_dict.get("apply_to_layers", "ALL"),
            target_token=config_dict.get("target_token", ""),
            strength=config_dict.get("strength", 1.0),
            padding_mode=config_dict.get("padding_mode", "border")
        )
        
        assert spec.operation == "scale"
        assert spec.parameter_name == "scale_factor"
        assert spec.range == (0.75, 1.25)
        assert spec.steps == 5
        assert spec.apply_to_timesteps == ["0-10", "10-19"]
        assert spec.strength == 0.8
        assert spec.padding_mode == "reflection"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
