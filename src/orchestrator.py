"""
Main orchestrator for WAN 1.3B video generation batches.
Coordinates configuration, prompt processing, and video generation.
"""
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import json

from src.config import ConfigManager, GenerationConfig
from src.prompts import PromptManager, PromptTemplate, PromptVariation
from src.generators import WAN13BVideoGenerator, BatchVideoGenerator
from src.utils import FileManager, LogManager, ProgressTracker, MetadataManager


class VideoGenerationOrchestrator:
    """
    Main orchestrator class that coordinates the entire video generation process.
    """
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.logger = None
        self.batch_dirs = None
        
        # Initialize components
        self.config_manager = ConfigManager()
        self.prompt_manager = PromptManager()
        self.video_generator = WAN13BVideoGenerator(config)
        self.batch_generator = BatchVideoGenerator(self.video_generator)
        
        # Setup progress tracking
        self.progress_tracker = None
        self._setup_progress_callback()
    
    def _setup_progress_callback(self):
        """Setup progress callback for batch generation."""
        def progress_callback(current: int, total: int, status: str):
            if self.progress_tracker:
                self.progress_tracker.update(current, status[:50])  # Limit status length
            else:
                print(f"Progress: {current}/{total} - {status}")
        
        self.batch_generator.set_progress_callback(progress_callback)
    
    def setup_batch(self, batch_name: Optional[str] = None) -> Dict[str, Path]:
        """Setup directory structure and logging for a new batch."""
        # Create batch directory structure
        self.batch_dirs = FileManager.create_batch_structure(
            base_dir=self.config.output_dir,
            batch_name=batch_name or self.config.batch_name,
            use_timestamp=self.config.use_timestamp
        )
        
        # Setup logging
        self.logger = LogManager.setup_logging(
            log_dir=str(self.batch_dirs["logs"]),
            log_level="INFO"
        )
        
        # Save configuration to batch
        config_file = self.batch_dirs["configs"] / "generation_config.yaml"
        self.config_manager.save_config(self.config, config_file)
        
        self.logger.info(f"Batch setup complete. Output directory: {self.batch_dirs['root']}")
        
        return self.batch_dirs
    
    def process_prompt_template(self, 
                              template: str, 
                              max_variations: Optional[int] = None) -> List[PromptVariation]:
        """Process prompt template and generate variations."""
        self.logger.info(f"Processing prompt template: {template}")
        
        # Load and validate template
        prompt_template = self.prompt_manager.load_template(template)
        
        # Generate variations
        variations = prompt_template.generate_variations()
        
        if max_variations and len(variations) > max_variations:
            self.logger.warning(f"Template produces {len(variations)} variations, limiting to {max_variations}")
            variations = variations[:max_variations]
        
        self.logger.info(f"Generated {len(variations)} prompt variations")
        
        # Save template and variations info
        if self.batch_dirs:
            template_file = self.batch_dirs["configs"] / "prompt_template.txt"
            with open(template_file, "w", encoding="utf-8") as f:
                f.write(template)
            
            variations_file = self.batch_dirs["configs"] / "prompt_variations.json"
            variations_data = [
                {
                    "id": var.variation_id,
                    "text": var.text,
                    "variables": var.variables
                }
                for var in variations
            ]
            
            with open(variations_file, "w", encoding="utf-8") as f:
                json.dump(variations_data, f, indent=2, ensure_ascii=False)
        
        return variations
    
    def generate_videos(self, 
                       variations: List[PromptVariation],
                       videos_per_variation: Optional[int] = None) -> Dict[str, List]:
        """Generate videos for all prompt variations."""
        if not self.batch_dirs:
            raise ValueError("Batch not setup. Call setup_batch() first.")
        
        videos_per_var = videos_per_variation or self.config.videos_per_variation
        total_videos = len(variations) * videos_per_var
        
        self.logger.info(f"Starting video generation: {len(variations)} variations × {videos_per_var} videos = {total_videos} total videos")
        
        # Setup progress tracker
        self.progress_tracker = ProgressTracker(
            total=total_videos,
            description="Generating videos"
        )
        
        # Prepare prompts for batch generation
        prompts = [var.text for var in variations]
        
        # Generate videos
        results = self.batch_generator.generate_batch(
            prompts=prompts,
            output_dir=str(self.batch_dirs["videos"]),
            videos_per_prompt=videos_per_var,
            filename_template="video_{video_num:03d}",
            # Pass model settings as generation parameters
            seed=self.config.model_settings.seed,
            sampler=self.config.model_settings.sampler,
            cfg_scale=self.config.model_settings.cfg_scale,
            steps=self.config.model_settings.steps,
            width=self.config.video_settings.width,
            height=self.config.video_settings.height,
            fps=self.config.video_settings.fps,
            frames=self.config.video_settings.frames
        )
        
        # Finish progress tracking
        if self.progress_tracker:
            self.progress_tracker.finish("Video generation complete")
        
        # Generate summary report
        report_file = self.batch_dirs["reports"] / "generation_summary.json"
        report = self.batch_generator.generate_summary_report(results, str(report_file))
        
        # Log summary
        summary = report["summary"]
        self.logger.info(f"Generation Summary:")
        self.logger.info(f"  Total videos: {summary['total_videos']}")
        self.logger.info(f"  Successful: {summary['successful_videos']}")
        self.logger.info(f"  Failed: {summary['failed_videos']}")
        self.logger.info(f"  Success rate: {summary['success_rate']:.1%}")
        self.logger.info(f"  Total time: {summary['total_generation_time']:.1f}s")
        self.logger.info(f"  Avg time per video: {summary['average_time_per_video']:.1f}s")
        
        return results
    
    def run_full_batch(self, 
                      template: str,
                      batch_name: Optional[str] = None,
                      videos_per_variation: Optional[int] = None,
                      max_variations: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a complete batch generation process.
        
        Args:
            template: Prompt template with variation syntax
            batch_name: Optional name for the batch
            videos_per_variation: Override config setting for videos per variation
            max_variations: Limit the number of variations to process
        
        Returns:
            Dictionary with batch results and metadata
        """
        try:
            # Setup batch
            batch_dirs = self.setup_batch(batch_name)
            
            # Process prompts
            variations = self.process_prompt_template(template, max_variations)
            
            # Generate videos
            results = self.generate_videos(variations, videos_per_variation)
            
            # Create final metadata
            batch_metadata = {
                "batch_info": {
                    "batch_name": batch_name,
                    "template": template,
                    "total_variations": len(variations),
                    "videos_per_variation": videos_per_variation or self.config.videos_per_variation,
                    "output_directory": str(batch_dirs["root"])
                },
                "configuration": self.config_manager._config_to_dict(self.config),
                "variations": [
                    {
                        "id": var.variation_id,
                        "text": var.text,
                        "variables": var.variables
                    }
                    for var in variations
                ],
                "results": results
            }
            
            # Save batch metadata
            metadata_file = batch_dirs["root"] / "batch_metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(batch_metadata, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Batch complete! Results saved to: {batch_dirs['root']}")
            
            return batch_metadata
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Batch generation failed: {e}")
            raise
    
    def validate_setup(self) -> List[str]:
        """Validate that all components are properly setup."""
        issues = []
        
        # Check video generator
        if not self.video_generator.is_available():
            issues.append("Video generator not available")
        
        # Validate configuration
        try:
            self.config_manager.validate_config(self.config)
        except ValueError as e:
            issues.append(f"Configuration validation failed: {e}")
        
        # Check output directory permissions
        try:
            test_dir = Path(self.config.output_dir) / "test"
            test_dir.mkdir(parents=True, exist_ok=True)
            test_dir.rmdir()
        except Exception as e:
            issues.append(f"Cannot write to output directory: {e}")
        
        return issues
    
    def preview_batch(self, template: str, max_preview: int = 5) -> Dict[str, Any]:
        """Preview what a batch would generate without actually generating."""
        prompt_template = PromptTemplate(template)
        variations = prompt_template.generate_variations()
        
        preview_info = {
            "template": template,
            "total_variations": len(variations),
            "videos_per_variation": self.config.videos_per_variation,
            "total_videos": len(variations) * self.config.videos_per_variation,
            "preview_variations": [var.text for var in variations[:max_preview]],
            "estimated_time": "Unknown",  # Would need model timing data
            "configuration": self.config_manager._config_to_dict(self.config)
        }
        
        return preview_info
