"""
Main orchestrator for WAN 1.3B video generation batches.
Coordinates configuration, prompt processing, and video generation.
"""
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import json

from src.config import ConfigManager, GenerationConfig
from src.prompts import PromptManager, PromptTemplate, PromptVariation, WeightingConfig
from src.generators import WAN13BVideoGenerator, BatchVideoGenerator
from src.utils import FileManager, LogManager, ProgressTracker, MetadataManager


class VideoGenerationOrchestrator:
    """
    Main orchestrator class that coordinates the entire video generation process.
    """
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)  # Initialize logger immediately
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
        
        # Setup enhanced logging for batch operations (but keep existing logger)
        batch_logger = LogManager.setup_logging(
            log_dir=str(self.batch_dirs["logs"]),
            log_level="INFO"
        )
        # Update our logger to use the batch logger if it's different
        if batch_logger != self.logger:
            self.logger = batch_logger
        
        # Save configuration to batch
        config_file = self.batch_dirs["configs"] / "generation_config.yaml"
        self.config_manager.save_config(self.config, config_file)
        
        self.logger.info(f"Batch setup complete. Output directory: {self.batch_dirs['root']}")
        
        return self.batch_dirs
    
    def process_prompt_template(self, 
                              template: str, 
                              max_variations: Optional[int] = None,
                              save_template_files: bool = True) -> List[PromptVariation]:
        """Process prompt template and generate variations."""
        self.logger.info(f"Processing prompt template: {template}")
        
        # Create weighting config from settings
        weighting_config = None
        if hasattr(self.config, 'prompt_settings') and self.config.prompt_settings.enable_weighting:
            weighting_config = WeightingConfig(
                enable_weighting=True,
                variation_weight=self.config.prompt_settings.variation_weight,
                base_weight=self.config.prompt_settings.base_weight
            )
            self.logger.info(f"Prompt weighting enabled: variation_weight={weighting_config.variation_weight}")
        
        # Load and validate template with weighting config
        prompt_template = self.prompt_manager.load_template(template, weighting_config)
        
        # Generate variations
        variations = prompt_template.generate_variations()
        
        if max_variations and len(variations) > max_variations:
            self.logger.warning(f"Template produces {len(variations)} variations, limiting to {max_variations}")
            variations = variations[:max_variations]
        
        self.logger.info(f"Generated {len(variations)} prompt variations")
        if weighting_config and weighting_config.enable_weighting:
            weighted_count = sum(1 for var in variations if var.weighted_text)
            self.logger.info(f"Created {weighted_count} weighted prompt variations")
        
        # Save template and variations info ONLY if not in continuation mode
        if self.batch_dirs and save_template_files:
            template_file = self.batch_dirs["configs"] / "prompt_template.txt"
            self.logger.info(f"Saving original template to: {template_file}")
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
        elif not save_template_files:
            self.logger.info("Skipping template file save (continuation mode - preserving original files)")
        
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
        # Use weighted prompts if available and enabled
        use_weighted = (hasattr(self.config, 'prompt_settings') and 
                       self.config.prompt_settings.enable_prompt_weighting)
        
        prompts = []
        for var in variations:
            if use_weighted and var.weighted_text:
                prompts.append(var.weighted_text)
                self.logger.debug(f"Using weighted prompt: {var.weighted_text}")
            else:
                prompts.append(var.text)
        
        self.logger.info(f"Using {'weighted' if use_weighted else 'standard'} prompts for generation")
        
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

    def continue_from_batch(self, 
                           batch_directory: str,
                           new_template: Optional[str] = None,
                           videos_per_variation: Optional[int] = None,
                           max_variations: Optional[int] = None) -> Dict[str, Any]:
        """
        Continue generation from an existing batch directory.
        
        Args:
            batch_directory: Path to existing batch directory
            new_template: Optional new template (if None, uses original template)
            videos_per_variation: Override config setting for videos per variation
            max_variations: Limit the number of variations to process
        
        Returns:
            Dictionary with batch results and metadata
        """
        batch_path = Path(batch_directory)
        if not batch_path.exists():
            raise ValueError(f"Batch directory does not exist: {batch_directory}")
        
        # Load existing configuration
        config_file = batch_path / "configs" / "generation_config.yaml"
        if not config_file.exists():
            raise ValueError(f"Configuration file not found in batch directory: {config_file}")
        
        self.logger.info(f"Loading original batch configuration from: {config_file}")
        
        # Replace current config with the batch's original config
        # This ensures we use the same settings as the original generation
        self.config = self.config_manager.load_config(config_file)
        
        # Update config with any new overrides while preserving original settings
        if videos_per_variation is not None:
            self.config.videos_per_variation = videos_per_variation
        
        # Recreate video generator with the loaded config
        self.video_generator = WAN13BVideoGenerator(self.config)
        self.batch_generator = BatchVideoGenerator(self.video_generator)
        self._setup_progress_callback()
        
        self.logger.info(f"Using original batch settings: videos_per_variation={self.config.videos_per_variation}")
        
        # Set up continuation with existing batch structure
        self.batch_dirs = {
            "root": batch_path,
            "videos": batch_path / "videos",
            "logs": batch_path / "logs",
            "configs": batch_path / "configs",
            "reports": batch_path / "reports"
        }
        
        # Setup enhanced logging for continuation (append to existing logs)
        batch_logger = LogManager.setup_logging(
            log_dir=str(self.batch_dirs["logs"]),
            log_level="INFO"
        )
        self.logger = batch_logger
        
        # Analyze existing batch to determine continuation point
        continuation_info = self._analyze_existing_batch(batch_path, new_template)
        
        if continuation_info['completed_videos'] == continuation_info['total_expected_videos']:
            self.logger.info("Batch is already complete. No additional generation needed.")
            
            # Still return metadata for consistency
            return {
                "batch_info": {
                    "batch_name": self.config.batch_name,
                    "template": continuation_info['template'],
                    "total_variations": len(continuation_info['variations']),
                    "videos_per_variation": self.config.videos_per_variation,
                    "output_directory": str(batch_path),
                    "continuation": True,
                    "already_complete": True
                },
                "configuration": self.config_manager._config_to_dict(self.config),
                "variations": continuation_info['variations'],
                "results": {"message": "Batch was already complete"}
            }
        
        self.logger.info(f"=== CONTINUING BATCH GENERATION ===")
        self.logger.info(f"Original template: {continuation_info['original_template']}")
        if new_template and new_template != continuation_info['original_template']:
            self.logger.info(f"New template: {new_template}")
        self.logger.info(f"Completed videos: {continuation_info['completed_videos']}")
        self.logger.info(f"Expected total: {continuation_info['total_expected_videos']}")
        
        # Process prompt template (use new template if provided, otherwise original)
        template_to_use = new_template if new_template else continuation_info['original_template']
        if not template_to_use:
            raise ValueError("No template available (neither new template provided nor original template found)")
        
        # CRITICAL: Do not overwrite original template files during continuation
        variations = self.process_prompt_template(template_to_use, max_variations, save_template_files=False)
        
        # Generate videos for continuation
        results = self._generate_videos_continuation(
            variations, 
            continuation_info, 
            videos_per_variation or self.config.videos_per_variation
        )
        
        # Create final metadata
        batch_metadata = {
            "batch_info": {
                "batch_name": self.config.batch_name,
                "template": template_to_use,
                "original_template": continuation_info['original_template'],
                "total_variations": len(variations),
                "videos_per_variation": videos_per_variation or self.config.videos_per_variation,
                "output_directory": str(batch_path),
                "continuation": True,
                "previously_completed": continuation_info['completed_videos']
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
        
        # Save updated batch metadata
        metadata_file = batch_path / "batch_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(batch_metadata, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Batch continuation complete! Results saved to: {batch_path}")
        
        return batch_metadata

    def _analyze_existing_batch(self, batch_path: Path, new_template: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze existing batch to determine what has been completed and what needs continuation.
        
        Args:
            batch_path: Path to existing batch directory
            new_template: Optional new template to analyze against
        
        Returns:
            Dictionary with batch analysis information
        """
        videos_dir = batch_path / "videos"
        
        # Try to find original template from batch files
        original_template = None
        
        # First try to read the original template file (most reliable)
        template_file = batch_path / "configs" / "prompt_template.txt"
        if template_file.exists():
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    original_template = f.read().strip()
                self.logger.info(f"Found original template file: {original_template[:100]}...")
            except Exception as e:
                self.logger.warning(f"Could not read template file: {e}")
        
        # If no template file, try batch metadata as fallback
        if not original_template:
            metadata_file = batch_path / "batch_metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        original_template = metadata.get('batch_info', {}).get('template')
                        if original_template:
                            self.logger.info(f"Found template in metadata: {original_template[:100]}...")
                except Exception as e:
                    self.logger.warning(f"Could not read batch metadata: {e}")
        
        # Last resort: try to infer from existing prompts (but this is dangerous!)
        if not original_template and videos_dir.exists():
            self.logger.warning("No original template found in files - attempting to infer from prompts (this may be incomplete)")
            prompt_dirs = [d for d in videos_dir.iterdir() if d.is_dir() and d.name.startswith('prompt_')]
            if prompt_dirs:
                # Read prompt files to see if we can infer a template
                prompt_texts = []
                for prompt_dir in sorted(prompt_dirs):
                    prompt_file = prompt_dir / "prompt.txt"
                    if prompt_file.exists():
                        with open(prompt_file, 'r', encoding='utf-8') as f:
                            prompt_texts.append(f.read().strip())
                
                if len(prompt_texts) == 1:
                    # Single prompt - use it directly
                    original_template = prompt_texts[0]
                    self.logger.warning(f"Single prompt batch detected, using: {original_template[:100]}...")
                else:
                    # Multiple prompts - this is problematic, we can't reconstruct the original template
                    self.logger.error(f"Multi-prompt batch detected ({len(prompt_texts)} prompts) - cannot safely reconstruct original template!")
                    self.logger.error("Original template may have been corrupted. Using first prompt as fallback.")
                    original_template = prompt_texts[0] if prompt_texts else None
        
        # Count existing videos
        completed_videos = 0
        prompt_variations = []
        
        if videos_dir.exists():
            prompt_dirs = [d for d in videos_dir.iterdir() if d.is_dir() and d.name.startswith('prompt_')]
            
            for prompt_dir in sorted(prompt_dirs):
                # Count video files in this prompt directory
                video_files = [f for f in prompt_dir.iterdir() if f.suffix in ['.mp4', '.avi', '.mov']]
                completed_videos += len(video_files)
                
                # Read prompt text
                prompt_file = prompt_dir / "prompt.txt"
                if prompt_file.exists():
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        prompt_text = f.read().strip()
                        prompt_variations.append({
                            'id': prompt_dir.name,
                            'text': prompt_text,
                            'completed_videos': len(video_files)
                        })
        
        # Calculate expected total based on config
        expected_videos_per_variation = self.config.videos_per_variation
        expected_total_videos = len(prompt_variations) * expected_videos_per_variation if prompt_variations else 0
        
        # Use template from argument if provided, otherwise use inferred
        template_to_use = new_template if new_template else original_template
        
        return {
            'original_template': original_template,
            'template': template_to_use,
            'variations': prompt_variations,
            'completed_videos': completed_videos,
            'total_expected_videos': expected_total_videos,
            'videos_per_variation': expected_videos_per_variation
        }

    def _generate_videos_continuation(self, 
                                    variations: List[PromptVariation],
                                    continuation_info: Dict[str, Any],
                                    videos_per_variation: int) -> Dict[str, Any]:
        """
        Generate videos for batch continuation, picking up where the previous generation left off.
        
        Args:
            variations: List of prompt variations to process
            continuation_info: Information about existing batch state
            videos_per_variation: Number of videos to generate per variation
        
        Returns:
            Generation results dictionary
        """
        # Map variations to existing prompt directories
        existing_variations = {var['text']: var for var in continuation_info['variations']}
        
        prompts_to_generate = []
        video_counts = []
        
        # For each variation, determine if we need to generate more videos
        for variation in variations:
            existing_var = existing_variations.get(variation.text)
            
            if existing_var:
                # This prompt already exists, check if we need more videos
                completed = existing_var['completed_videos']
                needed = videos_per_variation - completed
                
                if needed > 0:
                    prompts_to_generate.append(variation.text)
                    video_counts.append(needed)
                    self.logger.info(f"Prompt '{variation.text[:50]}...' needs {needed} more videos (has {completed}/{videos_per_variation})")
                else:
                    self.logger.info(f"Prompt '{variation.text[:50]}...' is complete ({completed}/{videos_per_variation})")
            else:
                # New prompt, generate all videos
                prompts_to_generate.append(variation.text)
                video_counts.append(videos_per_variation)
                self.logger.info(f"New prompt '{variation.text[:50]}...' needs {videos_per_variation} videos")
        
        if not prompts_to_generate:
            self.logger.info("No additional videos needed - all prompts are complete!")
            return {"message": "All prompts already complete", "prompts_processed": 0, "videos_generated": 0}
        
        # Generate videos using custom continuation logic
        results = self._generate_continuation_videos(prompts_to_generate, video_counts, videos_per_variation)
        
        return results

    def _generate_continuation_videos(self, 
                                    prompts: List[str], 
                                    video_counts: List[int],
                                    videos_per_variation: int) -> Dict[str, Any]:
        """
        Generate videos for continuation, handling existing prompt directories.
        """
        results = {}
        total_videos_to_generate = sum(video_counts)
        current_video = 0
        
        # Setup progress tracking
        if self.progress_tracker is None:
            self.progress_tracker = ProgressTracker(total_videos_to_generate, "Continuing video generation")
        
        for prompt_idx, (prompt, videos_needed) in enumerate(zip(prompts, video_counts)):
            prompt_results = []
            prompt_id = f"prompt_{prompt_idx:03d}"
            
            # Find existing prompt directory or create new one
            prompt_dir = None
            videos_dir = self.batch_dirs["videos"]
            
            # Look for existing directory with this prompt
            for existing_dir in videos_dir.iterdir():
                if existing_dir.is_dir() and existing_dir.name.startswith('prompt_'):
                    prompt_file = existing_dir / "prompt.txt"
                    if prompt_file.exists():
                        with open(prompt_file, 'r', encoding='utf-8') as f:
                            existing_prompt = f.read().strip()
                            if existing_prompt == prompt:
                                prompt_dir = existing_dir
                                break
            
            # If no existing directory found, create new one
            if prompt_dir is None:
                # Find next available prompt directory number
                existing_nums = []
                for existing_dir in videos_dir.iterdir():
                    if existing_dir.is_dir() and existing_dir.name.startswith('prompt_'):
                        try:
                            num = int(existing_dir.name.split('_')[1])
                            existing_nums.append(num)
                        except (IndexError, ValueError):
                            pass
                
                next_num = max(existing_nums) + 1 if existing_nums else 0
                prompt_dir = videos_dir / f"prompt_{next_num:03d}"
                prompt_dir.mkdir(exist_ok=True)
                
                # Save prompt to file
                with open(prompt_dir / "prompt.txt", "w", encoding="utf-8") as f:
                    f.write(prompt)
            
            # Count existing videos to determine starting number
            existing_videos = [f for f in prompt_dir.iterdir() if f.suffix in ['.mp4', '.avi', '.mov']]
            start_video_num = len(existing_videos) + 1
            
            # Generate the needed videos
            for video_num_offset in range(videos_needed):
                current_video += 1
                video_num = start_video_num + video_num_offset
                
                # Update progress
                if self.progress_tracker:
                    self.progress_tracker.update(current_video, f"Generating video {video_num} for prompt {prompt_idx}")
                
                # Generate filename
                filename = f"video_{video_num:03d}.mp4"
                
                # Calculate seed with continuation logic
                base_seed = self.config.model_settings.seed
                current_seed = base_seed + video_num - 1  # Adjust for continuation
                
                output_path = prompt_dir / filename
                
                self.logger.info(f"Generating continuation video {current_video}/{total_videos_to_generate}: {filename}")
                self.logger.info(f"Using seed: {current_seed} (base: {base_seed} + video_num: {video_num - 1})")
                
                # Generate video
                result = self.video_generator.generate(
                    prompt=prompt,
                    output_path=str(output_path),
                    seed=current_seed,
                    sampler=self.config.model_settings.sampler,
                    cfg_scale=self.config.model_settings.cfg_scale,
                    steps=self.config.model_settings.steps,
                    width=self.config.video_settings.width,
                    height=self.config.video_settings.height,
                    fps=self.config.video_settings.fps,
                    frames=self.config.video_settings.frames
                )
                
                # Log result
                if result.success:
                    self.logger.info(f"Successfully generated: {result.video_path}")
                else:
                    self.logger.error(f"Generation failed: {result.error_message}")
                
                prompt_results.append(result)
            
            results[prompt] = prompt_results
        
        # Finish progress tracking
        if self.progress_tracker:
            self.progress_tracker.finish("Video continuation complete")
        
        return results
