"""
VLM Batch Processor
Orchestrates video analysis across entire batches.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime
import time
import traceback

from .vlm_model_loader import VLMModelLoader
from .vlm_prompt_orchestrator import VLMPromptOrchestrator

logger = logging.getLogger(__name__)

class VLMBatchProcessor:
    """Processes video analysis across entire batches."""
    
    def __init__(
        self, 
        schema_path: str,
        max_retries: int = 1
    ):
        self.schema_path = schema_path
        self.max_retries = max_retries
        
        # Initialize components
        self.model_loader = VLMModelLoader()
        self.prompt_orchestrator = VLMPromptOrchestrator(schema_path)
        
        self.run_id = str(uuid.uuid4())
        
    def initialize(self):
        """Initialize the VLM model."""
        logger.info("Initializing VLM batch processor...")
        self.model_loader.load_model()
        logger.info(f"✅ Processor initialized with run_id: {self.run_id}")
        
    def extract_video_metadata(self, video_path: Path, prompt_text: str = "") -> Dict[str, Any]:
        """Extract basic metadata from video file."""
        try:
            # Try to get video duration/fps with opencv or similar
            # For now, use basic metadata
            video_id = video_path.stem
            
            # Extract prompt group and video number from path structure
            # Expected: batch_name/videos/prompt_XXX/video_XXX.mp4
            prompt_group = video_path.parent.name
            
            return {
                "video_id": video_id,
                "prompt_text": prompt_text,
                "source_model": "Wan",  # Default for this project
                "seed": "unknown",
                "duration_seconds": 3.2,  # Default Wan video length
                "fps_estimate": 12.0  # Wan videos are 12fps
            }
        except Exception as e:
            logger.warning(f"Failed to extract metadata for {video_path}: {e}")
            return {
                "video_id": str(video_path.stem),
                "prompt_text": prompt_text,
                "source_model": "unknown",
                "seed": "unknown", 
                "duration_seconds": 0.0,
                "fps_estimate": 12.0
            }
            
    def analyze_single_video(
        self, 
        video_path: Path, 
        output_path: Path,
        prompt_text: str = ""
    ) -> Dict[str, Any]:
        """Analyze a single video through the complete pipeline."""
        
        logger.info(f"Analyzing video: {video_path.name}")
        
        # Extract video metadata
        video_metadata = self.extract_video_metadata(video_path, prompt_text)
        
        # Initialize result structure
        result = {
            "schema_version": "1.0",
            "ok": True,
            "analysis_stages_completed": [],
            "errors": []
        }
        
        # Run through prompt sequence
        accumulated_data = {}
        
        for stage in self.prompt_orchestrator.prompt_sequence:
            try:
                logger.info(f"  Running stage: {stage}")
                
                # Get prompt for this stage
                prompt = self.prompt_orchestrator.get_prompt_for_stage(
                    stage=stage,
                    video_metadata=video_metadata,
                    prior_data=accumulated_data,
                    run_id=self.run_id
                )
                
                # Expected keys for validation
                stage_keys = self._get_expected_keys_for_stage(stage)
                
                # Attempt analysis with retry
                stage_data = self._analyze_with_retry(
                    video_path=video_path,
                    prompt=prompt,
                    expected_keys=stage_keys,
                    stage=stage
                )
                
                if stage_data:
                    accumulated_data.update(stage_data)
                    result["analysis_stages_completed"].append(stage)
                else:
                    result["ok"] = False
                    result["errors"].append(f"Failed to complete stage: {stage}")
                    
            except Exception as e:
                logger.error(f"Error in stage {stage}: {e}")
                result["ok"] = False
                result["errors"].append(f"Stage {stage} error: {str(e)}")
                
        # Merge accumulated data into result
        result.update(accumulated_data)
        
        # Save result
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"✅ Analysis saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save result: {e}")
            result["errors"].append(f"Save error: {str(e)}")
            
        return result
        
    def _get_expected_keys_for_stage(self, stage: str) -> List[str]:
        """Get expected top-level keys for each stage."""
        stage_keys = {
            "people_and_demographics": ["video_metadata", "people"],
            "scene_and_segments": ["scene_global", "temporal_segments"],
            "relationships_and_dyads": ["dyads"],
            "interpretation_and_flags": ["flags_and_readings", "scores_optional"],
            "finalize_provenance": ["provenance"]
        }
        return stage_keys.get(stage, [])
        
    def _analyze_with_retry(
        self, 
        video_path: Path, 
        prompt: str, 
        expected_keys: List[str],
        stage: str
    ) -> Optional[Dict[str, Any]]:
        """Analyze with retry logic for invalid JSON."""
        
        for attempt in range(self.max_retries + 1):
            try:
                # Get system prompt on first attempt only
                if attempt == 0:
                    system_prompt = self.prompt_orchestrator.get_system_prompt()
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                else:
                    # Retry with modified prompt
                    error_info = f"Attempt {attempt} failed JSON validation"
                    retry_prompt = self.prompt_orchestrator.get_retry_prompt(prompt, error_info)
                    full_prompt = retry_prompt
                    
                logger.info(f"    Attempt {attempt + 1}/{self.max_retries + 1}")
                
                # Call VLM with conservative parameters to avoid CUDA errors
                response = self.model_loader.analyze_video(
                    video_path=str(video_path),
                    text_prompt=full_prompt,
                    max_new_tokens=256,  # Reduced for testing
                    temperature=0.0,     # Force greedy decoding to avoid sampling errors
                    do_sample=False,     # Explicitly disable sampling
                    fps=12.0,
                    max_pixels=151200
                )
                
                logger.debug(f"VLM Response: {response[:200]}...")
                
                # Validate response
                validation = self.prompt_orchestrator.validate_response(response, expected_keys)
                
                if validation["valid"]:
                    logger.info(f"    ✅ Stage {stage} completed successfully")
                    return validation["data"]
                else:
                    logger.warning(f"    ❌ Validation failed: {validation['error']}")
                    if attempt == self.max_retries:
                        # Final attempt failed, save partial data
                        logger.error(f"Final attempt failed for stage {stage}")
                        return validation.get("data")  # Return partial data if available
                        
            except Exception as e:
                logger.error(f"    ❌ Analysis attempt {attempt + 1} failed: {e}")
                logger.error(traceback.format_exc())
                if attempt == self.max_retries:
                    return None
                    
        return None
        
    def process_batch(
        self, 
        batch_path: Path, 
        prompt_groups: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Process an entire batch of videos."""
        
        logger.info(f"Processing batch: {batch_path}")
        
        # Find video structure
        videos_dir = batch_path / "videos"
        if not videos_dir.exists():
            raise FileNotFoundError(f"Videos directory not found: {videos_dir}")
            
        # Get prompt groups
        if prompt_groups is None:
            prompt_groups = [d.name for d in videos_dir.iterdir() if d.is_dir()]
            
        # Create output directory
        output_dir = batch_path / "vlm_analysis"
        output_dir.mkdir(exist_ok=True)
        
        # Process each group
        batch_results = {
            "batch_name": batch_path.name,
            "run_id": self.run_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "prompt_groups": {},
            "summary": {
                "total_videos": 0,
                "successful_analyses": 0,
                "failed_analyses": 0
            }
        }
        
        for prompt_group in prompt_groups:
            logger.info(f"Processing prompt group: {prompt_group}")
            
            group_dir = videos_dir / prompt_group
            if not group_dir.exists():
                logger.warning(f"Prompt group directory not found: {group_dir}")
                continue
                
            # Find videos in group
            video_files = list(group_dir.glob("*.mp4"))
            if not video_files:
                logger.warning(f"No videos found in {group_dir}")
                continue
                
            # Create output directory for this group
            group_output_dir = output_dir / prompt_group
            group_output_dir.mkdir(exist_ok=True)
            
            # Process each video
            group_results = []
            for video_file in sorted(video_files):
                batch_results["summary"]["total_videos"] += 1
                
                # Define output path
                output_file = group_output_dir / f"{video_file.stem}.json"
                
                # Analyze video
                try:
                    result = self.analyze_single_video(
                        video_path=video_file,
                        output_path=output_file,
                        prompt_text=""  # Could extract from prompt group name
                    )
                    
                    if result.get("ok", False):
                        batch_results["summary"]["successful_analyses"] += 1
                    else:
                        batch_results["summary"]["failed_analyses"] += 1
                        
                    group_results.append({
                        "video_file": video_file.name,
                        "output_file": str(output_file.relative_to(batch_path)),
                        "success": result.get("ok", False),
                        "stages_completed": result.get("analysis_stages_completed", [])
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to process {video_file}: {e}")
                    batch_results["summary"]["failed_analyses"] += 1
                    group_results.append({
                        "video_file": video_file.name,
                        "output_file": None,
                        "success": False,
                        "error": str(e)
                    })
                    
            batch_results["prompt_groups"][prompt_group] = group_results
            
        # Save batch summary
        summary_file = output_dir / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(batch_results, f, indent=2)
            
        logger.info(f"✅ Batch processing complete: {batch_results['summary']}")
        return batch_results
        
    def cleanup(self):
        """Clean up resources."""
        self.model_loader.cleanup()
