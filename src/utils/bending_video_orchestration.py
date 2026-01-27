"""
Helper functions for orchestrating video generation with bending variations.

This module provides functionality to generate videos across multiple dimensions:
- Prompt variations
- Bending variations (parameter × timestep × layer)
- Seed variations

Separated from main orchestrator to keep code organized and testable.
"""

from typing import List, Dict, Optional, Any
from pathlib import Path
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)


def generate_videos_with_bending(
    video_generator,
    batch_dirs: Dict[str, Path],
    prompt_variations: List,
    bending_configs: List,  # List[BendingVariation | None]
    videos_per_var: int,
    use_weighted: bool,
    config,
    latent_storage=None,
    attention_storage=None,
    original_template: Optional[str] = None
) -> Dict[str, List]:
    """
    Generate videos with bending variation support.
    
    This creates videos in the following nested structure:
    - For each prompt variation
      - For each bending config (baseline + variations)
        - For each seed
          - Generate one video
    
    Returns results organized by prompt text.
    """
    results = {}
    video_metadata = []  # Track metadata for each video
    
    total_videos = len(prompt_variations) * len(bending_configs) * videos_per_var
    current_video = 0
    
    logger.info("="*70)
    logger.info("STARTING VIDEO GENERATION WITH BENDING VARIATIONS")
    logger.info("="*70)
    
    for prompt_idx, prompt_var in enumerate(prompt_variations):
        logger.info(f"\n{'─'*70}")
        logger.info(f"PROMPT VARIATION {prompt_idx + 1}/{len(prompt_variations)}")
        logger.info(f"{'─'*70}")
        
        prompt_text = prompt_var.weighted_text if use_weighted and prompt_var.weighted_text else prompt_var.text
        logger.info(f"Prompt: {prompt_text[:80]}{'...' if len(prompt_text) > 80 else ''}")
        
        prompt_results = []
        
        for bending_idx, bending_config in enumerate(bending_configs):
            # Determine if this is baseline or a variation
            is_baseline = bending_config is None
            
            if is_baseline:
                logger.info(f"\n  ┌─ BASELINE (no bending) - {bending_idx + 1}/{len(bending_configs)}")
                bending_display = "baseline"
                bending_metadata = None
            else:
                logger.info(f"\n  ┌─ BENDING VARIATION {bending_idx}/{len(bending_configs) - 1}: {bending_config.display_name}")
                logger.info(f"  │  ID: {bending_config.variation_id}")
                logger.info(f"  │  Operation: {bending_config.operation}")
                logger.info(f"  │  Parameter: {bending_config.parameter_name} = {bending_config.parameter_value}")
                logger.info(f"  │  Timesteps: {bending_config.timestep_spec}")
                logger.info(f"  │  Layers: {bending_config.layer_spec}")
                bending_display = bending_config.variation_id
                bending_metadata = bending_config.metadata
            
            for seed_offset in range(videos_per_var):
                current_video += 1
                base_seed = config.model_settings.seed
                current_seed = base_seed + seed_offset
                
                logger.info(f"  │")
                logger.info(f"  ├─ Seed {seed_offset + 1}/{videos_per_var}: {current_seed} (base: {base_seed} + {seed_offset})")
                logger.info(f"  │  Progress: Video {current_video}/{total_videos} ({100*current_video/total_videos:.1f}%)")
                
                # Generate filename that includes all dimensions
                video_filename = format_video_filename(
                    prompt_idx=prompt_idx,
                    bending_display=bending_display,
                    seed_offset=seed_offset,
                    video_num=current_video
                )
                
                # Create video ID for storage
                video_id = f"p{prompt_idx:03d}_b{bending_idx:03d}_s{seed_offset:03d}"
                
                # Build generation kwargs
                gen_kwargs = {
                    'seed': current_seed,
                    'sampler': config.model_settings.sampler,
                    'cfg_scale': config.model_settings.cfg_scale,
                    'steps': config.model_settings.steps,
                    'width': config.video_settings.width,
                    'height': config.video_settings.height,
                    'fps': config.video_settings.fps,
                    'frames': config.video_settings.frames,
                    'video_id': video_id
                }
                
                # Add latent storage if enabled
                if latent_storage:
                    gen_kwargs['latent_storage'] = latent_storage
                
                # Add attention storage if enabled
                if attention_storage:
                    gen_kwargs['attention_storage'] = attention_storage
                    if original_template:
                        gen_kwargs['attention_target_words'] = extract_target_words(original_template)
                
                # Add bending config if this is not baseline
                if not is_baseline:
                    gen_kwargs['bending_config'] = bending_config.config
                    logger.info(f"  │  ✓ Applying bending config: {bending_config.config.token} @ {bending_config.config.mode}")
                else:
                    logger.info(f"  │  ○ No bending applied (baseline)")
                
                # Generate the video
                output_path = batch_dirs["videos"] / video_filename
                logger.info(f"  │  Output: {video_filename}")
                
                result = video_generator.generate(
                    prompt=prompt_text,
                    output_path=str(output_path),
                    **gen_kwargs
                )
                
                # Log result
                if result.success:
                    logger.info(f"  └─ ✓ SUCCESS: Generated in {result.generation_time:.1f}s")
                else:
                    logger.error(f"  └─ ✗ FAILED: {result.error_message}")
                
                prompt_results.append(result)
                
                # Store metadata for this video
                video_meta = {
                    'video_id': video_id,
                    'video_num': current_video,
                    'filename': video_filename,
                    'prompt_variation': {
                        'index': prompt_idx,
                        'text': prompt_text,
                        'template_parts': prompt_var.template_parts if hasattr(prompt_var, 'template_parts') else None
                    },
                    'seed': current_seed,
                    'seed_offset': seed_offset,
                    'bending_metadata': bending_metadata,
                    'success': result.success,
                    'generation_time': result.generation_time if result.success else None,
                    'error': result.error_message if not result.success else None
                }
                video_metadata.append(video_meta)
        
        results[prompt_text] = prompt_results
    
    # Save comprehensive metadata
    save_video_metadata(batch_dirs, video_metadata)
    
    logger.info("\n" + "="*70)
    logger.info("VIDEO GENERATION COMPLETE")
    logger.info("="*70)
    
    return results


def format_video_filename(prompt_idx: int, bending_display: str, 
                         seed_offset: int, video_num: int) -> str:
    """Format filename that encodes all dimensions."""
    # Include all dimensions in filename for easy identification
    return f"video_{video_num:04d}_p{prompt_idx:02d}_{bending_display}_s{seed_offset:02d}.mp4"


def extract_target_words(template: str) -> List[str]:
    """Extract target words from parentheses in template."""
    return re.findall(r'\(([^)]+)\)', template)


def save_video_metadata(batch_dirs: Dict[str, Path], video_metadata: List[Dict]):
    """Save comprehensive metadata including bending variations."""
    metadata_file = batch_dirs["configs"] / "video_metadata.json"
    
    # Create structured metadata
    metadata = {
        'generation_date': datetime.now().isoformat(),
        'total_videos': len(video_metadata),
        'successful_videos': sum(1 for v in video_metadata if v['success']),
        'failed_videos': sum(1 for v in video_metadata if not v['success']),
        'has_bending_variations': any(v['bending_metadata'] is not None for v in video_metadata),
        'videos': video_metadata
    }
    
    import json
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"\n✓ Saved comprehensive metadata to: {metadata_file}")
    logger.info(f"  - Total videos: {metadata['total_videos']}")
    logger.info(f"  - Successful: {metadata['successful_videos']}")
    logger.info(f"  - Failed: {metadata['failed_videos']}")
    if metadata['has_bending_variations']:
        num_with_bending = sum(1 for v in video_metadata if v['bending_metadata'] is not None)
        logger.info(f"  - With bending variations: {num_with_bending}")
