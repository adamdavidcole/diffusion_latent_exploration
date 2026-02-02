"""
Utilities for resuming interrupted video generation batches.

Supports:
- Resuming after interruption (Ctrl+C, crash)
- Resuming with expanded configuration (more variations)
- Resuming with failed videos only
- Resuming dual GPU runs
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ResumeInfo:
    """Information about batch resume state."""
    batch_dir: Path
    existing_metadata: Dict[str, Any]
    completed_video_ids: Set[str]
    failed_video_ids: Set[str]
    max_video_num: int
    total_existing: int
    total_successful: int
    total_failed: int
    has_dual_gpu_metadata: bool


def load_existing_metadata(batch_dir: Path) -> Dict[str, Any]:
    """
    Load existing video metadata from batch directory.
    
    Handles both single and dual GPU metadata files:
    - video_metadata.json (merged or single GPU)
    - video_metadata_gpu0.json, video_metadata_gpu1.json (dual GPU)
    
    Args:
        batch_dir: Path to batch directory
    
    Returns:
        Merged metadata dictionary
    """
    configs_dir = batch_dir / "configs"
    
    # Try merged metadata first
    merged_file = configs_dir / "video_metadata.json"
    if merged_file.exists():
        logger.info(f"Loading merged metadata: {merged_file}")
        with open(merged_file, 'r') as f:
            return json.load(f)
    
    # Try dual GPU metadata
    gpu_files = list(configs_dir.glob("video_metadata_*gpu*.json"))
    if gpu_files:
        logger.info(f"Found {len(gpu_files)} GPU metadata files, merging...")
        return merge_gpu_metadata_files(gpu_files)
    
    # No metadata found
    logger.warning(f"No video metadata found in {configs_dir}")
    return {
        'generation_date': None,
        'total_videos': 0,
        'successful_videos': 0,
        'failed_videos': 0,
        'has_bending_variations': False,
        'videos': []
    }


def merge_gpu_metadata_files(gpu_files: List[Path]) -> Dict[str, Any]:
    """
    Merge multiple GPU metadata files into single metadata structure.
    
    Args:
        gpu_files: List of GPU metadata file paths
    
    Returns:
        Merged metadata dictionary
    """
    merged = {
        'generation_date': None,
        'total_videos': 0,
        'successful_videos': 0,
        'failed_videos': 0,
        'has_bending_variations': False,
        'videos': []
    }
    
    seen_video_ids = set()
    
    for gpu_file in gpu_files:
        logger.info(f"  Merging: {gpu_file.name}")
        with open(gpu_file, 'r') as f:
            data = json.load(f)
        
        # Update aggregate stats
        merged['total_videos'] += data.get('total_videos', 0)
        merged['successful_videos'] += data.get('successful_videos', 0)
        merged['failed_videos'] += data.get('failed_videos', 0)
        
        if data.get('has_bending_variations'):
            merged['has_bending_variations'] = True
        
        # Use earliest generation date
        gen_date = data.get('generation_date')
        if gen_date:
            if not merged['generation_date'] or gen_date < merged['generation_date']:
                merged['generation_date'] = gen_date
        
        # Merge video entries (avoid duplicates)
        for video in data.get('videos', []):
            video_id = video.get('video_id')
            if video_id and video_id not in seen_video_ids:
                merged['videos'].append(video)
                seen_video_ids.add(video_id)
    
    # Sort videos by video_num for consistency
    merged['videos'].sort(key=lambda x: x.get('video_num', 0))
    
    logger.info(f"  Merged {len(merged['videos'])} unique videos")
    
    return merged


def analyze_batch_state(batch_dir: Path, force_regenerate: Optional[Set[str]] = None) -> ResumeInfo:
    """
    Analyze existing batch to determine resume state.
    
    Args:
        batch_dir: Path to batch directory
        force_regenerate: Optional set of video_ids to force regenerate
    
    Returns:
        ResumeInfo with batch state
    """
    logger.info("="*70)
    logger.info("ANALYZING BATCH FOR RESUME")
    logger.info("="*70)
    logger.info(f"Batch directory: {batch_dir}")
    
    # Load existing metadata
    existing_metadata = load_existing_metadata(batch_dir)
    
    # Determine if dual GPU metadata exists
    configs_dir = batch_dir / "configs"
    has_dual_gpu = len(list(configs_dir.glob("video_metadata_*gpu*.json"))) > 1
    
    # Analyze videos
    videos = existing_metadata.get('videos', [])
    total_existing = len(videos)
    
    # Separate completed and failed
    completed_video_ids = set()
    failed_video_ids = set()
    max_video_num = 0
    
    for video in videos:
        video_id = video.get('video_id')
        video_num = video.get('video_num', 0)
        success = video.get('success', False)
        
        if video_num > max_video_num:
            max_video_num = video_num
        
        if video_id:
            if success and (not force_regenerate or video_id not in force_regenerate):
                completed_video_ids.add(video_id)
            else:
                failed_video_ids.add(video_id)
    
    total_successful = len(completed_video_ids)
    total_failed = total_existing - total_successful
    
    resume_info = ResumeInfo(
        batch_dir=batch_dir,
        existing_metadata=existing_metadata,
        completed_video_ids=completed_video_ids,
        failed_video_ids=failed_video_ids,
        max_video_num=max_video_num,
        total_existing=total_existing,
        total_successful=total_successful,
        total_failed=total_failed,
        has_dual_gpu_metadata=has_dual_gpu
    )
    
    # Log summary
    logger.info(f"\nBatch State Summary:")
    logger.info(f"  Total videos in metadata: {total_existing}")
    logger.info(f"  Successful: {total_successful}")
    logger.info(f"  Failed: {total_failed}")
    logger.info(f"  Max video_num: {max_video_num}")
    logger.info(f"  Dual GPU metadata: {has_dual_gpu}")
    
    if force_regenerate:
        logger.info(f"  Forced regeneration: {len(force_regenerate)} videos")
    
    logger.info("="*70)
    
    return resume_info


def construct_video_id(prompt_idx: int, bending_idx: int, seed_offset: int) -> str:
    """Construct deterministic video_id from indices."""
    return f"p{prompt_idx:03d}_b{bending_idx:03d}_s{seed_offset:03d}"


def match_variation_by_identity(
    existing_video: Dict[str, Any],
    current_variation
) -> bool:
    """
    Check if existing video matches current variation by operation identity.
    
    CRITICAL: This prevents config reordering bugs where video_id exists
    but refers to a different operation.
    
    Args:
        existing_video: Video metadata from existing batch
        current_variation: BendingVariation to check against (or None for baseline)
    
    Returns:
        True if same operation, False otherwise
    """
    existing_meta = existing_video.get('bending_metadata')
    
    # Both baseline (None)?
    if existing_meta is None and current_variation is None:
        return True
    
    # One baseline, one variation?
    if (existing_meta is None) != (current_variation is None):
        return False
    
    # Both variations - compare identity fields
    current_meta = current_variation.metadata
    
    # Compare all identity-defining fields
    identity_fields = [
        'operation',
        'parameter_name', 
        'parameter_value',
        'token',
        'timestep_start', 
        'timestep_end',
        'layer_start', 
        'layer_end'
    ]
    
    for field in identity_fields:
        if existing_meta.get(field) != current_meta.get(field):
            return False
    
    return True


def detect_config_changes(
    existing_variations: List[Dict[str, Any]],
    current_variations: List
) -> Dict[str, Any]:
    """
    Detect if configuration has changed in an unsafe way.
    
    Safe changes:
    - Adding variations at the END (append-only)
    - No changes (exact match)
    
    Unsafe changes:
    - Reordering variations
    - Adding variations at BEGINNING or MIDDLE
    - Changing variation parameters
    
    Args:
        existing_variations: List of bending_metadata from existing videos
        current_variations: List of BendingVariation objects from current config
    
    Returns:
        Dictionary with change analysis
    """
    # Build unique existing variations (deduplicate by identity)
    existing_set = []
    for existing in existing_variations:
        # Create identity tuple
        if existing is None:
            identity = ('baseline',)
        else:
            identity = (
                existing.get('operation'),
                existing.get('parameter_name'),
                existing.get('parameter_value'),
                existing.get('token'),
                existing.get('timestep_start'),
                existing.get('timestep_end'),
                existing.get('layer_start'),
                existing.get('layer_end')
            )
        if identity not in [e[0] for e in existing_set]:
            existing_set.append((identity, existing))
    
    # Build current set
    current_set = []
    for variation in current_variations:
        if variation is None:
            identity = ('baseline',)
        else:
            meta = variation.metadata
            identity = (
                meta.get('operation'),
                meta.get('parameter_name'),
                meta.get('parameter_value'),
                meta.get('token'),
                meta.get('timestep_start'),
                meta.get('timestep_end'),
                meta.get('layer_start'),
                meta.get('layer_end')
            )
        current_set.append((identity, variation))
    
    # Check for exact match
    if len(existing_set) == len(current_set):
        all_match = True
        for i, (existing_identity, _) in enumerate(existing_set):
            current_identity, _ = current_set[i]
            if existing_identity != current_identity:
                all_match = False
                break
        
        if all_match:
            return {
                'changed': False,
                'safe': True,
                'append_only': False,
                'message': 'Config unchanged'
            }
    
    # Check for append-only (existing is prefix of current)
    if len(current_set) >= len(existing_set):
        prefix_match = True
        for i, (existing_identity, _) in enumerate(existing_set):
            current_identity, _ = current_set[i]
            if existing_identity != current_identity:
                prefix_match = False
                break
        
        if prefix_match:
            added_count = len(current_set) - len(existing_set)
            return {
                'changed': True,
                'safe': True,
                'append_only': True,
                'added_count': added_count,
                'message': f'Config expanded: {added_count} variations added at end'
            }
    
    # Unsafe change detected
    return {
        'changed': True,
        'safe': False,
        'append_only': False,
        'message': 'UNSAFE: Config reordered or variations inserted/removed in middle'
    }


def filter_completed_variations(
    prompt_variations: List,
    bending_configs: List,
    num_seeds: int,
    completed_video_ids: Set[str],
    existing_videos_by_id: Dict[str, Dict],
    bending_index_offset: int = 0,
    strict_mode: bool = True
) -> Tuple[List[Tuple[int, int, int]], int]:
    """
    Filter to only uncompleted (prompt, bending, seed) combinations.
    
    CRITICAL: Validates that video_id collision means same operation identity.
    
    Args:
        prompt_variations: List of prompt variations
        bending_configs: List of bending configs (includes None for baseline)
        num_seeds: Number of seed variations
        completed_video_ids: Set of completed video_ids
        existing_videos_by_id: Dict mapping video_id to video metadata
        bending_index_offset: Offset for bending index (for dual GPU)
        strict_mode: If True, error on identity mismatch. If False, regenerate.
    
    Returns:
        Tuple of (list of (prompt_idx, bending_idx, seed_offset) tuples, count of todo items)
    
    Raises:
        ValueError: If strict_mode=True and config reordering detected
    """
    todo_combinations = []
    collisions = []
    
    for prompt_idx in range(len(prompt_variations)):
        for seed_offset in range(num_seeds):
            for bending_idx, bending_config in enumerate(bending_configs):
                # Calculate global bending index (local + offset)
                global_bending_idx = bending_idx + bending_index_offset
                
                # Construct video_id
                video_id = construct_video_id(prompt_idx, global_bending_idx, seed_offset)
                
                # Check if already completed
                if video_id in completed_video_ids:
                    existing_video = existing_videos_by_id[video_id]
                    
                    # CRITICAL: Validate that operation identity matches
                    if match_variation_by_identity(existing_video, bending_config):
                        # Same operation, safely skip
                        continue
                    else:
                        # COLLISION: Different operation, same video_id
                        collision_info = {
                            'video_id': video_id,
                            'existing_operation': existing_video.get('bending_metadata'),
                            'current_operation': bending_config.metadata if bending_config else None,
                            'prompt_idx': prompt_idx,
                            'bending_idx': bending_idx,
                            'seed_offset': seed_offset
                        }
                        collisions.append(collision_info)
                        
                        if strict_mode:
                            # In strict mode, reject config changes
                            continue  # Will error below
                        else:
                            # In permissive mode, regenerate with new video_num
                            # (This creates duplicate video_id but prevents corruption)
                            todo_combinations.append((prompt_idx, bending_idx, seed_offset))
                else:
                    # Not completed, add to todo list
                    todo_combinations.append((prompt_idx, bending_idx, seed_offset))
    
    # Error if collisions detected in strict mode
    if collisions and strict_mode:
        collision_summary = "\n".join([
            f"  - {c['video_id']}: existing={c['existing_operation']}, current={c['current_operation']}"
            for c in collisions[:5]  # Show first 5
        ])
        if len(collisions) > 5:
            collision_summary += f"\n  ... and {len(collisions) - 5} more"
        
        raise ValueError(
            f"\n{'='*70}\n"
            f"CONFIG REORDERING DETECTED!\n"
            f"{'='*70}\n"
            f"Found {len(collisions)} video_id collisions with different operations.\n"
            f"This means the bending variations list was reordered or modified.\n\n"
            f"Collisions:\n{collision_summary}\n\n"
            f"SOLUTION OPTIONS:\n"
            f"1. Use original config (don't reorder variations)\n"
            f"2. Add new variations at END of list only (append-only)\n"
            f"3. Start a new batch directory for different config\n"
            f"4. Use --force-regenerate to regenerate specific videos\n"
            f"{'='*70}"
        )
    
    return todo_combinations, len(todo_combinations)


def calculate_resume_offsets(resume_info: ResumeInfo) -> Dict[str, int]:
    """
    Calculate offsets for resuming generation.
    
    Args:
        resume_info: ResumeInfo with batch state
    
    Returns:
        Dictionary with offset values
    """
    return {
        'video_number_offset': resume_info.max_video_num,  # Next video starts here
        'bending_index_offset': 0  # No offset needed (filter by video_id instead)
    }


def validate_resume_safety(
    batch_dir: Path,
    resume_info: ResumeInfo,
    config,
    current_bending_configs: List = None
) -> List[str]:
    """
    Validate that resume is safe and won't cause conflicts.
    
    Args:
        batch_dir: Path to batch directory
        resume_info: ResumeInfo with batch state
        config: Generation config
        current_bending_configs: List of current bending configs (for change detection)
    
    Returns:
        List of validation warnings/errors (empty if safe)
    """
    issues = []
    
    # Check that batch directory exists
    if not batch_dir.exists():
        issues.append(f"ERROR: Batch directory does not exist: {batch_dir}")
        return issues  # Fatal, stop checking
    
    # Check that videos directory exists
    videos_dir = batch_dir / "videos"
    if not videos_dir.exists():
        issues.append(f"ERROR: Videos directory does not exist: {videos_dir}")
    
    # Check that config directory exists
    configs_dir = batch_dir / "configs"
    if not configs_dir.exists():
        issues.append(f"ERROR: Configs directory does not exist: {configs_dir}")
    
    # Warn if no existing videos
    if resume_info.total_existing == 0:
        issues.append("WARNING: No existing videos found. This is equivalent to a fresh start.")
    
    # Warn if all videos failed
    if resume_info.total_existing > 0 and resume_info.total_successful == 0:
        issues.append("WARNING: All existing videos failed. Will regenerate all.")
    
    # Check for dual GPU metadata that needs merging
    if resume_info.has_dual_gpu_metadata:
        merged_file = configs_dir / "video_metadata.json"
        if not merged_file.exists():
            issues.append("INFO: Dual GPU metadata detected. Will merge before resuming.")
    
    # CRITICAL: Check for config changes (if current config provided)
    if current_bending_configs is not None:
        # Extract existing variations from metadata
        existing_variations = []
        seen_bending_ids = set()
        
        for video in resume_info.existing_metadata.get('videos', []):
            bending_meta = video.get('bending_metadata')
            # Use video_id to determine order (b000, b001, etc.)
            video_id = video.get('video_id', '')
            if video_id and video.get('success'):
                bending_idx = int(video_id.split('_')[1][1:])  # Extract b### number
                
                if bending_idx not in seen_bending_ids:
                    existing_variations.append((bending_idx, bending_meta))
                    seen_bending_ids.add(bending_idx)
        
        # Sort by bending index to get original order
        existing_variations.sort(key=lambda x: x[0])
        existing_metas = [meta for _, meta in existing_variations]
        
        # Detect config changes
        change_analysis = detect_config_changes(existing_metas, current_bending_configs)
        
        if change_analysis['changed']:
            if change_analysis['safe']:
                issues.append(f"INFO: {change_analysis['message']}")
            else:
                issues.append(f"ERROR: {change_analysis['message']}")
                issues.append("       Resume is NOT SAFE - will cause video_id collisions")
                issues.append("       Use original config or start new batch directory")
    
    return issues


def merge_and_save_metadata(
    batch_dir: Path,
    existing_metadata: Dict[str, Any],
    new_videos: List[Dict[str, Any]],
    batch_name: str = ""
) -> None:
    """
    Merge new video metadata with existing and save.
    
    Args:
        batch_dir: Path to batch directory
        existing_metadata: Existing metadata dictionary
        new_videos: List of new video metadata entries
        batch_name: Optional batch name suffix
    """
    # Merge video lists
    all_videos = existing_metadata.get('videos', []) + new_videos
    
    # Remove duplicates by video_id (keep latest)
    videos_by_id = {}
    for video in all_videos:
        video_id = video.get('video_id')
        if video_id:
            videos_by_id[video_id] = video
    
    # Sort by video_num
    merged_videos = sorted(videos_by_id.values(), key=lambda x: x.get('video_num', 0))
    
    # Update metadata
    merged_metadata = {
        'generation_date': existing_metadata.get('generation_date'),
        'total_videos': len(merged_videos),
        'successful_videos': sum(1 for v in merged_videos if v.get('success', False)),
        'failed_videos': sum(1 for v in merged_videos if not v.get('success', False)),
        'has_bending_variations': any(v.get('bending_metadata') is not None for v in merged_videos),
        'videos': merged_videos
    }
    
    # Save merged metadata
    configs_dir = batch_dir / "configs"
    suffix = f"_{batch_name}" if batch_name else ""
    metadata_file = configs_dir / f"video_metadata{suffix}.json"
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(merged_metadata, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"\n✓ Merged and saved metadata: {metadata_file}")
    logger.info(f"  Total videos: {merged_metadata['total_videos']}")
    logger.info(f"  Successful: {merged_metadata['successful_videos']}")
    logger.info(f"  Failed: {merged_metadata['failed_videos']}")
