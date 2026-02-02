#!/usr/bin/env python3
"""
Merge metadata files from dual GPU parallel execution into unified files.

This script merges:
1. bending_variations_*_gpu*.json -> bending_variations.json
2. video_metadata_*_gpu*.json -> video_metadata.json

Usage:
    python scripts/merge_dual_gpu_metadata.py <output_directory>
    
Example:
    python scripts/merge_dual_gpu_metadata.py outputs/dual_gpu_comprehensive_20260128_125120
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def find_metadata_files(configs_dir: Path, pattern: str) -> List[Path]:
    """Find all metadata files matching pattern (e.g., 'video_metadata*.json')."""
    return sorted(configs_dir.glob(pattern))


def merge_bending_variations(configs_dir: Path) -> Dict[str, Any]:
    """Merge bending variation files from multiple GPUs."""
    files = find_metadata_files(configs_dir, "bending_variations*.json")
    
    if not files:
        print("No bending_variations files found to merge")
        return None
    
    print(f"\nMerging {len(files)} bending variation files:")
    for f in files:
        print(f"  - {f.name}")
    
    all_variations = []
    seen_ids = set()
    
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            
        # Handle both list and dict formats
        variations = data if isinstance(data, list) else data.get('variations', [])
        
        for var in variations:
            var_id = var.get('variation_id')
            if var_id and var_id not in seen_ids:
                all_variations.append(var)
                seen_ids.add(var_id)
    
    # Sort by variation_id for consistency
    all_variations.sort(key=lambda x: x.get('variation_id', ''))
    
    print(f"Merged {len(all_variations)} unique bending variations")
    return all_variations


def merge_video_metadata(configs_dir: Path) -> Dict[str, Any]:
    """Merge video metadata files from multiple GPUs."""
    files = find_metadata_files(configs_dir, "video_metadata*.json")
    
    if not files:
        print("No video_metadata files found to merge")
        return None
    
    print(f"\nMerging {len(files)} video metadata files:")
    for f in files:
        print(f"  - {f.name}")
    
    merged = {
        'generation_date': None,
        'total_videos': 0,
        'successful_videos': 0,
        'failed_videos': 0,
        'has_bending_variations': False,
        'videos': []
    }
    
    seen_video_ids = set()
    
    for file in files:
        with open(file, 'r') as f:
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
    
    print(f"Merged {len(merged['videos'])} unique videos")
    print(f"  Total: {merged['total_videos']}, Successful: {merged['successful_videos']}, Failed: {merged['failed_videos']}")
    
    return merged


def merge_metadata(output_dir: Path, verbose: bool = True) -> bool:
    """
    Merge metadata files from dual GPU execution.
    
    Args:
        output_dir: Path to the output directory containing configs folder
        verbose: Whether to print detailed progress messages
    
    Returns:
        True if merge was performed, False if no files to merge
    """
    configs_dir = output_dir / "configs"
    
    if not configs_dir.exists():
        if verbose:
            logger.warning(f"Configs directory not found: {configs_dir}")
        return False
    
    # Check if there are multiple GPU metadata files to merge
    gpu_metadata_files = list(configs_dir.glob("video_metadata_*gpu*.json"))
    if len(gpu_metadata_files) < 2:
        # Not a dual GPU run or already merged
        return False
    
    if verbose:
        logger.info("="*70)
        logger.info("MERGING DUAL GPU METADATA")
        logger.info("="*70)
    
    merged_any = False
    
    # Merge bending variations
    bending_variations = merge_bending_variations(configs_dir)
    if bending_variations:
        output_file = configs_dir / "bending_variations.json"
        with open(output_file, 'w') as f:
            json.dump(bending_variations, f, indent=2, ensure_ascii=False, default=str)
        if verbose:
            logger.info(f"✓ Merged bending variations: {output_file.name}")
        merged_any = True
    
    # Merge video metadata
    video_metadata = merge_video_metadata(configs_dir)
    if video_metadata:
        output_file = configs_dir / "video_metadata.json"
        with open(output_file, 'w') as f:
            json.dump(video_metadata, f, indent=2, ensure_ascii=False, default=str)
        if verbose:
            logger.info(f"✓ Merged video metadata: {output_file.name} ({len(video_metadata['videos'])} videos)")
        merged_any = True
    
    if verbose and merged_any:
        logger.info("="*70)
    
    return merged_any


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/merge_dual_gpu_metadata.py <output_directory>")
        print("\nExample:")
        print("  python scripts/merge_dual_gpu_metadata.py outputs/dual_gpu_comprehensive_20260128_125120")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    configs_dir = output_dir / "configs"
    
    if not configs_dir.exists():
        print(f"Error: Configs directory not found: {configs_dir}")
        sys.exit(1)
    
    print("="*70)
    print("DUAL GPU METADATA MERGER")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print(f"Configs directory: {configs_dir}")
    
    # Merge bending variations
    bending_variations = merge_bending_variations(configs_dir)
    if bending_variations:
        output_file = configs_dir / "bending_variations.json"
        with open(output_file, 'w') as f:
            json.dump(bending_variations, f, indent=2, ensure_ascii=False, default=str)
        print(f"✓ Written: {output_file}")
    
    # Merge video metadata
    video_metadata = merge_video_metadata(configs_dir)
    if video_metadata:
        output_file = configs_dir / "video_metadata.json"
        with open(output_file, 'w') as f:
            json.dump(video_metadata, f, indent=2, ensure_ascii=False, default=str)
        print(f"✓ Written: {output_file}")
    
    print("\n" + "="*70)
    print("MERGE COMPLETE")
    print("="*70)
    print("\nYou can now analyze this batch in the webapp:")
    print(f"  http://localhost:3000?experiment={output_dir.name}")
    print()


if __name__ == "__main__":
    main()
