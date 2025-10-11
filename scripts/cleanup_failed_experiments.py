#!/usr/bin/env python3
"""
Cleanup Failed Experiments Script

This script scans the outputs directory and removes experiment directories
that don't contain any successfully generated video files, indicating that
the generation process failed or was interrupted before completion.

The script handles both flat directory structures and nested hierarchical
folder structures (e.g., outputs/Category/Experiment).

Usage:
    python cleanup_failed_experiments.py [--dry-run] [--verbose]
    
Options:
    --dry-run    Show what would be deleted without actually deleting
    --verbose    Show detailed information about each directory
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple, Optional


def find_video_files(experiment_dir: Path) -> List[Path]:
    """
    Find all video files in an experiment directory.
    
    Args:
        experiment_dir: Path to the experiment directory
        
    Returns:
        List of video file paths found
    """
    video_files = []
    videos_dir = experiment_dir / 'videos'
    
    if not videos_dir.exists():
        return video_files
    
    # Look for video files in prompt_* subdirectories
    for prompt_dir in videos_dir.iterdir():
        if prompt_dir.is_dir() and prompt_dir.name.startswith('prompt_'):
            # Find video_*.mp4 files
            for video_file in prompt_dir.glob('video_*.mp4'):
                if video_file.is_file() and video_file.stat().st_size > 0:
                    video_files.append(video_file)
    
    return video_files


def get_directory_size(directory: Path) -> int:
    """Get the total size of a directory in bytes."""
    total_size = 0
    try:
        for item in directory.rglob('*'):
            if item.is_file():
                total_size += item.stat().st_size
    except (OSError, PermissionError):
        pass
    return total_size


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def scan_experiments(outputs_dir: Path, verbose: bool = False) -> Tuple[List[Path], List[Path]]:
    """
    Scan experiment directories and categorize them as successful or failed.
    Handles both flat structure and nested folder structure.
    
    Args:
        outputs_dir: Path to the outputs directory
        verbose: Whether to print detailed information
        
    Returns:
        Tuple of (successful_experiments, failed_experiments)
    """
    if not outputs_dir.exists():
        print(f"❌ Outputs directory not found: {outputs_dir}")
        return [], []
    
    successful_experiments = []
    failed_experiments = []
    
    print(f"🔍 Scanning experiments in: {outputs_dir}")
    print("=" * 60)
    
    def scan_directory(directory: Path, level: int = 0):
        """Recursively scan directories for experiments."""
        indent = "  " * level
        
        for item in directory.iterdir():
            if not item.is_dir():
                continue
                
            # Skip special directories
            if item.name.startswith('.') or item.name in ['successful_experiments']:
                if verbose:
                    print(f"{indent}⏭️  Skipping special directory: {item.name}")
                continue
            
            # Check if this directory contains video files (is an experiment)
            video_files = find_video_files(item)
            
            if video_files:
                # This is a successful experiment
                successful_experiments.append(item)
                if verbose:
                    dir_size = get_directory_size(item)
                    print(f"{indent}✅ {item.relative_to(outputs_dir)}")
                    print(f"{indent}   📹 Videos: {len(video_files)} files")
                    print(f"{indent}   💾 Size: {format_size(dir_size)}")
                    print()
            else:
                # Check if this directory contains subdirectories that might be experiments
                subdirs = [sub for sub in item.iterdir() if sub.is_dir() and not sub.name.startswith('.')]
                
                if subdirs:
                    # This is a folder containing other experiments/folders
                    if verbose:
                        print(f"{indent}📁 Folder: {item.relative_to(outputs_dir)}")
                    
                    # Recursively scan subdirectories
                    scan_directory(item, level + 1)
                else:
                    # This is a leaf directory with no videos and no subdirectories - failed experiment
                    failed_experiments.append(item)
                    if verbose:
                        dir_size = get_directory_size(item)
                        print(f"{indent}❌ {item.relative_to(outputs_dir)}")
                        print(f"{indent}   📹 Videos: 0 files")
                        print(f"{indent}   💾 Size: {format_size(dir_size)}")
                        print()
    
    scan_directory(outputs_dir)
    
    # Consolidate failed experiments: if a parent directory only contains failed experiments,
    # replace all child entries with just the parent
    failed_experiments = consolidate_failed_experiments(failed_experiments, outputs_dir, verbose)
    
    return successful_experiments, failed_experiments


def consolidate_failed_experiments(failed_experiments: List[Path], outputs_dir: Path, verbose: bool = False) -> List[Path]:
    """
    Consolidate failed experiments by replacing multiple failed child directories 
    with their parent directory when all children are failed.
    
    Args:
        failed_experiments: List of failed experiment paths
        outputs_dir: Base outputs directory
        verbose: Whether to print consolidation information
        
    Returns:
        Consolidated list of failed experiment paths
    """
    if not failed_experiments:
        return failed_experiments
    
    # Group failed experiments by their parent directory
    parent_to_children = {}
    for exp_dir in failed_experiments:
        parent = exp_dir.parent
        if parent not in parent_to_children:
            parent_to_children[parent] = []
        parent_to_children[parent].append(exp_dir)
    
    # Check each parent directory
    consolidated = set(failed_experiments)
    
    for parent, children in parent_to_children.items():
        # Skip if parent is the outputs directory itself
        if parent == outputs_dir:
            continue
            
        # Get all subdirectories of this parent
        try:
            all_subdirs = [sub for sub in parent.iterdir() if sub.is_dir() and not sub.name.startswith('.')]
        except (OSError, PermissionError):
            continue
        
        # If all subdirectories are failed experiments, consolidate to parent
        if len(all_subdirs) > 0 and len(children) == len(all_subdirs):
            # Remove all children from the consolidated set
            for child in children:
                consolidated.discard(child)
            # Add the parent instead
            consolidated.add(parent)
            
            if verbose:
                print(f"🔄 Consolidated {len(children)} failed experiments under: {parent.relative_to(outputs_dir)}")
    
    # Recursively consolidate in case we have multi-level consolidation opportunities
    consolidated_list = sorted(list(consolidated))
    
    # Check if we made any changes, if so, recurse to consolidate further up the tree
    if len(consolidated_list) < len(failed_experiments):
        return consolidate_failed_experiments(consolidated_list, outputs_dir, verbose)
    
    return consolidated_list


def cleanup_failed_experiments(failed_experiments: List[Path], dry_run: bool = True, outputs_dir: Optional[Path] = None) -> None:
    """
    Remove failed experiment directories.
    
    Args:
        failed_experiments: List of failed experiment directory paths
        dry_run: If True, only show what would be deleted without actually deleting
        outputs_dir: Base outputs directory for relative path calculation
    """
    if not failed_experiments:
        print("🎉 No failed experiments found! All directories contain videos.")
        return
    
    total_size = sum(get_directory_size(exp_dir) for exp_dir in failed_experiments)
    
    print(f"\n{'DRY RUN - ' if dry_run else ''}CLEANUP SUMMARY")
    print("=" * 60)
    print(f"Failed experiments to {'remove' if not dry_run else 'be removed'}: {len(failed_experiments)}")
    print(f"Total space to {'free' if not dry_run else 'be freed'}: {format_size(total_size)}")
    print()
    
    for exp_dir in failed_experiments:
        dir_size = get_directory_size(exp_dir)
        relative_path = exp_dir.relative_to(outputs_dir) if outputs_dir else exp_dir.name
        print(f"{'🗑️' if not dry_run else '👀'} {relative_path} ({format_size(dir_size)})")
        
        if not dry_run:
            try:
                shutil.rmtree(exp_dir)
                print(f"   ✅ Deleted successfully")
            except Exception as e:
                print(f"   ❌ Error deleting: {e}")
        else:
            print(f"   (would delete {format_size(dir_size)})")
    
    if dry_run:
        print(f"\n💡 Run without --dry-run to actually delete these {len(failed_experiments)} directories")
    else:
        print(f"\n✅ Cleanup complete! Removed {len(failed_experiments)} failed experiment directories")
        print(f"💾 Freed {format_size(total_size)} of disk space")


def main():
    parser = argparse.ArgumentParser(
        description="Clean up failed experiment directories that don't contain videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python cleanup_failed_experiments.py --dry-run --verbose
    python cleanup_failed_experiments.py --verbose
    python cleanup_failed_experiments.py
        """
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed information about each directory'
    )
    
    parser.add_argument(
        '--outputs-dir',
        type=Path,
        default=Path(__file__).parent.parent / 'outputs',
        help='Path to the outputs directory (default: ../outputs)'
    )
    
    args = parser.parse_args()
    
    print("🧹 WAN Video Generation - Cleanup Failed Experiments")
    print("=" * 60)
    
    # Scan experiments
    successful, failed = scan_experiments(args.outputs_dir, args.verbose)
    
    print(f"\n📊 RESULTS SUMMARY")
    print("=" * 60)
    print(f"✅ Successful experiments: {len(successful)}")
    print(f"❌ Failed experiments: {len(failed)}")
    
    if failed:
        print(f"\nFailed experiments (no videos found):")
        for exp_dir in failed:
            size = get_directory_size(exp_dir)
            # Show relative path from outputs directory for better readability
            relative_path = exp_dir.relative_to(args.outputs_dir)
            print(f"  • {relative_path} ({format_size(size)})")
    
    # Cleanup failed experiments
    if failed:
        if not args.dry_run:
            response = input(f"\n⚠️  Are you sure you want to delete {len(failed)} failed experiment directories? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("❌ Cleanup cancelled by user")
                return
        
        cleanup_failed_experiments(failed, args.dry_run, args.outputs_dir)
    
    print("\n🎯 Cleanup script finished!")


if __name__ == '__main__':
    main()
