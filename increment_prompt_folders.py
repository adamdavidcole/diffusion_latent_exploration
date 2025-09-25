#!/usr/bin/env python3
"""
Script to increment all prompt_XXX folder numbers by one.
This shifts existing prompts to make room for a new prompt_000.

Usage: python increment_prompt_folders.py <experiment_path>
Example: python increment_prompt_folders.py outputs/FinalScene/14b_final_scene_long_20250803_205637
"""

import os
import sys
import re
from pathlib import Path
import shutil

def increment_prompt_folders(experiment_path):
    """Increment all prompt_XXX folders in videos/ and vlm_analysis/ by one."""
    
    experiment_dir = Path(experiment_path)
    
    if not experiment_dir.exists():
        print(f"Error: Experiment directory '{experiment_path}' does not exist")
        return False
    
    # Directories to process
    target_dirs = ['videos', 'vlm_analysis']
    
    for target_dir in target_dirs:
        dir_path = experiment_dir / target_dir
        
        if not dir_path.exists():
            print(f"Warning: Directory '{dir_path}' does not exist, skipping")
            continue
        
        print(f"\nProcessing {dir_path}...")
        
        # Find all prompt_XXX folders
        prompt_folders = []
        for item in dir_path.iterdir():
            if item.is_dir():
                match = re.match(r'prompt_(\d{3})$', item.name)
                if match:
                    prompt_num = int(match.group(1))
                    prompt_folders.append((prompt_num, item))
        
        # Sort by prompt number in descending order (process highest numbers first)
        prompt_folders.sort(key=lambda x: x[0], reverse=True)
        
        if not prompt_folders:
            print(f"  No prompt_XXX folders found in {dir_path}")
            continue
        
        print(f"  Found {len(prompt_folders)} prompt folders")
        
        # Rename folders from highest to lowest number
        for prompt_num, folder_path in prompt_folders:
            new_num = prompt_num + 1
            new_name = f"prompt_{new_num:03d}"
            new_path = folder_path.parent / new_name
            
            print(f"  Renaming {folder_path.name} -> {new_name}")
            
            # Check if destination already exists
            if new_path.exists():
                print(f"    Error: Destination '{new_path}' already exists!")
                return False
            
            # Rename the folder
            folder_path.rename(new_path)
        
        print(f"  Successfully incremented all prompt folders in {dir_path}")
    
    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python increment_prompt_folders.py <experiment_path>")
        print("Example: python increment_prompt_folders.py outputs/FinalScene/14b_final_scene_long_20250803_205637")
        sys.exit(1)
    
    experiment_path = sys.argv[1]
    
    print(f"Incrementing prompt folders in: {experiment_path}")
    print("This will rename:")
    print("  prompt_000 -> prompt_001")
    print("  prompt_001 -> prompt_002")
    print("  prompt_002 -> prompt_003")
    print("  etc.")
    print()
    
    # Ask for confirmation
    response = input("Are you sure you want to proceed? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Operation cancelled.")
        sys.exit(0)
    
    success = increment_prompt_folders(experiment_path)
    
    if success:
        print("\n✅ Successfully incremented all prompt folders!")
        print("You can now add new content to prompt_000 folders.")
    else:
        print("\n❌ Operation failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()