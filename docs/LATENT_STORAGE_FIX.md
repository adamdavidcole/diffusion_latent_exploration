# Latent Storage Structure Fix: Individual Video Directories

## Issue Fixed

**Problem**: Multiple videos from the same prompt were overwriting each other's latent data because they were all stored in the same directory.

**Before (Broken)**:
```
latents_data/
└── prompt_000/
    ├── step_000.npy.gz  ← All videos overwrite here
    ├── step_001.npy.gz
    └── ...
```

**After (Fixed)**:
```
latents_data/
└── prompt_000/
    ├── vid_001/
    │   ├── step_000.npy.gz
    │   ├── step_001.npy.gz
    │   └── ...
    ├── vid_002/
    │   ├── step_000.npy.gz
    │   ├── step_001.npy.gz
    │   └── ...
    └── vid_003/
        ├── step_000.npy.gz
        ├── step_001.npy.gz
        └── ...
```

## Changes Made

### 1. **LatentStorage Class Updates**

#### `start_video_storage()`:
- Now creates individual video directories: `prompt_000/vid_001/`, `prompt_000/vid_002/`, etc.
- Converts video ID format: `prompt_000_vid001` → `prompt_000/vid_001/` directory

#### `load_latent()` & `load_metadata()`:
- Updated to look in the correct video-specific subdirectory
- Handles both old and new directory structures for backward compatibility

#### `list_stored_videos()`:
- Now scans the nested directory structure
- Returns actual video IDs like `prompt_000_vid001`, `prompt_000_vid002`, etc.
- No longer groups videos by prompt only

#### `list_steps_for_video()`:
- Updated to search in the specific video directory
- Works with the new nested structure

### 2. **Analysis Framework Updates**

#### `LatentTrajectoryAnalyzer`:
- `get_available_videos()`: Returns individual video IDs instead of just prompt directories
- `get_available_prompt_dirs()`: New method to get prompt-level groupings
- `discover_videos_in_prompt()`: Updated to work with the new structure
- `load_video_trajectory()`: Simplified to work with specific video IDs

### 3. **Directory Structure Benefits**

✅ **No More Overwrites**: Each video has its own isolated directory  
✅ **Clear Organization**: Matches video output structure exactly  
✅ **Individual Analysis**: Can analyze specific videos without interference  
✅ **Batch Analysis**: Can still analyze all videos from a prompt together  
✅ **Scalable**: Supports unlimited videos per prompt  

## Usage Examples

### **List All Available Videos**:
```bash
python analyze_latent_trajectories.py --batch-dir outputs/my_batch --list-videos
```
Output:
```
Found videos:
  prompt_000_vid001: 20 steps stored
  prompt_000_vid002: 20 steps stored
  prompt_001_vid001: 20 steps stored
  ...
```

### **Analyze Specific Video**:
```bash
python analyze_latent_trajectories.py --batch-dir outputs/my_batch --video-id prompt_000_vid001
```

### **Compare Videos from Same Prompt**:
```bash
python analyze_latent_trajectories.py --batch-dir outputs/my_batch --compare-videos prompt_000_vid001 prompt_000_vid002
```

## Backward Compatibility

The analysis tools maintain backward compatibility:
- Old storage format (flat structure) still works
- New storage format provides proper isolation
- Analysis scripts automatically detect the structure type

## Impact

- **Storage Integrity**: No more data loss from overwrites
- **Analysis Accuracy**: Each video's trajectory is preserved independently  
- **Research Quality**: Can now properly study variation within and between prompts
- **Data Organization**: Clear, hierarchical structure matching video outputs

This fix ensures that your latent trajectory analysis research data is properly preserved and organized for meaningful analysis of representation bias patterns.
