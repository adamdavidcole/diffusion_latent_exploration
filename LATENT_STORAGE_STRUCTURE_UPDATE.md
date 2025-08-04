# Latent Storage Structure Update

## Summary

Updated the latent storage system to organize latents in a hierarchical structure that matches the video output organization.

## Changes Made

### Directory Structure

**Before:**
```
outputs/batch_name/
├── latents/
│   ├── latent_prompt_000_vid001_step_000.npy.gz
│   ├── latent_prompt_000_vid001_step_001.npy.gz
│   └── ...
└── metadata/
    ├── latent_prompt_000_vid001_step_000_metadata.json
    └── ...
```

**After:**
```
outputs/batch_name/
├── latents_data/
│   └── prompt_000/
│       ├── step_000.npy.gz
│       ├── step_000_metadata.json
│       ├── step_001.npy.gz
│       ├── step_001_metadata.json
│       └── ...
└── videos/
    └── prompt_000/
        ├── prompt_000_001.mp4
        ├── prompt_000_002.mp4
        └── ...
```

### Key Benefits

1. **Consistent Organization**: Latents and videos now use the same hierarchical structure
2. **Cleaner Filenames**: Simplified from `latent_prompt_000_vid001_step_000.npy.gz` to `step_000.npy.gz`
3. **Better Grouping**: All latents from the same prompt are grouped together
4. **Easier Navigation**: Directory structure mirrors video organization

### Code Changes

#### LatentStorage Class (`src/utils/latent_storage.py`)
- Changed base directory from `latents/` to `latents_data/`
- Added prompt-based subdirectory creation
- Updated file naming scheme
- Modified load methods to work with new structure
- Added helper methods for video ID handling

#### Analysis Tools (`src/analysis/latent_trajectory_analysis.py`)
- Updated trajectory loading to work with new directory structure
- Added prompt-level analysis capabilities
- Enhanced video discovery methods

#### Standalone Script (`analyze_latent_trajectories.py`)
- Added support for both old and new directory structures
- Enhanced video listing to show prompt organization
- Improved error messages and user guidance

### Backward Compatibility

The analysis tools can handle both old (`latents/`) and new (`latents_data/`) directory structures:
- The analysis script automatically detects which structure is present
- Existing batches with old structure will continue to work
- New generations will use the improved structure

### Usage Examples

**Analyze all latents from a prompt:**
```bash
python analyze_latent_trajectories.py --batch-dir outputs/my_batch --video-id prompt_000
```

**List available prompt directories:**
```bash
python analyze_latent_trajectories.py --batch-dir outputs/my_batch --list-videos
```

**Analyze specific video (if video summaries are available):**
```bash
python analyze_latent_trajectories.py --batch-dir outputs/my_batch --video-id prompt_000_vid001
```

## Migration

No manual migration is needed:
- New generations will automatically use the new structure
- Old generations continue to work with existing analysis tools
- The system automatically detects and handles both structures
