# Attention Visual```yaml
attention_analysis:
  store_attention: true
  store_aggregated_attention: true  # NEW: Store step-averaged attention maps
  auto_generate_videos: true        # NEW: Automatically generate videos after batch completion
  auto_generate_per_video: true     # NEW: Generate videos immediately after each video completes
  
  visualization_params:
    figsize: [12, 8]
    colormap: "hot"
    overlay_alpha: 0.6
    static_duration: 3.0
    fps: 10
    interpolation_steps: 2
```

This directory contains a comprehensive suite of tools for visualizing attention maps captured during WAN 1.3B video generation.

## Overview

The attention visualization system provides three main components:

1. **AttentionAnalyzer** - Enhanced attention map loading and analysis with metadata support
2. **AttentionVisualizer** - Comprehensive visualization library with video generation capabilities  
3. **CLI Script** - Command-line interface for generating attention videos

## Quick Start

### NEW: Per-Video Visualization (Recommended for Long Batches)

For long-running batches that you might need to stop early, enable per-video visualization:

```yaml
attention_analysis:
  store_attention: true
  auto_generate_per_video: true     # Generate videos immediately after each video completes
  
  visualization_params:
    figsize: [12, 8]
    colormap: "hot"
    overlay_alpha: 0.6
```

This creates attention videos in a subdirectory alongside each generated video:
```
outputs/batch_xyz/videos/
├── prompt_001/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── attention_videos/              # NEW: Per-video attention videos
│       ├── prompt_001_vid001_kiss_attention.mp4
│       ├── prompt_001_vid001_kiss_overlay.mp4  # With original video overlay
│       ├── prompt_001_vid002_kiss_attention.mp4
│       └── ...
```

**Benefits:**
- ✅ **Immediate Results**: Get attention videos as soon as each video completes
- ✅ **Early Stopping**: Stop batches early and still have all completed videos visualized
- ✅ **Progress Monitoring**: See attention patterns evolving as the batch progresses
- ✅ **Organized Output**: Videos grouped with their source video for easy comparison

### Batch-End Visualization

First, configure your generation config to store attention maps:

```yaml
attention_analysis:
  store_attention: true
  store_aggregated_attention: true  # NEW: Store step-averaged attention maps
  auto_generate_videos: true        # NEW: Automatically generate videos after generation
  
  visualization_params:
    figsize: [12, 8]
    colormap: "hot"
    overlay_alpha: 0.6
    static_duration: 3.0
    fps: 10
    interpolation_steps: 2
```

### 2. Generate Videos with Attention Tracking

Use prompts with parenthetical tokens to mark attention targets:

```python
prompts = [
    "A romantic (kiss) between a man and woman with (flowers)",
    "A heroic (warrior) fighting a fierce (dragon)"
]
```

### 3. Auto-Generated Attention Videos

When `auto_generate_videos: true` is enabled, attention videos are automatically created in the `attention_videos` directory after batch completion.

### 4. Manual Video Generation

Use the CLI script for custom video generation:

```bash
# Generate attention video for a specific token
python scripts/generate_attention_video.py \
    --attention-dir outputs/batch_xyz/attention_maps \
    --video-id "prompt_001_vid_001" \
    --token "kiss" \
    --output attention_videos/kiss_analysis.mp4

# Overlay attention on original video
python scripts/generate_attention_video.py \
    --attention-dir outputs/batch_xyz/attention_maps \
    --video-id "prompt_001_vid_001" \
    --token "kiss" \
    --original-video outputs/batch_xyz/videos/prompt_001_vid_001.mp4 \
    --output attention_videos/kiss_overlay.mp4 \
    --overlay-alpha 0.6

# Generate using aggregated attention (static)
python scripts/generate_attention_video.py \
    --attention-dir outputs/batch_xyz/attention_maps \
    --video-id "prompt_001_vid_001" \
    --token "kiss" \
    --use-aggregated \
    --output attention_videos/kiss_aggregated.mp4
```

## Components

### AttentionAnalyzer

Enhanced attention map loader with metadata support:

```python
from src.visualization.attention_analyzer import AttentionAnalyzer

analyzer = AttentionAnalyzer("outputs/batch_xyz/attention_maps")

# Discover available videos and tokens
videos = analyzer.get_available_videos()
tokens = analyzer.get_available_tokens("prompt_001_vid_001")

# Load attention maps with metadata
attention_info = analyzer.load_step("prompt_001_vid_001", "kiss", 5)
print(f"Attention shape: {attention_info.attention_map.shape}")
print(f"Video dimensions: {attention_info.metadata.video_dimensions}")

# Load all steps for temporal analysis
all_steps = analyzer.load_all_steps("prompt_001_vid_001", "kiss")
```

### AttentionVisualizer

Comprehensive visualization with video generation:

```python
from src.visualization.attention_visualizer import AttentionVisualizer

visualizer = AttentionVisualizer(
    analyzer=analyzer,
    figsize=[12, 8],
    colormap="hot",
    fps=10
)

# Generate attention video
video_path = visualizer.generate_attention_video(
    video_id="prompt_001_vid_001",
    token_word="kiss",
    output_path="kiss_attention.mp4"
)

# Create overlay with original video
overlay_path = visualizer.generate_attention_video(
    video_id="prompt_001_vid_001", 
    token_word="kiss",
    output_path="kiss_overlay.mp4",
    original_video_path="original_video.mp4",
    overlay_alpha=0.6
)

# Generate temporal comparison
comparison = visualizer.generate_temporal_comparison(
    video_id="prompt_001_vid_001",
    tokens=["kiss", "flowers"],
    save_path="temporal_comparison.png"
)
```

## Storage Format

### Per-Step Storage

Individual diffusion steps are stored as:
```
attention_maps/
├── prompt_001_vid_001/
│   ├── kiss/
│   │   ├── step_000.npz        # Attention map for step 0
│   │   ├── step_001.npz        # Attention map for step 1
│   │   ├── ...
│   │   └── step_000_metadata.json  # Metadata for step 0
│   └── flowers/
│       ├── step_000.npz
│       └── ...
```

### Aggregated Storage (NEW)

Step-averaged attention maps are stored as:
```
attention_maps/
├── prompt_001_vid_001/
│   ├── aggregated/
│   │   ├── kiss_aggregated.npz         # Average across all steps
│   │   ├── kiss_aggregated_metadata.json
│   │   ├── flowers_aggregated.npz
│   │   └── flowers_aggregated_metadata.json
```

### Metadata Format

Each attention map includes rich metadata:

```json
{
  "shape": [320, 576],
  "dtype": "float32", 
  "num_steps": 20,
  "steps_used": [0, 1, 2, ..., 19],
  "aggregation_method": "mean",
  "token_word": "kiss",
  "video_dimensions": {
    "width": 576,
    "height": 320,
    "fps": 25,
    "frames": 65
  },
  "storage_timestamp": "2024-01-15T10:30:45",
  "attention_storage_version": "1.1.0"
}
```

## CLI Options

The CLI script provides extensive customization:

```bash
python scripts/generate_attention_video.py --help

Options:
  --attention-dir PATH     Directory containing attention maps [required]
  --video-id TEXT         Video identifier [required]  
  --token TEXT            Token word to visualize [required]
  --output PATH           Output video path [required]
  
  --original-video PATH   Original video for overlay
  --overlay-alpha FLOAT   Transparency for overlay (0.0-1.0) [default: 0.6]
  
  --use-aggregated        Use aggregated attention instead of per-step
  --colormap TEXT         Matplotlib colormap [default: hot]
  --fps INTEGER          Output video FPS [default: 10]
  --figsize TEXT         Figure size as "width,height" [default: 12,8]
  
  --interpolation-steps INT  Smoothing between steps [default: 2]
  --include-timestamp     Add timestamp to frames
  --video-format TEXT     Output format (mp4/avi) [default: mp4]
```

## Configuration Options

### Basic Attention Storage

```yaml
attention_analysis:
  store_attention: true
  storage_format: "numpy"      # numpy, torch
  compress_attention: true     # Use .npz compression
  storage_dtype: "float32"     # Storage precision
  
  # Aggregation during capture
  store_per_head: false        # Average across attention heads  
  store_per_block: false       # Average across transformer blocks
```

### NEW: Per-Video vs Batch-End Visualization

**Per-Video Visualization (Recommended for Long Batches):**
```yaml
attention_analysis:
  auto_generate_per_video: true   # Generate immediately after each video
```
- ✅ Immediate results - see attention videos as soon as each video completes
- ✅ Early stopping friendly - stop batches anytime and keep completed visualizations  
- ✅ Progress monitoring - watch attention patterns evolve during batch
- ✅ Memory efficient - doesn't accumulate all videos before processing

**Batch-End Visualization (Good for Short Batches):**
```yaml  
attention_analysis:
  auto_generate_videos: true      # Generate all videos after batch completion
```
- ✅ Centralized output - all videos in one directory
- ✅ Complete overview - process entire batch at once

### NEW: Aggregated Storage

```yaml
attention_analysis:
  # Generate step-averaged attention maps
  store_aggregated_attention: true
  aggregated_storage_format: "numpy"  # numpy, torch
```

### NEW: Auto-Visualization  

```yaml
attention_analysis:
  # Automatically generate videos after batch completion
  auto_generate_videos: true
  
  visualization_params:
    figsize: [12, 8]           # Figure dimensions
    colormap: "hot"            # Matplotlib colormap
    overlay_alpha: 0.6         # Overlay transparency
    static_duration: 3.0       # Duration for aggregated videos
    fps: 10                    # Video framerate
    interpolation_steps: 2     # Smoothing between steps
    include_colorbar: true     # Show attention scale
    video_format: "mp4"        # Output format
```

## Advanced Usage

### Custom Visualization Pipeline

```python
# Create custom analyzer and visualizer
analyzer = AttentionAnalyzer("attention_maps")
visualizer = AttentionVisualizer(
    analyzer=analyzer,
    figsize=[16, 10],
    colormap="viridis", 
    fps=15
)

# Batch process all tokens for a video
video_id = "prompt_001_vid_001"
tokens = analyzer.get_available_tokens(video_id)

for token in tokens:
    output_path = f"attention_videos/{video_id}_{token}.mp4"
    visualizer.generate_attention_video(
        video_id=video_id,
        token_word=token,
        output_path=output_path,
        interpolation_steps=3,
        include_timestamp=True
    )
```

### Temporal Analysis

```python
# Compare attention evolution across tokens
comparison_fig = visualizer.generate_temporal_comparison(
    video_id="prompt_001_vid_001",
    tokens=["kiss", "flowers", "garden"],
    save_path="temporal_analysis.png"
)

# Generate aggregated attention for analysis
aggregated = visualizer.generate_aggregated_attention_map(
    video_id="prompt_001_vid_001", 
    token_word="kiss",
    aggregate_steps=True
)
```

## Troubleshooting

### Common Issues

1. **No attention data found**
   - Ensure `store_attention: true` in config
   - Use parenthetical tokens like `(kiss)` in prompts
   - Check that attention_maps directory exists

2. **Video generation fails**
   - Verify ffmpeg/cv2 installation: `pip install opencv-python`
   - Check output directory permissions
   - Ensure attention files aren't corrupted

3. **Auto-visualization not working**
   - Enable with `auto_generate_videos: true` 
   - Check visualization module imports
   - Verify attention storage completed successfully

### Dependencies

```bash
pip install matplotlib opencv-python imageio torch numpy transformers
```

### Performance Tips

- Use `store_aggregated_attention: true` for faster analysis
- Set `interpolation_steps: 0` to disable smoothing
- Use compressed storage: `compress_attention: true`
- Lower resolution with `spatial_downsample_factor: 2`

## Examples

See `configs/attention_visualization_demo.yaml` for a complete example configuration demonstrating all features.
