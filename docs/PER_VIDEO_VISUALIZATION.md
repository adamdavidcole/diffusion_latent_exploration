# Per-Video Attention Visualization

## Overview

The attention visualization system now supports **per-video generation** - creating attention videos immediately after each individual video completes, rather than waiting for the entire batch to finish.

## Why Per-Video Visualization?

### 🎯 **Problem Solved**
- **Long-running batches**: Video generation can take hours or days
- **Early stopping**: Often need to stop batches before completion due to time constraints
- **Lost work**: Previously, stopping early meant losing all attention visualizations
- **Delayed feedback**: Had to wait until entire batch finished to see any results

### ✅ **Benefits**
- **Immediate Results**: See attention videos as soon as each video completes
- **Early Stopping Friendly**: Stop batches anytime and keep all completed visualizations
- **Progress Monitoring**: Watch attention patterns evolve during the batch
- **Memory Efficient**: Process videos individually rather than accumulating
- **Organized Output**: Videos grouped with their source video for easy comparison

## Configuration

### Enable Per-Video Visualization

```yaml
attention_analysis:
  store_attention: true
  auto_generate_per_video: true     # NEW: Generate immediately after each video
  
  visualization_params:
    figsize: [12, 8]
    colormap: "hot"
    overlay_alpha: 0.6
    static_duration: 3.0
    fps: 10
```

### Compare with Batch-End Visualization

```yaml
attention_analysis:
  auto_generate_videos: true        # Generate all videos after batch completion (old way)
  auto_generate_per_video: true     # Generate videos immediately after each video (new way)
```

You can enable both if desired - per-video for immediate feedback, batch-end for centralized collection.

## Output Structure

### Per-Video Output (NEW)

```
outputs/batch_xyz/videos/
├── prompt_001/
│   ├── video_001.mp4                           # Original generated video
│   ├── attention_videos/                       # NEW: Per-video attention directory
│   │   ├── prompt_001_vid001_kiss_attention.mp4    # Attention-only video
│   │   ├── prompt_001_vid001_kiss_overlay.mp4      # Attention overlaid on original
│   │   ├── prompt_001_vid001_flowers_attention.mp4
│   │   └── prompt_001_vid001_flowers_overlay.mp4
│   ├── video_002.mp4
│   ├── attention_videos/
│   │   ├── prompt_001_vid002_kiss_attention.mp4
│   │   └── ...
│   └── prompt.txt
├── prompt_002/
│   └── ...
```

### Batch-End Output (Original)

```
outputs/batch_xyz/
├── videos/
│   ├── prompt_001/
│   │   ├── video_001.mp4
│   │   └── video_002.mp4
│   └── prompt_002/
└── attention_videos/                           # Centralized attention videos
    ├── prompt_001_vid001_kiss_attention.mp4
    ├── prompt_001_vid002_kiss_attention.mp4
    └── ...
```

## Implementation Details

### When It Triggers

Per-video visualization triggers immediately after:
1. ✅ Video generation completes successfully
2. ✅ Attention maps are stored for the video
3. ✅ At least one attention token is tracked

### What It Generates

For each tracked token in each video:
1. **Attention-only video**: Pure attention map visualization
2. **Overlay video**: Attention maps overlaid on the original video (if available)
3. **Static aggregated video**: If aggregated attention storage is enabled

### Performance Characteristics

- **Speed**: Generates videos as fast as the visualization library (typically 10-30 seconds per token)
- **Memory**: Processes one video at a time, minimal memory overhead
- **Storage**: Videos stored alongside source video for easy organization

## Usage Examples

### For Long-Running Batches

```bash
# Start a long batch with per-video visualization
python main.py \
  --config configs/attention_visualization_demo.yaml \
  --template "a romantic (kiss) between [two men|two women|a man and woman]" \
  --videos-per-variation 10 \
  --batch-name "long_kiss_batch"

# As each video completes, you'll see:
# INFO: ✅ Generated attention video: prompt_001_vid001_kiss_attention.mp4
# INFO: ✅ Generated attention video: prompt_001_vid001_kiss_overlay.mp4

# Stop the batch anytime with Ctrl+C - all completed videos will have their attention visualizations
```

### Monitoring Progress

```bash
# Watch the attention_videos directories grow in real-time
watch -n 5 "find outputs/*/videos/*/attention_videos -name '*.mp4' | wc -l"

# Or monitor specific tokens
find outputs/*/videos/*/attention_videos -name '*kiss*' -type f
```

### Immediate Analysis

```bash
# As soon as a video completes, you can analyze it
ls outputs/long_kiss_batch_*/videos/prompt_001/attention_videos/
# prompt_001_vid001_kiss_attention.mp4
# prompt_001_vid001_kiss_overlay.mp4

# Play the overlay video to see attention on the original
mpv outputs/long_kiss_batch_*/videos/prompt_001/attention_videos/prompt_001_vid001_kiss_overlay.mp4
```

## Advanced Configuration

### Selective Generation

```yaml
attention_analysis:
  auto_generate_per_video: true
  
  visualization_params:
    # Only generate overlays for faster processing
    generate_attention_only: false
    generate_overlay: true
    generate_aggregated: false
    
    # Optimize for speed
    fps: 5                    # Lower FPS for faster generation
    interpolation_steps: 0    # Disable smoothing
    figsize: [8, 6]          # Smaller figures
```

### Error Handling

```yaml
attention_analysis:
  auto_generate_per_video: true
  
  visualization_params:
    # Continue batch even if visualization fails
    fail_on_visualization_error: false
    
    # Timeout for individual video processing
    visualization_timeout: 120  # seconds
```

## Migration Guide

### From Batch-End to Per-Video

**Old Configuration:**
```yaml
attention_analysis:
  store_attention: true
  auto_generate_videos: true  # Only at end
```

**New Configuration:**
```yaml
attention_analysis:
  store_attention: true
  auto_generate_per_video: true  # Immediate per-video
  # auto_generate_videos: false  # Optional: disable batch-end
```

### Hybrid Approach

```yaml
attention_analysis:
  store_attention: true
  auto_generate_per_video: true   # For immediate feedback
  auto_generate_videos: true      # For centralized collection
  
  # Different settings for each
  visualization_params:
    # Per-video: fast and basic
    per_video_fps: 5
    per_video_quality: "fast"
    
    # Batch-end: high quality
    batch_fps: 15
    batch_quality: "high"
```

## Troubleshooting

### Common Issues

1. **Videos not generating per-video**
   - Check `auto_generate_per_video: true` is set
   - Verify attention storage is working (look for attention_maps directory)
   - Check logs for import errors with visualization modules

2. **Slow per-video generation**
   - Reduce FPS: `fps: 5`
   - Disable interpolation: `interpolation_steps: 0`
   - Use smaller figures: `figsize: [8, 6]`

3. **Disk space concerns**
   - Enable aggregated storage: `store_aggregated_attention: true`
   - Use compression: `compress_attention: true`
   - Generate fewer video types (attention-only OR overlay, not both)

### Performance Tips

- **Aggregated attention**: Much faster than step-by-step visualization
- **SSD storage**: Faster I/O for video generation
- **GPU memory**: Visualization uses CPU/RAM, not GPU memory
- **Parallel processing**: Safe to run visualization while next video generates

## Comparison Summary

| Feature | Batch-End | Per-Video |
|---------|-----------|-----------|
| **Timing** | After entire batch | After each video |
| **Early stopping** | ❌ Lose all work | ✅ Keep completed videos |
| **Feedback** | Delayed | Immediate |
| **Organization** | Centralized | Grouped with source |
| **Memory usage** | Higher peak | Lower, constant |
| **Use case** | Small batches | Long batches |

## Recommendation

**For most users**: Enable `auto_generate_per_video: true` - it provides the best user experience with immediate feedback and early-stopping safety, with minimal downsides.
