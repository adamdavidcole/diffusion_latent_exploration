# Latent Step Decoder

This module provides functionality to decode stored latent tensors back into video files using the model's VAE decoder.

## Overview

During video generation experiments, latent tensors are stored at each diffusion step. This tool allows you to decode these stored latents back into videos to visualize the diffusion process progression.

## Features

- **Automatic Model Loading**: Reads the correct model from experiment configuration
- **Batch Processing**: Decode entire experiments or filter specific prompts/videos/steps
- **Memory Efficient**: Proper GPU memory management and cleanup
- **Progress Tracking**: Detailed logging and summary reports
- **Flexible Filtering**: Fine-grained control over what gets decoded

## Usage

### Basic Usage

```bash
# Decode all latents in an experiment
python scripts/decode_latent_steps.py outputs/MyExperiment_20250901_120000/

# Decode specific prompt and video
python scripts/decode_latent_steps.py outputs/MyExperiment_20250901_120000/ \
    --prompt-filter prompt_000 --video-filter vid_001

# Decode only first and last steps
python scripts/decode_latent_steps.py outputs/MyExperiment_20250901_120000/ \
    --step-filter step_000,step_019

# Use specific GPU device
python scripts/decode_latent_steps.py outputs/MyExperiment_20250901_120000/ \
    --device cuda:1
```

### Advanced Options

```bash
# Dry run to see what would be decoded
python scripts/decode_latent_steps.py outputs/MyExperiment_20250901_120000/ \
    --dry-run --verbose

# Custom output directory and FPS
python scripts/decode_latent_steps.py outputs/MyExperiment_20250901_120000/ \
    --output-dir /custom/path/ --fps 24

# Multiple filters
python scripts/decode_latent_steps.py outputs/MyExperiment_20250901_120000/ \
    --prompt-filter prompt_000,prompt_001 \
    --video-filter vid_001,vid_002 \
    --step-filter step_000,step_005,step_010,step_015,step_019
```

## Output Structure

The decoded videos maintain the same directory structure as the latents:

```
experiment_dir/
├── latents/
│   └── prompt_000/
│       └── vid_001/
│           ├── step_000.npy.gz
│           ├── step_001.npy.gz
│           └── ...
└── latents_videos/
    ├── decode_summary.json
    └── prompt_000/
        └── vid_001/
            ├── step_000.mp4
            ├── step_001.mp4
            └── ...
```

## Programmatic Usage

You can also use the decoder classes directly in your code:

```python
from src.visualization.latent_visualizer import ExperimentLatentDecoder

# Initialize decoder for an experiment
decoder = ExperimentLatentDecoder("outputs/MyExperiment_20250901_120000/")

# Decode specific video directory
results = decoder.decode_video_directory(
    video_dir=Path("outputs/MyExperiment_20250901_120000/latents/prompt_000/vid_001/"),
    step_filter="step_000,step_019"
)

# Decode entire experiment with filters
all_results = decoder.decode_experiment(
    prompt_filter="prompt_000",
    video_filter="vid_001",
    step_filter="step_000,step_019"
)

# Cleanup
decoder.cleanup()
```

## Requirements

- The experiment directory must contain:
  - `configs/generation_config.yaml` with the model ID
  - `latents/` directory with stored latent files
- The model specified in the config must be available
- Sufficient GPU memory to load the VAE

## Performance

- **Memory Usage**: ~3-4GB GPU memory for 1.3B models, ~8-10GB for 14B models
- **Speed**: ~10s per step for typical latents (61 frames, 848x480)
- **Storage**: Each decoded step is ~4-5MB as MP4

## Troubleshooting

1. **CUDA Out of Memory**: Use `--device cpu` or a different GPU with `--device cuda:1`
2. **Model Not Found**: Ensure the model ID in the config is correct and accessible
3. **No Latents Found**: Check that the experiment actually stored latents during generation
4. **Slow Decoding**: This is normal for large latents; consider filtering to specific steps

## Technical Details

- Latents are stored as compressed float16 tensors
- VAE decoding is performed in float32 for stability
- Video export uses 12 FPS by default (configurable)
- Output videos are in MP4 format with H.264 encoding
