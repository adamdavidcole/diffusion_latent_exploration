# WAN Video Generation Project

A comprehensive tool for generating video sets using WAN video models (1.3B and 14B) with configurable settings, prompt variations, and optimized memory management.

## Features

- **Multi-Model Support**: Compatible with WAN 1.3B and WAN 14B models
- **Advanced Memory Management**: Optimized for large models with intelligent memory allocation
- **Stable Configuration Management**: Define and reuse consistent generation settings (seed, sampler, CFG, etc.)
- **Prompt Variations**: Create prompts with variable keywords for systematic content generation
- **Batch Processing**: Generate multiple videos per prompt variation
- **Organized Output**: Automatically organize results in structured subfolders
- **Progress Tracking**: Monitor generation progress and handle errors gracefully
- **GPU Memory Optimization**: Automatic memory management for CUDA devices

## Project Structure

```
├── src/                    # Source code
│   ├── config/            # Configuration management
│   ├── generators/        # Video generation logic
│   ├── prompts/           # Prompt handling and variations
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── outputs/               # Generated videos (organized by batch)
├── logs/                  # Generation logs
├── scripts/               # Auxiliary scripts and utilities
├── webapp/                # Web interface for experiment management
├── requirements.txt       # Python dependencies
└── main.py               # Main entry point
```

## Usage

### Basic Usage with WAN 1.3B
```bash
python main.py --config configs/default.yaml --template "a romantic kiss between [two people|two men|two women|a man and a woman]"
```

### Using WAN 14B Model (Requires GPU with 48GB+ VRAM)
```bash
python main.py --config configs/wan_14b_optimized.yaml --template "a simple test video"
```

### With Custom Settings
```bash
python main.py --config configs/romantic_scenes.yaml --videos-per-variation 5 --output-dir outputs/romantic_batch_1
```

### Available Configurations
- `configs/default.yaml` - Standard WAN 1.3B settings
- `configs/wan_14b_optimized.yaml` - Optimized for WAN 14B with memory management
- `configs/fast_test.yaml` - Quick testing configuration
- `configs/high_quality.yaml` - High-quality generation settings

## Configuration

See configuration files in `configs/` directory for options including:
- **Model settings**: Model ID, seed, sampler, CFG scale, inference steps
- **Video parameters**: Resolution (width/height), fps, frame count, duration
- **Memory optimization**: GPU memory management, model reloading, cache clearing
- **Output organization**: Directory structure and file naming preferences

### Memory Requirements
- **WAN 1.3B**: Requires ~8GB VRAM
- **WAN 14B**: Requires 48GB+ VRAM (optimized for A100/A6000 GPUs)

## Requirements

- Python 3.8+
- PyTorch 2.6.0+ with CUDA 12.4 support
- NVIDIA GPU with sufficient VRAM
- WAN model dependencies (see requirements.txt)

### Installation
```bash
pip install -r requirements.txt
# For CUDA 12.4 compatibility:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Memory Optimization Features

The project includes advanced memory management for large models:

- **Intelligent Model Reloading**: Automatically unloads/reloads models based on available memory
- **GPU Memory Monitoring**: Real-time tracking of GPU memory usage
- **Gradient Checkpointing**: Reduces memory usage during training/inference
- **Expandable Memory Segments**: Prevents memory fragmentation issues
- **Smart Cache Management**: Clears GPU cache at optimal points during generation

### XFormers Support

XFormers is **optional** and **not required** for optimal performance:
- WAN transformer architecture already performs efficiently
- Current memory optimization provides sufficient performance
- XFormers may provide marginal improvements (5-10%) but adds complexity
- System works perfectly without XFormers - installation is not recommended unless you specifically need the minor performance gains

### Troubleshooting Memory Issues

If you encounter CUDA out-of-memory errors:

1. **Check CUDA version compatibility**:
   ```bash
   nvcc --version  # Should show CUDA 12.4
   python -c "import torch; print(torch.version.cuda)"  # Should show 12.4
   ```

2. **Ensure correct PyTorch version**:
   ```bash
   python -c "import torch; print(torch.__version__)"  # Should show 2.6.0+cu124
   ```

3. **Use memory-optimized configuration**:
   ```bash
   python main.py --config configs/wan_14b_optimized.yaml
   ```

4. **Monitor GPU memory**:
   ```bash
   nvidia-smi  # Check available VRAM before running
   ```
