# WAN 1.3B Video Generation Project

A comprehensive tool for generating video sets using the WAN 1.3B video model with configurable settings and prompt variations.

## Features

- **Stable Configuration Management**: Define and reuse consistent generation settings (seed, sampler, CFG, etc.)
- **Prompt Variations**: Create prompts with variable keywords for systematic content generation
- **Batch Processing**: Generate multiple videos per prompt variation
- **Organized Output**: Automatically organize results in structured subfolders
- **Progress Tracking**: Monitor generation progress and handle errors gracefully

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
├── requirements.txt       # Python dependencies
└── main.py               # Main entry point
```

## Usage

### Basic Usage
```bash
python main.py --config configs/default.yaml --prompt-template "a romantic kiss between [two people|two men|two women|a man and a woman]"
```

### With Custom Settings
```bash
python main.py --config configs/romantic_scenes.yaml --videos-per-variation 5 --output-dir outputs/romantic_batch_1
```

## Configuration

See `configs/default.yaml` for configuration options including:
- Model settings (seed, sampler, CFG scale, etc.)
- Generation parameters (resolution, fps, duration)
- Output organization preferences

## Requirements

- Python 3.8+
- PyTorch
- WAN 1.3B model dependencies (see requirements.txt)
