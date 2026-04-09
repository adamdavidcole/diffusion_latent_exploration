# WAN Video Generation Project

A comprehensive tool for generating video sets using WAN video models (1.3B and 14B) with configurable settings, prompt variations, optimized memory management, and **latent trajectory analysis** for studying diffusion model geometry.

## Features

- **Multi-Model Support**: Compatible with WAN 1.3B and WAN 14B models
- **Stable Configuration Management**: Define and reuse consistent generation settings (seed, sampler, CFG, etc.)
- **Prompt Variations**: Create prompts with variable keywords for systematic content generation
- **Batch Processing**: Generate multiple videos per prompt variation
- **Organized Output**: Automatically organize results in structured subfolders
- **Latent Trajectory Analysis**: Store and analyze latent representations during diffusion to study model geometry and potential biases
- **Attention Map Storage**: Capture and analyze cross-attention patterns between text tokens and spatial regions during generation
- **Attention Map Manipulation**: "Bend" cross-attention maps during generation to expand the generative possibilities of the model. 

## Project Structure

```
├── src/                    # Source code
│   ├── config/            # Configuration management
│   ├── generators/        # Video generation logic
│   ├── prompts/           # Prompt handling and variations
│   ├── analysis/          # Latent trajectory analysis tools
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── outputs/               # Generated videos (organized by batch)
├── logs/                  # Generation logs
├── scripts/               # Auxiliary scripts and utilities
├── webapp/                # Web interface for experiment management
├── docs/                  # Documentation
├── requirements.txt       # Python dependencies
├── main.py               # Main entry point
└── analyze_latent_trajectories.py  # Latent analysis script
```

## Usage

### Basic Usage with WAN 1.3B
```bash
python main.py --config configs/default.yaml --template "a romantic kiss between [two people|two men|two women|a man and a woman]"
```

### Using WAN 14B Model (Requires GPU with 48GB+ VRAM)
```bash
python main.py --config configs/wan_14b_optimized.yaml --template "a romantic kiss between [two people|two men|two women|a man and a woman]"
```

### 🆕 With Latent Trajectory Analysis
```bash
# Generate videos while storing latent representations for analysis
python main.py --template "a [romantic|platonic] kiss between [two people|two men|two women]" --store-latents

# Analyze stored latents to study diffusion geometry
python analyze_latent_trajectories.py --batch-dir outputs/your_batch_20250804_123456
```

### 🆕 With Attention Map Storage
```bash
# Generate videos while capturing attention maps for specific tokens
python main.py --template "a beautiful (flower) in a garden" --store-attention

# Generate with template variations tracking different tokens per variation
python main.py --template "a beautiful [(flower) | (tree)] in a park" --store-attention
```

### Available Configurations
- `configs/default.yaml` - Standard WAN 1.3B settings
- `configs/wan_14b_optimized.yaml` - Optimized for WAN 14B with memory management
- `configs/fast_test.yaml` - Quick testing configuration
- `configs/high_quality.yaml` - High-quality generation settings
- `configs/latent_analysis_example.yaml` - 🆕 Example config with latent storage enabled

## Latent Trajectory Analysis

This project includes advanced tools for studying the latent space geometry of diffusion models during the generation process. This can help understand potential biases and representation patterns in AI-generated content.

### Key Hypothesis
Dominant representations may occupy more area in the latent space, while "marginal" or "othered" representations might occupy less area. By analyzing trajectories, we can potentially measure the relative scale of certain representations.

### Documentation

- **[Latent Trajectory Analysis](docs/LATENT_TRAJECTORY_ANALYSIS.md)**: Comprehensive guide to latent space analysis and geometry studies
- **[Attention Map Storage](docs/ATTENTION_MAP_STORAGE.md)**: Complete documentation of attention map capture, storage format, and analysis
- **[FP16 Latent Storage](docs/FP16_LATENT_STORAGE.md)**: Memory-efficient storage with half precision
- **[Latent Storage Fix](docs/LATENT_STORAGE_FIX.md)**: Technical details on storage improvements

## Quick Start
1. **Generate with latent storage**: Add `--store-latents` flag
2. **Analyze trajectories**: Use `analyze_latent_trajectories.py` script
3. **Study results**: Examine metrics like trajectory linearity, volume, and dynamics

See [docs/LATENT_TRAJECTORY_ANALYSIS.md](docs/LATENT_TRAJECTORY_ANALYSIS.md) for detailed documentation.

### Analysis Metrics
- **Trajectory linearity**: How straight the path through latent space is
- **Volume estimation**: Space occupied by the trajectory
- **Temporal dynamics**: Velocity and acceleration patterns
- **Geometric properties**: PCA analysis and dimensionality reduction

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

- Python 3.10+ (tested on 3.12.9)
- NVIDIA GPU with CUDA support (**required**)
  - WAN 1.3B: ~8 GB VRAM minimum
  - WAN 14B: 48 GB+ VRAM (A100/A6000 class)
- Node.js 18+ and npm (for the web interface only)

---

## Installation

### 1. Create a conda environment (recommended)

```bash
conda create -n wan python=3.12
conda activate wan
```

### 2. Install PyTorch with CUDA

PyTorch must be installed with the index URL that matches your CUDA driver version. Check your driver with `nvidia-smi`, then pick the right command below:

```bash
# CUDA 12.9 (current — tested configuration)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# CUDA 12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify the install:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Optional extras**
> - UMAP trajectory visualizations: `pip install umap-learn`
> - Video similarity analysis (SSIM/MSE): `pip install scikit-image`

### 4. (Optional) Set up the web interface

The web interface has a Flask backend and a React/Vite frontend.

**Backend** — already covered by `requirements.txt` (Flask, Flask-CORS, PyYAML).

**Frontend** (requires Node.js 18+):
```bash
cd webapp/react-frontend
npm install
```

**Run both together** (from the repo root):
```bash
# Terminal 1 — Flask backend (port 5000)
cd webapp/backend
python app.py

# Terminal 2 — Vite dev server (port 5173)
cd webapp/react-frontend
npm run dev
```

Then open `http://localhost:5173` in your browser.

---

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
   nvcc --version
   python -c "import torch; print(torch.version.cuda)"
   ```

2. **Ensure correct PyTorch version**:
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

3. **Use memory-optimized configuration**:
   ```bash
   python main.py --config configs/wan_14b_optimized.yaml
   ```

4. **Monitor GPU memory**:
   ```bash
   nvidia-smi  # Check available VRAM before running
   ```
