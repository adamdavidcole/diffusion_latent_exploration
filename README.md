# WAN Video Generation Project

A research toolkit for systematic video generation and XAI analysis using WAN diffusion models (1.3B and 14B). Generate grids of videos across prompt variations, capture and visualize cross-attention maps, manipulate attention during generation, and explore latent space geometry — all configured via YAML and viewable through a built-in web UI.

## Features

- **Prompt Variations**: Create sequence of prompts with variable keywords for systematic content generation with organized results in structured subfolders.
- **Latent Trajectory Analysis**: Store and analyze latent representations during diffusion to study latent space geometry.
- **Attention Map Storage**: Capture and analyze cross-attention patterns between text tokens and spatial regions during generation.
- **Attention Map Manipulation**: "Bend" cross-attention maps during generation to expand the generative possibilities of the model.
- **Visualization**: Web UI interface to visualize grids of latent space variations, attention maps, and attention bending outputs. See a high level overview of generation process, or dig down into details of attention.

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
└── main.py                # Main entry point
```

## Usage

### Video Generation with Prompt Variations

Use templates to generate grids of videos with slight prompt variations. This is useful for evaluating how certain identities or demographics are represented under stable conditions. For example, the template `"a romantic kiss between [two people|two men|two women|a man and a woman]"` generates one video per seed for each of:

1. a romantic kiss between two people
2. a romantic kiss between two men
3. a romantic kiss between two women
4. a romantic kiss between a man and a woman

#### WAN 1.3B (recommended starting point — ~8 GB VRAM)
```bash
python main.py --config configs/wan_1-3b_optimized_long.yaml \
  --template "a romantic kiss between [two people|two men|two women|a man and a woman]"
```

#### WAN 14B (requires 48 GB+ VRAM)
```bash
python main.py --config configs/wan_14b_optimized_long.yaml \
  --template "a romantic kiss between [two people|two men|two women|a man and a woman]"
```

---

### Latent Trajectory Analysis

Store latent representations at each diffusion step, then analyze them to latent space geometry.

```bash
# Step 1 — generate videos and store latents
python main.py --config configs/wan_1-3b_optimized_long.yaml \
  --template "a romantic kiss between [two people|two men]" \
  --store-latents

# Step 2 — decode latents to videos for visual inspection
python scripts/decode_latent_steps.py <path_to_batch_folder>

# Step 3 — run trajectory analysis
python analyze_latent_trajectories.py --batch-dir <path_to_batch_folder>
```

---

### Attention Map Visualization

Wrap any token in parentheses in your template to capture cross-attention maps for it at each diffusion step. Use one of the provided attention configs to control storage settings.

#### Step 1 — Generate with an attention config

**Light** (per-step averages only — recommended starting point):
```bash
python main.py --config configs/attention_visualization_light.yaml \
  --template "a beautiful (flower) in a garden"
```

**Comprehensive** (per-step, per-layer, and per-head — generates many output videos):
```bash
python main.py --config configs/attention_visualization_comprehensive.yaml \
  --template "a beautiful (cat) in a garden"
```

The only difference between the two configs is `store_per_block` and `store_per_head` in the `attention_analysis_settings` section. Note that "Comprehensive" attention map generation can easily reach the tens of thousands of video outputs.

#### Step 2 — Decode attention maps to videos
```bash
python scripts/decode_attention_steps.py <path_to_batch_folder>
```

---

### Attention Bending

Modify cross-attention mechanism during generation to modulate the model behavior. See `configs/attention_bending_quick_sweep.yaml` for a full working example:

```bash
python main.py --config configs/attention_bending_quick_sweep.yaml \
  --template "a (cat) playing in a (garden)"
```

---

### Available Configurations

| Config | Description |
|--------|-------------|
| `configs/default.yaml` | Minimal defaults |
| `configs/wan_1-3b_optimized_short.yaml` | WAN 1.3B, short videos |
| `configs/wan_1-3b_optimized_long.yaml` | WAN 1.3B, longer videos with attention storage |
| `configs/wan_14b_optimized_short.yaml` | WAN 14B, optimized memory |
| `configs/wan_14b_optimized_long.yaml` | WAN 14B, longer videos |
| `configs/attention_visualization_light.yaml` | Attention maps, averages only |
| `configs/attention_visualization_comprehensive.yaml` | Attention maps, per-layer and per-head |
| `configs/attention_bending_quick_sweep.yaml` | Attention bending example |
| `configs/attention_bending_comprehensive_sweep.yaml` | Advanced Attention bending example |


---

## Configuration

All generation parameters are set in YAML config files under `configs/`. Key sections:

- **`model_settings`** — model ID, seed, sampler, CFG scale, inference steps
- **`video_settings`** — resolution (width/height), fps, frame count
- **`memory_settings`** — GPU memory management, cache clearing
- **`latent_analysis_settings`** — latent storage options
- **`attention_analysis_settings`** — attention capture and storage options
- **`attention_bending_settings` and `attention_bending_variations`** - enable attention bending and define modulation rules

---

## Requirements

- Python 3.10+ (tested on 3.12.9)
- NVIDIA GPU with CUDA support (**required**)
  - WAN 1.3B: ~8 GB VRAM minimum
  - WAN 14B: 48 GB+ VRAM (A100/A6000 class)
- Node.js 18+ and npm (for the web interface only)

---

## Installation

### 1. Create a conda environment

```bash
conda create -n wan python=3.12
conda activate wan
```

### 2. Install PyTorch with CUDA

PyTorch must be installed with the index URL matching your CUDA driver version. Check your driver with `nvidia-smi`, then use the appropriate command:

```bash
# CUDA 12.9 (tested configuration)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# CUDA 12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify:
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

### 4. (Optional) Web interface

The web interface has a Flask backend and a React/Vite frontend.


**Prep**
```bash
python scripts/generate_thumbnails.py
```

**Backend** 
```bash
cd webapp
npm install
```

**Frontend** (requires Node.js 18+):
```bash
cd webapp/react-frontend
npm install
```

**Run** (from the repo root):
```bash
# Terminal 1 — Flask backend (port 5000)
cd webapp/backend && python app.py

# Terminal 2 — Vite dev server (port 5174)
cd webapp/react-frontend && npm run dev
```

Open `http://localhost:5174` in your browser.

> **For large media collections (thousands of videos/thumbnails):** Flask's file serving can be slow. nginx handles this much more efficiently:
> ```bash
> # Install nginx if needed: sudo apt install nginx
> cd webapp && nginx -p $PWD -c nginx.conf
> VITE_MEDIA_SERVER=http://127.0.0.1:8888 npm run dev  # from webapp/react-frontend
> ```
> Stop nginx when done: `cd webapp && nginx -p $PWD -c nginx.conf -s stop`
