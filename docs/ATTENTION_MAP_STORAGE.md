# Attention Map Storage Documentation

This document describes the structure, format, and organization of attention maps stored during WAN video generation.

## Overview

The attention storage system captures cross-attention maps from the WAN transformer during video generation. These maps show how different spatial regions in the generated video attend to specific tokens in the text prompt.

## Storage Format

## Tensor Format

The attention storage system uses a **dynamic tensor format** that adapts based on your configuration settings:

### Shape Dependencies

The final tensor shape depends on your `store_per_block` and `store_per_head` settings:

- **`store_per_block: true, store_per_head: true`**: `[blocks, heads, spatial, tokens]`
  - Full resolution: 30 blocks × 12 heads
  - Example: `[30, 12, 23850, 1]`

- **`store_per_block: false, store_per_head: true`**: `[1, heads, spatial, tokens]`  
  - Averaged across blocks, preserves heads
  - Example: `[1, 12, 23850, 1]`

- **`store_per_block: true, store_per_head: false`**: `[blocks, 1, spatial, tokens]`
  - Preserves blocks, averaged across heads  
  - Example: `[30, 1, 23850, 1]`

- **`store_per_block: false, store_per_head: false`**: `[1, 1, spatial, tokens]`
  - Fully aggregated (minimal memory usage)
  - Example: `[1, 1, 23850, 1]`

### Data Types

- **Storage dtype**: `torch.float16` (configurable, can be `float32`)
- **File format**: NumPy arrays (`.npy` files) or PyTorch tensors (`.pt` files)
- **Compression**: Optional gzip compression available

## Directory Structure

```
outputs/batch_name/attention_maps/
├── prompt_000_vid001/           # Video identifier
│   ├── summary.json            # Video-level metadata
│   └── token_flower/           # Token-specific directory
│       ├── step_000.npy       # Attention tensor for step 0
│       ├── step_000_metadata.json
│       ├── step_001.npy       # Attention tensor for step 1
│       ├── step_001_metadata.json
│       └── ...
├── prompt_001_vid001/           # Next video
│   ├── summary.json
│   └── token_tree/
│       ├── step_000.npy
│       ├── step_000_metadata.json
│       └── ...
```

### File Naming Convention

- **Video directories**: `{prompt_id}_vid{video_number:03d}`
- **Token directories**: `token_{word}` (cleaned of weight syntax)
- **Step files**: `step_{step_number:03d}.npy`
- **Metadata files**: `step_{step_number:03d}_metadata.json`

## Metadata Structure

Each step includes comprehensive metadata in JSON format:

```json
{
  "video_id": "prompt_000_vid001",
  "step": 0,
  "timestep": 999.0,
  "total_steps": 3,
  "token_word": "flower",
  "token_ids": [93643],
  "token_texts": ["flower"],
  "aggregation_method": "per_step_full",
  "attention_shape": [30, 12, 1170, 1],
  "spatial_resolution": [1170, 1],
  "num_blocks": 30,
  "num_heads": 12,
  "threshold_applied": null,
  "block_idx": null,
  "head_idx": null,
  "dtype": "torch.float16",
  "prompt": "a beautiful flower near a tree",
  "seed": 999,
  "cfg_scale": 6.5
}
```

### Key Metadata Fields

- **token_ids**: Actual token IDs from the tokenizer
- **token_texts**: Human-readable token text
- **attention_shape**: Tensor dimensions `[blocks, heads, spatial, tokens]`
- **spatial_resolution**: Flattened spatial dimensions
- **aggregation_method**: Always `"per_step_full"` for consistent format
- **prompt**: Final processed prompt sent to the model

## Integration with Prompt Templates

The system correctly handles prompt templates with variations:

### Template Example
```
"a beautiful [(flower:2.5) near a tree|(tree:3) next to a flower]"
```

### Generated Variations
1. **Prompt**: `"a beautiful flower near a tree"`
   - **Tracks**: `flower` token only
   - **Weight**: 2.5 applied to "flower"

2. **Prompt**: `"a beautiful tree next to a flower"`  
   - **Tracks**: `tree` token only
   - **Weight**: 3.0 applied to "tree"

Each variation only stores attention maps for tokens that are actually present and weighted in that specific prompt.

## Usage Examples

### Loading Individual Steps

```python
from attention_analyzer import AttentionAnalyzer

analyzer = AttentionAnalyzer("outputs/batch_name/attention_maps")

# Load single step: [blocks, heads, spatial, tokens]
step_0 = analyzer.load_step("prompt_000_vid001", "flower", 0)
print(f"Shape: {step_0.shape}")  # [30, 12, 1170, 1]
```

### Loading Full Temporal Sequence

```python
# Load all steps: [steps, blocks, heads, spatial, tokens]
all_steps = analyzer.load_all_steps("prompt_000_vid001", "flower")
print(f"Shape: {all_steps.shape}")  # [3, 30, 12, 1170, 1]
```

### Aggregation Examples

```python
# Average across transformer blocks: [steps, heads, spatial, tokens]
block_averaged = analyzer.aggregate_blocks(all_steps)

# Average across attention heads: [steps, spatial, tokens]
head_averaged = analyzer.aggregate_heads(block_averaged)

# Get temporal evolution: [steps, spatial, tokens]
temporal = analyzer.get_temporal_evolution("prompt_000_vid001", "flower")

# Reshape to video dimensions: [steps, frames, height, width]
spatial_map = analyzer.get_spatial_attention_map(
    "prompt_000_vid001", "flower", step=0,
    height=240, width=416, frames=3
)
```

## Configuration Options

### Storage Settings

```yaml
attention_analysis_settings:
  store_attention: true
  storage_format: "numpy"           # "numpy" or "torch"
  compress_attention: false         # Enable gzip compression
  storage_interval: 1               # Store every N steps
  storage_dtype: "float16"          # "float16" or "float32"
  spatial_downsample_factor: 1      # Spatial downsampling
  
  # Aggregation control (affects memory usage)
  store_per_head: false             # false = average across heads (memory efficient)
  store_per_block: false            # false = average across blocks (memory efficient)
```

### Memory vs Detail Trade-offs

| Configuration | Shape | Memory Usage | Use Case |
|---------------|--------|--------------|----------|
| `store_per_head: false`<br>`store_per_block: false` | `[1, 1, spatial, tokens]` | **Minimal** (~100x smaller) | Basic visualization, memory-constrained |
| `store_per_head: true`<br>`store_per_block: false` | `[1, heads, spatial, tokens]` | **Low** (~30x smaller) | Head analysis, balanced approach |
| `store_per_head: false`<br>`store_per_block: true` | `[blocks, 1, spatial, tokens]` | **Medium** (~12x smaller) | Layer analysis, averaged heads |
| `store_per_head: true`<br>`store_per_block: true` | `[blocks, heads, spatial, tokens]` | **Full** (maximum detail) | Research, complete analysis |

### Memory Considerations

- **Per-step storage**: ~5.6MB for typical shape `[30, 12, 1170, 1]` in float16
- **Full video**: ~17MB for 3 steps with 1 token tracked
- **Multiple tokens**: Storage scales linearly with number of tracked tokens

## Research Applications

### Attention Evolution Analysis
- Study how attention patterns change across diffusion steps
- Compare attention for different tokens within the same prompt
- Analyze the effect of prompt weighting on attention distribution

### Spatial Attention Patterns
- Visualize which spatial regions attend to specific concepts
- Study attention localization vs. distribution
- Compare attention patterns between similar concepts

### Cross-Prompt Comparison
- Compare attention for the same token across different contexts
- Study how surrounding words affect token attention
- Analyze attention consistency across prompt variations

## Technical Notes

### Token Mapping
- Uses context-sensitive tokenization with T5 tokenizer (`google/umt5-xxl`)
- Token positions are mapped in the final processed prompt
- Preserves weight information in metadata while tracking clean token text

### Attention Extraction
- Captures cross-attention maps from `attn2` modules in WAN transformer
- Each block contributes one attention map per step
- Maps show spatial→text attention relationships

### Consistency Guarantees
- All steps use identical tensor format `[blocks, heads, spatial, tokens]`
- Compatible with existing analysis tools
- Enables flexible aggregation along any dimension
