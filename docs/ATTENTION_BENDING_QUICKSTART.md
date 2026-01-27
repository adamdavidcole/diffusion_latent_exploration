# 🎨 Attention Bending - Quick Start Guide

## What is This?

**Attention Bending** applies spatial transformations to cross-attention maps in Diffusion Transformers, inspired by Terence Broad's Network Bending for GANs. Think of it as "photoshopping" the attention mechanism during generation.

## TL;DR - 30 Second Setup

```python
from src.utils.attention_bending import AttentionBender, BendingConfig, BendingMode

# Amplify "kiss" token by 2.5x
config = BendingConfig(token="kiss", mode=BendingMode.SCALE, scale_factor=2.5, strength=0.8)
bender = AttentionBender(bending_configs=[config], device="cuda")

# PHASE 1: Visualization only (RECOMMENDED START)
attention_storage.configure_attention_bending(bender, apply_to_output=False)

# PHASE 2: Actually affects generation (after Phase 1 validation)
# attention_storage.configure_attention_bending(bender, apply_to_output=True)

attention_storage.wrap_attention_modules(pipe.transformer)
# Generate - bending happens automatically!
```

## Files You Need to Know

| File | Purpose |
|------|---------|
| **`ATTENTION_BENDING_INTEGRATION.md`** ⭐⭐⭐ | **How to use in video generator (START HERE!)** |
| **`ATTENTION_BENDING_TWO_PHASES.md`** ⭐ | Phase 1 vs Phase 2 guide |
| `src/utils/attention_bending.py` | Core implementation (700+ lines) |
| `src/utils/attention_storage.py` | Integration point (modified) |
| `configs/attention_bending_example.yaml` | Example configurations |
| `docs/ATTENTION_BENDING.md` | Full documentation (theory + recipes) |
| `docs/ATTENTION_BENDING_IMPLEMENTATION.md` | Implementation details |
| `scripts/demo_attention_bending.py` | Usage examples |
| `scripts/generate_with_bending.py` | Integration example |

## 9 Transformation Types

| Mode | What It Does | Best For | Example |
|------|-------------|----------|---------|
| **SCALE** | Multiply attention | Emphasis control | Amplify "kiss" 2.5x |
| **REGIONAL_MASK** ⭐ | Confine to region | Composition | Lock "sunset" to top half |
| **BLUR** | Soften attention | Background | Defocus "background" |
| **SHARPEN** | Enhance peaks | Focal elements | Sharpen "face" |
| **TRANSLATE** | Shift spatially | Repositioning | Move "romantic" right |
| **ROTATE** | Rotate pattern | Experimental | Rotate "cinematic" 15° |
| **SPATIAL_SCALE** | Zoom in/out | Extent control | Zoom "kiss" 1.5x |
| **FLIP** | Mirror H/V | Symmetry | Flip "people" horizontal |
| **FREQUENCY_FILTER** | Custom filter | Advanced | TBD |

⭐ = Most promising based on theory

## Quick Examples

### Example 1: Amplify a Concept

```yaml
# In attention_bending.yaml
attention_bending:
  enabled: true
  configs:
    - token: "kiss"
      mode: "scale"
      scale_factor: 2.5
      strength: 0.8
```

### Example 2: Spatial Composition

```yaml
configs:
  - token: "sunset"
    mode: "regional_mask"
    region: [0.0, 0.0, 1.0, 0.5]  # Top half
    region_feather: 0.2
  - token: "people"
    mode: "regional_mask"
    region: [0.0, 0.5, 1.0, 1.0]  # Bottom half
```

### Example 3: Depth of Field

```yaml
configs:
  - token: "kiss"
    mode: "sharpen"
    sharpen_amount: 2.0
    strength: 0.8
  - token: "background"
    mode: "blur"
    kernel_size: 7
    sigma: 3.0
```

## Integration Steps

### 1. Phase 1: Visualization Only (START HERE)

```python
from src.utils.attention_bending import AttentionBender, BendingConfig, BendingMode

# Create config
config = BendingConfig(
    token="kiss",
    mode=BendingMode.SCALE,
    scale_factor=2.5,
    strength=0.8
)

# Create bender
bender = AttentionBender(bending_configs=[config], device="cuda")

# Configure for Phase 1 (visualization only - SAFE)
attention_storage.configure_attention_bending(bender, apply_to_output=False)

# Wrap model
attention_storage.wrap_attention_modules(pipe.transformer)

# Generate (bent attention stored but not used)
prompt = "a romantic (kiss) between two people"
attention_storage.start_video_storage(video_id="test", prompt=prompt, target_words=["kiss"])
attention_storage.update_bending_token_map()

output = pipe(prompt=prompt, ...)

# Video is UNCHANGED, but bent attention maps are stored for analysis
```

### 2. Phase 2: Active Bending (AFTER Phase 1 validation)

```python
# Same setup, but now activate Phase 2
attention_storage.configure_attention_bending(bender, apply_to_output=True)

# Rest is identical - but now bent attention AFFECTS the video!
```

### 2. From YAML Config

```python
import yaml
from src.utils.attention_bending import create_bending_from_config

with open("configs/my_bending_config.yaml") as f:
    config = yaml.safe_load(f)

bender = create_bending_from_config(config["attention_bending"])
attention_storage.configure_attention_bending(bender)
# ... rest same as above
```

### 3. Using the Helper Script

```bash
# Simple test
python scripts/generate_with_bending.py \
  --config configs/wan_1-3b_optimized_short.yaml \
  --bending-config configs/attention_bending_example.yaml \
  --prompt "a romantic (kiss) at (sunset)"

# Quick test (inline config)
python scripts/generate_with_bending.py --quick-test

# Comparison batch (multiple scales)
python scripts/generate_with_bending.py --comparison-batch
```

## Key Parameters Explained

### `token` (string)
**Must match a parenthetical word in your prompt!**
- Prompt: `"a (romantic) (kiss) at (sunset)"`
- Valid tokens: `"romantic"`, `"kiss"`, `"sunset"`
- Invalid: `"love"` (not in prompt)

### `strength` (0.0 - 1.0)
**Blend between original and transformed**
- `0.0`: No effect (original attention)
- `0.5`: 50/50 blend
- `1.0`: Fully transformed
- Start with `0.6-0.8` for exploration

### `scale_factor` (for SCALE mode)
**Multiply attention weights**
- `1.0`: No change
- `2.0`: Double the influence
- `0.5`: Halve the influence
- `>3.0`: May cause instability
- Start with `1.5-2.5`

### `apply_to_timesteps` (tuple)
**[start, end] range of denoising steps**
- `[0, 15]`: Early (high noise, structure)
- `[15, 35]`: Mid (composition)
- `[35, 50]`: Late (details)
- `[0, 50]`: All steps

### `apply_to_layers` (list)
**Specific transformer layers**
- `[0, 1, 2]`: Early layers (semantics)
- `[4, 5, 6, 7]`: Mid layers (composition)
- `[8, 9, 10, 11]`: Late layers (details)
- `None`: All layers

### `region` (for REGIONAL_MASK)
**[x1, y1, x2, y2] in normalized 0-1 coordinates**
- `[0, 0]`: Top-left corner
- `[1, 1]`: Bottom-right corner
- `[0, 0, 1, 0.5]`: Top half
- `[0.25, 0.25, 0.75, 0.75]`: Center region

## Research Guidelines

### Start Conservative
1. ✅ Use `strength < 0.8`
2. ✅ Use `scale_factor < 3.0`
3. ✅ Test one token at a time
4. ✅ Compare against baseline (strength=0.0)

### Document Everything
```
Experiment Log Format:
- Prompt: [exact prompt]
- Config: [bending config details]
- Observations: [what happened]
- Metrics: [any measurements]
- Artifacts: [path to outputs]
- Success: [yes/partial/no]
```

### Expected Timeline
- **Day 1**: Setup & validation (null test)
- **Day 2-3**: Single-token scale sweep
- **Day 4-5**: Regional masking experiments
- **Day 6-7**: Multi-transformation composition
- **Week 2**: Analysis & documentation

### Success Criteria
✅ **Minimal Success**: System runs without errors
✅ **Partial Success**: Subtle visual effects detected
✅ **Full Success**: Strong, predictable semantic control
✅ **Unexpected Win**: Novel effects not anticipated

### Failure = Data
❌ No effect → Tells us attention isn't sufficient alone
❌ Instability → Tells us model stability constraints
❌ Opposite effect → Tells us about interaction dynamics

**All outcomes are valuable!**

## Troubleshooting

### "Token not found in map"
```python
# Check your prompt has parentheses
prompt = "a romantic (kiss) at (sunset)"  # ✅ Good
prompt = "a romantic kiss at sunset"       # ❌ Bad - no parens
```

### NaN or Inf in output
```python
# Reduce strength and scale
config = BendingConfig(
    token="kiss",
    mode=BendingMode.SCALE,
    scale_factor=2.0,  # Lower (was 4.0)
    strength=0.6,       # Lower (was 1.0)
    renormalize=True    # MUST be True
)
```

### No visible effect
```python
# Try stronger parameters
scale_factor=3.5  # Higher
strength=1.0      # Maximum
# Or try different transformation
mode=BendingMode.REGIONAL_MASK  # Often more visible
```

### Generation fails
```python
# Disable bending to isolate issue
attention_storage.configure_attention_bending(None)
# If this works, problem is in bending config
```

## What to Expect

### Optimistic Scenario 🎯
- Clear visual effects from SCALE
- Strong composition control from REGIONAL_MASK
- Interesting depth effects from BLUR/SHARPEN
- Publishable novel results

### Realistic Scenario 🤔
- Subtle effects requiring careful analysis
- Some transformations work better than others
- Need parameter tuning per concept type
- Interesting technical insights

### Learning Scenario 📚
- Effects are minimal or unclear
- Helps understand attention's true role
- Reveals model robustness mechanisms
- Negative results = valuable knowledge

## Next Steps

1. **Read**: `docs/ATTENTION_BENDING.md` for full theory
2. **Run**: `python scripts/demo_attention_bending.py` to see examples
3. **Test**: Start with null test (strength=0.0) to verify no regression
4. **Explore**: Try scale sweep on one token
5. **Document**: Keep detailed research log
6. **Analyze**: Visualize bent vs. original attention maps
7. **Iterate**: Refine based on observations

## Getting Help

- **Theory questions**: See `docs/ATTENTION_BENDING.md`
- **Implementation details**: See `docs/ATTENTION_BENDING_IMPLEMENTATION.md`
- **Usage examples**: See `scripts/demo_attention_bending.py`
- **Integration**: See `scripts/generate_with_bending.py`

## Citation

If this leads to publishable results:

```bibtex
@misc{attention_bending_2025,
  title={Attention Bending for Diffusion Transformers},
  author={Cole, Adam},
  year={2025},
  note={Inspired by Terence Broad's Network Bending}
}
```

---

**Remember**: This is research, not engineering. Embrace uncertainty, document thoroughly, and learn from all outcomes - especially the "failures". 🚀

**Good luck exploring the latent space!** 🎨✨
