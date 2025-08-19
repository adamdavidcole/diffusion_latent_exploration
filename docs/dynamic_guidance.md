# Dynamic Guidance Scale Scheduling

This feature allows you to modulate the classifier-free guidance scale during video generation, enabling creative effects like starting with high guidance for strong prompt adherence and reducing it over time for more creative/diverse results.

## How It Works

The system modifies the pipeline's `guidance_scale` property during each denoising step through a callback mechanism. This works because the WAN pipeline re-reads the guidance scale at the beginning of each loop iteration.

## Configuration

Add the `cfg_schedule_settings` section to your config YAML:

```yaml
cfg_schedule_settings:
  enabled: true
  schedule:
    0: 6.5    # Step 0: High guidance for strong prompt following
    10: 3.5   # Step 10: Medium guidance
    19: 0.0   # Step 19: No guidance for creative results
  interpolation: "linear"  # Options: "linear", "step", "cosine"
  apply_to_guidance_2: true  # Apply to both transformers (dual-model WAN)
  verbose: true  # Log guidance changes during generation
```

## Schedule Format

The `schedule` is a dictionary mapping step numbers (0-indexed) to guidance scale values:
- **Step 0**: First denoising step
- **Step N-1**: Last denoising step (where N is `num_inference_steps`)
- **Values**: Float guidance scale values (0.0 = no guidance, higher = stronger guidance)

## Interpolation Methods

### Linear (Default)
Smoothly interpolates between keyframe values using linear interpolation.

```yaml
interpolation: "linear"
```

### Step
Uses the previous keyframe value until reaching the next keyframe (step function).

```yaml
interpolation: "step"
```

### Cosine
Uses cosine interpolation for smoother transitions.

```yaml
interpolation: "cosine"
```

## Example Schedules

### High-to-Low Guidance
Start with strong prompt adherence, end with creative freedom:
```yaml
schedule:
  0: 7.5   # Strong guidance at start
  15: 2.0  # Reduced guidance near end
  19: 0.0  # No guidance at final steps
```

### Creative Middle
High guidance at start and end, creative exploration in middle:
```yaml
schedule:
  0: 6.0   # Strong guidance
  10: 1.0  # Creative middle phase
  19: 5.0  # Return to guided generation
```

### Gradual Reduction
Slowly reduce guidance throughout generation:
```yaml
schedule:
  0: 8.0
  5: 6.0
  10: 4.0
  15: 2.0
  19: 0.5
```

## Integration

The dynamic guidance system integrates seamlessly with:
- ✅ Latent trajectory analysis
- ✅ Attention map storage
- ✅ Weighted prompt embeddings
- ✅ Memory optimization features
- ✅ Batch generation

## Performance Impact

- **Minimal**: Only adds a simple callback function
- **No Model Changes**: Uses existing pipeline callback mechanism
- **Memory Efficient**: No additional memory overhead

## Troubleshooting

### Enable Verbose Logging
Set `verbose: true` to see guidance scale changes:
```
INFO:src.utils.dynamic_guidance:Step 0: Guidance scale 7.50 -> 6.50
INFO:src.utils.dynamic_guidance:Step 5: Guidance scale 6.50 -> 3.50
```

### Test Without Generation
Use the test script to verify scheduling:
```bash
python scripts/test_dynamic_guidance.py
```

### Common Issues

1. **Schedule validation fails**: Ensure all keys are integers, values are non-negative floats
2. **No effect observed**: Check that `enabled: true` and schedule has multiple keyframes
3. **Unexpected interpolation**: Verify interpolation method spelling and keyframe placement

## Technical Details

The implementation:
1. Creates a `GuidanceScheduler` that handles interpolation between keyframes
2. Wraps it in a `DynamicGuidanceCallback` that modifies pipeline properties
3. Integrates with existing callback chain (latent storage, attention storage)
4. Restores original guidance scale after generation

The callback modifies `pipe._guidance_scale` and optionally `pipe._guidance_scale_2` (for dual-model WAN) before each denoising step.
