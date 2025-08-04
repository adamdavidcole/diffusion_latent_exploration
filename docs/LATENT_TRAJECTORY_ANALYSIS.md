# Latent Trajectory Analysis

This feature enables you to study the latent trajectory of diffusion model generations as they develop over time during the denoising process. This can provide insights into the geometry of the diffusion latent space and potentially reveal biases in how different representations are encoded.

## Key Hypothesis

According to the hypothesis being tested, dominant representations may take up more area of the latent space, while "marginal" or "othered" representations might occupy less area. By analyzing trajectories, we can potentially measure the relative scale of certain representations.

## Quick Start

### 1. Generate Videos with Latent Storage

Use the `--store-latents` flag when generating videos:

```bash
# Basic usage with latent storage
python main.py --template "a [romantic|platonic] kiss between [two people|two men|two women]" --store-latents

# Using custom config with latent storage enabled
python main.py --config configs/latent_analysis_example.yaml --template "a [gentle|passionate] embrace"
```

### 2. Analyze Stored Latents

After generation, analyze the latent trajectories:

```bash
# Analyze all videos in a batch
python analyze_latent_trajectories.py --batch-dir outputs/latent_trajectory_test_20250804_123456

# Analyze specific video
python analyze_latent_trajectories.py --batch-dir outputs/latent_trajectory_test_20250804_123456 --video-id prompt_001_vid001

# Compare specific videos
python analyze_latent_trajectories.py --batch-dir outputs/latent_trajectory_test_20250804_123456 --compare-videos prompt_001_vid001 prompt_002_vid001
```

## Configuration

### Latent Analysis Settings

Add these settings to your YAML config file:

```yaml
latent_analysis_settings:
  store_latents: true                    # Enable latent storage
  latent_storage_format: "numpy"         # Storage format: "numpy" or "torch"
  storage_interval: 1                    # Store every N steps (1 = all steps)
  compress_latents: true                 # Use compression to save disk space
```

### Command Line Options

- `--store-latents`: Enable latent storage during generation
- All existing generation options work with latent storage

## Generated Files Structure

When latent storage is enabled, the following structure is created:

```
outputs/your_batch_20250804_123456/
├── videos/           # Generated videos
├── latents/          # Stored latent representations
│   ├── prompt_000/   # Prompt-specific directory
│   │   └── vid_001/  # Individual video directory
│   │       ├── step_000.npy.gz           # Latent tensor for step 0
│   │       ├── step_000_metadata.json    # Metadata for step 0
│   │       ├── step_001.npy.gz           # Latent tensor for step 1
│   │       ├── step_001_metadata.json    # Metadata for step 1
│   │       └── summary.json              # Video generation summary
│   └── analysis/     # Analysis results and visualizations
│       ├── visualizations/
│       └── trajectory_analysis_*.json
├── logs/             # Generation logs
├── configs/          # Configuration files
└── reports/          # Generation summary reports
```

## Analysis Metrics

The trajectory analysis computes various metrics to understand the latent space geometry:

### Basic Trajectory Properties
- **Number of steps**: Total denoising steps
- **Latent dimensions**: Shape and size of latent space
- **Total trajectory distance**: Cumulative movement through latent space
- **Trajectory linearity**: How "straight" the path is (0 = very curved, 1 = perfectly linear)

### Geometric Properties
- **Trajectory volume estimate**: Estimated volume occupied by the trajectory
- **PCA analysis**: Principal components and explained variance
- **Dimensionality reduction ratio**: How much variance is captured by the first principal component

### Temporal Dynamics
- **Velocity analysis**: Speed of movement through latent space
- **Acceleration analysis**: Changes in velocity
- **Variance evolution**: How latent variance changes over time
- **Mean evolution**: How latent mean changes over time

## Analysis Examples

### Single Video Analysis

```python
from src.analysis import LatentTrajectoryAnalyzer

# Initialize analyzer
analyzer = LatentTrajectoryAnalyzer("outputs/batch_20250804_123456/latents")

# Analyze single video
result = analyzer.analyze_single_video("prompt_001_vid001")

print(f"Trajectory linearity: {result.metrics['trajectory_linearity']:.4f}")
print(f"Total distance: {result.metrics['total_trajectory_distance']:.4f}")
print(f"Volume estimate: {result.metrics['trajectory_volume_estimate']:.6f}")
```

### Comparative Analysis

```python
# Compare multiple videos
video_ids = ["prompt_001_vid001", "prompt_002_vid001", "prompt_003_vid001"]
comparison = analyzer.compare_trajectories(video_ids)

# Access comparison metrics
linearity_data = comparison['comparison_data']['trajectory_linearity']
print(f"Mean linearity: {linearity_data['mean']:.4f} ± {linearity_data['std']:.4f}")
```

### Batch Analysis

```python
from src.analysis import analyze_latent_trajectories_from_batch

# Analyze entire batch
results = analyze_latent_trajectories_from_batch("outputs/batch_20250804_123456")

print(f"Analyzed {results['total_videos_analyzed']} videos")
print(f"Comparison metrics available for {len(results['comparison_analysis']['video_ids'])} videos")
```

## Visualizations

The analysis automatically generates several types of visualizations:

1. **PCA Trajectory Plot**: Shows the trajectory in 2D principal component space
2. **Temporal Dynamics**: Plots variance, mean, and distance evolution over time
3. **Step-wise Distance**: Shows movement speed between consecutive steps

Visualizations are saved as PNG files in the `analysis/visualizations/` directory.

## Research Applications

This latent trajectory analysis can be used to investigate:

1. **Representation Bias**: Do certain prompts (e.g., different demographic groups) follow different trajectory patterns?
2. **Latent Space Geometry**: How is the semantic space organized in the diffusion model?
3. **Generation Dynamics**: How do different types of content emerge during the denoising process?
4. **Model Comparison**: How do different models or configurations affect latent trajectories?

### Example Research Questions

- Do prompts mentioning marginalized groups have more constrained (smaller volume) trajectories?
- Are certain types of content generated through more linear vs. curved paths?
- How does prompt weighting affect the latent trajectory shape?
- Do different seeds for the same prompt create similar trajectory patterns?

## Performance Considerations

### Storage Requirements

Latent storage requires significant disk space:
- Each latent tensor: ~1-10 MB (depending on resolution and compression)
- 50 steps × 10 videos = ~500 MB - 5 GB per batch
- Use `compress_latents: true` to reduce storage by ~50-70%

### Memory Usage

- Latent storage adds minimal memory overhead during generation
- Analysis requires loading full trajectories into memory
- For large batches, analyze videos individually rather than all at once

### Optimization Tips

1. **Reduce steps**: Use fewer denoising steps (e.g., 20-30 instead of 50) for faster generation
2. **Storage interval**: Use `storage_interval: 2` to store every other step
3. **Compression**: Always enable `compress_latents: true`
4. **Batch size**: Generate smaller batches if storage space is limited

## Troubleshooting

### Common Issues

1. **"No latents directory found"**: Make sure you used `--store-latents` during generation
2. **"No stored latents found"**: Check that generation completed successfully
3. **Memory errors during analysis**: Analyze videos individually instead of all at once
4. **Missing dependencies**: Install required packages: `pip install scikit-learn matplotlib seaborn`

### Debugging

Use the `--verbose` flag for detailed logging:

```bash
python analyze_latent_trajectories.py --batch-dir outputs/my_batch --verbose
```

### Checking Available Videos

List all videos with stored latents:

```bash
python analyze_latent_trajectories.py --batch-dir outputs/my_batch --list-videos
```

## Advanced Usage

### Custom Metrics

You can specify which metrics to use for comparison:

```bash
python analyze_latent_trajectories.py --batch-dir outputs/my_batch \
  --metrics trajectory_linearity total_trajectory_distance trajectory_volume_estimate
```

### Programmatic Analysis

```python
from src.analysis import LatentTrajectoryAnalyzer

analyzer = LatentTrajectoryAnalyzer("path/to/latents")

# Custom analysis workflow
video_ids = analyzer.get_available_videos()
for video_id in video_ids:
    result = analyzer.analyze_single_video(video_id, create_visualizations=False)
    
    # Your custom analysis here
    linearity = result.metrics['trajectory_linearity']
    volume = result.metrics['trajectory_volume_estimate']
    
    print(f"{video_id}: linearity={linearity:.4f}, volume={volume:.6f}")
```

This framework provides a solid foundation for exploring the latent space geometry of diffusion models and investigating potential biases in representation.
