# GPU-Optimized Structure-Aware Analysis Output Documentation

## Overview
This document describes the structure and optimization of the GPU-accelerated structure-aware latent analysis output system.

## Current Output Structure

### Primary Results File: `gpu_optimized_analysis_results.json` (40MB)

**Why is it large?**
- Contains detailed statistical distributions for 1,440 latent samples (6 groups × 12 videos × 20 timesteps)
- Preserves full precision numerical data for reproducibility
- Includes comprehensive metrics across 8 analysis dimensions
- Each metric contains mean, std, min, max, and full distributions

### File Structure Breakdown:

```json
{
  "metadata": {
    "analysis_type": "gpu_optimized_structure_aware",
    "total_samples": 1440,
    "groups_analyzed": ["prompt_000", "prompt_001", "prompt_002", "prompt_003", "prompt_004", "prompt_008"],
    "latent_shape": [16, 16, 60, 106],
    "timestamp": "ISO format",
    "gpu_device": "NVIDIA RTX A6000"
  },
  
  "spatial_patterns": {
    "spatial_variance_maps": {
      "prompt_XXX": {
        "mean": 0.xxxx,
        "std": 0.xxxx,
        "min": 0.xxxx,
        "max": 0.xxxx,
        "distribution": [240 values],  // ← LARGE DATA
        "sample_statistics": {...}
      }
    }
  },
  
  "temporal_coherence": {
    "frame_correlation": {...},      // Similar structure
    "temporal_variance": {...},
    "motion_patterns": {...}
  },
  
  "channel_analysis": {
    "channel_variance": {
      "prompt_XXX": {
        "total_variance": 0.xxxx,
        "per_channel_mean": [16 values],
        "per_channel_std": [16 values],
        "channel_correlations": [16x16 matrix]  // ← MODERATE DATA
      }
    }
  },
  
  "statistical_significance": {
    "group_comparison_tests": {
      "metric_name": {
        "group1_vs_group2": {
          "statistic": 0.xxxx,
          "p_value": 0.xxxx,
          "effect_size": 0.xxxx,
          "significant": true/false
        }
      }
    }
  },
  
  "gpu_performance_stats": {...}
}
```

## Data Volume Analysis

| Component | Size Contribution | Rationale |
|-----------|-------------------|-----------|
| Distribution arrays | ~70% (28MB) | Full statistical distributions for each metric/group |
| Channel correlations | ~15% (6MB) | 16×16 correlation matrices per group |
| Metadata & stats | ~10% (4MB) | Summary statistics and test results |
| Performance data | ~5% (2MB) | GPU performance metrics |

## Optimization Strategies

### Option 1: Hierarchical Output (Recommended)
Split into multiple specialized files:

1. **`summary_results.json`** (500KB) - Key metrics and conclusions
2. **`detailed_statistics.json`** (5MB) - Statistical test results and effect sizes  
3. **`raw_distributions.json`** (35MB) - Full statistical distributions
4. **`performance_report.json`** (100KB) - GPU performance metrics

### Option 2: Compressed Storage
- Use HDF5 or compressed numpy arrays for numerical data
- Reduces file size by ~60-80%
- Requires additional dependencies

### Option 3: Sampled Distributions
- Store representative samples instead of full distributions
- Include quantiles (5%, 25%, 50%, 75%, 95%) instead of all values
- Reduces size to ~2-3MB while preserving statistical insights

## Recommended Implementation: Hierarchical + Compression

```python
Output Structure:
├── summary/
│   ├── analysis_summary.json          (Key findings, 500KB)
│   ├── group_comparisons.json         (Statistical tests, 1MB)
│   └── performance_metrics.json       (GPU stats, 100KB)
├── detailed/
│   ├── spatial_analysis.npz          (Compressed arrays, 5MB)
│   ├── temporal_analysis.npz         (Compressed arrays, 8MB)
│   ├── channel_analysis.npz          (Compressed arrays, 3MB)
│   └── global_structure.npz          (Compressed arrays, 2MB)
├── visualizations/
│   ├── *.png                         (Analysis plots)
│   └── dashboard.html                (Interactive summary)
└── metadata/
    ├── README.md                     (This documentation)
    └── schema.json                   (Data structure definition)
```

## Quick Access Patterns

### For Initial Review:
```bash
# Load key findings only
summary/analysis_summary.json  # 500KB
```

### For Statistical Analysis:
```bash
# Load statistical test results
summary/group_comparisons.json  # 1MB
```

### For Deep Investigation:
```python
# Load specific metric data
import numpy as np
spatial_data = np.load('detailed/spatial_analysis.npz')
```

## Benefits of Optimized Structure

1. **Faster Loading**: Load only needed components
2. **Better Organization**: Logical separation of concerns  
3. **Reduced Memory**: Don't load unused data
4. **Better Compatibility**: JSON for metadata, NPZ for arrays
5. **Progressive Detail**: Start with summary, drill down as needed

## Implementation Priority

**Phase 1** (Immediate): Create hierarchical JSON structure
- Split current 40MB file into logical components
- Maintain full compatibility with existing analysis

**Phase 2** (Future): Add compression and formats
- Implement NPZ storage for numerical arrays
- Add interactive HTML dashboard
- Create schema validation

This approach provides immediate benefits while maintaining data integrity and enabling future enhancements.
