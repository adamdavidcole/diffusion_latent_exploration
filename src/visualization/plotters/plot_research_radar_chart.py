import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from pathlib import Path

from src.analysis.data_structures import LatentTrajectoryAnalysis
from src.visualization.visualization_config import VisualizationConfig


def plot_research_radar_chart(
    results: LatentTrajectoryAnalysis, 
    viz_dir: Path, 
    results_full: Optional[LatentTrajectoryAnalysis] = None,
    viz_config: VisualizationConfig = None
) -> Path:
    """
    Multi-group radar comparison over key metrics.
    Metrics (normalized per-metric across groups):
    Scale (SNR-only): Length, Velocity
    Shape (Full): Acceleration, Late/Early, Turning Angle, Alignment
    """
    if viz_config is None:
        viz_config = VisualizationConfig()
        
    if results_full is None:
        results_full = results

    groups = sorted(results.temporal_analysis.keys())

    # Collect metrics
    length   = np.array([results.temporal_analysis[g]['trajectory_length']['mean_length'] for g in groups], dtype=float)
    velocity = np.array([results.temporal_analysis[g]['velocity_analysis']['overall_mean_velocity'] for g in groups], dtype=float)

    accel    = np.array([results_full.temporal_analysis[g]['acceleration_analysis']['overall_mean_acceleration'] for g in groups], dtype=float)
    late_ear = np.array([results_full.spatial_patterns['trajectory_spatial_evolution'][g]['evolution_ratio'] for g in groups], dtype=float)

    # Geometry (may be missing for some groups)
    geom = getattr(results_full, 'individual_trajectory_geometry', {})
    turning = np.array([float(geom[g]['turning_angle_stats']['mean']) if g in geom and 'error' not in geom[g] else np.nan for g in groups])
    align   = np.array([float(geom[g]['endpoint_alignment_stats']['mean']) if g in geom and 'error' not in geom[g] else np.nan for g in groups])

    # Normalize per metric (ignore NaNs)
    def norm01(a):
        b = a.astype(float)
        if np.all(np.isnan(b)): return np.zeros_like(b)
        m = np.nanmin(b); M = np.nanmax(b)
        if not np.isfinite(M-m) or (M-m) < 1e-12: return np.zeros_like(b)
        return (b - m) / (M - m + 1e-12)

    metrics = [
        ("Length",   norm01(length)),
        ("Velocity", norm01(velocity)),
        ("Acceleration", norm01(accel)),
        ("Late/Early",   norm01(late_ear)),
        ("Turning Angle", norm01(np.nan_to_num(turning, nan=np.nanmean(turning)))),
        ("Alignment",     norm01(np.nan_to_num(align,   nan=np.nanmean(align)))),
    ]

    labels = [m[0] for m in metrics]
    values = np.vstack([m[1] for m in metrics])  # [K, G]

    # colors
    cmap = plt.get_cmap('tab10')
    cols = [cmap(i % 10) for i in range(len(groups))]

    # Radar plot
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, projection='polar')
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=viz_config.fontsize_labels)

    for gi, g in enumerate(groups):
        vals = values[:, gi].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, color=cols[gi], label=g)
        ax.fill(angles, vals, color=cols[gi], alpha=0.15)

    ax.set_title("Prompt Group Comparison (normalized)", fontweight=viz_config.fontweight_title)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    
    output_path = viz_dir / f"research_radar_chart.{viz_config.save_format}"
    plt.savefig(output_path, dpi=viz_config.dpi, bbox_inches=viz_config.bbox_inches)
    plt.close()

    return output_path
