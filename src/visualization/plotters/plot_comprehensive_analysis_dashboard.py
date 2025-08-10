"""
Comprehensive analysis dashboard plotting functionality.
Extracted from LatentTrajectoryAnalyzer.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional

from src.analysis.data_structures import LatentTrajectoryAnalysis
from src.config.visualization_config import VisualizationConfig


def plot_comprehensive_analysis_dashboard(results: LatentTrajectoryAnalysis, viz_dir: Path, 
                                        viz_config: VisualizationConfig = None, 
                                        labels_map: dict = None, 
                                        results_full: Optional[LatentTrajectoryAnalysis] = None,
                                        **kwargs) -> Path:
    """
    Hierarchical insight board:
    Row 1: Radar (spans 2 cols) + Final-state scatter + Key insights box
    Row 2: Per-timestep curves (Spatial variance; Global variance; Global magnitude)
    Row 3: Bars (Length, Velocity) [SNR], (Acceleration, Late/Early) [Full]
    """
    # Set defaults for optional parameters
    if viz_config is None:
        viz_config = VisualizationConfig()
    
    if results_full is None:
        results_full = results
    
    output_path = viz_dir / f"comprehensive_analysis_dashboard.{viz_config.save_format}"
    
    groups = sorted(results.temporal_analysis.keys())
    cmap = plt.get_cmap('tab10')
    cols = [cmap(i % 10) for i in range(len(groups))]

    # ------ Gather metrics ------
    # SNR track (scale)
    length   = np.array([results.temporal_analysis[g]['trajectory_length']['mean_length'] for g in groups], dtype=float)
    velocity = np.array([results.temporal_analysis[g]['velocity_analysis']['overall_mean_velocity'] for g in groups], dtype=float)
    ge = results.global_structure['trajectory_global_evolution']
    final_var = np.array([ge[g]['variance_progression'][-1] for g in groups], dtype=float)
    final_mag = np.array([ge[g]['magnitude_progression'][-1] for g in groups], dtype=float)

    # Full track (shape)
    accel  = np.array([results_full.temporal_analysis[g]['acceleration_analysis']['overall_mean_acceleration'] for g in groups], dtype=float)
    late_e = np.array([results_full.spatial_patterns['trajectory_spatial_evolution'][g]['evolution_ratio'] for g in groups], dtype=float)

    # Per-timestep curves
    # spatial variance curve (robust keys)
    def spatial_curve(g):
        spg = results_full.spatial_patterns['trajectory_spatial_evolution'][g]
        for k in ('spatial_variance_curve', 'spatial_variance_by_step', 'variance_curve'):
            if k in spg: return np.array(spg[k], dtype=float)
        return None
    spatial_curves = {g: spatial_curve(g) for g in groups}
    var_prog = {g: np.array(results.global_structure['trajectory_global_evolution'][g]['variance_progression'], dtype=float) for g in groups}
    mag_prog = {g: np.array(results.global_structure['trajectory_global_evolution'][g]['magnitude_progression'], dtype=float) for g in groups}

    # Correlations vs specificity index
    idx = np.arange(len(groups), dtype=float)
    def corr(y):
        y = np.array(y, dtype=float)
        if len(y) < 3 or np.allclose(y, y[0]): return 0.0
        return float(np.corrcoef(idx, y)[0,1])

    insights = [
        f"Length↑ specificity r={corr(length):.2f}",
        f"Velocity↑ specificity r={corr(velocity):.2f}",
        f"Acceleration↑ specificity r={corr(accel):.2f}",
        f"Late/Early ratio↑ specificity r={corr(late_e):.2f}",
    ]

    # ------ Layout ------
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.1, 1.0, 0.9], hspace=0.4, wspace=0.35)

    # Row 1: Radar (2 cols)
    ax_radar = fig.add_subplot(gs[0, 0:2], projection='polar')

    # Build radar values (normalized per metric)
    def norm01(a):
        a = np.asarray(a, dtype=float)
        if np.allclose(a, a[0]): return np.zeros_like(a)
        m, M = float(np.min(a)), float(np.max(a))
        return (a - m) / (M - m + 1e-12)

    labels = ['Length', 'Velocity', 'Acceleration', 'Late/Early']
    mat = np.vstack([norm01(length), norm01(velocity), norm01(accel), norm01(late_e)])  # [K, G]
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    ax_radar.set_theta_offset(np.pi / 2); ax_radar.set_theta_direction(-1)
    ax_radar.set_xticks(angles[:-1]); ax_radar.set_xticklabels(labels, fontsize=viz_config.fontsize_labels)
    for gi, g in enumerate(groups):
        vals = mat[:, gi].tolist(); vals += vals[:1]
        ax_radar.plot(angles, vals, linewidth=2, color=cols[gi], label=g)
        ax_radar.fill(angles, vals, color=cols[gi], alpha=0.15)
    ax_radar.set_title("Group Comparison (normalized)", fontweight=viz_config.fontweight_title)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.25, 1.10), fontsize=8)

    # Row 1 right: final-state scatter + insights box
    ax_fs = fig.add_subplot(gs[0, 2])
    ax_fs.scatter(final_var, final_mag, s=40)
    for i, g in enumerate(groups):
        ax_fs.annotate(g, (final_var[i], final_mag[i]), fontsize=8)
    ax_fs.set_xlabel("Final Variance"); ax_fs.set_ylabel("Final Magnitude")
    ax_fs.set_title("Final State")

    # Insights box
    txt = "Key Insights:\n" + "\n".join("• "+s for s in insights)
    ax_fs.text(0.02, 0.02, txt, transform=ax_fs.transAxes,
            fontsize=10, va='bottom', ha='left',
            bbox=dict(facecolor='white', alpha=0.9, boxstyle='round'))

    # Row 2: per-timestep curves
    ax_sp = fig.add_subplot(gs[1, 0])
    for c, g in zip(cols, groups):
        y = spatial_curves[g]
        if y is not None:
            ax_sp.plot(range(len(y)), y, color=c, label=g, alpha=0.9)
    ax_sp.set_title("Spatial Variance over Diffusion")
    ax_sp.set_xlabel("Step"); ax_sp.set_ylabel("Spatial variance")
    ax_sp.grid(True, alpha=0.3)

    ax_vp = fig.add_subplot(gs[1, 1])
    for c, g in zip(cols, groups):
        y = var_prog[g]
        ax_vp.plot(range(len(y)), y, color=c, alpha=0.9)
    ax_vp.set_title("Global Variance Progression")
    ax_vp.set_xlabel("Step"); ax_vp.set_ylabel("Variance")
    ax_vp.grid(True, alpha=0.3)

    ax_mp = fig.add_subplot(gs[1, 2])
    for c, g in zip(cols, groups):
        y = mag_prog[g]
        ax_mp.plot(range(len(y)), y, color=c, alpha=0.9)
    ax_mp.set_title("Global Magnitude Progression")
    ax_mp.set_xlabel("Step"); ax_mp.set_ylabel("Magnitude")
    ax_mp.grid(True, alpha=0.3)

    # Row 3: bar summaries
    ax_l = fig.add_subplot(gs[2, 0]); ax_l.bar(groups, length, color=cols); ax_l.set_title("Trajectory Length (SNR)") ; ax_l.tick_params(axis='x', rotation=45)
    ax_v = fig.add_subplot(gs[2, 1]); ax_v.bar(groups, velocity, color=cols); ax_v.set_title("Mean Velocity (SNR)"); ax_v.tick_params(axis='x', rotation=45)
    ax_a = fig.add_subplot(gs[2, 2]); ax_a.bar(groups, accel, color=cols); ax_a.set_title("Mean Acceleration (Full)"); ax_a.tick_params(axis='x', rotation=45)
    # Optionally replace ax_a with Late/Early; we keep Accel here, Late/Early is in radar + could add a 4th panel if desired

    plt.tight_layout()
    plt.savefig(output_path, dpi=viz_config.dpi, bbox_inches=viz_config.bbox_inches)
    plt.close()
    
    return output_path
