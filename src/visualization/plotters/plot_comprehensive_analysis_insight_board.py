# src/visualization/plotters/plot_comprehensive_analysis_insight_board.py
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import matplotlib.pyplot as plt

from src.visualization.visualization_config import VisualizationConfig
from src.analysis.data_structures import LatentTrajectoryAnalysis

def plot_comprehensive_analysis_insight_board(
    results: LatentTrajectoryAnalysis,
    viz_dir: Path,
    viz_config: Optional[VisualizationConfig] = None,
    results_full: Optional[LatentTrajectoryAnalysis] = None,
    labels_map: Optional[Dict[str, str]] = None,
    video_grid_path: Optional[str] = None,
) -> Path:
    viz_config = viz_config or VisualizationConfig()
    viz_config.apply_style_settings()

    if results_full is None:
        results_full = results

    groups = results.analysis_metadata['prompt_groups']
    labels = [labels_map.get(g, g) if labels_map else g for g in groups]
    cmap = plt.get_cmap(viz_config.name_cmap or 'tab10')
    cols = [cmap(i % 10) for i in range(len(groups))]

    def norm01(a):
        a = np.asarray(a, float)
        if a.size == 0 or np.allclose(a, a[0]): return np.zeros_like(a)
        m, M = np.nanmin(a), np.nanmax(a)
        return (a - m) / ((M - m) if M > m else 1.0)

    def corr_vs_rung(y):
        y = np.asarray(y, float)
        
        # Correct way to specify the data type
        x = np.arange(len(y), dtype=float)
        
        if len(y) < 3 or np.allclose(y, y[0]): 
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    # --- metrics ---
    ta_snr  = results.temporal_analysis
    ta_full = results_full.temporal_analysis
    ge      = results.global_structure['trajectory_global_evolution']

    length   = np.array([ta_snr[g]['trajectory_length']['mean_length'] for g in groups], float)
    velocity = np.array([ta_snr[g]['velocity_analysis']['overall_mean_velocity'] for g in groups], float)
    accel    = np.array([ta_full[g]['acceleration_analysis']['overall_mean_acceleration'] for g in groups], float)
    late_e   = np.array([results_full.spatial_patterns['trajectory_spatial_evolution'][g]['evolution_ratio'] for g in groups], float)

    geom = getattr(results_full, 'individual_trajectory_geometry', {})
    turning = np.array([float(geom[g]['turning_angle_stats']['mean']) if g in geom and 'error' not in geom[g] else np.nan for g in groups])
    align   = np.array([float(geom[g]['endpoint_alignment_stats']['mean']) if g in geom and 'error' not in geom[g] else np.nan for g in groups])
    circ_m1 = np.array([float(np.nanmean(np.array(geom[g]['circuitousness_stats']['individual_values'], float) - 1.0))
                        if g in geom and 'error' not in geom[g] else np.nan for g in groups])

    final_var = np.array([ge[g]['variance_progression'][-1]  for g in groups], float)
    final_mag = np.array([ge[g]['magnitude_progression'][-1] for g in groups], float)

    # CIs if available
    CIs = getattr(results, 'confidence_intervals', None) or {}
    def yerr_ci(key):
        lo, hi = [], []
        for g in groups:
            ci = CIs.get(g, {}).get(key)
            if ci:
                mean, low, high = ci
                lo.append(mean - low); hi.append(high - mean)
            else:
                lo.append(0.0); hi.append(0.0)
        return np.array([lo, hi])

    # --- layout ---
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.15, 1.0, 1.0], wspace=0.35, hspace=0.45)

    # Radar
    ax_radar = fig.add_subplot(gs[0, 0:2], projection='polar')
    radar_labels = ['Length', 'Velocity', 'Acceleration', 'Late/Early', 'Turning', 'Alignment', 'Circ−1']
    mat = np.vstack([
        norm01(length), norm01(velocity), norm01(accel), norm01(late_e),
        norm01(np.nan_to_num(turning, nan=np.nanmean(turning))),
        norm01(np.nan_to_num(align,   nan=np.nanmean(align))),
        norm01(np.nan_to_num(circ_m1, nan=np.nanmean(circ_m1))),
    ])
    N = len(radar_labels)
    ang = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
    ax_radar.set_theta_offset(np.pi/2); ax_radar.set_theta_direction(-1)
    ax_radar.set_xticks(ang[:-1]); ax_radar.set_xticklabels(radar_labels, fontsize=viz_config.fontsize_labels)
    for gi, g in enumerate(groups):
        vals = mat[:, gi].tolist(); vals += vals[:1]
        ax_radar.plot(ang, vals, lw=2, color=cols[gi], label=labels[gi])
        ax_radar.fill(ang, vals, color=cols[gi], alpha=0.12)
    ax_radar.set_title("Group comparison (normalized)", fontweight=viz_config.fontweight_title)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.35, 1.10), fontsize=8, frameon=False)

    # Final manifold + insights
    ax_fs = fig.add_subplot(gs[0, 2])
    ax_fs.scatter(final_var, final_mag, s=46, c=cols)
    for i, g in enumerate(groups): ax_fs.annotate(labels[i], (final_var[i], final_mag[i]), fontsize=8)
    ax_fs.set_xlabel("Final Variance"); ax_fs.set_ylabel("Final Magnitude")
    ax_fs.set_title("Final-state manifold")
    ax_note = fig.add_subplot(gs[0, 3]); ax_note.axis('off')
    ins = [
        f"Length r={corr_vs_rung(length):.2f}",
        f"Velocity r={corr_vs_rung(velocity):.2f}",
        f"Acceleration r={corr_vs_rung(accel):.2f}",
        f"Late/Early r={corr_vs_rung(late_e):.2f}",
        f"Turning ↑, Alignment ↓ with specificity",
        f"corr(Var,Mag)={np.corrcoef(final_var, final_mag)[0,1]:.3f}",
    ]
    ax_note.text(0, 1, "Key Insights", fontsize=12, fontweight='bold', va='top')
    ax_note.text(0, 0.92, "\n".join("• "+s for s in ins), fontsize=10, va='top')
    if video_grid_path:
        try:
            import matplotlib.image as mpimg
            img = mpimg.imread(video_grid_path)
            ax_note.imshow(img); ax_note.axis('off')
        except Exception:
            pass

    # Spatial variance curve – use the fields you actually emit
    ax_sp = fig.add_subplot(gs[1, 0])
    any_curve = False
    for c, g, lab in zip(cols, groups, labels):
        spg = results_full.spatial_patterns.get('trajectory_spatial_evolution', {}).get(g, {})
        y = None
        # your analyzer writes 'trajectory_pattern'
        if 'trajectory_pattern' in spg:
            y = np.asarray(spg['trajectory_pattern'], float)
        else:
            # fallbacks if schema changes in the future
            alt = results_full.spatial_patterns.get('spatial_progression_patterns', {}).get(g, {})
            y = np.asarray(alt.get('step_deltas_mean', []), float) if 'step_deltas_mean' in alt else None
        if y is not None and y.size:
            ax_sp.plot(range(len(y)), y, color=c, lw=2, alpha=0.95, label=lab)
            any_curve = True
    ax_sp.set_title("Spatial variance over steps"); ax_sp.set_xlabel("Step"); ax_sp.set_ylabel("Variance / ΔVariance")
    if any_curve: ax_sp.legend(fontsize=8, frameon=False)
    else: ax_sp.text(0.5, 0.5, "No spatial variance curves recorded", ha='center', va='center', transform=ax_sp.transAxes)

    # Global variance & magnitude curves
    ax_vp = fig.add_subplot(gs[1, 1])
    for c, g in zip(cols, groups):
        y = np.asarray(ge[g]['variance_progression'], float)
        ax_vp.plot(range(len(y)), y, color=c, lw=2, alpha=0.95)
    ax_vp.set_title("Global variance progression"); ax_vp.set_xlabel("Step"); ax_vp.set_ylabel("Variance")

    ax_mp = fig.add_subplot(gs[1, 2])
    for c, g in zip(cols, groups):
        y = np.asarray(ge[g]['magnitude_progression'], float)
        ax_mp.plot(range(len(y)), y, color=c, lw=2, alpha=0.95)
    ax_mp.set_title("Global magnitude progression"); ax_mp.set_xlabel("Step"); ax_mp.set_ylabel("Magnitude")

    fig.add_subplot(gs[1, 3]).axis('off')  # breathing room

    # Bars with CIs
    def bar_with_ci(ax, y, title, ci_key=None):
        ax.bar(labels, y, color=cols)
        if ci_key is not None:
            lohi = yerr_ci(ci_key)
            if np.any(lohi):
                ax.errorbar(np.arange(len(labels)), y, yerr=lohi, fmt='none',
                            ecolor='k', capsize=3, lw=1)
        ax.set_title(title); ax.tick_params(axis='x', rotation=45)

    ax_l  = fig.add_subplot(gs[2, 0]); bar_with_ci(ax_l,  length,   "Trajectory Length (SNR)",  'length')
    ax_v  = fig.add_subplot(gs[2, 1]); bar_with_ci(ax_v,  velocity, "Mean Velocity (SNR)",      'velocity')
    ax_a  = fig.add_subplot(gs[2, 2]); bar_with_ci(ax_a,  accel,    "Mean Acceleration (Full)", 'acceleration')

    ax_le = fig.add_subplot(gs[2, 3])
    ax_le.bar(labels, late_e, color=cols)
    ax_le.set_title("Late/Early Ratio (Full)"); ax_le.tick_params(axis='x', rotation=45)

    out = viz_dir / f"comprehensive_analysis_dashboard.{viz_config.save_format}"
    plt.tight_layout()
    plt.savefig(out, dpi=viz_config.dpi, bbox_inches=viz_config.bbox_inches)
    plt.close()
    return out
