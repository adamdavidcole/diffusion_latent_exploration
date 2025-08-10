"""
Individual Trajectory Geometry Dashboard plotting functionality.
Extracted from LatentTrajectoryAnalyzer.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.analysis.data_structures import LatentTrajectoryAnalysis
from src.visualization.visualization_config import VisualizationConfig

def plot_individual_trajectory_geometry_dashboard(
        results: LatentTrajectoryAnalysis, 
        viz_dir: Path, 
        viz_config: VisualizationConfig = VisualizationConfig(), 
        labels_map: dict = None, 
        **kwargs
) -> Path:
    """
    Restored + improved geometry dashboard:
    • Trajectory speed (per-group mean)
    • Per-trajectory log volumes (violin)
    • Circuitousness − 1.0 (mean bar)
    • Scatter: Speed vs Log Volume (points = trajectories)
    • Scatter: Speed vs Circuitousness (points = trajectories)
    • Turning angle distribution (violin) + endpoint alignment overlay
    • Convex-hull proxies: Δ% vs baseline for log-volume & effective side
    """
    # ---------- helpers ----------
    def _palette(n):
        base = plt.get_cmap('tab10')
        return [base(i % 10) for i in range(n)]

    ta = results.temporal_analysis
    geom = results.individual_trajectory_geometry
    hull = results.convex_hull_analysis if hasattr(results, 'convex_hull_analysis') else {}

    groups = sorted(ta.keys())
    colors = _palette(len(groups))

    # -------- per-group scalars for bars --------
    speed_mean = [ta[g]['velocity_analysis']['overall_mean_velocity'] for g in groups]
    circuit_means = []
    for g in groups:
        if g in geom and 'error' not in geom[g]:
            vals = np.array(geom[g]['circuitousness_stats']['individual_values'], dtype=float)
            circuit_means.append(float(np.nanmean(vals - 1.0)))
        else:
            circuit_means.append(np.nan)

    # -------- per-trajectory arrays for scatters/violins --------
    logvol_by_group = []
    circ_by_group = []
    speed_by_group = []
    for g in groups:
        # log-volumes
        if g in geom and 'error' not in geom[g]:
            logv = np.array(geom[g]['log_volume_stats']['individual_values'], dtype=float)
            logvol_by_group.append(logv)
            circ = np.array(geom[g]['circuitousness_stats']['individual_values'], dtype=float)
            circ_by_group.append(circ)
        else:
            logvol_by_group.append(np.array([]))
            circ_by_group.append(np.array([]))

        # speeds: per-video mean velocity
        mv = np.array(ta[g]['velocity_analysis'].get('mean_velocity_by_video',
                                                    ta[g]['velocity_analysis'].get('mean_velocity', [])),
                    dtype=float)
        speed_by_group.append(mv)

    # ---------- convex hull proxies (Δ% vs baseline) ----------
    def _pct_delta(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0: return arr
        base = arr[0]
        return 100.0 * (arr - base) / (base + 1e-12)

    if hull:
        logvol_group = np.array([hull[g].get('log_bbox_volume', np.nan) for g in groups], dtype=float)
        eff_group    = np.array([hull[g].get('effective_side',   np.nan) for g in groups], dtype=float)
        # NOTE: If you want strict consistency, derive eff_group from logvol_group here:
        # D = <same dimension used in analysis>; if unknown, we skip to avoid wrong scaling.
        logvol_delta = _pct_delta(logvol_group)
        eff_delta    = _pct_delta(eff_group)
    else:
        logvol_delta = eff_delta = np.array([])

    # ---------- figure ----------
    fig = plt.figure(figsize=(18, 14))

    # Row 1: speed bar, per-traject log-vol violin, circuit-1 bar
    ax1 = plt.subplot(3,3,1)
    ax1.bar(groups, speed_mean, color=colors)
    ax1.set_title("Trajectory Speed (mean per group)")
    ax1.tick_params(axis='x', rotation=45)

    ax2 = plt.subplot(3,3,2)
    valid = [lv if lv.size else np.array([np.nan]) for lv in logvol_by_group]
    parts = ax2.violinplot(valid, showmeans=True, showextrema=False)
    ax2.set_xticks(np.arange(1, len(groups)+1)); ax2.set_xticklabels(groups, rotation=45)
    ax2.set_title("Per-trajectory Log BBox Volume (violin)")

    ax3 = plt.subplot(3,3,3)
    ax3.bar(groups, circuit_means, color=colors)
    # Tighten y to highlight small differences
    if np.isfinite(circuit_means).any():
        arr = np.array([x for x in circuit_means if np.isfinite(x)], dtype=float)
        span = max(0.02, (arr.max() - arr.min()) * 1.3)
        ax3.set_ylim(arr.min() - 0.1*span, arr.min() + span)
    ax3.set_title("Trajectory Circuitousness − 1.0 (mean)")
    ax3.tick_params(axis='x', rotation=45)

    # Row 2: scatters
    ax4 = plt.subplot(3,3,4)
    for g, c, v_speed, v_log in zip(groups, colors, speed_by_group, logvol_by_group):
        n = min(len(v_speed), len(v_log))
        if n > 0:
            ax4.scatter(v_speed[:n], v_log[:n], s=18, alpha=0.65, color=c, label=g)
    ax4.set_xlabel("Speed (mean per trajectory)")
    ax4.set_ylabel("Log BBox Volume")
    ax4.set_title("Speed vs Log Volume (points = trajectories)")
    ax4.legend(fontsize=8, loc='best')

    ax5 = plt.subplot(3,3,5)
    for g, c, v_speed, v_circ in zip(groups, colors, speed_by_group, circ_by_group):
        n = min(len(v_speed), len(v_circ))
        if n > 0:
            ax5.scatter(v_speed[:n], v_circ[:n]-1.0, s=18, alpha=0.65, color=c, label=g)
    ax5.set_xlabel("Speed (mean per trajectory)")
    ax5.set_ylabel("Circuitousness − 1.0")
    ax5.set_title("Speed vs Circuitousness (points = trajectories)")

    ax6 = plt.subplot(3,3,6)
    turn_vals = [np.array(geom[g]['turning_angle_stats']['individual_values'], dtype=float)
                if g in geom and 'error' not in geom[g] else np.array([np.nan]) for g in groups]
    try:
        ax6.violinplot([v[~np.isnan(v)] if v.size else np.array([np.nan]) for v in turn_vals],
                    showmeans=True, showextrema=False)
    except Exception:
        pass
    ax6.set_xticks(np.arange(1, len(groups)+1)); ax6.set_xticklabels(groups, rotation=45)
    ax6.set_title("Turning Angle distribution (violin)")
    # Overlay endpoint alignment
    ax6b = ax6.twinx()
    align_means = [float(np.nanmean(np.array(geom[g]['endpoint_alignment_stats']['individual_values'], dtype=float)))
                if g in geom and 'error' not in geom[g] else np.nan for g in groups]
    ax6b.plot(np.arange(1, len(groups)+1), align_means, 's--', linewidth=2, label='Endpoint Alignment')
    ax6b.legend(loc='upper right', fontsize=8)

    # Row 3: convex-hull Δ% bars
    ax7 = plt.subplot(3,3,7)
    if logvol_delta.size:
        ax7.bar(groups, logvol_delta, color=colors)
        ax7.set_title("Convex Hull: Log BBox Volume Δ% vs baseline")
        ax7.set_ylabel("% change")
        ax7.tick_params(axis='x', rotation=45)
        ax7.grid(True, alpha=0.3)
    else:
        ax7.set_axis_off(); ax7.text(0.5, 0.5, "No convex-hull data", ha='center', va='center')

    ax8 = plt.subplot(3,3,8)
    if eff_delta.size:
        ax8.bar(groups, eff_delta, color=colors)
        ax8.set_title("Convex Hull: Effective Side Δ% vs baseline")
        ax8.set_ylabel("% change")
        ax8.tick_params(axis='x', rotation=45)
        ax8.grid(True, alpha=0.3)
    else:
        ax8.set_axis_off(); ax8.text(0.5, 0.5, "No convex-hull data", ha='center', va='center')

    # Keep last panel free for future (or show a legend/color key)
    ax9 = plt.subplot(3,3,9)
    ax9.axis('off')
    lines = [plt.Line2D([0], [0], color=c, lw=6) for c in colors]
    ax9.legend(lines, groups, title="Groups", loc='center', fontsize=8, ncol=2, frameon=False)

    plt.tight_layout()

    output_path = viz_dir / f"individual_trajectory_geometry_dashboard.{viz_config.save_format}"
    plt.savefig(output_path, dpi=viz_config.dpi, bbox_inches=viz_config.bbox_inches)
    plt.close()

    return output_path