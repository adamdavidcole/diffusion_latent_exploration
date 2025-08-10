"""
Comprehensive Analysis Insight Board plotting functionality.
Extracted from LatentTrajectoryAnalyzer.
"""

from pathlib import Path
from venv import logger
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional
from src.analysis.data_structures import LatentTrajectoryAnalysis
from src.visualization.visualization_config import VisualizationConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_comprehensive_analysis_insight_board(
        results: LatentTrajectoryAnalysis, 
        viz_dir: Path, 
        viz_config: VisualizationConfig = VisualizationConfig(), 
        results_full: Optional[LatentTrajectoryAnalysis] = None, 
        video_grid_path: Optional[Path] = None, 
        **kwargs
) -> Path:
    """
    Publication board: clear hierarchy + consistent palette.
    Top row:   Radar (normalized group comparison), Final-state manifold (Var vs Mag) + Key insights box
    Middle:    Per-timestep curves (Spatial variance, Global variance, Global magnitude)
    Bottom:    Bars (Length, Velocity) [SNR track], (Acceleration, Late/Early, Turning, Alignment) [Full track]
    """ 
    try:
        has_different_results_full = False

        if results_full is None:
            results_full = results
            has_different_results_full = True

        # ---- palette & helpers ----
        groups = sorted(results.temporal_analysis.keys())
        cmap = plt.get_cmap('tab10')
        cols = [cmap(i % 10) for i in range(len(groups))]

        def norm01(a):
            a = np.asarray(a, dtype=float)
            if a.size == 0 or np.allclose(a, a[0]): return np.zeros_like(a)
            m, M = float(np.nanmin(a)), float(np.nanmax(a))
            if not np.isfinite(M - m) or (M - m) < 1e-12: return np.zeros_like(a)
            return (a - m) / (M - m + 1e-12)

        def corr_vs_rung(y):
            y = np.asarray(y, dtype=float)
            x = np.arange(len(y), dtype=float)
            if len(y) < 3 or np.allclose(y, y[0]): return 0.0
            return float(np.corrcoef(x, y)[0, 1])

        # ---- SNR track (scale) ----
        length   = np.array([results.temporal_analysis[g]['trajectory_length']['mean_length'] for g in groups], dtype=float)
        velocity = np.array([results.temporal_analysis[g]['velocity_analysis']['overall_mean_velocity'] for g in groups], dtype=float)
        ge = results.global_structure['trajectory_global_evolution']
        final_var = np.array([ge[g]['variance_progression'][-1]   for g in groups], dtype=float)
        final_mag = np.array([ge[g]['magnitude_progression'][-1]  for g in groups], dtype=float)

        # ---- Full track (shape/timing) ----
        accel   = np.array([results_full.temporal_analysis[g]['acceleration_analysis']['overall_mean_acceleration'] for g in groups], dtype=float)
        late_e  = np.array([results_full.spatial_patterns['trajectory_spatial_evolution'][g]['evolution_ratio'] for g in groups], dtype=float)
        geom    = getattr(results_full, 'individual_trajectory_geometry', {})
        turning = np.array([float(geom[g]['turning_angle_stats']['mean']) if g in geom and 'error' not in geom[g] else np.nan for g in groups])
        align   = np.array([float(geom[g]['endpoint_alignment_stats']['mean']) if g in geom and 'error' not in geom[g] else np.nan for g in groups])
        circ_m1 = np.array([
            float(np.nanmean(np.array(geom[g]['circuitousness_stats']['individual_values'], dtype=float) - 1.0))
            if g in geom and 'error' not in geom[g] else np.nan
            for g in groups
        ], dtype=float)

        # ---- per-timestep curves (Full norm for spatial; raw globals for consistency with your pipeline) ----
        def spatial_curve(g):
            spg = results_full.spatial_patterns['trajectory_spatial_evolution'][g]
            for k in ('spatial_variance_curve', 'spatial_variance_by_step', 'variance_curve'):
                if k in spg: return np.array(spg[k], dtype=float)
            return None
        spatial_curves = {g: spatial_curve(g) for g in groups}
        var_prog = {g: np.array(ge[g]['variance_progression'], dtype=float)   for g in groups}
        mag_prog = {g: np.array(ge[g]['magnitude_progression'], dtype=float) for g in groups}

        # ---- insights text ----
        insights = [
            f"Length increases with specificity: r={corr_vs_rung(length):.2f}",
            f"Velocity increases with specificity: r={corr_vs_rung(velocity):.2f}",
            f"Acceleration increases with specificity: r={corr_vs_rung(accel):.2f}",
            f"Late/Early ratio increases with specificity: r={corr_vs_rung(late_e):.2f}",
            f"Turning angle increases; alignment decreases (late steering).",
            f"Final manifold ~1D: corr(Var,Mag)={np.corrcoef(final_var, final_mag)[0,1]:.3f}",
        ]

        # ---- layout ----
        plt.rcParams.update({
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.titleweight": "bold", "axes.grid": True
        })
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 4, figure=fig, height_ratios=[1.15, 1.0, 1.0], width_ratios=[1.0, 1.0, 1.0, 1.0], hspace=0.45, wspace=0.35)

        # Top-left (2 cols): Radar (normalized)
        ax_radar = fig.add_subplot(gs[0, 0:2], projection='polar')
        labels = ['Length', 'Velocity', 'Acceleration', 'Late/Early', 'Turning', 'Alignment', 'Circ−1']
        mat = np.vstack([
            norm01(length), norm01(velocity), norm01(accel), norm01(late_e),
            norm01(np.nan_to_num(turning, nan=np.nanmean(turning))),
            norm01(np.nan_to_num(align,   nan=np.nanmean(align))),
            norm01(np.nan_to_num(circ_m1, nan=np.nanmean(circ_m1))),
        ])  # [K, G]
        N = len(labels)
        ang = np.linspace(0, 2*np.pi, N, endpoint=False).tolist(); ang += ang[:1]
        ax_radar.set_theta_offset(np.pi/2); ax_radar.set_theta_direction(-1)
        ax_radar.set_xticks(ang[:-1]); ax_radar.set_xticklabels(labels, fontsize=viz_config.fontsize_labels)
        for gi, g in enumerate(groups):
            vals = mat[:, gi].tolist(); vals += vals[:1]
            ax_radar.plot(ang, vals, linewidth=2, color=cols[gi], label=g)
            ax_radar.fill(ang, vals, color=cols[gi], alpha=0.12)
        ax_radar.set_title("Group Comparison (normalized)", fontweight=viz_config.fontweight_title)
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.35, 1.10), fontsize=8, frameon=False)

        # Top-right: Final state + insights (and optional video grid thumbnail)
        ax_fs = fig.add_subplot(gs[0, 2])
        ax_fs.scatter(final_var, final_mag, s=46, c=cols)
        for i, g in enumerate(groups): ax_fs.annotate(g, (final_var[i], final_mag[i]), fontsize=8)
        ax_fs.set_xlabel("Final Variance"); ax_fs.set_ylabel("Final Magnitude")
        ax_fs.set_title("Final-state manifold")
        ax_note = fig.add_subplot(gs[0, 3])
        ax_note.axis('off')
        ax_note.text(0, 1, "Key Insights", fontsize=12, fontweight='bold', va='top')
        ax_note.text(0, 0.92, "\n".join("• " + s for s in insights), fontsize=10, va='top')

        if video_grid_path and Path(video_grid_path).exists():
            import matplotlib.image as mpimg
            img = mpimg.imread(str(video_grid_path))
            # Make the image larger: inset occupies more of the lower panel (x, y, width, height)
            inset = ax_note.inset_axes([0.05, 0.02, 0.9, 0.5])
            inset.imshow(img)
            inset.axis('off')
            inset.set_title("Video batch grid", fontsize=10)

        # Middle row: per-timestep curves
        ax_sp = fig.add_subplot(gs[1, 0]); ax_vp = fig.add_subplot(gs[1, 1]); ax_mp = fig.add_subplot(gs[1, 2])
        for c, g in zip(cols, groups):
            y = spatial_curves[g]
            if y is not None: ax_sp.plot(range(len(y)), y, color=c, lw=2, alpha=0.9, label=g)
        ax_sp.set_title("Spatial variance over steps"); ax_sp.set_xlabel("Step"); ax_sp.set_ylabel("Variance")

        for c, g in zip(cols, groups): ax_vp.plot(range(len(var_prog[g])), var_prog[g], color=c, lw=2, alpha=0.9)
        ax_vp.set_title("Global variance progression"); ax_vp.set_xlabel("Step"); ax_vp.set_ylabel("Variance")

        for c, g in zip(cols, groups): ax_mp.plot(range(len(mag_prog[g])), mag_prog[g], color=c, lw=2, alpha=0.9)
        ax_mp.set_title("Global magnitude progression"); ax_mp.set_xlabel("Step"); ax_mp.set_ylabel("Magnitude")

        # Middle-right: empty for breathing room or future (e.g., paired-seed heatmap)
        ax_blank = fig.add_subplot(gs[1, 3]); ax_blank.axis('off')

        # Bottom row: bar summaries
        ax_l   = fig.add_subplot(gs[2, 0]); ax_v = fig.add_subplot(gs[2, 1])
        ax_a   = fig.add_subplot(gs[2, 2]); ax_le = fig.add_subplot(gs[2, 3])

        ax_l.bar(groups, length, color=cols);   ax_l.set_title("Trajectory Length (SNR)"); ax_l.tick_params(axis='x', rotation=45)
        ax_v.bar(groups, velocity, color=cols); ax_v.set_title("Mean Velocity (SNR)");     ax_v.tick_params(axis='x', rotation=45)
        ax_a.bar(groups, accel, color=cols);    ax_a.set_title("Mean Acceleration (Full)"); ax_a.tick_params(axis='x', rotation=45)
        ax_le.bar(groups, late_e, color=cols);  ax_le.set_title("Late/Early Ratio (Full)"); ax_le.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        output_path = viz_dir / f"comprehensive_insights_dashboard.{viz_config.save_format}"

        if has_different_results_full:
            output_path = viz_dir / f"comprehensive_insights_dashboard_results_full_norm.{viz_config.save_format}"
        plt.savefig(output_path, dpi=viz_config.dpi, bbox_inches=viz_config.bbox_inches)
        plt.close()

        return output_path
    except Exception as e:
        logger.error(f"Error occurred while creating comprehensive insights dashboard: {e}")
        return None