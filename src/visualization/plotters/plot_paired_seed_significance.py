"""
Paired-seed significance plotting functionality.
Extracted from LatentTrajectoryAnalyzer.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.analysis.data_structures import LatentTrajectoryAnalysis
from src.visualization.visualization_config import VisualizationConfig

def plot_paired_seed_significance(
        results: LatentTrajectoryAnalysis, 
        viz_dir: Path, 
        viz_config: VisualizationConfig = VisualizationConfig(), 
        labels_map: dict = None, 
        **kwargs
) -> Path:
    '''Paired-seed tests for adjacent rungs: length (SNR), velocity, acceleration (if arrays available).'''
    import numpy as np
    import matplotlib.pyplot as plt
    try:
        from scipy.stats import ttest_rel, wilcoxon
        HAVE_SCIPY = True
    except Exception:
        HAVE_SCIPY = False

    ta = results.temporal_analysis
    groups = sorted(ta.keys())
    if len(groups) < 2:
        return

    def get_per_video(key_chain):
        out = {}
        for g in groups:
            d = results.temporal_analysis[g]
            cur = d
            ok = True
            for k in key_chain:
                if k not in cur:
                    ok=False; break
                cur = cur[k]
            if not ok or cur is None:
                out[g] = None
                continue
            arr = np.array(cur, dtype=float)
            out[g] = arr if arr.ndim==1 else arr.reshape(-1)
        return out

    lengths = get_per_video(['trajectory_length','individual_lengths'])
    vels    = get_per_video(['velocity_analysis','mean_velocity'])
    accels  = get_per_video(['acceleration_analysis','mean_acceleration'])

    rows = []
    for i in range(len(groups)-1):
        g1, g2 = groups[i], groups[i+1]
        for label, series in [('Length', lengths), ('Velocity', vels), ('Acceleration', accels)]:
            a, b = series.get(g1), series.get(g2)
            if a is None or b is None or a.size==0 or b.size==0 or a.size!=b.size:
                continue
            diff = b - a
            d = float(diff.mean() / (diff.std(ddof=1)+1e-12))
            if HAVE_SCIPY:
                t_p = float(ttest_rel(b, a).pvalue)
                try:
                    w_p = float(wilcoxon(b, a, zero_method='wilcox').pvalue) if not np.allclose(diff, 0) else None
                except Exception:
                    w_p = None
            else:
                t_p, w_p = None, None
            rows.append((f"{g1}→{g2}", label, -np.log10(t_p) if t_p else np.nan, d))

    if not rows:
        return

    pairs = sorted({r[0] for r in rows})
    metrics = sorted({r[1] for r in rows})
    heat = np.full((len(metrics), len(pairs)), np.nan)
    annot = np.empty_like(heat, dtype=object)
    for (pair, label, logp, d) in rows:
        i = metrics.index(label); j = pairs.index(pair)
        heat[i,j] = logp
        annot[i,j] = f"{d:.2f}"

    fig, ax = plt.subplots(figsize=(1.8*len(pairs)+3, 1.2*len(metrics)+2))
    im = ax.imshow(heat, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(pairs))); ax.set_xticklabels(pairs, rotation=45)
    ax.set_yticks(range(len(metrics))); ax.set_yticklabels(metrics)
    ax.set_title("Paired-seed significance (−log10 p) with Cohen's d")
    for i in range(len(metrics)):
        for j in range(len(pairs)):
            if not np.isnan(heat[i,j]):
                ax.text(j, i, annot[i,j], ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax, label="−log10 p (paired t-test)")
    plt.tight_layout()

    output_path = viz_dir / f'paired_seed_significance.{viz_config.save_format}'
    plt.savefig(output_path, dpi=viz_config.dpi, bbox_inches=viz_config.bbox_inches)
    plt.close()

    return output_path

