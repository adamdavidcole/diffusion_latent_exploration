# TODO: NEEDS TO HANDLE NORMALIZATION

"""
Trajectory Corridor Atlas plotting functionality.
Extracted from LatentTrajectoryAnalyzer._plot_trajectory_corridor_atlas.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import logging
from typing import Optional, Dict, Any, List
from src.analysis.data_structures import GroupTensors
from src.visualization.visualization_config import VisualizationConfig

from src.analysis.latent_trajectory_analysis.utils.apply_normalization import apply_normalization

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_trajectory_corridor_atlas(
    group_tensors: GroupTensors = None,
    viz_dir: Path = None,
    viz_config: Optional[VisualizationConfig] = VisualizationConfig(),
    steps_keep: Optional[List[int]] = None,
    max_seeds_per_group: int = 12,
    reducer: str = "umap",
    norm_cfg: Optional[Dict[str, Any]] = None
) -> Optional[Path]:
    """
        Visualizes the *corridor* structure:
        • Fit reducer on a sampled set of flattened step latents (Full norm) across all groups & seeds
        • For each group, plot the *mean path* (polyline across steps)
        • Add translucent 1σ ellipses per step representing cross-seed spread (corridor width)
        • Color encodes step index; legend encodes group
        """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    try:
        from sklearn.decomposition import PCA
        HAVE_SK = True
    except Exception:
        HAVE_SK = False

    HAVE_UMAP = False
    if reducer == "umap":
        try:
            import umap
            HAVE_UMAP = True
        except Exception:
            HAVE_UMAP = False

    # ---- load tensors on demand ----
    if group_tensors is None:
        logger.warning("⚠️ Group tensors not available, returning early")
        return None

    if viz_dir is None:
        logger.warning("⚠️ Output path not provided, returning early")
        return None

    # logger.info("plot_trajectory_corridor_atlas not implemented yet, return early")
    # return 


    groups = sorted(group_tensors.keys())
    # Determine step set
    sample = next(iter(group_tensors.values()))['trajectory_tensor']
    T = int(sample.shape[1])
    if steps_keep is None:
        steps_keep = sorted(set([0, max(1, T//5), max(2, T//5), max(3, T//5), T-1]))

    # ---- collect normalized flattened latents ----
    X_blocks, labels_step, labels_group = [], [], []
    per_group_step_arrays = {}  # for later means/ellipses

    for gi, g in enumerate(groups):
        tens = group_tensors[g]['trajectory_tensor']  # [N, T, C, F, H, W]
        N = min(tens.shape[0], max_seeds_per_group)
        tens = tens[:N]
        flat = apply_normalization(tens, group_tensors[g], norm_cfg=norm_cfg)  # [N, T, D]

        per_group_step_arrays[g] = {}
        for si in steps_keep:
            pts = flat[:, si, :].float().cpu().numpy()  # [N, D]
            per_group_step_arrays[g][si] = pts
            X_blocks.append(pts)
            labels_step.extend([si] * pts.shape[0])
            labels_group.extend([gi] * pts.shape[0])

    X = np.concatenate(X_blocks, axis=0)
    if X.shape[0] < 10: return

    # ---- reduce ----
    if HAVE_SK:
        X50 = PCA(n_components=min(50, X.shape[1]), random_state=42).fit_transform(X)
    else:
        X50 = X

    if reducer == "umap" and HAVE_UMAP:
        X2 = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, metric='cosine', random_state=42).fit_transform(X50)
    else:
        if HAVE_SK and X50.shape[1] > 2:
            X2 = PCA(n_components=2, random_state=42).fit_transform(X50)
        else:
            X2 = X50[:, :2]

    labels_step = np.asarray(labels_step)
    labels_group = np.asarray(labels_group)

    # ---- compute per-group mean polylines and step-wise ellipses (corridor width) ----
    cmap = plt.get_cmap('viridis')
    group_colors = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(10, 8))

    # draw step-wise global corridor ellipse (across ALL groups/seeds) lightly
    for si in steps_keep:
        mask = labels_step == si
        P = X2[mask]
        if P.shape[0] < 5: continue
        mu = P.mean(axis=0)
        cov = np.cov(P.T)
        # Eigen-decomp for ellipse axes
        w, v = np.linalg.eigh(cov + 1e-9*np.eye(2))
        order = np.argsort(w)[::-1]; w = w[order]; v = v[:, order]
        angle = np.degrees(np.arctan2(v[1,0], v[0,0]))
        # 1σ ellipse
        ell = Ellipse(xy=mu, width=2*np.sqrt(w[0]), height=2*np.sqrt(w[1]),
                    angle=angle, facecolor=cmap(si / max(1, T-1)), alpha=0.12, edgecolor='none')
        ax.add_artist(ell)

    # overlay per-group mean polylines through steps
    for gi, g in enumerate(groups):
        means = []
        for si in steps_keep:
            mask = (labels_group == gi) & (labels_step == si)
            pts = X2[mask]
            if pts.shape[0] == 0:
                means.append([np.nan, np.nan])
            else:
                means.append(pts.mean(axis=0))
        means = np.array(means, dtype=float)
        ax.plot(means[:,0], means[:,1], '-o', lw=2, ms=5,
                color=group_colors(gi % 10), label=g, alpha=0.95)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(steps_keep), vmax=max(steps_keep)))
    cbar = plt.colorbar(sm, ax=ax); cbar.set_label("Diffusion Step (sampled)")
    ax.legend(title='Prompt Group', fontsize=9, frameon=False)
    ax.set_title("Trajectory Corridor Atlas (mean paths + 1σ corridor)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = viz_dir / f"trajectory_corridor_atlas.{viz_config.save_format}"
    plt.savefig(output_path,
                dpi=viz_config.dpi, bbox_inches=viz_config.bbox_inches)
    plt.close()

    return output_path