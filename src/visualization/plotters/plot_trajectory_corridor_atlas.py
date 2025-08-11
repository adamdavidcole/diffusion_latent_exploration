# TODO: NEEDS TO HANDLE NORMALIZATION

"""
Trajectory Corridor Atlas plotting functionality.
Extracted from LatentTrajectoryAnalyzer._plot_trajectory_corridor_atlas.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import LineCollection
import logging
from typing import Optional, Dict, Any, List
from src.analysis.data_structures import GroupTensors
from src.visualization.visualization_config import VisualizationConfig

from src.analysis.latent_trajectory_analysis.utils.apply_normalization import apply_normalization

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_trajectory_corridor_atlas(
    group_tensors: Dict[str, Dict[str, 'torch.Tensor']],
    viz_dir: Path,
    viz_config: Optional[VisualizationConfig] = None,
    labels_map: Optional[Dict[str, str]] = None,
    reducer: str = "umap",
    max_seeds_per_group: int = 12,
    norm_cfg: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Corridor atlas with:
      - faint seed-level polylines colored by step
      - per-group mean centerlines
      - global 1σ ellipses per step (shared corridor width)
    Axes are the reduced coordinates (UMAP/PCA) => abstract; label them as such.
    """
    import torch
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

    viz_config = viz_config or VisualizationConfig()
    viz_config.apply_style_settings()

    # Load tensors if needed
    if group_tensors is None:
        return None  # keep this function pure in the new split; call from group_tensors_visualizer

    groups = sorted(group_tensors.keys())
    sample = next(iter(group_tensors.values()))['trajectory_tensor']
    T = int(sample.shape[1])

    # gather flattened Full-norm embeddings [N,T,D]
    X_blocks, marks_g, marks_s = [], [], []
    per_group_flat = {}
    for gi, g in enumerate(groups):
        tens = group_tensors[g]['trajectory_tensor']
        N = min(tens.shape[0], max_seeds_per_group)
        tens = tens[:N]
        # Full normalization from analysis (you can import your shared util if you have one)
        flat = tens.reshape(N, T, -1).float()  # assume inputs already in the chosen norm for the atlas call site
        per_group_flat[g] = flat
        X_blocks.append(flat.reshape(N*T, -1))
        marks_g.extend([gi] * (N*T))
        # step indices for coloring segments
        marks_s.extend(list(np.tile(np.arange(T), N)))

    X = torch.cat(X_blocks, dim=0).cpu().numpy()
    if X.shape[0] < 10:
        return None

    # reduce to 2D
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

    marks_g = np.asarray(marks_g)
    marks_s = np.asarray(marks_s)
    cmap_steps = plt.get_cmap(viz_config.step_cmap or 'viridis')
    cmap_groups = plt.get_cmap(viz_config.name_cmap or 'tab10')

    fig, ax = plt.subplots(figsize=(10, 8))

    # Global 1σ ellipse per step
    for si in range(T):
        P = X2[marks_s == si]
        if P.shape[0] < 5: 
            continue
        mu = P.mean(axis=0)
        cov = np.cov(P.T)
        w, v = np.linalg.eigh(cov + 1e-9*np.eye(2))
        order = np.argsort(w)[::-1]; w, v = w[order], v[:, order]
        angle = np.degrees(np.arctan2(v[1,0], v[0,0]))
        ell = Ellipse(xy=mu, width=2*np.sqrt(w[0]), height=2*np.sqrt(w[1]),
                      angle=angle, facecolor=cmap_steps(si/(T-1)), alpha=0.12, edgecolor='none')
        ax.add_artist(ell)

    # Seed polylines colored by step (faint)
    for gi, g in enumerate(groups):
        flat = per_group_flat[g].cpu().numpy()  # [N,T,D]
        N = flat.shape[0]
        # reduce the same way: re-embed these rows by mapping indices
        # Build index window for this group's block inside X2
        offset = 0
        for gj in range(gi):
            offset += per_group_flat[groups[gj]].shape[0]*T
        M = N*T
        P2 = X2[offset:offset+M, :].reshape(N, T, 2)
        for n in range(N):
            segs = np.stack([P2[n, :-1, :], P2[n, 1:, :]], axis=1)   # [T-1, 2, 2]
            lc = LineCollection(segs, cmap=cmap_steps, array=np.arange(T-1), linewidths=1.25, alpha=0.35)
            ax.add_collection(lc)

    # Group mean centerlines (solid)
    for gi, g in enumerate(groups):
        flat = per_group_flat[g].cpu().numpy()
        N = flat.shape[0]
        offset = 0
        for gj in range(gi):
            offset += per_group_flat[groups[gj]].shape[0]*T
        M = N*T
        P2 = X2[offset:offset+M, :].reshape(N, T, 2)
        mean_path = P2.mean(axis=0)  # [T,2]
        ax.plot(mean_path[:,0], mean_path[:,1], '-o', color=cmap_groups(gi%10), lw=2.0, ms=4, label=(labels_map.get(g,g) if labels_map else g))

    sm = plt.cm.ScalarMappable(cmap=cmap_steps, norm=plt.Normalize(vmin=0, vmax=T-1))
    cbar = plt.colorbar(sm, ax=ax); cbar.set_label("Diffusion step")

    ax.legend(title='Prompt Group', fontsize=9, frameon=False)
    ax.set_title("Trajectory Corridor Atlas (seed polylines, mean centerlines, global 1σ)")
    ax.set_xlabel("Atlas dim 1 (UMAP/PCA)"); ax.set_ylabel("Atlas dim 2 (UMAP/PCA)")
    ax.grid(True, alpha=0.3)

    out = viz_dir / f"trajectory_corridor_atlas.{viz_config.save_format}"
    plt.tight_layout(); plt.savefig(out, dpi=viz_config.dpi, bbox_inches=viz_config.bbox_inches); plt.close()
    return out