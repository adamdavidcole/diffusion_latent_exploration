# src/visualization/plotters/plot_trajectory_atlas_umap.py
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from src.visualization.visualization_config import VisualizationConfig
from src.analysis.latent_trajectory_analysis.utils.apply_normalization import apply_normalization

def plot_trajectory_atlas_umap(
    group_tensors,
    viz_dir: Path,
    viz_config: VisualizationConfig = VisualizationConfig(),
    norm_cfg: Optional[Dict[str, Any]] = None,
    max_seeds_per_group: int = 12,
) -> Path:
    from sklearn.decomposition import PCA
    import umap
    viz_config.apply_style_settings()

    groups = sorted(group_tensors.keys())
    sample = next(iter(group_tensors.values()))['trajectory_tensor']; T = sample.shape[1]
    steps_keep = sorted(set([0, max(1,T//4), max(2,T//4), max(3,T//4), T-1]))

    # normalize → anchor at global step0 → per-step z-whiten
    import torch
    with torch.no_grad():
        flats = {}
        for g in groups:
            tens = group_tensors[g]['trajectory_tensor'][:max_seeds_per_group]
            X = apply_normalization(tens, group_tensors[g], norm_cfg)  # [N,T,D]
            flats[g] = X.cpu()
        step0_all = torch.cat([flats[g][:,0,:] for g in groups], dim=0).mean(dim=0)
        for g in groups:
            flats[g] = flats[g] - step0_all
        for t in range(T):
            Xt = torch.cat([flats[g][:,t,:] for g in groups], dim=0)
            mu, sd = Xt.mean(dim=0), Xt.std(dim=0) + 1e-8
            for g in groups:
                flats[g][:,t,:] = (flats[g][:,t,:] - mu) / sd

    # stack sampled steps to fit PCA→UMAP
    Xs, step_colors, group_ids = [], [], []
    for gi,g in enumerate(groups):
        for si in steps_keep:
            Xs.append(flats[g][:, si, :].numpy())
            step_colors.extend([si]*flats[g].shape[0])
            group_ids.extend([gi]*flats[g].shape[0])
    X = np.concatenate(Xs, axis=0)

    p50 = PCA(n_components=min(50, X.shape[1]), random_state=42).fit_transform(X)
    X2  = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, metric='euclidean', random_state=42).fit_transform(p50)

    # left: colored by step
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    sc1 = ax1.scatter(X2[:,0], X2[:,1], c=step_colors, cmap=viz_config.step_cmap or 'viridis', s=10, alpha=0.7)
    cb = plt.colorbar(sc1, ax=ax1); cb.set_label('Diffusion Step (sampled)')
    # overlay group centroids with color
    cmap_groups = plt.get_cmap(viz_config.name_cmap or 'tab10')
    groups_arr = np.array(group_ids)
    for gi,g in enumerate(groups):
        mask = groups_arr == gi
        cx, cy = X2[mask,0].mean(), X2[mask,1].mean()
        ax1.scatter([cx],[cy], s=120, edgecolor='k', facecolor=cmap_groups(gi%10), label=g)
    ax1.legend(title='Prompt Group', fontsize=8, frameon=False)
    ax1.set_title('Trajectory Atlas (anchored, PCA→UMAP) — colored by step')
    ax1.grid(True, alpha=0.3)

    # right: colored by group + hulls (footprints/area)
    for gi,g in enumerate(groups):
        mask = groups_arr == gi
        ax2.scatter(X2[mask,0], X2[mask,1], s=10, color=cmap_groups(gi%10), alpha=0.25, label=g)
        try:
            hull = ConvexHull(X2[mask])
            poly = X2[mask][hull.vertices]
            ax2.fill(poly[:,0], poly[:,1], color=cmap_groups(gi%10), alpha=0.12, lw=0)
            ax2.plot(np.r_[poly[:,0], poly[0,0]], np.r_[poly[:,1], poly[0,1]], color=cmap_groups(gi%10), lw=1.0)
        except Exception:
            pass
    ax2.legend(title='Prompt Group', fontsize=8, frameon=False)
    ax2.set_title('Group footprints (2D hulls) — colored by group')
    ax2.grid(True, alpha=0.3)

    out = viz_dir / f"trajectory_atlas_umap.{viz_config.save_format}"
    plt.tight_layout(); plt.savefig(out, dpi=viz_config.dpi, bbox_inches=viz_config.bbox_inches); plt.close()
    return out
