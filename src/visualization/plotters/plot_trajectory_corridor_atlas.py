from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse

from src.visualization.visualization_config import VisualizationConfig
from src.analysis.data_structures import LatentTrajectoryAnalysis
from src.analysis.latent_trajectory_analysis.utils.apply_normalization import apply_normalization

def _fit_project_pca(X: np.ndarray, k: int = 50, out_dim: int = 2, seed: int = 42) -> np.ndarray:
    """Fast PCA using sklearn; X is [M,D]."""
    from sklearn.decomposition import PCA
    k = min(k, X.shape[1])
    P = PCA(n_components=k, random_state=seed).fit_transform(X)
    if out_dim < k:
        P2 = PCA(n_components=out_dim, random_state=seed).fit_transform(P)
    else:
        P2 = P[:, :out_dim]
    return P2

def plot_trajectory_corridor_atlas(
    viz_dir: Path,
    viz_config: Optional[VisualizationConfig] = None,
    labels_map: Optional[Dict[str, str]] = None,
    group_tensors: Optional[Dict[str, Dict[str, 'torch.Tensor']]] = None,
    norm_cfg: Optional[Dict[str, Any]] = None,
    reducer: str = "pca",
    max_seeds_per_group: int = 12,
    sample_frac_for_fit: float = 0.25,
) -> Path:
    """
    Anchored + step-whitened atlas:
      • Full-norm, flatten → subtract global step-0 mean → per-step whiten
      • Fit PCA/UMAP on a random subset; transform all
      • Draw seed polylines (colored by step), group mean, and global 1σ ellipses
    """
    import torch
    viz_config = viz_config or VisualizationConfig()
    viz_config.apply_style_settings()

    assert group_tensors is not None, "group_tensors required for atlas"

    groups = sorted(group_tensors.keys())
    cmap_steps  = plt.get_cmap(viz_config.step_cmap or 'viridis')
    cmap_groups = plt.get_cmap(viz_config.name_cmap or 'tab10')

    # ---- gather Full-norm flattened data ----
    flats = {}
    with torch.no_grad():
        for g in groups:
            tens = group_tensors[g]['trajectory_tensor']
            N = min(tens.shape[0], max_seeds_per_group)
            X = apply_normalization(tens[:N], group_tensors[g], norm_cfg)  # [N,T,D]
            flats[g] = X.cpu()

    T = flats[groups[0]].shape[1]

    # ---- anchor: subtract global step-0 mean ----
    step0_all = torch.cat([flats[g][:, 0, :] for g in groups], dim=0).mean(dim=0)  # [D]
    for g in groups:
        flats[g] = flats[g] - step0_all  # broadcast to [N,T,D]

    # ---- per-step whitening across all groups/seeds ----
    # compute mean/std per step on the fly
    all_step_means = []
    all_step_stds  = []
    for t in range(T):
        Xt = torch.cat([flats[g][:, t, :] for g in groups], dim=0)  # [M,D]
        mu = Xt.mean(dim=0)
        sd = Xt.std(dim=0) + 1e-8
        all_step_means.append(mu); all_step_stds.append(sd)
    for g in groups:
        X = flats[g]
        for t in range(T):
            flats[g][:, t, :] = (X[:, t, :] - all_step_means[t]) / all_step_stds[t]

    # ---- build big matrix for fitting reducer (subsample for speed) ----
    mats = []
    for g in groups:
        X = flats[g].reshape(-1, flats[g].shape[-1])  # [N*T,D]
        if sample_frac_for_fit < 1.0:
            m = X.shape[0]
            idx = np.random.default_rng(42).choice(m, size=max(64, int(m * sample_frac_for_fit)), replace=False)
            mats.append(X[idx])
        else:
            mats.append(X)
    X_fit = torch.cat(mats, dim=0).numpy()

    # ---- fit reducer & transform all points to 2D ----
    if reducer == "pca":
        # 50D then 2D PCA; robust & fast
        _ = _fit_project_pca(X_fit, k=min(50, X_fit.shape[1]), out_dim=2)
        # fit again to get components; simpler path: just refit on all for clean transform
        from sklearn.decomposition import PCA
        p50 = PCA(n_components=min(50, X_fit.shape[1]), random_state=42).fit(X_fit)
        def proj(A: np.ndarray) -> np.ndarray:
            Z = p50.transform(A)
            p2 = PCA(n_components=2, random_state=42).fit(Z)
            return p2.transform(Z)
        project = proj
    else:
        # UMAP on 50D PCA for stability
        from sklearn.decomposition import PCA
        import umap
        p50 = PCA(n_components=min(50, X_fit.shape[1]), random_state=42).fit(X_fit)
        um = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, metric='euclidean', random_state=42).fit(p50.transform(X_fit))
        def project(A: np.ndarray) -> np.ndarray:
            return um.transform(p50.transform(A))

    # transform each group’s [N,T,D] to 2D
    P2 = {g: project(flats[g].reshape(-1, flats[g].shape[-1]).numpy()).reshape(flats[g].shape[0], T, 2) for g in groups}

    # --- standardize XY for display (does not change relative geometry) ---
    ALL = np.concatenate([P2[g].reshape(-1,2) for g in groups], axis=0)
    mu, sd = ALL.mean(axis=0), ALL.std(axis=0) + 1e-8
    for g in groups:
        P2[g] = (P2[g] - mu) / sd



    # ---- plot ----
    fig, ax = plt.subplots(figsize=(10, 8))
    # global 1σ per step (in 2D space)
    for t in range(T):
        P = np.concatenate([P2[g][:, t, :] for g in groups], axis=0)
        if P.shape[0] < 5: continue
        mu = P.mean(axis=0); C = np.cov(P.T)
        w, v = np.linalg.eigh(C + 1e-9*np.eye(2)); o = np.argsort(w)[::-1]; w, v = w[o], v[:, o]
        ang = np.degrees(np.arctan2(v[1,0], v[0,0]))
        ell = Ellipse(mu, 2*np.sqrt(w[0]), 2*np.sqrt(w[1]), angle=ang,
                      facecolor=cmap_steps(t/(T-1)), alpha=0.10, edgecolor='none')
        ax.add_artist(ell)

    # seed polylines colored by step; group mean
    for gi, g in enumerate(groups):
        col = cmap_groups(gi % 10)
        Q = P2[g]  # [N,T,2]
        for n in range(Q.shape[0]):
            segs = np.stack([Q[n, :-1, :], Q[n, 1:, :]], axis=1)  # [T-1,2,2]
            lc = LineCollection(segs, cmap=cmap_steps, array=np.arange(T-1), linewidths=1.25, alpha=0.35)
            ax.add_collection(lc)
        ax.plot(Q.mean(axis=0)[:,0], Q.mean(axis=0)[:,1], '-o', color=col, lw=2.0, ms=4,
                label=(labels_map.get(g,g) if labels_map else g))


    sm = plt.cm.ScalarMappable(cmap=cmap_steps, norm=plt.Normalize(vmin=0, vmax=T-1))
    cbar = plt.colorbar(sm, ax=ax); cbar.set_label("Diffusion step")
    ax.legend(title='Prompt Group', fontsize=9, frameon=False)
    ax.set_title("Trajectory Corridor Atlas (anchored, step-whitened)")
    ax.set_xlabel("Atlas dim 1 (UMAP/PCA)"); ax.set_ylabel("Atlas dim 2 (UMAP/PCA)")
    ax.grid(True, alpha=0.3)
    out = viz_dir / f"trajectory_corridor_atlas.{viz_config.save_format}"
    plt.tight_layout(); plt.savefig(out, dpi=viz_config.dpi, bbox_inches=viz_config.bbox_inches); plt.close()
    return out
