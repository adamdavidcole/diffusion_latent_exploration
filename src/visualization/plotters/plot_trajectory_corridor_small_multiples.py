from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import torch

from src.visualization.visualization_config import VisualizationConfig
from src.analysis.latent_trajectory_analysis.utils.apply_normalization import apply_normalization

def plot_trajectory_corridor_small_multiples(
    group_tensors: Dict[str, Dict[str, torch.Tensor]],
    viz_dir: Path,
    viz_config: Optional[VisualizationConfig] = None,
    norm_cfg: Optional[Dict[str, Any]] = None,
    reducer: str = "pca",
    max_seeds_per_group: int = 12,
) -> Path:
    """
    Small-multiples: per-group bouquet of seed polylines (anchored, whitened), with a shared projection.
    """
    from sklearn.decomposition import PCA
    viz_config = viz_config or VisualizationConfig()
    viz_config.apply_style_settings()
    groups = sorted(group_tensors.keys())

    # --- prepare the same anchored, whitened 2D projection as atlas ---
    import torch
    with torch.no_grad():
        flats = {}
        for g in groups:
            tens = group_tensors[g]['trajectory_tensor']
            N = min(tens.shape[0], max_seeds_per_group)
            X = apply_normalization(tens[:N], group_tensors[g], norm_cfg)  # [N,T,D]
            flats[g] = X.cpu()
    T = flats[groups[0]].shape[1]
    step0_all = torch.cat([flats[g][:, 0, :] for g in groups], dim=0).mean(dim=0)
    for g in groups:
        flats[g] = flats[g] - step0_all
    for t in range(T):
        Xt = torch.cat([flats[g][:, t, :] for g in groups], dim=0)
        mu, sd = Xt.mean(dim=0), Xt.std(dim=0) + 1e-8
        for g in groups:
            flats[g][:, t, :] = (flats[g][:, t, :] - mu) / sd
    X_fit = torch.cat([flats[g].reshape(-1, flats[g].shape[-1]) for g in groups], dim=0).numpy()
    p50 = PCA(n_components=min(50, X_fit.shape[1]), random_state=42).fit(X_fit)
    p2  = PCA(n_components=2, random_state=42).fit(p50.transform(X_fit))
    def proj(A: np.ndarray) -> np.ndarray:
        return p2.transform(p50.transform(A))

    P2 = {g: proj(flats[g].reshape(-1, flats[g].shape[-1]).numpy()).reshape(flats[g].shape[0], T, 2) for g in groups}

    # --- plot grid ---
    n = len(groups); cols = min(4, n); rows = int(np.ceil(n/cols))
    cmap_steps = plt.get_cmap(viz_config.step_cmap or 'viridis')
    fig, axes = plt.subplots(rows, cols, figsize=(4.5*cols, 4.5*rows), squeeze=False)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]; ax.set_axisbelow(True)
            if idx >= n:
                ax.axis('off'); continue
            g = groups[idx]; Q = P2[g]
            for nseed in range(Q.shape[0]):
                segs = np.stack([Q[nseed, :-1, :], Q[nseed, 1:, :]], axis=1)
                lc = LineCollection(segs, cmap=cmap_steps, array=np.arange(T-1), linewidths=1.25, alpha=0.45)
                ax.add_collection(lc)
            ax.plot(Q.mean(axis=0)[:,0], Q.mean(axis=0)[:,1], '-k', lw=2.0, alpha=0.8)
            ax.set_title(g); ax.grid(True, alpha=0.3)
            ax.set_xlabel("Atlas dim 1"); ax.set_ylabel("Atlas dim 2")
            idx += 1
    out = viz_dir / f"trajectory_corridor_small_multiples.{viz_config.save_format}"
    plt.tight_layout(); plt.savefig(out, dpi=viz_config.dpi, bbox_inches=viz_config.bbox_inches); plt.close()
    return out
