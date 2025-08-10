"""
Trajectory Atlas UMAP plotting functionality.
Extracted from LatentTrajectoryAnalyzer.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import logging
import matplotlib.pyplot as plt
from src.analysis.data_structures import LatentTrajectoryAnalysis
from src.visualization.visualization_config import VisualizationConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def plot_trajectory_atlas_umap(
    results: LatentTrajectoryAnalysis, 
    viz_dir: Path, 
    viz_config: VisualizationConfig =  VisualizationConfig(), 
    labels_map: dict = None, 
    group_tensors: Optional[Dict[str, Dict[str, Any]]] = None,
    **kwargs
) -> Path:
    """2-D map (UMAP/PCA) of step embeddings, colored by step index; centroids per prompt group."""
    
    try:
        from sklearn.decomposition import PCA
        HAVE_SK = True
    except Exception:
        logger.warning("⚠️ HAVE_SK PCA not available")
        HAVE_SK = False
    try:
        import umap
        HAVE_UMAP = True
    except Exception:
        logger.warning("⚠️ HAVE UMAP not available")
        HAVE_UMAP = False

    # Load on demand to avoid RAM spikes
    if group_tensors is None:
        logger.warning("⚠️ Group tensors not available, returning early")
        return

    logger.warning("⚠️ Trajectory Atlas UMAP not implemented")
    return

    # Sample a few canonical steps across the schedule
    sample = next(iter(group_tensors.values()))['trajectory_tensor']
    T = sample.shape[1]
    steps_keep = sorted(set([0, max(1, T//5), max(2, T//5), max(3, T//5), T-1]))

    Xs, cols, marks = [], [], []
    groups = sorted(group_tensors.keys())
    for gi, g in enumerate(groups):
        tens = group_tensors[g]['trajectory_tensor']   # [N, T, C, F, H, W]
        # TODO: figure out what to do about _apply_normalization
        flat = self._apply_normalization(tens, group_tensors[g])  # [N, T, D]
        for si in steps_keep:
            pts = flat[:, si, :]
            Xs.append(pts.float().cpu().numpy())
            cols.extend([si] * pts.shape[0])
            marks.extend([gi] * pts.shape[0])

    if not Xs:
        return
    X = np.concatenate(Xs, axis=0)
    if X.shape[0] < 10:
        return

    # Dimensionality reduction
    X50 = None
    X2 = None

    try: 
        if HAVE_SK:
            X50 = PCA(n_components=min(50, X.shape[1]), random_state=42).fit_transform(X)
        else:
            X50 = X

        if HAVE_UMAP:
            X2 = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, metric='cosine',
                        random_state=42).fit_transform(X50)
        else:
            if HAVE_SK and X50.shape[1] > 2:
                X2 = PCA(n_components=2, random_state=42).fit_transform(X50)
            else:
                X2 = X50[:, :2]
                logger.warning("⚠️ UMAP not available, using PCA for 2D projection")

    except Exception as e:
        logger.error(f"Error during dimensionality reduction: {e}")
        traceback.print_exc()
        return

    # Plot atlas
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(X2[:, 0], X2[:, 1], c=cols, cmap='viridis', alpha=0.6, s=14)
    cb = plt.colorbar(sc, ax=ax); cb.set_label('Diffusion Step (sampled)')

    groups_arr = np.array(marks)
    for gi, g in enumerate(groups):
        mask = groups_arr == gi
        if mask.any():
            cx, cy = X2[mask, 0].mean(), X2[mask, 1].mean()
            ax.scatter([cx], [cy], s=120, edgecolor='k', facecolor='none', label=g, marker='o')

    ax.legend(title='Prompt Group', loc='best', fontsize=8)
    ax.set_title('Trajectory Atlas (UMAP/PCA)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = viz_dir / f'trajectory_atlas_umap.{viz_config.save_format}'
    plt.savefig(output_path, dpi=viz_config.dpi, bbox_inches=viz_config.bbox_inches)
    plt.close()

    return output_path

