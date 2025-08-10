import torch
import logging
from typing import Dict, Any, List
import math
import numpy as np
from typing import Optional
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)    


def analyze_intrinsic_dimension(
    group_tensors: Dict[str, Dict[str, torch.Tensor]],
) -> Dict[str, Any]:
    """
    Intrinsic dimension analysis that is GPU/CPU memory-safe.

    Pipeline per group:
    1) Flatten features to [N*T, F_full].
    2) (GPU) Optional Gaussian random projection to K << F_full in float32.
    3) (GPU) Row subsample to at most `max_points`.
    4) (CPU) Standardize and run PCA (randomized) for explained variance + #components @ 90/95/99%.
    5) (CPU) Estimate ID via:
            - TwoNN (Facco et al.) on a small subset.
            - Levina–Bickel MLE with k in [k_min, k_max] (averaged), also on a small subset.
    """

    id_cfg =  {
        "use_random_projection": True,
        "rp_dim": 512,            # target feature dim after RP (256–512 is good)
        "max_points": 2000,       # cap rows for PCA/ID
        "corr_max_points": 600,   # cap rows for NN / pairwise distance based estimators
        "pca_components": 50,     # cap PCA components (randomized)
        "center": True,
        "standardize": True,
        "random_seed": 42,
        "twoNN_min_frac": 0.05,   # fraction of tail to ignore in TwoNN fit (robustness)
        "twoNN_max_frac": 0.95,
        "mle_k_min": 10,
        "mle_k_max": 20,
    }

    use_rp: bool = bool(id_cfg.get("use_random_projection", True))
    rp_dim: int = int(id_cfg.get("rp_dim", 512))
    max_points: int = int(id_cfg.get("max_points", 2000))
    corr_max_points: int = int(id_cfg.get("corr_max_points", 600))
    pca_components_cap: int = int(id_cfg.get("pca_components", 50))
    center: bool = bool(id_cfg.get("center", True))
    standardize: bool = bool(id_cfg.get("standardize", True))
    seed: int = int(id_cfg.get("random_seed", 42))
    twoNN_min_frac: float = float(id_cfg.get("twoNN_min_frac", 0.05))
    twoNN_max_frac: float = float(id_cfg.get("twoNN_max_frac", 0.95))
    mle_k_min: int = int(id_cfg.get("mle_k_min", 10))
    mle_k_max: int = int(id_cfg.get("mle_k_max", 20))


    def _twonn_id(points: np.ndarray) -> float:
        """
        Facco et al. TwoNN estimator.
        - Compute for each point the ratio mu = d2/d1 (2nd / 1st NN).
        - Fit: log(1 - F(mu)) vs log(mu) over central quantiles -> slope ~ -ID.
        """
        n = points.shape[0]
        if n < 20:
            return float("nan")

        # Pairwise distances (small n, already projected)
        # Use chunking-free broadcast since corr_max_points <= ~600
        diffs = points[:, None, :] - points[None, :, :]
        D = np.sqrt(np.sum(diffs * diffs, axis=2))  # (n, n)
        np.fill_diagonal(D, np.inf)
        # For each row, get smallest and second smallest distances
        d1 = np.partition(D, 0, axis=1)[:, 0]
        d2 = np.partition(D, 1, axis=1)[:, 1]
        # Guard against zeros
        d1 = np.clip(d1, 1e-12, None)
        mu = d2 / d1

        # Sort mu and compute empirical CDF
        mu_sorted = np.sort(mu)
        F = (np.arange(1, n + 1) - 0.5) / n

        # Trim tails for robustness
        lo = int(max(1, math.floor(twoNN_min_frac * n)))
        hi = int(min(n, math.ceil(twoNN_max_frac * n)))
        if hi - lo < 10:
            return float("nan")

        x = np.log(mu_sorted[lo:hi])
        y = np.log(1.0 - F[lo:hi])

        # Linear fit y = a + b x; slope b ≈ -ID
        # Use robust guard against NaN/inf
        msk = np.isfinite(x) & np.isfinite(y)
        if msk.sum() < 10:
            return float("nan")
        b, a = np.polyfit(x[msk], y[msk], 1)
        id_est = -float(b)
        return id_est if np.isfinite(id_est) and id_est > 0 else float("nan")

    def _mle_id(points: np.ndarray, k_min: int = 10, k_max: int = 20) -> float:
        """
        Levina–Bickel MLE (average over k in [k_min, k_max]).
        Implementation uses full pairwise distances (OK for <= ~600).
        """
        n = points.shape[0]
        if n <= k_max + 1:
            return float("nan")

        diffs = points[:, None, :] - points[None, :, :]
        D = np.sqrt(np.sum(diffs * diffs, axis=2))  # (n, n)
        np.fill_diagonal(D, np.inf)
        # Sort distances per row
        D_sorted = np.sort(D, axis=1)  # increasing, D_sorted[:, 0] is 1-NN

        ids = []
        for k in range(k_min, k_max + 1):
            # Levina–Bickel: m_hat = [ 1 / ( (1/(n*(k-1))) * sum_i sum_{j=1..k-1} log(d_{ik} / d_{ij}) ) ]
            # Using natural log.
            d_k = D_sorted[:, k]  # distance to k-th neighbor (0-based)
            d_js = D_sorted[:, 1:k]  # distances to 1..(k-1)-th neighbors
            ratio = d_k[:, None] / np.clip(d_js, 1e-12, None)
            s = np.log(np.clip(ratio, 1e-12, None)).sum(axis=1)  # per i
            denom = np.mean(s)  # average over i
            if denom <= 0 or not np.isfinite(denom):
                continue
            m_hat = (1.0 / ((1.0 / (k - 1)) * denom))
            if np.isfinite(m_hat) and m_hat > 0:
                ids.append(m_hat)

        if len(ids) == 0:
            return float("nan")
        return float(np.mean(ids))