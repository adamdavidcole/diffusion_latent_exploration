from typing import Dict, Any, Tuple
import numpy as np

def bootstrap_ci(arr: np.ndarray, n_boot: int = 2000, alpha: float = 0.05) -> Tuple[float,float,float]:
    x = np.asarray(arr, float)
    x = x[np.isfinite(x)]
    if x.size == 0: return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(123)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = x[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha/2, 1-alpha/2])
    return (float(x.mean()), float(lo), float(hi))