import numpy as np

def bootstrap_ci(
    arr: np.ndarray, 
    n_boot: int = 2000, 
    alpha: float = 0.05
):
    """Return (mean, lo, hi) with (1-alpha) CI via bootstrap; NaN-safe."""
    import numpy as np
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(123)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        s = rng.choice(x, size=x.size, replace=True)
        boots[i] = float(np.mean(s))
    lo, hi = np.quantile(boots, [alpha/2, 1 - alpha/2])
    return (float(np.mean(x)), float(lo), float(hi))