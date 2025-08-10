import numpy as np
import torch
from typing import List

def find_peaks(signal: torch.Tensor) -> List[int]:
    """Simple peak detection on GPU tensor."""
    if len(signal) < 3:
        return []
    
    # If signal is multi-dimensional, use the norm for peak detection
    if signal.dim() > 1:
        signal = torch.norm(signal, dim=tuple(range(1, signal.dim())))
    
    # Find local maxima using tensor operations
    peaks = []
    for i in range(1, len(signal) - 1):
        # Convert tensor comparisons to boolean values properly
        is_peak = (signal[i] > signal[i-1]).item() and (signal[i] > signal[i+1]).item()
        if is_peak:
            peaks.append(i)
    
    return peaks