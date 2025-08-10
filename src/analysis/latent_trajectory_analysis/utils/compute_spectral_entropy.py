import torch

def compute_spectral_entropy(power_spectrum: torch.Tensor) -> float:
    """Compute spectral entropy of power spectrum."""
    # Normalize to probability distribution
    probs = power_spectrum / (torch.sum(power_spectrum) + 1e-8)
    probs = probs[probs > 1e-8]  # Remove zeros
    
    if len(probs) > 1:
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        return entropy.item()
    return 0.0