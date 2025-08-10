import torch

def corrcoef(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated correlation coefficient."""
    if x.numel() < 2:
        return torch.tensor(0.0, device=x.device)
    
    x_centered = x - torch.mean(x)
    y_centered = y - torch.mean(y)
    
    numerator = torch.sum(x_centered * y_centered)
    denominator = torch.sqrt(torch.sum(x_centered**2) * torch.sum(y_centered**2))
    
    if denominator > 1e-8:
        return numerator / denominator
    else:
        return torch.tensor(0.0, device=x.device)