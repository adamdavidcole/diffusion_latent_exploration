from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_confidence_ellipse(x, y, ax, color, alpha=0.15, n_std=2.0):
    """Add confidence ellipse to plot."""
    try:
        if len(x) < 2 or len(y) < 2:
            return
        
        # Calculate covariance matrix
        cov = np.cov(x, y)
        
        # Check for degenerate cases
        if np.isclose(np.std(x), 0) or np.isclose(np.std(y), 0):
            return
        
        # Calculate ellipse parameters
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        
        # Ellipse radii
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        
        # Create ellipse
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                        facecolor=color, edgecolor=color, alpha=alpha)
        
        # Scale and position
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)
        
        # Apply transformation
        transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
        ellipse.set_transform(transf + ax.transData)
        
        return ax.add_patch(ellipse)
        
    except Exception as e:
        logger.warning(f"Failed to add confidence ellipse: {e}")
        return None