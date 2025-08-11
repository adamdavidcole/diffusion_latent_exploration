import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

@dataclass
class VisualizationConfig:
    """Configuration for visualization generation with consistent design system."""
    # Figure settings
    dpi: int = 300
    figsize_standard: tuple = (15, 12)
    figsize_wide: tuple = (20, 8)
    figsize_dashboard: tuple = (20, 24)
    save_format: str = "png"
    bbox_inches: str = "tight"
    
    # Design system settings
    color_palette: str = "husl"
    alpha: float = 0.8
    linewidth: float = 2.0
    markersize: float = 3.0
    
    # Typography settings
    fontsize_labels: int = 8
    fontsize_legend: int = 9
    fontsize_title: int = 10
    fontweight_title: str = "bold"
    
    # Layout settings
    legend_bbox_anchor: tuple = (1.05, 1)
    legend_loc: str = "upper left"
    grid_alpha: float = 0.3
    
    # Color variations for different plot types
    heatmap_cmap: str = "RdYlBu_r"
    diverging_cmap: str = "coolwarm"
    sequential_cmap: str = "YlOrRd"
    
    step_cmap: str = "viridis"
    name_cmap: str = "tab10"

    def get_colors(self, n_groups: int) -> list:
        """Get color palette for n groups."""
        return sns.color_palette(self.color_palette, n_groups)
    
    def apply_style_settings(self):
        """Apply style settings to matplotlib."""
        plt.rcParams['figure.dpi'] = self.dpi
        plt.rcParams['savefig.dpi'] = self.dpi
        plt.rcParams['font.size'] = self.fontsize_labels
        plt.rcParams['axes.titlesize'] = self.fontsize_title
        plt.rcParams['axes.labelsize'] = self.fontsize_labels
        plt.rcParams['legend.fontsize'] = self.fontsize_legend