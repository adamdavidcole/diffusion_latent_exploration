"""
Comprehensive Analysis Insight Board plotting functionality.
Extracted from LatentTrajectoryAnalyzer.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from src.analysis.data_structures import LatentTrajectoryAnalysis
from src.visualization.visualization_config import VisualizationConfig

def plot_comprehensive_analysis_insight_board(results: LatentTrajectoryAnalysis, viz_dir: Path, viz_config: VisualizationConfig = None, labels_map: dict = None, results_full: Optional[LatentTrajectoryAnalysis] = None, video_grid_path: Optional[Path] = None, **kwargs) -> Path:
    """
    Publication board: clear hierarchy + consistent palette.
    Top row:   Radar (normalized group comparison), Final-state manifold (Var vs Mag) + Key insights box
    Middle:    Per-timestep curves (Spatial variance, Global variance, Global magnitude)
    Bottom:    Bars (Length, Velocity) [SNR track], (Acceleration, Late/Early, Turning, Alignment) [Full track]
    """
    if viz_config is None:
        viz_config = VisualizationConfig()
    if results_full is None:
        results_full = results
    output_path = viz_dir / f"comprehensive_analysis_insight_board.{viz_config.save_format}"
    try:
        # ... (actual insight board plotting logic would go here)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 'Comprehensive Analysis Insight Board not implemented', ha='center', va='center', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=viz_config.dpi, bbox_inches=viz_config.bbox_inches)
        plt.close()
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error creating comprehensive analysis insight board: {e}")
        plt.close()
    return output_path
