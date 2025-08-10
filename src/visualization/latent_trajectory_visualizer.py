import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, fields



from .plotters.test import test_visualizer
from .plotters.plot_trajectory_spatial_evolution import plot_trajectory_spatial_evolution
from .plotters.plot_cross_trajectory_synchronization import plot_cross_trajectory_synchronization

from src.analysis.data_structures import LatentTrajectoryAnalysis

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


class LatentTrajectoryVisualizer:
    def __init__(
        self, 
        batch_dir, 
        output_dir,
        viz_config: Optional[VisualizationConfig] = None,
        use_prompt_variation_text_label: Optional[bool] = False
    ):
        self.batch_dir = batch_dir
        self.output_dir = output_dir
        
        self.viz_config = viz_config or VisualizationConfig()
        self.viz_config.apply_style_settings()

        self.use_prompt_variation_text_label = use_prompt_variation_text_label

        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        handler = logging.FileHandler(self.output_dir / "visualization.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def visualize(self):
        test_visualizer(self.batch_dir, self.output_dir)

    def _get_prompt_group_label(self, results: 'LatentTrajectoryAnalysis', group_name: str) -> str:
        """Get the label for a prompt group based on instance configuration."""
        if self.use_prompt_variation_text_label:
            # Safely navigate the nested dictionary
            prompt_meta = results.analysis_metadata.get('prompt_metadata', {})
            group_meta = prompt_meta.get(group_name, {})
            return group_meta.get('prompt_var_text', group_name)
        return group_name

    def create_comprehensive_visualizations(self, results: 'LatentTrajectoryAnalysis'):
        """Create comprehensive visualizations for all key statistical analyses."""
        try:
            import seaborn as sns
            
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            self.output_dir.mkdir(exist_ok=True)
            
            self.logger.info("Creating comprehensive analysis visualizations...")

            test_visualizer(self.batch_dir, self.output_dir)

            field_names = [f.name for f in fields(results)]
            self.logger.info(f"Loaded results fields: {field_names}")

            all_group_names = results.temporal_analysis.keys()
            labels_map = {
                name: self._get_prompt_group_label(results, name)
                for name in all_group_names
            }
            self.logger.info(f"labels_map: {labels_map}")


            # # 1. Trajectory Spatial Evolution (U-shaped pattern)
            output_path = plot_trajectory_spatial_evolution(results, self.output_dir, labels_map)
            self.logger.info(f"Saved trajectory spatial evolution plot to {output_path}")   

            # # 2. Cross-Trajectory Synchronization
            plot_cross_trajectory_synchronization(results, self.output_dir)
            
            # # 3. Temporal Momentum Analysis
            # self._plot_temporal_momentum_analysis(results, viz_dir)
            
            # # 4. Phase Transition Detection
            # self._plot_phase_transition_detection(results, viz_dir)
            
            # # 5. Temporal Frequency Signatures
            # self._plot_temporal_frequency_signatures(results, viz_dir)
            
            # # 6. Group Separability Analysis
            # self._plot_group_separability(results, viz_dir)
            
            # # 7. Spatial Progression Patterns
            # self._plot_spatial_progression_patterns(results, viz_dir)
            
            # # 8. Edge Density Evolution
            # self._plot_edge_density_evolution(results, viz_dir)
            
            # # 8b. Edge Formation Trends Dashboard (extracted from spatial progression)
            # self._plot_edge_formation_trends_dashboard(results, viz_dir)
            
            # # 9. Spatial Coherence Patterns
            # self._plot_spatial_coherence_patterns(results, viz_dir)
            
            # # 9b. Individual Video Coherence Dashboard (extracted from spatial coherence)
            # self._plot_individual_video_coherence_dashboard(results, viz_dir)
            
            # # 9c. Spatial Coherence Individual Trajectories (new detailed view)
            # self._plot_spatial_coherence_individual(results, viz_dir)
            
            # # 9d. Research-focused Radar Chart (multi-metric profiles)
            # self._plot_research_radar_chart(results, viz_dir)
            
            # # 9e. Endpoint Constellation Analysis (final latent space positions)
            # self._plot_endpoint_constellations(results, viz_dir)
            
            # # NEW: Advanced Geometric Analysis Visualizations
            # # 10. Convex Hull Volume Analysis
            # self._plot_convex_hull_analysis(results, viz_dir)
            
            # # 11. Functional PCA Analysis
            # self._plot_functional_pca_analysis(results, viz_dir)
            
            # # 12. Individual Trajectory Geometry Dashboard
            # self._plot_individual_trajectory_geometry_dashboard(results, viz_dir)
            
            # # 13. Intrinsic Dimension Analysis
            # self._plot_intrinsic_dimension_analysis(results, viz_dir)
            
            # # 14. Temporal Stability Windows
            # self._plot_temporal_stability_windows(results, viz_dir)
            
            # # 15. Channel Evolution Patterns
            # self._plot_channel_evolution_patterns(results, viz_dir)
            
            # # 16. Global Structure Analysis
            # self._plot_global_structure_analysis(results, viz_dir)
            
            # # 17. Information Content Analysis
            # self._plot_information_content_analysis(results, viz_dir)
            
            # # 18. Complexity Measures
            # self._plot_complexity_measures(results, viz_dir)
            
            # # 19. Statistical Significance Tests
            # self._plot_statistical_significance(results, viz_dir)
            
            # # 16. Temporal Analysis Visualizations
            # self._plot_temporal_analysis(results, viz_dir)
            
            # # 17. Structural Analysis Visualizations
            # self._plot_structural_analysis(results, viz_dir)
            
            # # Paired-seed significance
            # self._plot_paired_seed_significance(results, viz_dir)
            
            # # 18. Comprehensive Dashboard
            # self._plot_comprehensive_analysis_dashboard(results, viz_dir)
            # self._plot_trajectory_atlas_umap(results, viz_dir, self.group_tensors)

            # self._plot_log_volume_delta_panel(results, viz_dir)

            # self._create_batch_image_grid(results, viz_dir)

            # batch_image_grid_path = self._get_batch_image_grid_path()
            # self._plot_comprehensive_analysis_insight_board(results, viz_dir, results_full=None, video_grid_path=batch_image_grid_path)

            # self._plot_trajectory_corridor_atlas(results, viz_dir)


            # self.logger.info(f"✅ Visualizations saved to: {viz_dir}")
            
        except Exception as e:
            import traceback
            self.logger.error(f"Visualization creation failed: {e}")
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")

 