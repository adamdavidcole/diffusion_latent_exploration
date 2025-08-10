import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, fields

from src.visualization.visualization_config import VisualizationConfig
from src.analysis.data_structures import LatentTrajectoryAnalysis

from .plotters.test import test_visualizer
from .plotters.plot_trajectory_spatial_evolution import plot_trajectory_spatial_evolution
from .plotters.plot_cross_trajectory_synchronization import plot_cross_trajectory_synchronization
from .plotters.plot_temporal_momentum import plot_temporal_momentum_analysis
from .plotters.plot_phase_transition_detection import plot_phase_transition_detection
from .plotters.plot_temporal_frequency_signatures import plot_temporal_frequency_signatures
from .plotters.plot_group_separability import plot_group_separability
from .plotters.plot_spatial_progression_patterns import plot_spatial_progression_patterns
from .plotters.plot_edge_formation_trends_dashboard import plot_edge_formation_trends_dashboard
from .plotters.plot_edge_density_evolution import plot_edge_density_evolution
from .plotters.plot_spatial_coherence_patterns import plot_spatial_coherence_patterns
from .plotters.plot_individual_video_coherence_dashboard import plot_individual_video_coherence_dashboard
from .plotters.plot_spatial_coherence_individual import plot_spatial_coherence_individual
from .plotters.plot_research_radar_chart import plot_research_radar_chart
from .plotters.plot_endpoint_constellations import plot_endpoint_constellations
from .plotters.plot_temporal_stability_windows import plot_temporal_stability_windows
from .plotters.plot_channel_evolution_patterns import plot_channel_evolution_patterns
from .plotters.plot_global_structure_analysis import plot_global_structure_analysis
from .plotters.plot_information_content_analysis import plot_information_content_analysis
from .plotters.plot_complexity_measures import plot_complexity_measures
from .plotters.plot_statistical_significance import plot_statistical_significance
from .plotters.plot_temporal_analysis import plot_temporal_analysis
from .plotters.plot_convex_hull_analysis import plot_convex_hull_analysis

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
            output_path = plot_trajectory_spatial_evolution(results, self.output_dir, labels_map=labels_map)
            self.logger.info(f"Saved trajectory spatial evolution plot to {output_path}")

            # # 2. Cross-Trajectory Synchronization
            output_path = plot_cross_trajectory_synchronization(results, self.output_dir)
            self.logger.info(f"Saved cross-trajectory synchronization plot to {output_path}")

            # # 3. Temporal Momentum Analysis
            output_path = plot_temporal_momentum_analysis(results, self.output_dir, labels_map=labels_map, viz_config=self.viz_config)
            self.logger.info(f"Saved temporal momentum analysis plot to {output_path}")

            # # 4. Phase Transition Detection
            output_path = plot_phase_transition_detection(results, self.output_dir, labels_map=labels_map)
            self.logger.info(f"Saved phase transition detection plot to {output_path}")

            # # 5. Temporal Frequency Signatures
            output_path = plot_temporal_frequency_signatures(results, self.output_dir, viz_config=self.viz_config)
            self.logger.info(f"Saved temporal frequency signatures plot to {output_path}")
            
            # # 6. Group Separability Analysis
            output_path = plot_group_separability(results, self.output_dir)
            self.logger.info(f"Saved group separability plot to {output_path}")
            
            # # 7. Spatial Progression Patterns
            output_path = plot_spatial_progression_patterns(results, self.output_dir, labels_map=labels_map)
            self.logger.info(f"Saved spatial progression patterns plot to {output_path}")
            
            # # 8a. Edge Formation Trends Dashboard (extracted from spatial progression)
            output_path = plot_edge_formation_trends_dashboard(results, self.output_dir, labels_map=labels_map)
            self.logger.info(f"Saved edge formation trends dashboard plot to {output_path}")
            
            # # 8b. Edge Density Evolution
            output_path = plot_edge_density_evolution(results, self.output_dir, viz_config=self.viz_config)
            self.logger.info(f"Saved edge density evolution plot to {output_path}")
            
            # # 9a. Spatial Coherence Patterns
            output_path = plot_spatial_coherence_patterns(results, self.output_dir, logger=self.logger)
            self.logger.info(f"Saved spatial coherence patterns plot to {output_path}")
            
            # # 9b. Individual Video Coherence Dashboard (extracted from spatial coherence)
            output_path = plot_individual_video_coherence_dashboard(results, self.output_dir)
            self.logger.info(f"Saved individual video coherence dashboard plot to {output_path}")
            
            # # 9c. Spatial Coherence Individual Trajectories (new detailed view)
            output_path = plot_spatial_coherence_individual(results, self.output_dir, viz_config=self.viz_config)
            self.logger.info(f"Saved spatial coherence individual trajectories to {output_path}")
            
            # # 9d. Research-focused Radar Chart (multi-metric profiles)
            output_path = plot_research_radar_chart(results, self.output_dir, results_full=None, viz_config=self.viz_config)
            self.logger.info(f"Saved research radar chart plot to {output_path}")
            
            # # 9e. Endpoint Constellation Analysis (final latent space positions)
            output_path = plot_endpoint_constellations(results, self.output_dir, labels_map=labels_map, viz_config=self.viz_config)
            self.logger.info(f"Saved endpoint constellations plot to {output_path}")
            
            # # NEW: Advanced Geometric Analysis Visualizations
            # # 10. Convex Hull Volume Analysis
            output_path = plot_convex_hull_analysis(results, self.output_dir, viz_config=self.viz_config, labels_map=labels_map)
            self.logger.info(f"Saved convex hull volume analysis plot to {output_path}")
            

    
            
            # # 11. Functional PCA Analysis
            # self._plot_functional_pca_analysis(results, viz_dir)
            
            # # 12. Individual Trajectory Geometry Dashboard
            # self._plot_individual_trajectory_geometry_dashboard(results, viz_dir)
            
            # # 13. Intrinsic Dimension Analysis
            # self._plot_intrinsic_dimension_analysis(results, viz_dir)
            
            # # 15. Temporal Stability Windows
            # output_path = plot_temporal_stability_windows(results, self.output_dir, viz_config=self.viz_config, labels_map=labels_map)
            # self.logger.info(f"Saved temporal stability windows plot to {output_path}")
            
            # # 16. Channel Evolution Patterns
            output_path = plot_channel_evolution_patterns(results, self.output_dir, viz_config=self.viz_config, labels_map=labels_map)
            self.logger.info(f"Saved channel evolution patterns plot to {output_path}")
            
            # # 17. Global Structure Analysis
            output_path = plot_global_structure_analysis(results, self.output_dir, viz_config=self.viz_config, labels_map=labels_map)
            self.logger.info(f"Saved global structure analysis plot to {output_path}")
            
            # # 18. Information Content Analysis
            output_path = plot_information_content_analysis(results, self.output_dir, viz_config=self.viz_config, labels_map=labels_map)
            self.logger.info(f"Saved information content analysis plot to {output_path}")
            
            # # 19. Complexity Measures
            output_path = plot_complexity_measures(results, self.output_dir, viz_config=self.viz_config, labels_map=labels_map)
            self.logger.info(f"Saved complexity measures plot to {output_path}")
            
            # # 20. Statistical Significance Tests
            output_path = plot_statistical_significance(results, self.output_dir, viz_config=self.viz_config, labels_map=labels_map)
            self.logger.info(f"Saved statistical significance plot to {output_path}")
            
            # # 21. Temporal Analysis Visualizations
            output_path = plot_temporal_analysis(results, self.output_dir, viz_config=self.viz_config, labels_map=labels_map)
            self.logger.info(f"Saved temporal analysis plot to {output_path}")

            # # 21. Structural Analysis Visualizations
            # self._plot_structural_analysis(results, viz_dir)
            
            # # 22. Paired-seed significance
            # self._plot_paired_seed_significance(results, viz_dir)
            
            # # 23. Comprehensive Dashboard
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

 