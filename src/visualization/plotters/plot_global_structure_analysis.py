"""
Global structure analysis plotting functionality.
Extracted from LatentTrajectoryAnalyzer.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from src.analysis.data_structures import LatentTrajectoryAnalysis
from src.config.visualization_config import VisualizationConfig


def plot_global_structure_analysis(results: LatentTrajectoryAnalysis, viz_dir: Path, 
                                  viz_config: VisualizationConfig = None, 
                                  labels_map: dict = None, **kwargs) -> Path:
    """Plot global structure analysis with comprehensive error handling."""
    # Set defaults for optional parameters
    if viz_config is None:
        viz_config = VisualizationConfig()
    
    output_path = viz_dir / f"global_structure_analysis.{viz_config.save_format}"
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🔧 Starting global structure analysis visualization...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=viz_config.figsize_standard)
        
        # Validate data structure
        if 'trajectory_global_evolution' not in results.global_structure:
            logger.error("❌ Missing 'trajectory_global_evolution' in global_structure")
            raise KeyError("Global evolution data not found in results")
        
        global_data = results.global_structure['trajectory_global_evolution']
        if not global_data:
            logger.warning("⚠️ Global structure evolution data is empty")
            
        sorted_group_names = sorted(global_data.keys())
        logger.info(f"📊 Found {len(sorted_group_names)} groups: {sorted_group_names}")
        
        colors = viz_config.get_colors(len(sorted_group_names))
        
        # Debug: Log data structure for first group
        if sorted_group_names:
            sample_group = sorted_group_names[0]
            sample_data = global_data[sample_group]
            logger.info(f"🔍 Global structure data for '{sample_group}': {list(sample_data.keys())}")
            
            # Log sample values
            for key, value in sample_data.items():
                if isinstance(value, (list, np.ndarray)):
                    logger.info(f"  {key}: length={len(value)}, sample={value[:3] if len(value) > 0 else 'empty'}")
                else:
                    logger.info(f"  {key}: {type(value).__name__}={value}")
        
        # Apply labels_map if provided
        def get_label(group_name):
            if labels_map and group_name in labels_map:
                return labels_map[group_name]
            return group_name
        
        # Plot 1: Variance progression
        variance_count = 0
        for i, group_name in enumerate(sorted_group_names):
            try:
                data = global_data[group_name]
                variance_progression = data.get('variance_progression', [])
                if variance_progression and len(variance_progression) > 0:
                    steps = list(range(len(variance_progression)))
                    label = get_label(group_name)

                    ax1.plot(steps, variance_progression, 'o-', label=label, 
                            alpha=viz_config.alpha, color=colors[i], 
                            linewidth=viz_config.linewidth, markersize=viz_config.markersize)
                    variance_count += 1
                    logger.debug(f"✅ Plotted variance for '{group_name}': {len(variance_progression)} steps")
                else:
                    logger.warning(f"⚠️ Empty variance progression for '{group_name}'")
            except Exception as e:
                logger.error(f"❌ Error plotting variance for '{group_name}': {e}")
        
        logger.info(f"📈 Successfully plotted variance for {variance_count}/{len(sorted_group_names)} groups")
        
        if variance_count > 0:
            ax1.set_xlabel('Diffusion Step', fontsize=viz_config.fontsize_labels)
            ax1.set_ylabel('Global Variance', fontsize=viz_config.fontsize_labels)
            ax1.set_title('Global Variance Progression', 
                         fontsize=viz_config.fontsize_title, fontweight=viz_config.fontweight_title)
            ax1.legend(fontsize=viz_config.fontsize_legend, 
                      bbox_to_anchor=viz_config.legend_bbox_anchor, loc=viz_config.legend_loc)
            ax1.grid(True, alpha=viz_config.grid_alpha)
            ax1.tick_params(axis='both', labelsize=viz_config.fontsize_labels)
        else:
            ax1.text(0.5, 0.5, 'No variance progression data available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Global Variance Progression (No Data)')
            logger.warning("⚠️ Plot 1: No variance data available")
        
        # Plot 2: Magnitude progression
        magnitude_count = 0
        for i, group_name in enumerate(sorted_group_names):
            try:
                data = global_data[group_name]
                magnitude_progression = data.get('magnitude_progression', [])
                if magnitude_progression and len(magnitude_progression) > 0:
                    steps = list(range(len(magnitude_progression)))
                    label = get_label(group_name)

                    ax2.plot(steps, magnitude_progression, 's-', label=label, 
                            alpha=viz_config.alpha, color=colors[i], 
                            linewidth=viz_config.linewidth, markersize=viz_config.markersize)
                    magnitude_count += 1
                    logger.debug(f"✅ Plotted magnitude for '{group_name}': {len(magnitude_progression)} steps")
                else:
                    logger.warning(f"⚠️ Empty magnitude progression for '{group_name}'")
            except Exception as e:
                logger.error(f"❌ Error plotting magnitude for '{group_name}': {e}")
        
        logger.info(f"📈 Successfully plotted magnitude for {magnitude_count}/{len(sorted_group_names)} groups")
        
        if magnitude_count > 0:
            ax2.set_xlabel('Diffusion Step', fontsize=viz_config.fontsize_labels)
            ax2.set_ylabel('Global Magnitude', fontsize=viz_config.fontsize_labels)
            ax2.set_title('Global Magnitude Progression', 
                         fontsize=viz_config.fontsize_title, fontweight=viz_config.fontweight_title)
            ax2.legend(fontsize=viz_config.fontsize_legend, 
                      bbox_to_anchor=viz_config.legend_bbox_anchor, loc=viz_config.legend_loc)
            ax2.grid(True, alpha=viz_config.grid_alpha)
            ax2.tick_params(axis='both', labelsize=viz_config.fontsize_labels)
        else:
            ax2.text(0.5, 0.5, 'No magnitude progression data available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Global Magnitude Progression (No Data)')
            logger.warning("⚠️ Plot 2: No magnitude data available")
        
        # Plot 3: Convergence patterns
        try:
            if 'convergence_patterns' not in results.global_structure:
                logger.error("❌ Missing 'convergence_patterns' in global_structure")
                raise KeyError("Convergence patterns data not found")
            
            convergence_data = results.global_structure['convergence_patterns']
            diversity_scores = []
            valid_groups = []
            
            for group in sorted_group_names:
                try:
                    if group in convergence_data and 'overall_diversity_score' in convergence_data[group]:
                        score = convergence_data[group]['overall_diversity_score']
                        diversity_scores.append(score)
                        valid_groups.append(group)
                        logger.debug(f"✅ Added diversity score for '{group}': {score}")
                    else:
                        logger.warning(f"⚠️ Missing diversity score for '{group}'")
                except Exception as e:
                    logger.error(f"❌ Error processing diversity score for '{group}': {e}")
            
            if diversity_scores:
                bars = ax3.bar(valid_groups, diversity_scores, alpha=viz_config.alpha, 
                              color=colors[:len(valid_groups)])
                ax3.set_xlabel('Prompt Group', fontsize=viz_config.fontsize_labels)
                ax3.set_ylabel('Diversity Score', fontsize=viz_config.fontsize_labels)
                ax3.set_title('Overall Trajectory Diversity', 
                             fontsize=viz_config.fontsize_title, fontweight=viz_config.fontweight_title)
                ax3.tick_params(axis='x', rotation=45, labelsize=viz_config.fontsize_labels)
                ax3.tick_params(axis='y', labelsize=viz_config.fontsize_labels)
                ax3.grid(True, alpha=viz_config.grid_alpha)
                
                # Add value labels
                for bar, score in zip(bars, diversity_scores):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + max(diversity_scores) * 0.01,
                            f'{score:.3f}', ha='center', va='bottom', fontsize=viz_config.fontsize_labels)
                
                logger.info(f"✅ Created diversity plot with {len(diversity_scores)} groups")
            else:
                ax3.text(0.5, 0.5, 'No diversity score data available', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Overall Trajectory Diversity (No Data)')
                logger.warning("⚠️ Plot 3: No diversity data available")
                
        except Exception as e:
            logger.error(f"❌ Error in convergence patterns plot: {e}")
            ax3.text(0.5, 0.5, f'Convergence analysis failed:\n{str(e)}', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Convergence Patterns (Error)')
        
        # Plot 4: Variance vs Magnitude correlation
        try:
            if global_data:
                final_variances = []
                final_magnitudes = []
                valid_scatter_groups = []
                
                for group_name in sorted_group_names:
                    try:
                        data = global_data[group_name]
                        var_prog = data.get('variance_progression', [])
                        mag_prog = data.get('magnitude_progression', [])
                        if var_prog and mag_prog and len(var_prog) > 0 and len(mag_prog) > 0:
                            final_variances.append(var_prog[-1])
                            final_magnitudes.append(mag_prog[-1])
                            valid_scatter_groups.append(group_name)
                            logger.debug(f"✅ Added scatter point for '{group_name}': var={var_prog[-1]}, mag={mag_prog[-1]}")
                        else:
                            logger.warning(f"⚠️ Missing final values for '{group_name}'")
                    except Exception as e:
                        logger.error(f"❌ Error processing scatter data for '{group_name}': {e}")
                
                if final_variances and final_magnitudes:
                    scatter = ax4.scatter(final_variances, final_magnitudes, s=100, alpha=viz_config.alpha, 
                                        c=colors[:len(final_variances)], edgecolors='black', linewidth=1)
                    for i, group in enumerate(valid_scatter_groups):
                        ax4.annotate(group, (final_variances[i], final_magnitudes[i]), 
                                   xytext=(3, 3), textcoords='offset points', 
                                   fontsize=viz_config.fontsize_labels, fontweight='bold')
                    
                    ax4.set_xlabel('Final Global Variance', fontsize=viz_config.fontsize_labels)
                    ax4.set_ylabel('Final Global Magnitude', fontsize=viz_config.fontsize_labels)
                    ax4.set_title('Final State: Variance vs Magnitude', 
                                 fontsize=viz_config.fontsize_title, fontweight=viz_config.fontweight_title)
                    ax4.grid(True, alpha=viz_config.grid_alpha)
                    ax4.tick_params(axis='both', labelsize=viz_config.fontsize_labels)
                    
                    logger.info(f"✅ Created scatter plot with {len(final_variances)} points")
                else:
                    ax4.text(0.5, 0.5, 'No final state data available', 
                            ha='center', va='center', transform=ax4.transAxes)
                    ax4.set_title('Final State Analysis (No Data)')
                    logger.warning("⚠️ Plot 4: No scatter data available")
            else:
                ax4.text(0.5, 0.5, 'No global data available', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Final State Analysis (No Data)')
                logger.warning("⚠️ Plot 4: No global data available")
                
        except Exception as e:
            logger.error(f"❌ Error in final state scatter plot: {e}")
            ax4.text(0.5, 0.5, f'Final state analysis failed:\n{str(e)}', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Final State Analysis (Error)')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=viz_config.dpi, bbox_inches=viz_config.bbox_inches)
        plt.close()
        
        logger.info(f"✅ Global structure analysis visualization saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Critical error in global structure analysis visualization: {e}")
        logger.exception("Full traceback:")
        
        # Create a fallback error visualization
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, f'Global Structure Analysis Visualization Failed\n\nError: {str(e)}\n\nCheck logs for details', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
            ax.set_title('Global Structure Analysis - Error')
            ax.axis('off')
            
            plt.tight_layout()
            error_output_path = viz_dir / f"global_structure_analysis_ERROR.{viz_config.save_format}"
            plt.savefig(error_output_path, dpi=viz_config.dpi, bbox_inches=viz_config.bbox_inches)
            plt.close()
            logger.info(f"💥 Error visualization saved to: {error_output_path}")
        except:
            logger.error("Failed to create even the error visualization")
        
        raise
    plt.close()
    
    return output_path
