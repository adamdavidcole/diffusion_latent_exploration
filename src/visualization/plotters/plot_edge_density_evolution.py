import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from pathlib import Path

from src.analysis.data_structures import LatentTrajectoryAnalysis
from src.visualization.visualization_config import VisualizationConfig


def plot_edge_density_evolution(
    results: LatentTrajectoryAnalysis, 
    viz_dir: Path, 
    viz_config: VisualizationConfig = None
) -> Path:
    """Plot edge density evolution analysis with comprehensive error handling."""
    if viz_config is None:
        viz_config = VisualizationConfig()
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🔧 Starting edge density evolution visualization...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=viz_config.figsize_standard)
        
        # Validate data structure
        if 'edge_density_evolution' not in results.spatial_patterns:
            logger.error("❌ Missing 'edge_density_evolution' in spatial_patterns")
            raise KeyError("Edge density evolution data not found in results")
        
        edge_data = results.spatial_patterns['edge_density_evolution']
        if not edge_data:
            logger.warning("⚠️ Edge density evolution data is empty")
            
        prompt_names = sorted(edge_data.keys())
        logger.info(f"📊 Found {len(prompt_names)} prompt groups: {prompt_names}")
        
        colors = viz_config.get_colors(len(prompt_names))
        
        # Debug: Log data structure for first prompt
        if prompt_names:
            sample_prompt = prompt_names[0]
            sample_data = edge_data[sample_prompt]
            logger.info(f"🔍 Edge density data structure for '{sample_prompt}': {list(sample_data.keys())}")
            
            # Log sample values
            for key, value in sample_data.items():
                if isinstance(value, (list, np.ndarray)):
                    logger.info(f"  {key}: length={len(value)}, sample={value[:3] if len(value) > 0 else 'empty'}")
                else:
                    logger.info(f"  {key}: {type(value).__name__}={value}")
        
        # Plot 1: Edge density evolution over diffusion steps
        has_evolution_data = False
        evolution_count = 0
        
        for i, prompt_name in enumerate(prompt_names):
            try:
                if 'mean_evolution_pattern' in edge_data[prompt_name]:
                    evolution = edge_data[prompt_name]['mean_evolution_pattern']
                    if evolution and len(evolution) > 0:
                        steps = range(len(evolution))
                        ax1.plot(steps, evolution, 'o-', label=prompt_name, 
                                color=colors[i], alpha=viz_config.alpha, 
                                linewidth=viz_config.linewidth, markersize=viz_config.markersize)
                        has_evolution_data = True
                        evolution_count += 1
                        logger.debug(f"✅ Plotted evolution for '{prompt_name}': {len(evolution)} steps")
                    else:
                        logger.warning(f"⚠️ Empty evolution pattern for '{prompt_name}'")
                else:
                    logger.warning(f"⚠️ Missing 'mean_evolution_pattern' for '{prompt_name}'")
            except Exception as e:
                logger.error(f"❌ Error plotting evolution for '{prompt_name}': {e}")
        
        logger.info(f"📈 Successfully plotted evolution for {evolution_count}/{len(prompt_names)} prompts")
        
        if has_evolution_data:
            ax1.set_xlabel('Diffusion Step', fontsize=viz_config.fontsize_labels)
            ax1.set_ylabel('Mean Edge Density', fontsize=viz_config.fontsize_labels)
            ax1.set_title('Edge Density Evolution by Prompt\n(Mean Evolution Pattern)', 
                         fontsize=viz_config.fontsize_title, fontweight=viz_config.fontweight_title)
            ax1.legend(bbox_to_anchor=viz_config.legend_bbox_anchor, loc=viz_config.legend_loc, 
                      fontsize=viz_config.fontsize_legend)
            ax1.grid(True, alpha=viz_config.grid_alpha)
        else:
            ax1.text(0.5, 0.5, 'No edge density evolution data available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Edge Density Evolution (No Data)', 
                         fontsize=viz_config.fontsize_title, fontweight=viz_config.fontweight_title)
            logger.warning("⚠️ Plot 1: No evolution data available")
        
        # Plot 2: Evolution variability 
        variability_count = 0
        variabilities = []
        valid_prompts = []
        
        for prompt in prompt_names:
            try:
                if 'evolution_variability' in edge_data[prompt] and edge_data[prompt]['evolution_variability'] is not None:
                    var_data = edge_data[prompt]['evolution_variability']
                    if isinstance(var_data, (list, np.ndarray)):
                        scalar_var = np.mean(var_data)
                    else:
                        scalar_var = var_data
                    variabilities.append(scalar_var)
                    valid_prompts.append(prompt)
                    variability_count += 1
                    logger.debug(f"✅ Added variability for '{prompt}': {scalar_var}")
            except Exception as e:
                logger.error(f"❌ Error processing variability for '{prompt}': {e}")
        
        logger.info(f"📊 Successfully processed variability for {variability_count}/{len(prompt_names)} prompts")
        
        if variabilities:
            bars2 = ax2.bar(valid_prompts, variabilities, alpha=viz_config.alpha, 
                           color=colors[:len(valid_prompts)])
            ax2.set_xlabel('Prompt ID', fontsize=viz_config.fontsize_labels)
            ax2.set_ylabel('Evolution Variability', fontsize=viz_config.fontsize_labels)
            ax2.set_title('Edge Density Evolution Variability', 
                         fontsize=viz_config.fontsize_title, fontweight=viz_config.fontweight_title)
            ax2.tick_params(axis='x', rotation=45, labelsize=viz_config.fontsize_labels)
            ax2.grid(True, alpha=viz_config.grid_alpha)
            
            # Add value labels
            for bar, var in zip(bars2, variabilities):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(variabilities) * 0.01,
                        f'{var:.3f}', ha='center', va='bottom', fontsize=viz_config.fontsize_labels)
        else:
            ax2.text(0.5, 0.5, 'No evolution variability data available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Evolution Variability (No Data)', 
                         fontsize=viz_config.fontsize_title, fontweight=viz_config.fontweight_title)
            logger.warning("⚠️ Plot 2: No variability data available")
        
        # Plot 3: Edge formation trend
        trend_count = 0
        trends = []
        trend_labels = []
        valid_trend_prompts = []
        
        for prompt in prompt_names:
            try:
                if 'edge_formation_trend' in edge_data[prompt] and edge_data[prompt]['edge_formation_trend'] is not None:
                    trend_value = edge_data[prompt]['edge_formation_trend']
                    if isinstance(trend_value, str):
                        if trend_value.lower() in ['increasing', 'inc']:
                            numeric_trend = 1.0
                        elif trend_value.lower() in ['decreasing', 'dec']:
                            numeric_trend = -1.0
                        elif trend_value.lower() in ['stable', 'constant']:
                            numeric_trend = 0.0
                        else:
                            numeric_trend = 0.0
                        trend_labels.append(trend_value)
                    else:
                        numeric_trend = float(trend_value)
                        trend_labels.append(f'{numeric_trend:.3f}')
                    
                    trends.append(numeric_trend)
                    valid_trend_prompts.append(prompt)
                    trend_count += 1
                    logger.debug(f"✅ Added trend for '{prompt}': {trend_value} -> {numeric_trend}")
            except Exception as e:
                logger.error(f"❌ Error processing trend for '{prompt}': {e}")
        
        logger.info(f"📈 Successfully processed trends for {trend_count}/{len(prompt_names)} prompts")
        
        if trends:
            bars3 = ax3.bar(valid_trend_prompts, trends, alpha=viz_config.alpha, color=colors[:len(valid_trend_prompts)])
            ax3.set_xlabel('Prompt ID', fontsize=viz_config.fontsize_labels)
            ax3.set_ylabel('Edge Formation Trend', fontsize=viz_config.fontsize_labels)
            ax3.set_title('Edge Formation Trend Direction\n(+1=Increasing, 0=Stable, -1=Decreasing)', 
                         fontsize=viz_config.fontsize_title, fontweight=viz_config.fontweight_title)
            ax3.tick_params(axis='x', rotation=45, labelsize=viz_config.fontsize_labels)
            ax3.grid(True, alpha=viz_config.grid_alpha)
            
            # Add value labels
            for bar, label in zip(bars3, trend_labels):
                height = bar.get_height()
                y_offset = 0.05 if height >= 0 else -0.15
                ax3.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                        label, ha='center', va='bottom' if height >= 0 else 'top', 
                        fontsize=viz_config.fontsize_labels)
        else:
            ax3.text(0.5, 0.5, 'No edge formation trend data available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Edge Formation Trend (No Data)', 
                         fontsize=viz_config.fontsize_title, fontweight=viz_config.fontweight_title)
            logger.warning("⚠️ Plot 3: No trend data available")
        
        # Plot 4: Heatmap of edge evolution patterns
        if has_evolution_data:
            evolution_matrix = []
            valid_heatmap_prompts = []
            for prompt_name in prompt_names:
                try:
                    if 'mean_evolution_pattern' in edge_data[prompt_name] and edge_data[prompt_name]['mean_evolution_pattern']:
                        evolution = edge_data[prompt_name]['mean_evolution_pattern']
                        if evolution and len(evolution) > 0:
                            evolution_matrix.append(evolution)
                            valid_heatmap_prompts.append(prompt_name)
                except Exception as e:
                    logger.error(f"❌ Error adding to heatmap matrix for '{prompt_name}': {e}")
            
            if evolution_matrix:
                try:
                    evolution_array = np.array(evolution_matrix)
                    im = ax4.imshow(evolution_array, cmap=viz_config.sequential_cmap, aspect='auto')
                    ax4.set_yticks(range(len(valid_heatmap_prompts)))
                    ax4.set_yticklabels(valid_heatmap_prompts, fontsize=viz_config.fontsize_labels)
                    ax4.set_xlabel('Diffusion Step', fontsize=viz_config.fontsize_labels)
                    ax4.set_ylabel('Prompt ID', fontsize=viz_config.fontsize_labels)
                    ax4.set_title('Edge Density Evolution Heatmap\n(All Prompts)', 
                                 fontsize=viz_config.fontsize_title, fontweight=viz_config.fontweight_title)
                    plt.colorbar(im, ax=ax4, label='Edge Density')
                    logger.info(f"✅ Created heatmap with {len(evolution_matrix)} prompts")
                except Exception as e:
                    logger.error(f"❌ Error creating heatmap: {e}")
                    ax4.text(0.5, 0.5, f'Heatmap creation failed:\n{str(e)}', 
                            ha='center', va='center', transform=ax4.transAxes)
                    ax4.set_title('Edge Evolution Heatmap (Error)')
            else:
                ax4.text(0.5, 0.5, 'No evolution matrix data available', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Edge Evolution Heatmap (No Data)')
                logger.warning("⚠️ Plot 4: No heatmap data available")
        else:
            ax4.text(0.5, 0.5, 'No edge density evolution data available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Edge Evolution Heatmap (No Data)')
            logger.warning("⚠️ Plot 4: No evolution data for heatmap")
        
        plt.tight_layout()
        output_path = viz_dir / f"edge_density_evolution.{viz_config.save_format}"
        plt.savefig(output_path, dpi=viz_config.dpi, bbox_inches=viz_config.bbox_inches)
        plt.close()
        
        logger.info(f"✅ Edge density evolution visualization saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Critical error in edge density evolution visualization: {e}")
        logger.exception("Full traceback:")
        
        # Create a fallback error visualization
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, f'Edge Density Evolution Visualization Failed\n\nError: {str(e)}\n\nCheck logs for details', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
            ax.set_title('Edge Density Evolution - Error')
            ax.axis('off')
            
            plt.tight_layout()
            error_output_path = viz_dir / f"edge_density_evolution_ERROR.{viz_config.save_format}"
            plt.savefig(error_output_path, dpi=viz_config.dpi, bbox_inches=viz_config.bbox_inches)
            plt.close()
            logger.info(f"💥 Error visualization saved to: {error_output_path}")
        except:
            logger.error("Failed to create even the error visualization")
        
        raise

    return output_path
