from pathlib import Path
import traceback
from typing import Optional, Dict, Any, List
import logging

from src.analysis.data_structures import GroupTensors, NormCfg, DEFAULT_NORMALIZATION_CONFIG
from src.analysis.load_and_batch_trajectory_data import load_and_batch_trajectory_data

from .plotters.plot_trajectory_atlas_umap import plot_trajectory_atlas_umap
from .plotters.plot_trajectory_corridor_atlas import plot_trajectory_corridor_atlas 


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def group_tensors_visualizer(
        # Pass EITHER group_tensors OR latents_dir
        group_tensors: Optional[GroupTensors],
        latents_dir: Optional[Path] = None,
        output_dir: Path = None,
        prompt_groups: Optional[List[str]] = None,
        norm_cfg: Optional[NormCfg] = DEFAULT_NORMALIZATION_CONFIG
):
    # Visualize the latent tensors using your preferred method
    logger.info(f"Visualizing group tensors {norm_cfg}")
    if not group_tensors and not latents_dir:
        logger.warning("⚠️ No group tensors or latents directory provided, returning early")
        return
    
    if not output_dir:
        logger.warning("⚠️ No output directory provided, returning early")
        return
    
    if not group_tensors:
        logger.info(f"Loading group tensors from {latents_dir}")
        # Load group tensors from the directory
        group_tensors = load_and_batch_trajectory_data(
            latents_dir, 
            prompt_groups=prompt_groups
        )
    
    if group_tensors is None:
        logger.info("No group tensors loaded, returning early")
        return
    
    try:
        output_path = plot_trajectory_atlas_umap(group_tensors, viz_dir=output_dir, norm_cfg=norm_cfg)
        logger.info(f"🗺️  Atlas UMAP plot saved to {output_path}")

        output_path = plot_trajectory_corridor_atlas(group_tensors, viz_dir=output_dir, norm_cfg=norm_cfg)
        logger.info(f"💠 Trajectory corridor plot saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to plot latent tensors: {e}")
        logger.error(traceback.format_exc())