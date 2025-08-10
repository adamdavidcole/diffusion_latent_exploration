from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

import logging

from src.analysis.data_structures import LatentTrajectoryAnalysis

logger = logging.getLogger(__name__)

def plot_phase_transition_detection(results: LatentTrajectoryAnalysis, viz_dir: Path, labels_map: Dict[str, str]) -> Path:
    """Plot phase transition patterns."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    phase_data = results.temporal_coherence['phase_transition_detection']
    group_names = sorted(phase_data.keys())
    colors = sns.color_palette("husl", len(group_names))
    
    # Debug: Check data structure
    sample_group = group_names[0]
    sample_data = phase_data[sample_group]
    logger.info(f"Phase transition data structure for {sample_group}: {list(sample_data.keys())}")
    if 'p75_transitions' in sample_data:
        p75_shape = np.array(sample_data['p75_transitions']).shape
        logger.info(f"p75_transitions shape: {p75_shape}")
    
    # Plot 1: 75th percentile transitions
    for i, group_name in enumerate(group_names):
        data = phase_data[group_name]
        p75_transitions = data['p75_transitions']
        
        # Handle different data structures
        if isinstance(p75_transitions, (list, np.ndarray)):
            p75_array = np.array(p75_transitions)
            if p75_array.ndim > 1:
                # If 2D, take mean across first dimension (videos)
                p75_transitions = np.mean(p75_array, axis=0)
            else:
                p75_transitions = p75_array
        
        steps = list(range(len(p75_transitions)))
        label = labels_map[group_name]
        ax1.plot(steps, p75_transitions, 'o-', label=label, 
                color=colors[i], alpha=0.8, linewidth=2, markersize=4)
    
    ax1.set_xlabel('Diffusion Step')
    ax1.set_ylabel('Mean Transition Count')
    ax1.set_title('Phase Transitions (75th Percentile)\n(Moderate Changes - Average per Group)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: 95th percentile transitions (major changes)
    for i, group_name in enumerate(group_names):
        data = phase_data[group_name]
        p95_transitions = data['p95_transitions']
        
        # Handle different data structures
        if isinstance(p95_transitions, (list, np.ndarray)):
            p95_array = np.array(p95_transitions)
            if p95_array.ndim > 1:
                # If 2D, take mean across first dimension (videos)
                p95_transitions = np.mean(p95_array, axis=0)
            else:
                p95_transitions = p95_array
        
        steps = list(range(len(p95_transitions)))
        label = labels_map[group_name]
        ax2.plot(steps, p95_transitions, '^-', label=label, 
                color=colors[i], alpha=0.8, linewidth=2, markersize=4)
    
    ax2.set_xlabel('Diffusion Step')
    ax2.set_ylabel('Mean Transition Count')
    ax2.set_title('Major Phase Transitions (95th Percentile)\n(Dramatic Changes - Average per Group)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Transition intensity heatmap
    p90_data_list = []
    for group_name in group_names:
        p90_transitions = phase_data[group_name]['p90_transitions']
        p90_array = np.array(p90_transitions)
        
        # Handle different data structures
        if p90_array.ndim > 1:
            # If shape is (steps, features), take mean across features
            if p90_array.shape[1] > p90_array.shape[0]:
                # Likely (features, steps) - transpose and take mean
                p90_transitions = np.mean(p90_array.T, axis=1)
            else:
                # Likely (steps, features) - take mean across features
                p90_transitions = np.mean(p90_array, axis=1)
        else:
            p90_transitions = p90_array
            
        p90_data_list.append(p90_transitions)
    
    # Create heatmap matrix
    p90_data = np.array(p90_data_list)
    
    im = ax3.imshow(p90_data, cmap='YlOrRd', aspect='auto')
    ax3.set_yticks(range(len(group_names)))
    ax3.set_yticklabels(group_names)
    ax3.set_xlabel('Diffusion Step')
    ax3.set_ylabel('Prompt Group')
    ax3.set_title('Phase Transition Intensity Map\n(90th Percentile)')
    plt.colorbar(im, ax=ax3, label='Transition Count')
    
    # Plot 4: Total transitions by group with better calculation
    total_transitions = []
    for group_name in group_names:
        data = phase_data[group_name]
        
        # Calculate totals more carefully
        p75_total = np.sum(np.array(data['p75_transitions']).flatten())
        p90_total = np.sum(np.array(data['p90_transitions']).flatten())
        p95_total = np.sum(np.array(data['p95_transitions']).flatten())
        
        total = p75_total + p90_total + p95_total
        total_transitions.append(total)
        
        # Debug output
        logger.info(f"{group_name}: p75={p75_total:.2f}, p90={p90_total:.2f}, p95={p95_total:.2f}, total={total:.2f}")
    
    bars = ax4.bar(group_names, total_transitions, alpha=0.7, 
                    color=sns.color_palette("rocket", len(group_names)))
    ax4.set_xlabel('Prompt Group')
    ax4.set_ylabel('Total Transition Events')
    ax4.set_title('Overall Phase Transition Activity')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, total in zip(bars, total_transitions):
        height = bar.get_height()
        if total_transitions:  # Avoid division by zero
            max_total = max(total_transitions)
            ax4.text(bar.get_x() + bar.get_width()/2., height + max_total * 0.01,
                    f'{total:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_path = viz_dir / "phase_transition_detection.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path