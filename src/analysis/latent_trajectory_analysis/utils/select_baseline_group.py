from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)    


def select_baseline_group(
    self, prompt_groups: List[str], 
    strategy: str = "auto"
) -> str:
    """
    Select baseline group using different strategies for research comparison.
    
    Args:
        prompt_groups: List of prompt group names
        strategy: "auto", "empty_prompt", "first_class_specific", or "alphabetical"
    """
    if strategy == "empty_prompt":
        # Look for empty/no prompt - typically prompt_000 or similar
        empty_candidates = [p for p in prompt_groups if '000' in p or 'empty' in p.lower() or 'no_prompt' in p.lower()]
        if empty_candidates:
            baseline = sorted(empty_candidates)[0]
            logger.info(f"Selected empty prompt baseline: {baseline}")
            return baseline
    
    elif strategy == "first_class_specific":
        # Look for first class-specific prompt (e.g., "flower" vs more specific variants)
        # This would be prompt_001 in your flower specificity sequence
        sorted_groups = sorted(prompt_groups)
        if len(sorted_groups) > 1:
            baseline = sorted_groups[1]  # Second group (001) assuming 000 is empty
            logger.info(f"Selected first class-specific baseline: {baseline}")
            return baseline
    
    elif strategy == "alphabetical":
        baseline = sorted(prompt_groups)[0]
        logger.info(f"Selected alphabetical baseline: {baseline}")
        return baseline
    
    # Auto strategy: prefer empty prompt if available, otherwise alphabetical
    empty_candidates = [p for p in prompt_groups if '000' in p]
    if empty_candidates:
        baseline = sorted(empty_candidates)[0]
        logger.info(f"Auto-selected empty prompt baseline: {baseline}")
    else:
        baseline = sorted(prompt_groups)[0]
        logger.info(f"Auto-selected alphabetical baseline: {baseline}")
    
    return baseline