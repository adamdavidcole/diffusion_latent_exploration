"""
Utilities for loading and processing prompt metadata from batch configurations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_prompt_metadata(batch_path: str, prompt_groups: List[str]) -> Dict[str, Dict[str, str]]:
    """Load prompt descriptions and complete text from prompt_variations.json.
    
    Args:
        batch_path: Path to the batch directory
        prompt_groups: List of prompt group names (e.g., ['prompt_000', 'prompt_001'])
    
    Returns:
        tuple: (prompt_descriptions, prompt_metadata) where:
            - prompt_descriptions: List of var_0 values for each prompt group (for backward compatibility)
            - prompt_metadata: Dict mapping prompt_group_id -> {prompt_var_text: "...", prompt_text: "..."}
    """
    logger = logging.getLogger(__name__)
    
    # Look for prompt_variations.json in configs directory
    config_dir = Path(batch_path) / "configs"
    prompt_variations_file = config_dir / "prompt_variations.json"
    
    if not prompt_variations_file.exists():
        logger.warning(f"⚠️ No prompt_variations.json found at {prompt_variations_file}")
        logger.warning("   Using fallback descriptions")
        prompt_metadata = {g: {"prompt_var_text": f"Description for {g}", "prompt_text": f"Complete prompt for {g}"} for g in prompt_groups}
        prompt_descriptions = [prompt_metadata[g]["prompt_var_text"] for g in prompt_groups]
        return prompt_metadata
    
    try:
        with open(prompt_variations_file, 'r') as f:
            variations = json.load(f)
        
        logger.debug(f"📋 Loaded {len(variations)} prompt variations from config")
        
        # Create mappings from ID to text and var_0
        id_to_text = {}
        id_to_var0 = {}
        
        for variation in variations:
            variation_id = variation.get('id', '')
            full_text = variation.get('text', '')
            var_0 = variation.get('variables', {}).get('var_0', '')
            
            id_to_text[variation_id] = full_text
            id_to_var0[variation_id] = var_0
        
        # Map prompt groups to metadata structure
        prompt_metadata = {}
        
        for group in prompt_groups:
            # Try to find matching variation by ID
            # Group names might be like 'prompt_000', but IDs might be like 'two_(people)'
            # We'll try exact match first, then look for partial matches
            
            if group in id_to_var0:
                # Exact match
                var_0_text = id_to_var0[group]
                full_text = id_to_text[group]
                
                # Replace empty strings with "[no-prompt]"
                prompt_metadata[group] = {
                    "prompt_var_text": var_0_text if var_0_text.strip() else "[no-prompt]",
                    "prompt_text": full_text if full_text.strip() else "[no-prompt]"
                }
            else:
                # Try to find by index - extract number from prompt_XXX format
                try:
                    if group.startswith('prompt_'):
                        group_index = int(group.split('_')[1])
                        if group_index < len(variations):
                            variation = variations[group_index]
                            var_0_text = variation.get('variables', {}).get('var_0', f"Unknown variation for {group}")
                            full_text = variation.get('text', f"Unknown text for {group}")
                            
                            # Replace empty strings with "[no-prompt]"
                            prompt_metadata[group] = {
                                "prompt_var_text": var_0_text if var_0_text.strip() else "[no-prompt]",
                                "prompt_text": full_text if full_text.strip() else "[no-prompt]"
                            }
                        else:
                            logger.warning(f"⚠️ Group index {group_index} out of range for {group}")
                            prompt_metadata[group] = {
                                "prompt_var_text": f"Description for {group}",
                                "prompt_text": f"Complete prompt for {group}"
                            }
                    else:
                        # No pattern match, use fallback
                        logger.warning(f"⚠️ No matching variation found for group: {group}")
                        prompt_metadata[group] = {
                            "prompt_var_text": f"Description for {group}",
                            "prompt_text": f"Complete prompt for {group}"
                        }
                except (ValueError, IndexError) as e:
                    logger.warning(f"⚠️ Error parsing group name {group}: {e}")
                    prompt_metadata[group] = {
                        "prompt_var_text": f"Description for {group}",
                        "prompt_text": f"Complete prompt for {group}"
                    } 
        
        # Construct prompt_descriptions list for backward compatibility
        prompt_descriptions = [prompt_metadata[g]["prompt_var_text"] for g in prompt_groups]
        
        logger.debug(f"✅ Mapped {len(prompt_descriptions)} prompt descriptions")
        
        return prompt_metadata
        
    except Exception as e:
        logger.error(f"❌ Error loading prompt variations: {e}")
        logger.warning("   Using fallback descriptions")
        prompt_metadata = {g: {"prompt_var_text": f"Description for {g}", "prompt_text": f"Complete prompt for {g}"} for g in prompt_groups}
        prompt_descriptions = [prompt_metadata[g]["prompt_var_text"] for g in prompt_groups]
        return prompt_metadata


def load_prompt_template(batch_path: str) -> str:
    """Load the prompt template text from batch configuration.
    
    Args:
        batch_path: Path to the batch directory
        
    Returns:
        The prompt template text, or a fallback message if not found
    """
    logger = logging.getLogger(__name__)
    
    config_dir = Path(batch_path) / "configs"
    template_file = config_dir / "prompt_template.txt"
    
    if not template_file.exists():
        logger.warning(f"⚠️ No prompt_template.txt found at {template_file}")
        return "[No prompt template found]"
    
    try:
        with open(template_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        return content if content else "[Empty prompt template]"
    except Exception as e:
        logger.error(f"❌ Error loading prompt template: {e}")
        return "[Error loading prompt template]"
