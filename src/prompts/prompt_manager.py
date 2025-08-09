"""
Enhanced Prompt Manager with support for nested bracket syntax.

This module provides PromptTemplate class that can handle complex nested bracket
syntax for generating prompt variations, including:
- Up to 3 layers of recursive depth
- Weighted options using (option:weight) syntax
- Proper validation and error handling
- Only '|' as separator (no '/' support)

Examples:
- Simple: "[cat|dog] on the beach"
- Weighted: "[cat:2.0|dog] on the beach"
- Nested: "[scary [movie|book]|funny [sign:2.0|poster]] scene"
- Complex: "[a [big [red|blue]|small [green|yellow]] [car|truck]|motorcycle] driving"
"""

import re
import itertools
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
@dataclass
class PromptVariation:
    """Single prompt variation with metadata."""
    text: str
    variation_id: str
    variables: Dict[str, str]
    weighted_text: Optional[str] = None  # For prompt weighting version
    

@dataclass 
class WeightingConfig:
    """Configuration for prompt weighting."""
    enable_weighting: bool = False
    variation_weight: float = 1.5
    base_weight: float = 1.0


class PromptValidationError(Exception):
    """Custom exception for prompt template validation errors."""
    pass


class PromptTemplate:
    """
    Enhanced template class supporting nested bracket syntax for prompt variations.
    
    Supports up to 3 layers of nested brackets with weighted options and proper
    validation of malformed syntax.
    """
    
    def __init__(self, template: str, max_depth: int = 3):
        """
        Initialize the prompt template.
        
        Args:
            template: The template string with bracket syntax
            max_depth: Maximum nesting depth allowed (default: 3)
        """
        self.template = template
        self.max_depth = max_depth
        self.variables = []
        self.parsed_structure = None
        
        # Validate and parse the template
        self.validate_template()
        self._parse_template()
    
    def validate_template(self) -> bool:
        """
        Validate the template syntax for proper bracket matching and nesting depth.
        
        Returns:
            True if template is valid
            
        Raises:
            PromptValidationError: If template has malformed syntax
        """
        if not self.template:
            raise PromptValidationError("Template cannot be empty")
        
        # Check for forbidden '/' separator
        if '/' in self.template and '[' in self.template:
            # Only check for '/' inside brackets
            bracket_content = re.findall(r'\[([^\[\]]*)\]', self.template)
            for content in bracket_content:
                if '/' in content:
                    raise PromptValidationError("Only '|' separator is supported, found '/' in bracket content")
        
        # Check bracket matching and nesting depth
        stack = []
        depth = 0
        max_depth_found = 0
        
        for i, char in enumerate(self.template):
            if char == '[':
                stack.append(i)
                depth += 1
                max_depth_found = max(max_depth_found, depth)
                
                if depth > self.max_depth:
                    raise PromptValidationError(
                        f"Nesting depth {depth} exceeds maximum allowed depth of {self.max_depth} at position {i}"
                    )
            elif char == ']':
                if not stack:
                    raise PromptValidationError(f"Unmatched closing bracket ']' at position {i}")
                stack.pop()
                depth -= 1
        
        if stack:
            raise PromptValidationError(f"Unmatched opening bracket '[' at position {stack[-1]}")
        
        # Validate bracket contents
        self._validate_bracket_contents()
        
        return True
    
    def _is_bracket_weight_syntax(self, option: str) -> bool:
        """
        Determine if a colon in an option represents bracket weight syntax.
        
        Returns True only if the colon appears to be for bracket weighting (option:weight)
        rather than diffusion model syntax like (word:weight).
        """
        # Find the last colon
        last_colon = option.rfind(':')
        if last_colon == -1:
            return False
        
        # Check if the part after the colon looks like a weight (just a number)
        after_colon = option[last_colon + 1:].strip()
        
        # If there's any non-numeric character (except .), it's probably not a weight
        try:
            float(after_colon)
        except ValueError:
            return False
        
        # Check if the colon is inside parentheses
        before_colon = option[:last_colon]
        
        # Count unmatched parentheses before the colon
        paren_count = 0
        for char in before_colon:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
        
        # If we're inside parentheses (paren_count > 0), this is likely diffusion syntax
        if paren_count > 0:
            return False
        
        # Check if the colon is immediately after a closing parenthesis
        # This would indicate diffusion syntax like "(word:weight)"
        if before_colon.rstrip().endswith(')'):
            return False
        
        # Otherwise, it's likely bracket weight syntax
        return True
    
    def _validate_bracket_contents(self):
        """Validate the contents within brackets for proper syntax."""
        # Find all bracket contents (including nested)
        pattern = r'\[([^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*)\]'
        matches = re.finditer(pattern, self.template)
        
        for match in matches:
            content = match.group(1)
            
            # Allow completely empty brackets like [] - this will be treated as a single empty option
            if not content.strip():
                continue
            
            # Check for proper option syntax
            options = content.split('|')
            
            # Allow empty options (they represent empty strings)
            for option in options:
                option_stripped = option.strip()
                
                # Empty options are allowed and represent empty strings
                if not option_stripped:
                    continue
                
                # Validate weight syntax if present
                # Only consider it weight syntax if colon is not inside parentheses
                if ':' in option_stripped:
                    # Check if this looks like actual bracket weight syntax (option:weight)
                    # vs. diffusion model syntax like (word:weight)
                    if self._is_bracket_weight_syntax(option_stripped):
                        parts = option_stripped.rsplit(':', 1)  # Split from the right
                        if len(parts) != 2:
                            raise PromptValidationError(f"Invalid weight syntax in option: '{option_stripped}'")
                        
                        try:
                            weight = float(parts[1])
                            if weight <= 0:
                                raise PromptValidationError(f"Weight must be positive, got {weight} in option: '{option_stripped}'")
                        except ValueError:
                            raise PromptValidationError(f"Invalid weight value in option: '{option_stripped}'")
    
    def _parse_template(self):
        """Parse the template into a structured format for variation generation."""
        self.parsed_structure = self._parse_nested_brackets(self.template)
        self.variables = self._extract_all_variables(self.parsed_structure)
    
    def _parse_nested_brackets(self, text: str, depth: int = 0) -> Dict[str, Any]:
        """
        Recursively parse nested bracket structures.
        
        Args:
            text: Text to parse
            depth: Current nesting depth
            
        Returns:
            Parsed structure as nested dictionaries
        """
        if depth > self.max_depth:
            raise PromptValidationError(f"Maximum nesting depth {self.max_depth} exceeded")
        
        result = {
            'type': 'text',
            'content': text,
            'variables': [],
            'depth': depth
        }
        
        # Find brackets at the current level only (not nested ones)
        variables = []
        i = 0
        var_index = 0
        
        while i < len(text):
            if text[i] == '[':
                # Find the matching closing bracket
                bracket_count = 1
                start = i
                j = i + 1
                
                while j < len(text) and bracket_count > 0:
                    if text[j] == '[':
                        bracket_count += 1
                    elif text[j] == ']':
                        bracket_count -= 1
                    j += 1
                
                if bracket_count == 0:
                    # Found complete bracket
                    end = j
                    var_content = text[start + 1:end - 1]
                    var_id = f"var_{depth}_{var_index}"
                    
                    # Parse options within this bracket
                    options = []
                    option_parts = self._split_options(var_content)
                    
                    # Handle completely empty brackets
                    if not option_parts or (len(option_parts) == 1 and not option_parts[0].strip()):
                        # Create a single empty option
                        empty_option = {
                            'type': 'text',
                            'content': '',
                            'variables': [],
                            'depth': depth + 1,
                            'weight': 1.0
                        }
                        options.append(empty_option)
                    else:
                        for option_text in option_parts:
                            # Don't strip yet - we need to detect truly empty options
                            original_option = option_text
                            option_text = option_text.strip()
                            
                            # Handle empty options (including just whitespace)
                            if not original_option.strip():
                                # This is an empty option - create an empty parsed option
                                empty_option = {
                                    'type': 'text',
                                    'content': '',
                                    'variables': [],
                                    'depth': depth + 1,
                                    'weight': 1.0
                                }
                                options.append(empty_option)
                                continue
                            
                            # Check for weight
                            weight = 1.0
                            if ':' in option_text:
                                # Use the same logic as validation to detect bracket weight syntax
                                if self._is_bracket_weight_syntax(option_text):
                                    parts = option_text.rsplit(':', 1)  # Split from the right
                                    try:
                                        weight = float(parts[1])
                                        option_text = parts[0].strip()
                                    except ValueError:
                                        weight = 1.0
                            
                            # Recursively parse nested content
                            parsed_option = self._parse_nested_brackets(option_text, depth + 1)
                            parsed_option['weight'] = weight
                            options.append(parsed_option)
                    
                    variable = {
                        'id': var_id,
                        'start': start,
                        'end': end,
                        'options': options,
                        'depth': depth
                    }
                    variables.append(variable)
                    var_index += 1
                    i = end
                else:
                    i += 1
            else:
                i += 1
        
        result['variables'] = variables
        return result
    
    def _split_options(self, content: str) -> List[str]:
        """Split option content by '|' while respecting nested brackets."""
        options = []
        current_option = ""
        bracket_count = 0
        
        for char in content:
            if char == '[':
                bracket_count += 1
                current_option += char
            elif char == ']':
                bracket_count -= 1
                current_option += char
            elif char == '|' and bracket_count == 0:
                # This '|' is at the top level, so it's a separator
                options.append(current_option)
                current_option = ""
            else:
                current_option += char
        
        # Add the last option
        if current_option:
            options.append(current_option)
        
        return options
    
    def _extract_all_variables(self, structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all variables from the parsed structure."""
        variables = []
        
        if 'variables' in structure:
            for var in structure['variables']:
                variables.append(var)
                # Recursively extract from options
                for option in var['options']:
                    variables.extend(self._extract_all_variables(option))
        
        return variables
    
    def generate_variations(self, max_variations: int = 100) -> List['PromptVariation']:
        """
        Generate all possible variations of the template.
        
        Args:
            max_variations: Maximum number of variations to generate
            
        Returns:
            List of PromptVariation objects
        """
        if not self.variables:
            return [PromptVariation(
                text=self.template,
                variation_id="single",
                variables={"var_0": "single variation"}
            )]
        
        variations = []
        
        try:
            # Generate combinations of variable choices
            variation_count = 0
            for i, (variation_text, selected_choices) in enumerate(self._generate_all_combinations_with_choices()):
                if variation_count >= max_variations:
                    logger.warning(f"Reached maximum variations limit of {max_variations}")
                    break
                
                # Create variation ID from the variation text
                variation_id = self._sanitize_for_filename(variation_text)
                if not variation_id:
                    variation_id = f"variation_{i}"
                
                # Create meaningful var_0 summary
                var_0_summary = self._create_variation_summary(variation_text, selected_choices)
                
                variables = {
                    "var_0": var_0_summary
                }
                
                variations.append(PromptVariation(
                    text=variation_text,
                    variation_id=variation_id,
                    variables=variables
                ))
                variation_count += 1
                
        except Exception as e:
            logger.error(f"Error generating variations: {e}")
            # Return at least the original template
            return [PromptVariation(
                text=self.template,
                variation_id="error_fallback",
                variables={"var_0": "error generating variations"}
            )]
        
        return variations if variations else [PromptVariation(
            text=self.template,
            variation_id="empty_fallback",
            variables={"var_0": "no variations generated"}
        )]
    
    def _sanitize_for_filename(self, text: str) -> str:
        """Sanitize text for use in filenames."""
        # Remove or replace problematic characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', text)
        sanitized = re.sub(r'\s+', '_', sanitized)
        sanitized = sanitized.strip('_')
        return sanitized[:50]  # Limit length
    
    def preview_variations(self, max_preview: int = 10) -> List[str]:
        """Preview the first few variations without generating all."""
        variations = self.generate_variations(max_variations=max_preview)
        return [var.text for var in variations]
    
    def _generate_all_combinations_with_choices(self):
        """Generate all possible combinations with tracking of selected choices."""
        def get_all_choice_combinations(structure: Dict[str, Any], prefix: str = "") -> List[Dict[str, int]]:
            """Get all possible choice combinations for a structure."""
            combinations = []
            
            if not structure.get('variables'):
                return [{}]
            
            # Get combinations for first variable
            first_var = structure['variables'][0]
            var_id = first_var['id']
            
            for i, option in enumerate(first_var['options']):
                # Get combinations for this option
                option_combinations = get_all_choice_combinations(option, f"{prefix}{var_id}_{i}_")
                
                # Add current choice to each combination
                for combo in option_combinations:
                    combo[var_id] = i
                    combinations.append(combo)
            
            # Process remaining variables at this level
            if len(structure['variables']) > 1:
                rest_structure = {
                    'variables': structure['variables'][1:],
                    'content': structure['content']
                }
                rest_combinations = get_all_choice_combinations(rest_structure, prefix)
                
                # Combine with existing combinations
                new_combinations = []
                for combo1 in combinations:
                    for combo2 in rest_combinations:
                        merged = {**combo1, **combo2}
                        new_combinations.append(merged)
                combinations = new_combinations
            
            return combinations
        
        def apply_choices_with_tracking(structure: Dict[str, Any], choices: Dict[str, int]) -> Tuple[str, List[str]]:
            """Apply choice selections to generate final text and track selected options."""
            text = structure['content']
            selected_options = []
            
            # Process variables in reverse order to maintain positions
            for var in reversed(structure.get('variables', [])):
                var_id = var['id']
                choice_idx = choices.get(var_id, 0)
                
                if choice_idx < len(var['options']):
                    chosen_option = var['options'][choice_idx]
                    # Recursively apply choices to the chosen option
                    replacement, nested_options = apply_choices_with_tracking(chosen_option, choices)
                    
                    # Track the selected option's content
                    if replacement.strip():  # Only track non-empty replacements
                        selected_options.append(replacement.strip())
                    
                    # Also track nested selections
                    selected_options.extend(nested_options)
                    
                    # Replace the bracket with the processed content
                    text = text[:var['start']] + replacement + text[var['end']:]
            
            return text, selected_options
        
        # Generate all possible choice combinations
        all_combinations = get_all_choice_combinations(self.parsed_structure)
        
        if not all_combinations:
            yield self.template, []
            return
        
        # Generate text for each combination
        for choices in all_combinations:
            try:
                result_text, selected_options = apply_choices_with_tracking(self.parsed_structure, choices)
                yield result_text, selected_options
            except Exception as e:
                logger.warning(f"Failed to generate variation with choices {choices}: {e}")
                continue
    
    def _create_variation_summary(self, variation_text: str, selected_choices: List[str]) -> str:
        """Create a meaningful summary for var_0 based on the selected choices."""
        if not selected_choices:
            # For empty variations (when first option is empty)
            if not variation_text.strip():
                return "[empty]"
            else:
                return "single option"
        
        # Extract the most distinctive/specific choices
        # Look for the leaf-level selections (the actual variant options)
        leaf_choices = []
        
        for choice in selected_choices:
            choice_clean = choice.strip()
            if choice_clean:
                # Keep the diffusion model weight syntax intact
                # Only clean up extra parentheses that aren't part of weights
                clean_choice = choice_clean.strip()
                
                # Skip choices that contain too much template text
                template_words = ['romantic', 'cinematic', 'kiss', 'between']
                # Remove weight syntax temporarily just for counting template words
                temp_clean = re.sub(r'\([^)]*:\d+\.?\d*\)', '', clean_choice.lower())
                choice_words = temp_clean.split()
                template_word_count = sum(1 for word in choice_words if word in template_words)
                
                # Only include choices that are more content than template
                if (len(choice_words) <= 5 or template_word_count < len(choice_words) / 2):
                    leaf_choices.append(clean_choice)
        
        if leaf_choices:
            # Find the most specific/distinctive choice
            # Prefer choices with character/subject information
            subject_keywords = ['people', 'man', 'woman', 'men', 'women', 'gay', 'lesbian', 'dog', 'cat']
            
            # Look for the shortest, most specific choice (likely a leaf choice)
            best_choice = None
            best_score = -1
            
            for choice in leaf_choices:
                choice_lower = choice.lower()
                score = 0
                
                # Higher score for subject/character words
                if any(keyword in choice_lower for keyword in subject_keywords):
                    score += 10
                
                # Prefer shorter, more specific choices
                word_count = len(choice.split())
                if word_count <= 3:
                    score += 5
                elif word_count <= 5:
                    score += 2
                
                # Bonus for distinctive words
                if any(word in choice_lower for word in ['two', 'and']):
                    score += 3
                
                if score > best_score:
                    best_score = score
                    best_choice = choice
            
            if best_choice:
                # Keep the choice as-is, preserving diffusion model syntax
                summary = best_choice.strip()
                
                # Limit length
                if len(summary) > 40:
                    summary = summary[:37] + "..."
                
                return summary
        
        # Fallback: try to identify the key distinguishing elements from the full text
        if variation_text.strip():
            # Find parts that are likely the variable content
            words = variation_text.strip().split()
            
            # Look for distinctive sequences, preserving weights
            skip_words = {'a', 'an', 'the', 'and', 'or', 'romantic', 'cinematic', 'kiss', 'between'}
            distinctive_sequences = []
            
            i = 0
            while i < len(words):
                # Check if this word (without weights) is distinctive
                base_word = re.sub(r'\([^)]*:\d+\.?\d*\)', '', words[i]).strip()
                clean_word = re.sub(r'[^\w]', '', base_word)
                
                if clean_word.lower() not in skip_words and len(clean_word) > 2:
                    # Found a distinctive word, collect sequence (preserving original format)
                    sequence = []
                    j = i
                    while j < len(words) and len(sequence) < 4:
                        sequence.append(words[j])  # Keep original word with weights
                        j += 1
                        # Stop at common words (but check base word without weights)
                        if j < len(words):
                            next_base = re.sub(r'\([^)]*:\d+\.?\d*\)', '', words[j]).strip()
                            next_clean = re.sub(r'[^\w]', '', next_base)
                            if next_clean.lower() in skip_words:
                                break
                    
                    if sequence:
                        distinctive_sequences.append(' '.join(sequence))
                        i = j
                    else:
                        i += 1
                else:
                    i += 1
            
            if distinctive_sequences:
                # Use the first distinctive sequence
                result = distinctive_sequences[0]
                if len(result) > 40:
                    result = result[:37] + "..."
                return result
        
        return "variation"
        
    def _generate_all_combinations(self):
        """Generate all possible combinations using the parsed structure."""
        def get_all_choice_combinations(structure: Dict[str, Any], prefix: str = "") -> List[Dict[str, int]]:
            """Get all possible choice combinations for a structure."""
            combinations = []
            
            if not structure.get('variables'):
                return [{}]
            
            # Get combinations for first variable
            first_var = structure['variables'][0]
            var_id = first_var['id']
            
            for i, option in enumerate(first_var['options']):
                # Get combinations for this option
                option_combinations = get_all_choice_combinations(option, f"{prefix}{var_id}_{i}_")
                
                # Add current choice to each combination
                for combo in option_combinations:
                    combo[var_id] = i
                    combinations.append(combo)
            
            # Process remaining variables at this level
            if len(structure['variables']) > 1:
                rest_structure = {
                    'variables': structure['variables'][1:],
                    'content': structure['content']
                }
                rest_combinations = get_all_choice_combinations(rest_structure, prefix)
                
                # Combine with existing combinations
                new_combinations = []
                for combo1 in combinations:
                    for combo2 in rest_combinations:
                        merged = {**combo1, **combo2}
                        new_combinations.append(merged)
                combinations = new_combinations
            
            return combinations
        
        def apply_choices(structure: Dict[str, Any], choices: Dict[str, int]) -> str:
            """Apply choice selections to generate final text."""
            text = structure['content']
            
            # Process variables in reverse order to maintain positions
            for var in reversed(structure.get('variables', [])):
                var_id = var['id']
                choice_idx = choices.get(var_id, 0)
                
                if choice_idx < len(var['options']):
                    chosen_option = var['options'][choice_idx]
                    # Recursively apply choices to the chosen option
                    replacement = apply_choices(chosen_option, choices)
                    # Replace the bracket with the processed content
                    text = text[:var['start']] + replacement + text[var['end']:]
            
            return text
        
        # Generate all possible choice combinations
        all_combinations = get_all_choice_combinations(self.parsed_structure)
        
        if not all_combinations:
            yield self.template
            return
        
        # Generate text for each combination
        for choices in all_combinations:
            try:
                result = apply_choices(self.parsed_structure, choices)
                yield result
            except Exception as e:
                logger.warning(f"Failed to generate variation with choices {choices}: {e}")
                continue
    
    def get_variable_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all variables in the template.
        
        Returns:
            List of dictionaries containing variable information
        """
        info = []
        for var in self.variables:
            var_info = {
                'id': var['id'],
                'depth': var['depth'],
                'position': (var['start'], var['end']),
                'options': []
            }
            
            for option in var['options']:
                option_info = {
                    'text': option['content'],
                    'weight': option.get('weight', 1.0)
                }
                var_info['options'].append(option_info)
            
            info.append(var_info)
        
        return info


class PromptManager:
    """Manages prompt templates and variations."""
    
    def __init__(self):
        self.templates = {}
        self.current_template = None
    
    def load_template(self, template: str, weighting_config: Optional[WeightingConfig] = None, name: Optional[str] = None) -> PromptTemplate:
        """Load a prompt template with optional weighting configuration."""
        prompt_template = PromptTemplate(template)
        
        if name:
            self.templates[name] = prompt_template
            self.current_template = prompt_template
        
        return prompt_template
    
    def load_template_from_file(self, file_path: str, weighting_config: Optional[WeightingConfig] = None, name: Optional[str] = None) -> PromptTemplate:
        """Load prompt template from file with optional weighting configuration."""
        with open(file_path, 'r', encoding='utf-8') as f:
            template = f.read().strip()
        
        return self.load_template(template, weighting_config, name)
    
    def save_template(self, template: PromptTemplate, file_path: str):
        """Save template to file."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(template.template)
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a named template."""
        return self.templates.get(name)
    
    def list_templates(self) -> List[str]:
        """List all loaded template names."""
        return list(self.templates.keys())
    
    def validate_template(self, template: str) -> Tuple[bool, List[str]]:
        """Validate template syntax and return any issues."""
        issues = []
        
        try:
            # Use the enhanced validation from PromptTemplate
            pt = PromptTemplate(template)
            return True, []
        except PromptValidationError as e:
            return False, [str(e)]
        except Exception as e:
            return False, [f"Unexpected error: {str(e)}"]


# Example usage and utility functions
def create_example_templates() -> Dict[str, str]:
    """Create example templates for common use cases."""
    examples = {
        "romance": "a romantic kiss between [two people|two men|two women|a man and a woman]",
        "action": "a [dramatic|intense|explosive] [chase scene|fight scene|battle] in [the city|the countryside|space]",
        "nature": "a [beautiful|serene|dramatic] [sunset|sunrise|storm] over [mountains|ocean|forest|desert]",
        "portrait": "a [professional|artistic|casual] portrait of [a man|a woman|a child] with [blue eyes|brown eyes|green eyes]",
        "animals": "a [cute|majestic|playful] [cat|dog|lion|elephant] in [natural habitat|urban environment|studio setting]",
        # Enhanced examples with nested syntax
        "complex_scene": "[a [beautiful [sunset:3.0|sunrise]|mysterious [forest|cave]] with [golden|silver] light|stormy [ocean:2.0|mountain] landscape]",
        "character_description": "[a [tall [dark|fair]|short [blonde|red-haired]] [man|woman] with [piercing [blue|green]|warm [brown|hazel]] eyes]",
        "empty_options": "My favorite [|dog|cat] in the [park|street|]"
    }
    return examples


def analyze_template_complexity(template: str) -> Dict[str, int]:
    """Analyze template to understand variation complexity."""
    try:
        prompt_template = PromptTemplate(template)
        variations = prompt_template.generate_variations(max_variations=1000)
        
        return {
            "total_variations": len(variations),
            "variable_count": len(prompt_template.variables),
            "max_variation_length": max(len(var) for var in variations) if variations else 0,
            "min_variation_length": min(len(var) for var in variations) if variations else 0,
            "max_depth": prompt_template.max_depth,
            "template_length": len(template)
        }
    except Exception as e:
        return {
            "error": str(e),
            "total_variations": 0,
            "variable_count": 0,
            "max_variation_length": 0,
            "min_variation_length": 0,
            "max_depth": 0,
            "template_length": len(template)
        }


def test_prompt_template():
    """Test function to demonstrate the enhanced PromptTemplate functionality."""
    
    print("=== Enhanced Prompt Template Test ===\n")
    
    # Test cases with different complexity levels
    test_cases = [
        # Simple cases
        "[cat|dog] on the beach",
        "[red:2.0|blue|green] car",
        
        # Empty option cases
        "My favorite [ | dog | cat ]",
        "this [|[(dog)|(cat)]] is good",
        
        # Nested cases
        "[scary [movie|book]|funny [sign:2.0|poster]] scene",
        "[big [red|blue] car|small truck] driving",
        
        # Complex 3-level nesting
        "[a [big [red|blue]|small [green|yellow]] [car|truck]|motorcycle] driving fast",
        
        # Edge cases
        "simple text without brackets",
        "[single] option",
        "[]",  # Empty bracket (single empty option)
    ]
    
    # Error cases to test validation
    error_cases = [
        "[unclosed bracket",
        "unopened bracket]",
        "[nested [too [deep [level [four]]]]]",  # Too deep
        "[option1/option2]",  # Forbidden separator
        "[option:invalid_weight]",  # Invalid weight
    ]
    
    # Test valid cases
    print("Testing valid templates:")
    print("-" * 50)
    
    for i, template in enumerate(test_cases, 1):
        print(f"\nTest {i}: {template}")
        try:
            pt = PromptTemplate(template)
            variations = pt.generate_variations(max_variations=10)
            print(f"Generated {len(variations)} variations:")
            for j, var in enumerate(variations[:5], 1):  # Show first 5
                print(f"  {j}. {var}")
            if len(variations) > 5:
                print(f"  ... and {len(variations) - 5} more")
                
            # Show variable info
            var_info = pt.get_variable_info()
            if var_info:
                print(f"Variables found: {len(var_info)}")
                for var in var_info:
                    print(f"  - {var['id']} (depth {var['depth']}): {len(var['options'])} options")
                    
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Test error cases
    print("\n\nTesting error cases:")
    print("-" * 50)
    
    for i, template in enumerate(error_cases, 1):
        print(f"\nError Test {i}: {template}")
        try:
            pt = PromptTemplate(template)
            print(f"  UNEXPECTED: Should have failed but didn't")
        except PromptValidationError as e:
            print(f"  EXPECTED ERROR: {e}")
        except Exception as e:
            print(f"  UNEXPECTED ERROR: {e}")
    
    # Demonstrate complex nested template
    print("\n\nAdvanced Example:")
    print("-" * 50)
    
    complex_template = "[a [beautiful [sunset:3.0|sunrise]|mysterious [forest|cave]] with [golden|silver] light|stormy [ocean:2.0|mountain] landscape] in [photorealistic|artistic] style"
    print(f"Template: {complex_template}")
    
    try:
        pt = PromptTemplate(complex_template)
        variations = pt.generate_variations(max_variations=20)
        print(f"\nGenerated {len(variations)} variations:")
        for i, var in enumerate(variations[:10], 1):
            print(f"  {i:2d}. {var}")
        if len(variations) > 10:
            print(f"     ... and {len(variations) - 10} more")
            
        print(f"\nStructure analysis:")
        var_info = pt.get_variable_info()
        for var in var_info:
            print(f"  Variable {var['id']} (depth {var['depth']}):")
            for j, opt in enumerate(var['options']):
                weight_str = f" (weight: {opt['weight']})" if opt['weight'] != 1.0 else ""
                print(f"    {j+1}. '{opt['text']}'{weight_str}")
                
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    test_prompt_template()
