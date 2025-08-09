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
import logging

logger = logging.getLogger(__name__)


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
    
    def _validate_bracket_contents(self):
        """Validate the contents within brackets for proper syntax."""
        # Find all bracket contents (including nested)
        pattern = r'\[([^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*)\]'
        matches = re.finditer(pattern, self.template)
        
        for match in matches:
            content = match.group(1)
            if not content.strip():
                raise PromptValidationError(f"Empty bracket content at position {match.start()}")
            
            # Check for proper option syntax
            options = content.split('|')
            for option in options:
                option = option.strip()
                if not option:
                    raise PromptValidationError(f"Empty option in bracket content: '{content}'")
                
                # Validate weight syntax if present
                if ':' in option:
                    parts = option.split(':')
                    if len(parts) != 2:
                        raise PromptValidationError(f"Invalid weight syntax in option: '{option}'")
                    
                    try:
                        weight = float(parts[1])
                        if weight <= 0:
                            raise PromptValidationError(f"Weight must be positive, got {weight} in option: '{option}'")
                    except ValueError:
                        raise PromptValidationError(f"Invalid weight value in option: '{option}'")
    
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
                    
                    for option_text in option_parts:
                        option_text = option_text.strip()
                        if not option_text:
                            continue
                        
                        # Check for weight
                        weight = 1.0
                        if ':' in option_text and not '[' in option_text.split(':')[1]:
                            # Only consider it a weight if there's no bracket after the colon
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
    
    def generate_variations(self, max_variations: int = 100) -> List[str]:
        """
        Generate all possible variations of the template.
        
        Args:
            max_variations: Maximum number of variations to generate
            
        Returns:
            List of generated prompt variations
        """
        if not self.variables:
            return [self.template]
        
        variations = []
        
        try:
            # Generate combinations of variable choices
            variation_count = 0
            for variation in self._generate_all_combinations():
                if variation_count >= max_variations:
                    logger.warning(f"Reached maximum variations limit of {max_variations}")
                    break
                
                variations.append(variation)
                variation_count += 1
                
        except Exception as e:
            logger.error(f"Error generating variations: {e}")
            # Return at least the original template
            return [self.template]
        
        return variations if variations else [self.template]
    
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


def test_prompt_template():
    """Test function to demonstrate the enhanced PromptTemplate functionality."""
    
    print("=== Enhanced Prompt Template Test ===\n")
    
    # Test cases with different complexity levels
    test_cases = [
        # Simple cases
        "[cat|dog] on the beach",
        "[red:2.0|blue|green] car",
        
        # Nested cases
        "[scary [movie|book]|funny [sign:2.0|poster]] scene",
        "[big [red|blue] car|small truck] driving",
        
        # Complex 3-level nesting
        "[a [big [red|blue]|small [green|yellow]] [car|truck]|motorcycle] driving fast",
        
        # Edge cases
        "simple text without brackets",
        "[single] option",

        "[|[a (romantic) (cinematic) (kiss) between [two (people) | a (man:2.5) and a (woman:2.5)| two (men:2.5) | two (women:2.5) | two (gay:2.5) (men) | two (lesbian:2.5) (women) | a (dog:2.5) and a (cat:2.5)]]"
    ]
    
    # Error cases to test validation
    error_cases = [
        "[unclosed bracket",
        "unopened bracket]",
        "[nested [too [deep [level [four]]]]]",  # Too deep
        "[option1/option2]",  # Forbidden separator
        "[]",  # Empty bracket
        "[option1||option3]",  # Empty option
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
