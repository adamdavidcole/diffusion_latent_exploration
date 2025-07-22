"""
Prompt template and variation handling for video generation.
Supports creating multiple prompt variations from templates with variable keywords.
"""
import re
import itertools
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PromptVariation:
    """Single prompt variation with metadata."""
    text: str
    variation_id: str
    variables: Dict[str, str]
    

class PromptTemplate:
    """Handles prompt templates with variable sections."""
    
    def __init__(self, template: str):
        self.template = template
        self.variables = self._extract_variables()
    
    def _extract_variables(self) -> Dict[str, List[str]]:
        """Extract variable sections from template."""
        # Pattern to match [option1|option2|option3] syntax
        pattern = r'\[([^\]]+)\]'
        variables = {}
        
        matches = re.finditer(pattern, self.template)
        for i, match in enumerate(matches):
            var_name = f"var_{i}"
            options = [opt.strip() for opt in match.group(1).split('|')]
            variables[var_name] = options
        
        return variables
    
    def generate_variations(self) -> List[PromptVariation]:
        """Generate all possible prompt variations."""
        if not self.variables:
            return [PromptVariation(
                text=self.template,
                variation_id="single",
                variables={}
            )]
        
        variations = []
        variable_names = list(self.variables.keys())
        variable_options = [self.variables[name] for name in variable_names]
        
        # Generate all combinations
        for i, combination in enumerate(itertools.product(*variable_options)):
            # Create variable mapping
            var_map = dict(zip(variable_names, combination))
            
            # Replace variables in template
            text = self.template
            pattern = r'\[([^\]]+)\]'
            
            def replace_func(match):
                options = [opt.strip() for opt in match.group(1).split('|')]
                var_key = f"var_{len([m for m in re.finditer(pattern, self.template[:match.start()])])}"
                return var_map.get(var_key, options[0])
            
            text = re.sub(pattern, replace_func, text)
            
            # Create variation ID
            variation_id = "_".join([
                self._sanitize_for_filename(val) for val in combination
            ])
            
            variations.append(PromptVariation(
                text=text,
                variation_id=variation_id,
                variables=var_map
            ))
        
        return variations
    
    def _sanitize_for_filename(self, text: str) -> str:
        """Sanitize text for use in filenames."""
        # Remove or replace problematic characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', text)
        sanitized = re.sub(r'\s+', '_', sanitized)
        sanitized = sanitized.strip('_')
        return sanitized[:50]  # Limit length
    
    def preview_variations(self, max_preview: int = 10) -> List[str]:
        """Preview the first few variations without generating all."""
        variations = self.generate_variations()
        return [var.text for var in variations[:max_preview]]


class PromptManager:
    """Manages prompt templates and variations."""
    
    def __init__(self):
        self.templates = {}
        self.current_template = None
    
    def load_template(self, template: str, name: Optional[str] = None) -> PromptTemplate:
        """Load a prompt template."""
        prompt_template = PromptTemplate(template)
        
        if name:
            self.templates[name] = prompt_template
            self.current_template = prompt_template
        
        return prompt_template
    
    def load_template_from_file(self, file_path: str, name: Optional[str] = None) -> PromptTemplate:
        """Load prompt template from file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            template = f.read().strip()
        
        return self.load_template(template, name)
    
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
        
        # Check for unmatched brackets
        bracket_count = 0
        for char in template:
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count < 0:
                    issues.append("Unmatched closing bracket ']'")
                    bracket_count = 0
        
        if bracket_count > 0:
            issues.append("Unmatched opening bracket '['")
        
        # Check for empty variable sections
        empty_sections = re.findall(r'\[\s*\]', template)
        if empty_sections:
            issues.append("Empty variable sections found")
        
        # Check for sections without options
        single_option_sections = re.findall(r'\[([^\|\]]*)\]', template)
        if single_option_sections:
            for section in single_option_sections:
                if '|' not in section:
                    issues.append(f"Single option in variable section: [{section}]")
        
        return len(issues) == 0, issues


# Example usage and utility functions
def create_example_templates() -> Dict[str, str]:
    """Create example templates for common use cases."""
    examples = {
        "romance": "a romantic kiss between [two people|two men|two women|a man and a woman]",
        "action": "a [dramatic|intense|explosive] [chase scene|fight scene|battle] in [the city|the countryside|space]",
        "nature": "a [beautiful|serene|dramatic] [sunset|sunrise|storm] over [mountains|ocean|forest|desert]",
        "portrait": "a [professional|artistic|casual] portrait of [a man|a woman|a child] with [blue eyes|brown eyes|green eyes]",
        "animals": "a [cute|majestic|playful] [cat|dog|lion|elephant] in [natural habitat|urban environment|studio setting]"
    }
    return examples


def analyze_template_complexity(template: str) -> Dict[str, int]:
    """Analyze template to understand variation complexity."""
    prompt_template = PromptTemplate(template)
    variations = prompt_template.generate_variations()
    
    return {
        "total_variations": len(variations),
        "variable_count": len(prompt_template.variables),
        "max_variation_length": max(len(var.text) for var in variations) if variations else 0,
        "min_variation_length": min(len(var.text) for var in variations) if variations else 0
    }
