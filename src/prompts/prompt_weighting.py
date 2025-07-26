"""
Prompt weighting module for WAN video generation.
Provides safe, embedding-free approaches to prompt emphasis.
"""

import re
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WeightedSegment:
    """Represents a text segment with its weight."""
    text: str
    weight: float
    
    def __str__(self):
        if self.weight == 1.0:
            return self.text
        return f"({self.text}:{self.weight})"


class PromptWeightingStrategy:
    """Base class for different prompt weighting strategies."""
    
    def apply_weights(self, segments: List[WeightedSegment]) -> str:
        """Apply weighting strategy to segments and return modified prompt."""
        raise NotImplementedError


class RepetitionStrategy(PromptWeightingStrategy):
    """Apply weights by repeating emphasized words."""
    
    def __init__(self, max_repetitions: int = 4):
        self.max_repetitions = max_repetitions
    
    def apply_weights(self, segments: List[WeightedSegment]) -> str:
        """Apply weights through word repetition."""
        result_parts = []
        
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
                
            if segment.weight > 1.0:
                # Calculate repetitions: 1.5 -> 2, 2.0 -> 3, etc.
                repetitions = min(int(segment.weight + 0.5), self.max_repetitions)
                repeated_text = ", ".join([text] * repetitions)
                result_parts.append(repeated_text)
                logger.debug(f"Repeated '{text}' {repetitions} times (weight {segment.weight})")
            else:
                result_parts.append(text)
        
        return " ".join(result_parts)


class EnhancedLanguageStrategy(PromptWeightingStrategy):
    """Apply weights by using stronger descriptive language."""
    
    def __init__(self):
        # Mapping of words to their enhanced versions
        self.enhancement_map = {
            "dancing": ["energetically dancing", "vibrantly dancing", "passionately dancing"],
            "running": ["swiftly running", "rapidly running", "powerfully running"],
            "walking": ["confidently walking", "gracefully walking", "purposefully walking"],
            "jumping": ["dynamically jumping", "athletically jumping", "explosively jumping"],
            "romantic": ["deeply romantic", "intensely romantic", "passionately romantic"],
            "beautiful": ["stunningly beautiful", "breathtakingly beautiful", "magnificently beautiful"],
            "strong": ["very strong", "extremely strong", "powerfully strong"],
            "fast": ["very fast", "extremely fast", "lightning fast"],
            "bright": ["brilliantly bright", "radiantly bright", "dazzlingly bright"],
            "dark": ["deeply dark", "intensely dark", "profoundly dark"],
            "happy": ["joyfully happy", "radiantly happy", "blissfully happy"],
            "sad": ["deeply sad", "profoundly sad", "heartbreakingly sad"],
        }
    
    def apply_weights(self, segments: List[WeightedSegment]) -> str:
        """Apply weights through enhanced language."""
        result_parts = []
        
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
                
            if segment.weight > 1.0:
                enhanced_text = self._enhance_text(text, segment.weight)
                result_parts.append(enhanced_text)
                if enhanced_text != text:
                    logger.debug(f"Enhanced '{text}' -> '{enhanced_text}' (weight {segment.weight})")
            else:
                result_parts.append(text)
        
        return " ".join(result_parts)
    
    def _enhance_text(self, text: str, weight: float) -> str:
        """Enhance text based on weight."""
        text_lower = text.lower()
        
        if text_lower in self.enhancement_map:
            enhancement_levels = self.enhancement_map[text_lower]
            # Choose enhancement level based on weight: 1.5->0, 2.0->1, 2.5->2, etc.
            level = min(int((weight - 1.0) * 2), len(enhancement_levels) - 1)
            return enhancement_levels[level]
        
        return text


class CleanPromptStrategy(PromptWeightingStrategy):
    """Remove all weight syntax and return clean prompt."""
    
    def apply_weights(self, segments: List[WeightedSegment]) -> str:
        """Return clean prompt without any weight syntax."""
        return "".join(segment.text for segment in segments)


class PromptWeightingProcessor:
    """Main processor for handling weighted prompts."""
    
    def __init__(self, strategy: Optional[PromptWeightingStrategy] = None):
        self.strategy = strategy or CleanPromptStrategy()
    
    def parse_weighted_prompt(self, prompt: str) -> List[WeightedSegment]:
        """
        Parse a prompt with weight syntax like (text:weight) into segments.
        
        Examples:
            "a beautiful (landscape:1.2)" -> [("a beautiful ", 1.0), ("landscape", 1.2)]
            "(romantic:1.5) kiss between (two people:0.8)" -> [("romantic", 1.5), (" kiss between ", 1.0), ("two people", 0.8)]
        
        Returns:
            List of WeightedSegment objects
        """
        # Pattern to match (text:weight) syntax
        pattern = r'\(([^:)]+):([0-9]*\.?[0-9]+)\)'
        
        segments = []
        last_end = 0
        
        for match in re.finditer(pattern, prompt):
            # Add text before this match with weight 1.0
            if match.start() > last_end:
                text_before = prompt[last_end:match.start()]
                if text_before:  # Don't add empty segments
                    segments.append(WeightedSegment(text_before, 1.0))
            
            # Add the weighted segment
            text = match.group(1)
            weight = float(match.group(2))
            segments.append(WeightedSegment(text, weight))
            
            last_end = match.end()
        
        # Add remaining text with weight 1.0
        if last_end < len(prompt):
            remaining_text = prompt[last_end:]
            if remaining_text:  # Don't add empty segments
                segments.append(WeightedSegment(remaining_text, 1.0))
        
        return segments
    
    def process_prompt(self, prompt: str) -> str:
        """Process a weighted prompt using the current strategy."""
        segments = self.parse_weighted_prompt(prompt)
        
        if not segments:
            return prompt
        
        # Check if any weights are present
        has_weights = any(segment.weight != 1.0 for segment in segments)
        
        if not has_weights:
            return prompt
        
        result = self.strategy.apply_weights(segments)
        logger.debug(f"Processed prompt: '{prompt}' -> '{result}'")
        return result
    
    def has_weights(self, prompt: str) -> bool:
        """Check if prompt contains weight syntax."""
        return '(' in prompt and ':' in prompt and bool(re.search(r'\([^:)]+:[0-9]*\.?[0-9]+\)', prompt))
    
    def get_weight_summary(self, prompt: str) -> Dict[str, float]:
        """Get a summary of weights in the prompt."""
        segments = self.parse_weighted_prompt(prompt)
        weights = {}
        
        for segment in segments:
            if segment.weight != 1.0:
                weights[segment.text.strip()] = segment.weight
        
        return weights


# Convenience functions
def create_repetition_processor(max_repetitions: int = 3) -> PromptWeightingProcessor:
    """Create a processor that uses repetition for emphasis."""
    return PromptWeightingProcessor(RepetitionStrategy(max_repetitions))


def create_enhanced_language_processor() -> PromptWeightingProcessor:
    """Create a processor that uses enhanced language for emphasis."""
    return PromptWeightingProcessor(EnhancedLanguageStrategy())


def create_clean_processor() -> PromptWeightingProcessor:
    """Create a processor that removes weight syntax (safe fallback)."""
    return PromptWeightingProcessor(CleanPromptStrategy())


# Example usage and testing
if __name__ == "__main__":
    # Test different strategies
    test_prompts = [
        "a person (dancing:1.5) in the park",
        "(romantic:1.8) kiss between two people",
        "a (beautiful:2.0) and (fast:1.3) car",
        "multiple (strong:2.0) weights and (subtle:1.2) emphasis"
    ]
    
    print("🔄 Prompt Weighting Strategy Comparison")
    print("=" * 60)
    
    strategies = {
        "Clean (Safe)": create_clean_processor(),
        "Repetition": create_repetition_processor(),
        "Enhanced Language": create_enhanced_language_processor()
    }
    
    for prompt in test_prompts:
        print(f"\nOriginal: {prompt}")
        for name, processor in strategies.items():
            result = processor.process_prompt(prompt)
            print(f"  {name:15}: {result}")
