"""
VLM Prompts Module
Dynamic prompt generation for video analysis workflow.
"""

from typing import Dict, Any, List
import json


def getVLMPrompts(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate the sequence of VLM prompts for video analysis.
    
    Args:
        schema: The analysis schema containing vocabulary and structure
        
    Returns:
        Dictionary with prompts array and system prompt
    """

  
    
    # Extract vocabulary for easy reference
    vocab = schema.get("vocab", {})
    
    # System prompt - sets the overall context
    system_prompt = """You are an expert film analyst with a PhD in Media Studies. You analyze video content with scholarly precision.

CRITICAL INSTRUCTIONS:
- Respond with ONLY valid JSON - no explanations, markdown, or additional text
- Use the exact vocabulary tokens provided when specified
- Use "Undetectable" when you cannot determine a value with confidence
- Include confidence scores (0.0 to 1.0) where requested
- Be precise and analytical in your observations"""

    # Prompt sequence - starting with demographics, then composition
    prompts = [
        {
            "stage": "demographics",
            "text": f"""Analyze this video and identify all people visible. For each person, determine their demographic characteristics using the provided vocabulary.

OUTPUT ONLY THIS JSON STRUCTURE:
{{
    "people": [
        {{
            "id": "P1",
            "demographics": {{
                "age": "choose ONE age range like '25-34', '35-44', etc.",
                "race": "choose ONE from: {', '.join(vocab.get('race', []))}",
                "gender": "choose ONE from: {', '.join(vocab.get('gender', []))}",
                "confidence": 0.8
            }}
        }}
    ]
}}

RULES:
- Add additional people as P2, P3, etc. if multiple people are visible
- Choose exactly ONE token from each vocabulary list
- Use confidence scores between 0.7-1.0 for clear observations, 0.4-0.6 for uncertain
- Estimate age ranges like "18-24", "25-34", "35-44", etc.""",
            "expected_keys": ["people"],
            "enabled": True,
            "retry_guidance": "Ensure you select exactly ONE vocabulary token for each demographic field."
        },
        {
            "stage": "composition",
            "text": f"""Now analyze the cinematographic and visual composition of this same video. Focus on technical film elements.

OUTPUT ONLY THIS JSON STRUCTURE:
{{
    "composition": {{
        "shot_scale": "choose ONE from: {', '.join(vocab.get('shot_scale', []))}",
        "camera_angle": "choose ONE from: {', '.join(vocab.get('angle', []))}",
        "camera_movement": "choose ONE from: {', '.join(vocab.get('movement', []))}",
        "framing": "choose ONE from: {', '.join(vocab.get('framing', []))}",
        "color_temperature": "choose ONE from: {', '.join(vocab.get('color_temp', []))}",
        "lighting_style": "choose ONE from: {', '.join(vocab.get('lighting', []))}",
        "visual_style": "choose ONE from: {', '.join(vocab.get('style', []))}",
        "confidence": 0.8
    }}
}}

RULES:
- Choose exactly ONE token from each vocabulary list
- Consider the overall dominant characteristics of the video
- Use confidence scores between 0.7-1.0 for clear observations, 0.4-0.6 for uncertain""",
            "expected_keys": ["composition"],
            "enabled": True,
            "retry_guidance": "Ensure you select exactly ONE vocabulary token for each composition field."
        }
    ]
    
    return {
        "system_prompt": system_prompt,
        "prompts": prompts
    }


def get_retry_prompt(original_prompt: str, error_info: str, retry_guidance: str = None) -> str:
    """Generate a retry prompt with specific error guidance."""
    
    guidance = retry_guidance or "Please ensure your response is valid JSON with the exact structure requested."
    
    return f"""{original_prompt}

            RETRY INSTRUCTION: The previous response failed validation: {error_info}

            {guidance}

            Try again and ensure your JSON response is properly formatted and uses the exact vocabulary tokens specified."""