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
        schema: The new flexible schema with type-based field definitions
        
    Returns:
        Dictionary with prompts array and system prompt
    """
    
    def get_field_instruction(field_name: str, field_def: Dict[str, Any]) -> str:
        """Generate instruction text for a field based on its type."""
        field_type = field_def.get("type", "open")
        
        if field_type == "options":
            options = field_def.get("options", [])
            return f'"{field_name}": "choose ONE from: {", ".join(options)}"'
        elif field_type == "range":
            examples = field_def.get("examples", [0.0, 1.0])
            return f'"{field_name}": {examples[1]} (range {examples[0]} to {examples[1]})'
        elif field_type == "open":
            examples = field_def.get("examples", [])
            if examples:
                return f'"{field_name}": "free response, examples: {", ".join(examples)}"'
            else:
                return f'"{field_name}": "free response"'
        else:
            return f'"{field_name}": "value"'
    
    def get_nested_instructions(obj: Dict[str, Any], indent: str = "    ") -> str:
        """Generate nested field instructions for complex objects."""
        instructions = []
        for key, value in obj.items():
            if isinstance(value, dict):
                if "type" in value:
                    # This is a field definition
                    instructions.append(f"{indent}{get_field_instruction(key, value)}")
                else:
                    # This is a nested object
                    instructions.append(f'{indent}"{key}": {{')
                    instructions.append(get_nested_instructions(value, indent + "    "))
                    instructions.append(f"{indent}}}")
        return ",\n".join(instructions)
    
    # System prompt - sets the overall context
    system_prompt = """You are an expert film analyst with a PhD in Media Studies. You analyze video content with scholarly precision.

CRITICAL INSTRUCTIONS:
- Respond with ONLY valid JSON - no explanations, markdown, or additional text
- Use the exact vocabulary tokens provided when specified for "options" type fields
- For "open" type fields, provide descriptive free responses using examples as guidance
- For "range" type fields, provide numerical values within the specified range
- Use "Undetectable" when you cannot determine a value with confidence
- Include confidence scores (0.0 to 1.0) where requested
- Be precise and analytical in your observations"""

    # Extract people template structure for generating instructions
    people_template = schema.get("people", [{}])[0] if schema.get("people") else {}
    
    # Prompt sequence - 5 stages based on main schema categories
    prompts = [
        {
            "stage": "people",
            "text": f"""Analyze this video and identify all people visible. For each person, determine their demographics, appearance, and role using the provided field types.

OUTPUT ONLY THIS JSON STRUCTURE:
{{
    "people": [
        {{
            "person_id": "p1",
            "demographics": {{
                {get_nested_instructions(people_template.get("demographics", {}))}
            }},
            "appearance": {{
                {get_nested_instructions(people_template.get("appearance", {}))}
            }},
            "role_and_agency": {{
                {get_nested_instructions(people_template.get("role_and_agency", {}))}
            }}
        }}
    ]
}}

RULES:
- Add additional people as p2, p3, etc. if multiple people are visible
- For "options" type fields, choose exactly ONE from the provided options
- For "open" type fields, provide descriptive responses (can reference examples)
- For "range" type fields, provide numerical values within the range
- Use confidence scores between 0.7-1.0 for clear observations, 0.4-0.6 for uncertain
- Analyze demographics, appearance, and role carefully for each person""",
            "expected_keys": ["people"],
            "enabled": True,
            "retry_guidance": "Ensure you follow the field type requirements: exact options for 'options' fields, descriptive text for 'open' fields, numerical values for 'range' fields."
        },
        {
            "stage": "composition",
            "text": f"""Now analyze the cinematographic and visual composition of this same video. Focus on technical film elements.

OUTPUT ONLY THIS JSON STRUCTURE:
{{
    "composition": {{
        {get_nested_instructions(schema.get("composition", {}))}
    }}
}}

RULES:
- For "options" type fields, choose exactly ONE from the provided options
- For "range" type fields, provide confidence scores between 0.0 and 1.0
- Consider the overall dominant characteristics of the video
- Use confidence scores between 0.7-1.0 for clear observations, 0.4-0.6 for uncertain""",
            "expected_keys": ["composition"],
            "enabled": True,
            "retry_guidance": "Ensure you select exactly ONE option for each 'options' field and provide numerical confidence scores."
        },
        {
            "stage": "setting",
            "text": f"""Now analyze the setting and environmental context of this video. Focus on location, time, atmosphere and environmental details.

OUTPUT ONLY THIS JSON STRUCTURE:
{{
    "setting": {{
        {get_nested_instructions(schema.get("setting", {}))}
    }}
}}

RULES:
- For "options" type fields, choose exactly ONE from the provided options
- For "open" type fields, provide descriptive responses about objects and cultural icons you observe
- For "range" type fields, provide confidence scores between 0.0 and 1.0
- Consider visual cues for weather even if unclear (lighting, shadows, clothing)
- Assess atmosphere from visual mood and emotional tone""",
            "expected_keys": ["setting"],
            "enabled": True,
            "retry_guidance": "Ensure you select exactly ONE option for each 'options' field and provide descriptive text for 'open' fields."
        },
        {
            "stage": "cultural_flags",
            "text": f"""Now perform a cultural and critical analysis of this video. Look for visual cues, tropes, and cultural readings. This is an interpretive analysis of the representation.

OUTPUT ONLY THIS JSON STRUCTURE:
{{
    "cultural_flags": {{
        {get_nested_instructions(schema.get("cultural_flags", {}))}
    }}
}}

RULES:
- For "options" type fields (level fields), choose exactly ONE from the provided options
- For "open" type fields (context fields), provide descriptive responses about what you observe
- Use the provided examples as guidance but describe what you actually see
- Use "Undetectable" if no clear cues are present
- Focus on visual evidence - camera work, framing, lighting, body positioning, etc.
- Be analytical and objective in identifying cultural patterns and representations
- For "range" type fields, provide confidence scores between 0.0 and 1.0""",
            "expected_keys": ["cultural_flags"],
            "enabled": True,
            "retry_guidance": "Choose exact options for level fields, provide descriptive text for context fields based on visual evidence."
        },
        {
            "stage": "overall_notes",
            "text": f"""Finally, provide overall notes and observations about this video analysis.

OUTPUT ONLY THIS JSON STRUCTURE:
{{
    "overall_notes": {{
        {get_nested_instructions(schema.get("overall_notes", {}))}
    }}
}}

RULES:
- For "salient_actions", describe the most important actions or moments in the video
- For "coder_discussion", note any ambiguities or analytical challenges you encountered
- Provide descriptive, detailed responses for both fields
- Reference specific people by their person_id (p1, p2, etc.) when describing actions""",
            "expected_keys": ["overall_notes"],
            "enabled": True,
            "retry_guidance": "Provide detailed descriptive text for both salient actions and any analytical ambiguities."
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

            Try again and ensure your JSON response is properly formatted and follows the field type requirements."""