"""
VLM Prompt Orchestrator
Manages the multi-prompt workflow for video analysis.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class VLMPromptOrchestrator:
    """Manages the sequence of prompts for comprehensive video analysis."""
    
    def __init__(self, schema_path: str):
        """Initialize with the JSON schema."""
        self.schema_path = Path(schema_path)
        self.schema = self._load_schema()
        self.vocab = self.schema.get("vocab", {})
        
        # Configurable prompt sequence
        self.prompt_sequence = [
            "people_and_demographics",
            "scene_and_segments", 
            "relationships_and_dyads",
            "interpretation_and_flags",
            "finalize_provenance"
        ]
        
    def _load_schema(self) -> Dict[str, Any]:
        """Load the JSON schema."""
        try:
            with open(self.schema_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load schema: {e}")
            raise
            
    def _get_vocab_tokens(self, vocab_key: str) -> str:
        """Get formatted vocabulary tokens for a specific key."""
        if vocab_key not in self.vocab:
            return "Undetectable"
        return ", ".join(self.vocab[vocab_key])
        
    def _build_system_prompt(self) -> str:
        """Build the system prompt with constraints and vocabulary."""
        return f"""You are a film-scholar VLM analyzing video content. 

CRITICAL CONSTRAINTS:
- Output must be valid JSON only - no prose, explanations, or markdown
- Use supplied vocabulary tokens EXACTLY as provided
- Use "Undetectable" + confidence score for uncertainty
- Follow the schema structure precisely
- Include confidence scores (0.0-1.0) where specified

VOCABULARY REFERENCE:
{json.dumps(self.vocab, indent=2)}

Analyze the video systematically and provide structured data only."""

    def _build_prompt_1_people_demographics(self, video_metadata: Dict[str, Any]) -> str:
        """Build prompt for detecting people and demographics."""
        return f"""Analyze this video and identify all people present. Extract demographic and appearance information.

VIDEO METADATA: {json.dumps(video_metadata)}

Output ONLY this JSON structure:
{{
  "video_metadata": {json.dumps(video_metadata)},
  "people": [
    {{
      "id": "P1",
      "presence_confidence": 0.0,
      "demographics": {{
        "perceived_age": "25-35",
        "perceived_race": "{self._get_vocab_tokens('race')}",
        "perceived_gender": "{self._get_vocab_tokens('gender')}",
        "perceived_ses": "{self._get_vocab_tokens('ses')}",
        "body_type": "{self._get_vocab_tokens('body_type')}",
        "notable_features": ["glasses", "visible-disability", "tattoos", "Undetectable"]
      }},
      "appearance": {{
        "clothing_style": "{self._get_vocab_tokens('clothing_style')}",
        "clothing_coverage": "{self._get_vocab_tokens('clothing_coverage')}",
        "makeup": "{self._get_vocab_tokens('makeup')}",
        "hair": "{self._get_vocab_tokens('hair')}",
        "skin_sheen": ["Matte", "Glossy", "Sweaty", "Undetectable"]
      }}
    }}
  ]
}}

Use the exact vocabulary tokens. Add more people objects as needed (P2, P3, etc.)."""

    def _build_prompt_2_scene_segments(self, prior_data: Dict[str, Any]) -> str:
        """Build prompt for scene global analysis and temporal segments."""
        people_ids = [person["id"] for person in prior_data.get("people", [])]
        
        return f"""Continue analysis. Fill in scene_global and temporal_segments using prior people data.

PRIOR DATA: {json.dumps(prior_data)}

Output ONLY this JSON structure:
{{
  "scene_global": {{
    "location_type": "{self._get_vocab_tokens('loc_type')}",
    "location_category": "{self._get_vocab_tokens('location_cat')}",
    "era_setting": ["Past", "Contemporary", "Future", "Undetectable"],
    "era_look": "{self._get_vocab_tokens('era_look')}",
    "time_of_day": "{self._get_vocab_tokens('time_of_day')}",
    "public_private": "{self._get_vocab_tokens('public_private')}",
    "weather": ["Sunny", "Overcast", "Rain", "Snow", "Fog", "Undetectable"],
    "scene_tokens": ["comma,separated,keywords"]
  }},
  "temporal_segments": [
    {{
      "segment_id": "S1",
      "start_sec": 0.0,
      "end_sec": 3.2,
      "shot_scale": "{self._get_vocab_tokens('shot_scale')}",
      "camera_angle": "{self._get_vocab_tokens('angle')}",
      "camera_movement": "{self._get_vocab_tokens('movement')}",
      "framing": "{self._get_vocab_tokens('framing')}",
      "color_temperature": "{self._get_vocab_tokens('color_temp')}",
      "lighting_style": "{self._get_vocab_tokens('lighting')}",
      "visual_style": "{self._get_vocab_tokens('style')}",
      "people_present": {people_ids},
      "gaze_gesture": {{
        "eye_contact": "{self._get_vocab_tokens('eye_contact')}",
        "touch_type": "{self._get_vocab_tokens('touch')}",
        "touch_initiator": {people_ids + ["Both", "Undetectable"]},
        "facial_expression_P1": "{self._get_vocab_tokens('facial_expr')}",
        "facial_expression_P2": "{self._get_vocab_tokens('facial_expr')}",
        "posture_relation": "{self._get_vocab_tokens('posture_relation')}"
      }},
      "events": ["Kiss-start", "Kiss-end", "Hug", "Hand-hold", "None"],
      "confidence": 0.0
    }}
  ]
}}

Break video into logical segments. Use exact vocabulary tokens."""

    def _build_prompt_3_relationships_dyads(self, prior_data: Dict[str, Any]) -> str:
        """Build prompt for relationship and dyad analysis."""
        people_ids = [person["id"] for person in prior_data.get("people", [])]
        
        return f"""Continue analysis. Analyze relationships between people using prior data.

PRIOR DATA: {json.dumps(prior_data)}

Output ONLY this JSON structure:
{{
  "dyads": [
    {{
      "pair": {people_ids[:2] if len(people_ids) >= 2 else ["P1", "P2"]},
      "relationship_inferred": ["Romantic", "Friends", "Family", "Undetectable"],
      "dominance_pattern": ["None", "P1-dominant", "P2-dominant", "Alternating", "Undetectable"],
      "driver_seat_context": ["P1-driver", "P2-driver", "N/A", "Undetectable"],
      "aggregate_interactions": {{
        "mutual_gaze_ratio": 0.0,
        "touch_frequency": 0,
        "kiss_count": 0
      }}
    }}
  ]
}}

Analyze all possible pairs. Count actual interactions from the video."""

    def _build_prompt_4_interpretation_flags(self, prior_data: Dict[str, Any]) -> str:
        """Build prompt for interpretation and cultural analysis."""
        return f"""Continue analysis. Identify cultural cues, tropes, and interpretive readings using prior data.

PRIOR DATA: {json.dumps(prior_data)}

Output ONLY this JSON structure:
{{
  "flags_and_readings": {{
    "sexualization_cues": ["{self._get_vocab_tokens('sexualization_cues')}"],
    "respectability_cues": ["{self._get_vocab_tokens('respectability_cues')}"],
    "moralizing_cues": ["{self._get_vocab_tokens('moralizing_cues')}"],
    "cinematic_tropes": ["{self._get_vocab_tokens('cinematic_tropes')}"],
    "objectification_cues": ["{self._get_vocab_tokens('objectification_cues')}"]
  }},
  "scores_optional": {{
    "intimacy_score_0to1": 0.0,
    "sexualization_score_0to1": 0.0,
    "publicness_score_0to1": 0.0
  }}
}}

Use exact vocabulary tokens. Provide thoughtful cultural analysis."""

    def _build_prompt_5_finalize_provenance(self, prior_data: Dict[str, Any], run_id: str) -> str:
        """Build prompt for finalizing provenance and overall confidence."""
        return f"""Finalize analysis with provenance metadata. Review all prior data for overall confidence.

PRIOR DATA: {json.dumps(prior_data)}

Output ONLY this JSON structure:
{{
  "provenance": {{
    "analyzer_model": "Qwen2.5-VL-32B-Instruct",
    "run_id": "{run_id}",
    "timestamp_utc": "{datetime.utcnow().isoformat()}Z",
    "reviewer": "auto",
    "vlm_warnings": ["low-light", "motion-blur", "partial-occlusion"],
    "overall_confidence": 0.0
  }}
}}

Assess video quality issues and provide overall confidence in the analysis."""

    def get_prompt_for_stage(
        self, 
        stage: str, 
        video_metadata: Dict[str, Any],
        prior_data: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None
    ) -> str:
        """Get the prompt for a specific analysis stage."""
        
        if stage == "people_and_demographics":
            return self._build_prompt_1_people_demographics(video_metadata)
        elif stage == "scene_and_segments":
            return self._build_prompt_2_scene_segments(prior_data or {})
        elif stage == "relationships_and_dyads":
            return self._build_prompt_3_relationships_dyads(prior_data or {})
        elif stage == "interpretation_and_flags":
            return self._build_prompt_4_interpretation_flags(prior_data or {})
        elif stage == "finalize_provenance":
            return self._build_prompt_5_finalize_provenance(prior_data or {}, run_id or str(uuid.uuid4()))
        else:
            raise ValueError(f"Unknown stage: {stage}")
            
    def get_system_prompt(self) -> str:
        """Get the system prompt."""
        return self._build_system_prompt()
        
    def get_retry_prompt(self, original_prompt: str, error_info: str) -> str:
        """Get a retry prompt with additional guidance."""
        return f"""{original_prompt}

RETRY INSTRUCTION: The previous response had issues: {error_info}
Please try again and ensure the response matches the JSON structure exactly. 
Double-check all vocabulary tokens against the provided lists."""

    def validate_response(self, response: str, expected_keys: List[str]) -> Dict[str, Any]:
        """Validate and parse a JSON response."""
        try:
            data = json.loads(response)
            
            # Check for expected top-level keys
            missing_keys = [key for key in expected_keys if key not in data]
            if missing_keys:
                return {
                    "valid": False,
                    "error": f"Missing keys: {missing_keys}",
                    "data": data
                }
                
            return {
                "valid": True,
                "data": data
            }
            
        except json.JSONDecodeError as e:
            return {
                "valid": False,
                "error": f"Invalid JSON: {str(e)}",
                "data": None
            }
