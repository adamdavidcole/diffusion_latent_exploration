"""
Conversation-based VLM Processor
Manages video analysis through a multi-turn conversation approach.
"""

import json
import logging
from pathlib import Path
import traceback
from typing import Dict, Any, List, Optional, Tuple
import uuid
from datetime import datetime
import time

from .vlm_model_loader import VLMModelLoader
from .vlm_prompts import getVLMPrompts, get_retry_prompt

logger = logging.getLogger(__name__)


CONVERSATION_MAX_NEW_TOKENS = 1024
CONVERSATION_FPS = 1.0
CONVERSATION_MAX_PIXELS = 151200

class ConversationVLMProcessor:
    """Processes video analysis through conversation-based prompting."""
    
    def __init__(
        self, 
        schema_path: str,
        max_retries: int = 2,
        enable_conversation_log: bool = True
    ):
        self.schema_path = Path(schema_path)
        self.max_retries = max_retries
        self.enable_conversation_log = enable_conversation_log
        
        # Load schema
        self.schema = self._load_schema()
        
        # Initialize model loader
        self.model_loader = VLMModelLoader(
            model_id="Qwen/Qwen2.5-VL-32B-Instruct",
            use_flash_attention=True,
            device_map="auto"
        )
        
        # Generate prompts from schema
        self.prompt_config = getVLMPrompts(self.schema)
        
        self.run_id = str(uuid.uuid4())
        
    def _load_schema(self) -> Dict[str, Any]:
        """Load the analysis schema."""
        try:
            with open(self.schema_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load schema from {self.schema_path}: {e}")
            raise
            
    def initialize(self):
        """Initialize the VLM model."""
        logger.info("Initializing conversation-based VLM processor...")
        self.model_loader.load_model()
        logger.info(f"✅ Processor initialized with run_id: {self.run_id}")
        
    def extract_video_metadata(self, video_path: Path) -> Dict[str, Any]:
        """Extract basic metadata from video file."""
        return {
            "video_id": video_path.stem,
            "video_path": str(video_path),
            "source_model": "Wan",
            "duration_seconds": 4.4,  # Standard Wan video length
            "fps_estimate": 12.0
        }
        
    def analyze_video(
        self, 
        video_path: Path, 
        output_path: Path
    ) -> Dict[str, Any]:
        """
        Analyze a video through conversation-based prompting.
        
        Args:
            video_path: Path to video file
            output_path: Path for output JSON
            
        Returns:
            Analysis results
        """
        
        logger.info(f"Starting conversation-based analysis: {video_path.name}")
        
        # Initialize conversation and results
        conversation = []
        results = {
            "schema_version": self.schema.get("schema_version", "2.0"),
            "video_metadata": self.extract_video_metadata(video_path),
            "run_id": self.run_id,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "stages_completed": [],
            "errors": [],
            "ok": True
        }
        
        # Conversation log setup
        conversation_log_path = None
        if self.enable_conversation_log:
            conversation_log_path = output_path.with_suffix('.conversation.log')
            
        try:
            # Process each enabled prompt stage
            for i, prompt_config in enumerate(self.prompt_config["prompts"]):
                if not prompt_config.get("enabled", True):
                    continue
                    
                stage = prompt_config["stage"]
                logger.info(f"  Processing stage: {stage}")
                
                # Execute stage with retry logic
                stage_success, stage_data = self._execute_stage_with_retry(
                    video_path=video_path,
                    prompt_config=prompt_config,
                    conversation=conversation,
                    stage=stage,
                    is_first_stage=(i == 0)
                )
                
                if stage_success and stage_data:
                    # Merge stage data into results
                    results.update(stage_data)
                    results["stages_completed"].append(stage)
                    logger.info(f"    ✅ Stage {stage} completed successfully")
                else:
                    results["ok"] = False
                    error_msg = f"Failed to complete stage: {stage}"
                    results["errors"].append(error_msg)
                    logger.error(f"    ❌ {error_msg}")
                    
            # Save conversation log if enabled
            if self.enable_conversation_log and conversation:
                self._save_conversation_log(conversation, conversation_log_path)
                
            # Save main results
            self._save_results(results, output_path)
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            logger.error(traceback.format_exc())
            results["ok"] = False
            results["errors"].append(f"Analysis exception: {str(e)}")
            
        return results
        
    def _execute_stage_with_retry(
        self,
        video_path: Path,
        prompt_config: Dict[str, Any],
        conversation: List[Dict[str, Any]],
        stage: str,
        is_first_stage: bool = False
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Execute a single stage with retry logic."""
        
        stage_prompt = prompt_config["text"]
        expected_keys = prompt_config["expected_keys"]
        retry_guidance = prompt_config.get("retry_guidance", "")
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"    Attempt {attempt + 1}/{self.max_retries + 1}")
                
                # Build messages for this attempt
                if is_first_stage and attempt == 0:
                    # Very first stage - include video, system prompt, and initial question
                    messages = self._build_initial_messages(video_path, stage_prompt)
                    conversation.extend(messages)
                elif attempt == 0:
                    # Subsequent stage - add new user question to existing conversation
                    conversation.append({
                        "role": "user",
                        "content": stage_prompt
                    })
                    messages = conversation.copy()
                else:
                    # Retry attempt - modify the last user message
                    error_info = f"Attempt {attempt} failed validation"
                    retry_prompt = get_retry_prompt(stage_prompt, error_info, retry_guidance)
                    
                    # Replace the last user message with retry prompt
                    if conversation and conversation[-1]["role"] == "user":
                        conversation[-1]["content"] = retry_prompt
                    else:
                        conversation.append({
                            "role": "user",
                            "content": retry_prompt
                        })
                    messages = conversation.copy()
                    
                # Execute VLM call with full conversation context
                response = self._call_vlm_with_conversation(messages)
                
                # Add assistant response to conversation
                conversation.append({
                    "role": "assistant", 
                    "content": response
                })
                
                # Validate response
                validation_result = self._validate_json_response(response, expected_keys)
                
                if validation_result["valid"]:
                    return True, validation_result["data"]
                else:
                    logger.warning(f"    ❌ Validation failed: {validation_result['error']}")
                    if attempt == self.max_retries:
                        # Final attempt failed
                        return False, validation_result.get("data")
                        
            except Exception as e:
                logger.error(f"    ❌ Stage execution attempt {attempt + 1} failed: {e}")
                logger.error(traceback.format_exc())
                conversation.append({
                    "role": "system",
                    "content": f"Error in attempt {attempt + 1}: {str(e)}"
                })
                
                if attempt == self.max_retries:
                    return False, None
                    
        return False, None
        
    def _build_initial_messages(self, video_path: Path, prompt: str) -> List[Dict[str, Any]]:
        """Build the initial messages including video and system prompt."""
        
        return [
            {
                "role": "system",
                "content": self.prompt_config["system_prompt"]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(video_path),
                        "max_pixels": 151200,  # Conservative for stability
                        "fps": 1.0,
                    },
                    {
                        "type": "text", 
                        "text": prompt
                    }
                ],
            }
        ]
        
    def _call_vlm_with_conversation(self, messages: List[Dict[str, Any]]) -> str:
        """Call the VLM with full conversation context."""
        
        # For now, we'll use the existing analyze_video method but pass conversation context
        # Extract the video from the first message that contains it
        video_path = None
        for message in messages:
            if message["role"] == "user" and isinstance(message["content"], list):
                for item in message["content"]:
                    if isinstance(item, dict) and item.get("type") == "video":
                        video_path = item["video"]
                        break
                if video_path:
                    break
                    
        if not video_path:
            raise ValueError("Could not find video path in conversation")
            
        # Build a combined prompt from the conversation
        combined_prompt = self._build_combined_prompt_from_conversation(messages)
        
        # Call VLM with combined prompt
        return self.model_loader.analyze_video(
            video_path=video_path,
            text_prompt=combined_prompt,
            max_new_tokens=CONVERSATION_MAX_NEW_TOKENS,
            fps=CONVERSATION_FPS,
            max_pixels=CONVERSATION_MAX_PIXELS
        )
        
    def _build_combined_prompt_from_conversation(self, messages: List[Dict[str, Any]]) -> str:
        """Build a combined prompt from the conversation history."""
        
        prompt_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt_parts.append(f"SYSTEM: {content}")
            elif role == "user":
                if isinstance(content, str):
                    prompt_parts.append(f"USER: {content}")
                elif isinstance(content, list):
                    # Extract text from multimodal content
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            prompt_parts.append(f"USER: {item['text']}")
            elif role == "assistant":
                prompt_parts.append(f"ASSISTANT: {content}")
                
        return "\n\n".join(prompt_parts)
        
    def _call_vlm_with_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Call the VLM with a conversation format."""
        
        # For now, use the existing analyze_video method
        # In future, we could modify VLMModelLoader to support conversation format
        
        # Extract the video and latest prompt
        video_path = None
        latest_prompt = None
        
        for message in reversed(messages):
            if message["role"] == "user":
                content = message["content"]
                if isinstance(content, list):
                    for item in content:
                        if item["type"] == "video":
                            video_path = item["video"]
                        elif item["type"] == "text":
                            latest_prompt = item["text"]
                elif isinstance(content, str):
                    latest_prompt = content
                break
                
        if not video_path or not latest_prompt:
            raise ValueError("Could not extract video path or prompt from messages")
            
        # Build combined prompt (system + user prompt)
        system_content = ""
        for message in messages:
            if message["role"] == "system":
                system_content = message["content"]
                break
                
        combined_prompt = f"{system_content}\n\n{latest_prompt}"
        
        # Call VLM
        return self.model_loader.analyze_video(
            video_path=video_path,
            text_prompt=combined_prompt,
            max_new_tokens=CONVERSATION_MAX_NEW_TOKENS,  # Conservative for JSON responses
            fps=CONVERSATION_FPS,
            max_pixels=CONVERSATION_MAX_PIXELS
        )
        
    def _validate_json_response(self, response: str, expected_keys: List[str]) -> Dict[str, Any]:
        """Validate and parse JSON response for new flexible schema."""
        try:
            # Strip markdown formatting
            cleaned_response = response.strip()
            
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            elif cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]
                
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
                
            cleaned_response = cleaned_response.strip()
            
            # Parse JSON
            data = json.loads(cleaned_response)
            
            # Validate expected keys
            missing_keys = [key for key in expected_keys if key not in data]
            if missing_keys:
                return {
                    "valid": False,
                    "error": f"Missing expected keys: {missing_keys}",
                    "data": data
                }
                
            # Validate against new schema structure
            validation_errors = self._validate_data_against_schema(data)
            if validation_errors:
                return {
                    "valid": False,
                    "error": f"Schema validation errors: {'; '.join(validation_errors)}",
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
            
    def _validate_data_against_schema(self, data: Dict[str, Any]) -> List[str]:
        """Validate data against the new flexible schema structure."""
        errors = []
        
        # Validate people array
        if "people" in data:
            people_template = self.schema.get("people", [{}])[0] if self.schema.get("people") else {}
            for i, person in enumerate(data["people"]):
                person_errors = self._validate_person_data(person, people_template, f"people[{i}]")
                errors.extend(person_errors)
        
        # Validate composition
        if "composition" in data:
            comp_errors = self._validate_object_data(
                data["composition"], 
                self.schema.get("composition", {}), 
                "composition"
            )
            errors.extend(comp_errors)
            
        # Validate setting
        if "setting" in data:
            setting_errors = self._validate_object_data(
                data["setting"], 
                self.schema.get("setting", {}), 
                "setting"
            )
            errors.extend(setting_errors)
            
        # Validate cultural_flags
        if "cultural_flags" in data:
            flags_errors = self._validate_object_data(
                data["cultural_flags"], 
                self.schema.get("cultural_flags", {}), 
                "cultural_flags"
            )
            errors.extend(flags_errors)
            
        # Validate overall_notes
        if "overall_notes" in data:
            notes_errors = self._validate_object_data(
                data["overall_notes"], 
                self.schema.get("overall_notes", {}), 
                "overall_notes"
            )
            errors.extend(notes_errors)
            
        return errors
        
    def _validate_person_data(self, person: Dict[str, Any], template: Dict[str, Any], context: str) -> List[str]:
        """Validate a person object against the template."""
        errors = []
        
        # Validate demographics
        if "demographics" in person and "demographics" in template:
            demo_errors = self._validate_object_data(
                person["demographics"], 
                template["demographics"], 
                f"{context}.demographics"
            )
            errors.extend(demo_errors)
            
        # Validate appearance
        if "appearance" in person and "appearance" in template:
            app_errors = self._validate_object_data(
                person["appearance"], 
                template["appearance"], 
                f"{context}.appearance"
            )
            errors.extend(app_errors)
            
        # Validate role_and_agency
        if "role_and_agency" in person and "role_and_agency" in template:
            role_errors = self._validate_object_data(
                person["role_and_agency"], 
                template["role_and_agency"], 
                f"{context}.role_and_agency"
            )
            errors.extend(role_errors)
            
        return errors
        
    def _validate_object_data(self, obj: Dict[str, Any], schema_obj: Dict[str, Any], context: str) -> List[str]:
        """Validate an object against its schema definition."""
        errors = []
        
        for field_name, field_def in schema_obj.items():
            if not isinstance(field_def, dict) or "type" not in field_def:
                # Skip non-field definitions (like nested objects)
                if isinstance(field_def, dict) and field_name in obj:
                    nested_errors = self._validate_object_data(
                        obj[field_name], 
                        field_def, 
                        f"{context}.{field_name}"
                    )
                    errors.extend(nested_errors)
                continue
                
            field_type = field_def["type"]
            value = obj.get(field_name)
            
            if value is None:
                continue  # Optional field
                
            # Validate based on field type
            if field_type == "options":
                valid_options = field_def.get("options", [])
                if value not in valid_options:
                    errors.append(f"{context}.{field_name}: '{value}' not in valid options {valid_options}")
                    
            elif field_type == "range":
                examples = field_def.get("examples", [0.0, 1.0])
                if not isinstance(value, (int, float)):
                    errors.append(f"{context}.{field_name}: must be numeric, got {type(value)}")
                elif len(examples) >= 2 and not (examples[0] <= value <= examples[1]):
                    errors.append(f"{context}.{field_name}: {value} not in range [{examples[0]}, {examples[1]}]")
                    
            elif field_type == "open":
                # Open fields are flexible - just check they're strings or reasonable types
                if not isinstance(value, (str, list)):
                    errors.append(f"{context}.{field_name}: expected string or list, got {type(value)}")
                    
        return errors
            
    def _save_conversation_log(self, conversation: List[Dict[str, Any]], log_path: Path):
        """Save conversation history for debugging."""
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            log_content = {
                "run_id": self.run_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "conversation": conversation
            }
            
            with open(log_path, 'w') as f:
                json.dump(log_content, f, indent=2)
                
            logger.info(f"Conversation log saved: {log_path}")
            
        except Exception as e:
            logger.error(f"Failed to save conversation log: {e}")
            
    def _save_results(self, results: Dict[str, Any], output_path: Path):
        """Save the final analysis results."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"✅ Results saved: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
            
    def cleanup(self):
        """Clean up resources."""
        self.model_loader.cleanup()
        logger.info("✅ Conversation processor cleaned up")
