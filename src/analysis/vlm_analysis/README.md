# VLM Analysis Module

This module provides a comprehensive pipeline for analyzing video content using Vision-Language Models (VLM), specifically Qwen2.5-VL-32B-Instruct.

## Components

### 1. VLMModelLoader (`vlm_model_loader.py`)
- Handles loading and configuration of the Qwen2.5-VL model
- Manages model resources and GPU memory
- Provides video input preparation and response generation
- Supports local model loading and flash attention optimization

### 2. VLMPromptOrchestrator (`vlm_prompt_orchestrator.py`)
- Manages the multi-prompt workflow for comprehensive video analysis
- Breaks down analysis into configurable stages:
  1. People & Demographics detection
  2. Scene & Temporal Segments analysis
  3. Relationship & Dyad analysis
  4. Cultural Interpretation & Flags
  5. Provenance & Confidence finalization
- Provides JSON validation and retry logic for failed responses
- Uses structured vocabulary from the analysis schema

### 3. VLMBatchProcessor (`vlm_batch_processor.py`)
- Orchestrates analysis across entire video batches
- Handles file structure navigation and output organization
- Provides error handling and partial result saving
- Generates batch summaries and processing reports

### 4. CLI Interface (`scripts/run_vlm_analysis.py`)
- Command-line interface for both single video and batch processing
- Configurable model paths, retry logic, and output options
- Verbose logging and error reporting

## Usage

### Single Video Analysis
```bash
python scripts/run_vlm_analysis.py \
  --batch-path path/to/video.mp4 \
  --single-video \
  --output-path analysis_output.json
```

### Batch Processing
```bash
python scripts/run_vlm_analysis.py \
  --batch-path path/to/batch_directory \
  --prompt-groups prompt_000 prompt_001
```

### Configuration Options
- `--schema-path`: Path to analysis schema JSON
- `--model-path`: Path to local VLM model
- `--max-retries`: Retry attempts for failed responses
- `--verbose`: Enable debug logging

## Expected Directory Structure

### Input (Batch)
```
batch_name/
├── videos/
│   ├── prompt_000/
│   │   ├── video_000.mp4
│   │   └── video_001.mp4
│   └── prompt_001/
│       └── video_000.mp4
```

### Output
```
batch_name/
├── vlm_analysis/
│   ├── prompt_000/
│   │   ├── video_000.json
│   │   └── video_001.json
│   ├── prompt_001/
│   │   └── video_000.json
│   └── batch_summary.json
```

## Analysis Schema

The system uses a comprehensive JSON schema that captures:
- **Demographics**: Age, race, gender, SES, body type
- **Appearance**: Clothing, makeup, hair, skin
- **Scene Context**: Location, time, weather, era
- **Cinematography**: Shot scale, angle, movement, lighting
- **Interactions**: Gaze, touch, facial expressions, relationships
- **Cultural Analysis**: Sexualization, respectability, tropes, objectification
- **Confidence Scores**: Per-element and overall confidence ratings

## Multi-Prompt Strategy

The system uses a staged approach for reliability:

1. **Stage 1**: Extract people and demographics information
2. **Stage 2**: Analyze scene context and temporal segments
3. **Stage 3**: Identify relationships and dyadic interactions
4. **Stage 4**: Perform cultural interpretation and flag analysis
5. **Stage 5**: Finalize provenance and confidence metadata

Each stage builds on previous results, allowing for focused analysis and better error recovery.

## Error Handling

- **JSON Validation**: Ensures responses match expected schema
- **Retry Logic**: Re-prompts VLM with clarification for failed attempts
- **Partial Results**: Saves incomplete data with error flags
- **Confidence Tracking**: Records analysis confidence at multiple levels

## Memory and Performance

- **Local Model Loading**: Uses pre-downloaded models to avoid re-downloading
- **GPU Memory Management**: Optimized for NVIDIA RTX A6000
- **Sequential Processing**: Designed for single-video processing (batch optimization planned)
- **Flash Attention**: Optional optimization for better memory usage

## Dependencies

- `transformers`: Hugging Face transformers library
- `qwen_vl_utils`: Qwen-specific utilities for vision processing
- `torch`: PyTorch for model inference
- Standard Python libraries: `json`, `pathlib`, `logging`, etc.
