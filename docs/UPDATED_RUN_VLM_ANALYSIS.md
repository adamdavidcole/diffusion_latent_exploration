# Updated run_vlm_analysis.py - Conversation-Based VLM Analysis

## ✅ New Features

### 1. **Dual Processing Modes**
- **Single Video**: `--video-path path/to/video.mp4 --output-path results.json`
- **Batch Processing**: `--batch-path path/to/batch_directory`

### 2. **Batch Directory Structure Support**
```
batch_path/
├── videos/
│   ├── prompt_000/
│   │   ├── video_000.mp4
│   │   ├── video_001.mp4
│   │   └── ...
│   ├── prompt_001/
│   │   └── ...
│   └── ...
└── vlm_analysis/          # Auto-created output
    ├── prompt_000/
    │   ├── video_000.json
    │   ├── video_001.json
    │   └── ...
    └── ...
```

### 3. **New Conversation-Based Analysis**
- Uses `ConversationVLMProcessor` instead of old batch processor
- Supports new flexible schema (`vlm_analysis_schema_new.json`)
- Falls back to old schema if new one not found
- 5-stage conversation analysis (people, composition, setting, cultural_flags, overall_notes)

## 📋 Usage Examples

### Single Video Analysis
```bash
python scripts/run_vlm_analysis.py \
  --video-path outputs/batch/videos/prompt_001/video_005.mp4 \
  --output-path analysis_results.json \
  --verbose
```

### Batch Processing
```bash
python scripts/run_vlm_analysis.py \
  --batch-path outputs/14b_kiss_latent_attention_no_weight_20250811_003646 \
  --max-retries 3 \
  --verbose
```

### Custom Schema
```bash
python scripts/run_vlm_analysis.py \
  --video-path video.mp4 \
  --output-path results.json \
  --schema-path src/analysis/vlm_analysis/vlm_analysis_schema.json
```

## 🔧 Command Line Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--video-path` | string | Path to single video file (mutually exclusive with --batch-path) |
| `--batch-path` | string | Path to batch directory containing videos/ folder |
| `--output-path` | string | Output path for single video (required with --video-path) |
| `--schema-path` | string | Schema file path (defaults to new flexible schema) |
| `--model-id` | string | Hugging Face model ID (default: Qwen/Qwen2.5-VL-32B-Instruct) |
| `--max-retries` | int | Max retries per conversation stage (default: 2) |
| `--disable-conversation-log` | flag | Disable conversation logging |
| `--verbose` | flag | Enable verbose logging |

## 🚀 Key Improvements

### **Automatic Video Discovery**
- Scans `videos/prompt_*/` directories for `.mp4` files
- Maintains directory structure in output

### **Robust Error Handling**
- Per-video error tracking in batch mode
- Detailed success/failure reporting
- Graceful cleanup on interruption

### **Progress Reporting**
```
🚀 Starting batch processing...
Processing: prompt_003/video_007.mp4
  ✅ Success: vlm_analysis/prompt_003/video_007.json
...
✅ Batch processing completed in 245.3s
   Total videos: 64
   Successful: 62
   Failed: 2
```

### **Schema Flexibility**
- Defaults to new flexible schema
- Automatic fallback to old schema
- Custom schema support

## 🧪 Testing Results

### Batch Structure Test
```
🔍 Testing batch structure: outputs/14b_kiss_latent_attention_no_weight_20250811_003646
  Found 64 videos:
    - prompt_003/video_007.mp4
    - prompt_003/video_006.mp4
    - prompt_003/video_005.mp4
    ... and 61 more
  Example output: vlm_analysis/prompt_003/video_007.json
```

### Integration Status
- ✅ **Single video processing**: Complete
- ✅ **Batch processing**: Complete  
- ✅ **New schema support**: Complete
- ✅ **Conversation-based analysis**: Complete
- ✅ **Error handling & cleanup**: Complete
- ✅ **Progress reporting**: Complete

The updated script is ready for production use with both single video and batch processing capabilities using the new conversation-based VLM analysis system.
