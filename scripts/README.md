# Scripts Directory

This directory contains auxiliary scripts for the WAN Video Generation project. These are utility and testing scripts that support the main application but are not part of the core functionality.

## Scripts

### 🧪 `test_memory_optimization.py`
**Purpose**: Test memory optimization features for WAN 14B model  
**Usage**: 
```bash
python scripts/test_memory_optimization.py
```
**Description**: Tests memory management features including GPU memory info, optimization settings, and model initialization without running a full batch. Useful for validating memory configurations before large generations.

### 🧹 `cleanup_failed_experiments.py`
**Purpose**: Clean up failed experiment directories  
**Usage**: 
```bash
# See what would be cleaned (safe)
python scripts/cleanup_failed_experiments.py --dry-run --verbose

# Actually clean up failed experiments
python scripts/cleanup_failed_experiments.py
```
**Description**: Scans the `outputs/` directory and removes experiment directories that don't contain any successfully generated video files. This helps free up disk space from failed or interrupted generations.

**Features**:
- Smart detection of failed experiments (no video files)
- Dry-run mode for safety
- Detailed reporting of space savings
- Interactive confirmation before deletion

### 🎬 `demo.py`
**Purpose**: Demonstrate system capabilities  
**Usage**: 
```bash
python scripts/demo.py
```
**Description**: Runs a series of demo commands to showcase the video generation system's features, including template parsing, batch generation, and system validation.

**Demonstrations**:
- Creating example prompt templates
- Analyzing prompt complexity
- Previewing batch generations
- Running test generations
- System validation

### ⭐ `demo_prompt_weighting.py` **NEW**
**Purpose**: Demonstrate the new prompt weighting feature  
**Usage**: 
```bash
python scripts/demo_prompt_weighting.py
```
**Description**: Shows how to use the new prompt weighting system to emphasize specific parts of prompts for more focused and distinct video outputs.

**Features Demonstrated**:
- Prompt weighting syntax `(text:weight)`
- Configuration-based weighting
- Manual weight specification examples
- Integration with existing templates
- Best practices and tips

**What is Prompt Weighting?**
Allows you to control the emphasis on different parts of your prompts:
- `"a romantic kiss between (two men:1.8)"` - Strong emphasis on "two men"
- `"a landscape with (mountains:1.3)"` - Subtle emphasis on "mountains"
- `"(dramatic:2.0) action scene"` - Very strong emphasis on "dramatic"

This is especially powerful for making prompt variations more distinct and ensuring the model focuses on the key differences between variations.

## Running Scripts

All scripts should be run from the project root directory or will automatically handle path resolution to find the correct project files and directories.

### From Project Root:
```bash
python scripts/script_name.py
```

### From Scripts Directory:
```bash
cd scripts
python script_name.py
```

## Adding New Scripts

When adding new auxiliary scripts to this directory:

1. **File naming**: Use descriptive snake_case names
2. **Documentation**: Add docstrings and help text
3. **Path handling**: Ensure scripts work when called from different directories
4. **Update this README**: Document the new script's purpose and usage
5. **Executable permissions**: Make scripts executable with `chmod +x script_name.py`

## Script Categories

- **Testing Scripts**: `test_*.py` - Scripts for testing specific features
- **Utility Scripts**: `*_utility.py` or descriptive names - Helper/maintenance scripts  
- **Demo Scripts**: `demo*.py` - Scripts that demonstrate functionality
- **Cleanup Scripts**: `cleanup_*.py` - Scripts for maintenance and cleanup tasks
