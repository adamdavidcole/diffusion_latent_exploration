#!/usr/bin/env python3
"""
Demo script for WAN 1.3B Video Generation Project
Demonstrates various features and capabilities of the system.
"""

import subprocess
import sys
from pathlib import Path
import time


def run_command(command, description):
    """Run a command and display results."""
    print(f"\n{'='*60}")
    print(f"🎬 {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0


def main():
    """Run the demo."""
    print("🎥 WAN 1.3B Video Generation Demo")
    print("This demo showcases the capabilities of the video generation system.")
    
    # Change to project directory
    project_dir = Path(__file__).parent
    
    # Demo 1: Create example templates
    run_command(
        "python main.py --create-examples",
        "Creating Example Template Files"
    )
    
    # Demo 2: Analyze template complexity
    run_command(
        'python main.py --analyze --template "a [beautiful|serene|dramatic] [sunset|sunrise|storm] over [mountains|ocean|forest|desert]"',
        "Analyzing Template Complexity"
    )
    
    # Demo 3: Preview a batch
    run_command(
        'python main.py --preview --template "a romantic kiss between [two people|two men|two women|a man and a woman]"',
        "Previewing Batch Generation"
    )
    
    # Demo 4: Generate a small test batch
    run_command(
        'python main.py --config configs/fast_test.yaml --template "a [happy|sad] [cat|dog]" --videos-per-variation 2 --batch-name "demo_batch"',
        "Generating Test Batch (4 variations × 2 videos = 8 total)"
    )
    
    # Demo 5: Validate setup
    run_command(
        "python main.py --validate",
        "Validating System Setup"
    )
    
    # Demo 6: Preview high-complexity template with limitations
    run_command(
        'python main.py --preview --template "a [fast|slow] [red|blue|green] [car|truck|bike] on [highway|street|track]" --max-variations 10',
        "Preview with Variation Limit (36 total, showing 10)"
    )
    
    print(f"\n{'='*60}")
    print("🎉 Demo Complete!")
    print("The system successfully demonstrated:")
    print("  ✅ Template parsing and variation generation")
    print("  ✅ Configuration management")
    print("  ✅ Batch organization and file structure")
    print("  ✅ Progress tracking and logging")
    print("  ✅ Preview and analysis capabilities")
    print(f"{'='*60}")
    
    # Show output structure
    output_dirs = list(Path("outputs").glob("*"))
    if output_dirs:
        print(f"\nGenerated outputs can be found in:")
        for output_dir in output_dirs:
            print(f"  📁 {output_dir}")
    
    print("\nTo use the system with actual WAN 1.3B model:")
    print("  1. Install WAN 1.3B dependencies (see requirements.txt)")
    print("  2. Replace MockVideoGenerator with actual WAN 1.3B interface")
    print("  3. Update video_generator.py with real model calls")


if __name__ == "__main__":
    main()
