"""
Main entry point for WAN 1.3B video generation.
Command-line interface for batch video generation with prompt variations.
"""
import os
# Suppress TensorFlow warnings before importing anything else
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Suppress tokenizer fork warning

import argparse
import sys
from pathlib import Path
import json

from src.config import ConfigManager, GenerationConfig
from src.orchestrator import VideoGenerationOrchestrator
from src.prompts import create_example_templates, analyze_template_complexity
from src.utils import estimate_disk_space, format_duration


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate videos using WAN 1.3B with prompt variations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with prompt template
  python main.py --template "a romantic kiss between [two people|two men|two women|a man and a woman]"
  
  # Use custom config and specify output
  python main.py --config configs/romantic.yaml --template "a [gentle|passionate] kiss" --output outputs/kiss_series
  
  # Continue from existing batch (same template)
  python main.py --continue-from outputs/14b_kiss_5frame_weighted_20250726_020105
  
  # Continue from existing batch with new template
  python main.py --continue-from outputs/14b_kiss_5frame_weighted_20250726_020105 --template "a [tender|passionate] embrace"
  
  # Preview without generating
  python main.py --template "a [cat|dog] playing in [garden|house]" --preview
  
  # Generate example templates
  python main.py --create-examples
        """
    )
    
    # Template and generation options
    parser.add_argument('--template', '-t', type=str,
                       help='Prompt template with [option1|option2] syntax')
    
    parser.add_argument('--config', '-c', type=str, default='configs/default.yaml',
                       help='Configuration file path')
    
    parser.add_argument('--continue-from', type=str,
                       help='Continue generation from existing batch directory')
    
    parser.add_argument('--output', '-o', type=str,
                       help='Output directory (overrides config)')
    
    parser.add_argument('--batch-name', type=str,
                       help='Name for this batch (for organization)')
    
    parser.add_argument('--videos-per-variation', type=int,
                       help='Number of videos per prompt variation')
    
    parser.add_argument('--max-variations', type=int,
                       help='Limit number of variations to process')
    
    # Preview and analysis
    parser.add_argument('--preview', action='store_true',
                       help='Preview batch without generating videos')
    
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze template complexity')
    
    # Utility functions
    parser.add_argument('--create-examples', action='store_true',
                       help='Create example template files')
    
    parser.add_argument('--create-default-config', action='store_true',
                       help='Create default configuration file')
    
    parser.add_argument('--validate', action='store_true',
                       help='Validate configuration and setup')
    
    # Model settings overrides
    parser.add_argument('--device', type=str,
                       help='Override device (auto, cuda:0, cuda:1, cpu, etc.)')
    
    parser.add_argument('--seed', type=int,
                       help='Override seed value')
    
    parser.add_argument('--cfg-scale', type=float,
                       help='Override CFG scale')
    
    parser.add_argument('--steps', type=int,
                       help='Override number of steps')
    
    parser.add_argument('--sampler', type=str,
                       help='Override sampler type')
    
    # Video settings overrides
    parser.add_argument('--width', type=int,
                       help='Override video width')
    
    parser.add_argument('--height', type=int,
                       help='Override video height')
    
    parser.add_argument('--fps', type=int,
                       help='Override video FPS')
    
    parser.add_argument('--duration', type=float,
                       help='Override video duration (seconds)')
    
    # Verbose output
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    return parser


def load_or_create_config(config_path: str, args) -> GenerationConfig:
    """Load configuration file or create default."""
    config_manager = ConfigManager()
    
    # Try to load existing config
    config_path = Path(config_path)
    if config_path.exists():
        print(f"Loading configuration from: {config_path}")
        config = config_manager.load_config(config_path)
    else:
        print(f"Configuration file not found, creating default: {config_path}")
        config = config_manager.create_default_config(config_path)
    
    # Apply command line overrides
    if args.output:
        config.output_dir = args.output
    
    if args.videos_per_variation:
        config.videos_per_variation = args.videos_per_variation
    
    if args.batch_name:
        config.batch_name = args.batch_name
    
    # Model setting overrides
    if args.device is not None:
        config.model_settings.device = args.device
    
    if args.seed is not None:
        config.model_settings.seed = args.seed
    
    if args.cfg_scale is not None:
        config.model_settings.cfg_scale = args.cfg_scale
    
    if args.steps is not None:
        config.model_settings.steps = args.steps
    
    if args.sampler:
        config.model_settings.sampler = args.sampler
    
    # Video setting overrides
    if args.width is not None:
        config.video_settings.width = args.width
    
    if args.height is not None:
        config.video_settings.height = args.height
    
    if args.fps is not None:
        config.video_settings.fps = args.fps
    
    if args.duration is not None:
        config.video_settings.duration = args.duration
    
    return config


def create_example_templates_files():
    """Create example template files."""
    examples = create_example_templates()
    templates_dir = Path("configs/templates")
    templates_dir.mkdir(parents=True, exist_ok=True)
    
    for name, template in examples.items():
        template_file = templates_dir / f"{name}.txt"
        with open(template_file, 'w', encoding='utf-8') as f:
            f.write(template)
        print(f"Created example template: {template_file}")
    
    # Create a combined examples file
    combined_file = templates_dir / "all_examples.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    print(f"Created combined examples file: {combined_file}")


def analyze_template(template: str):
    """Analyze and display template complexity."""
    complexity = analyze_template_complexity(template)
    
    print(f"\nTemplate Analysis:")
    print(f"  Template: {template}")
    print(f"  Total variations: {complexity['total_variations']}")
    print(f"  Variable sections: {complexity['variable_count']}")
    print(f"  Variation length range: {complexity['min_variation_length']}-{complexity['max_variation_length']} chars")
    
    if complexity['total_variations'] > 20:
        print(f"  ⚠️  Large number of variations - consider using --max-variations")


def preview_batch(orchestrator: VideoGenerationOrchestrator, template: str, max_preview: int = 10):
    """Preview batch generation."""
    preview = orchestrator.preview_batch(template, max_preview)
    
    print(f"\nBatch Preview:")
    print(f"  Template: {preview['template']}")
    print(f"  Total variations: {preview['total_variations']}")
    print(f"  Videos per variation: {preview['videos_per_variation']}")
    print(f"  Total videos to generate: {preview['total_videos']}")
    
    # Estimate disk space
    config = orchestrator.config
    estimated_space = estimate_disk_space(
        num_videos=preview['total_videos'],
        video_duration=config.video_settings.duration,
        resolution=(config.video_settings.width, config.video_settings.height),
        fps=config.video_settings.fps
    )
    print(f"  Estimated disk space: {estimated_space}")
    
    print(f"\nSample variations (showing first {len(preview['preview_variations'])}):")
    for i, variation in enumerate(preview['preview_variations'], 1):
        print(f"  {i:2d}: {variation}")
    
    if preview['total_variations'] > len(preview['preview_variations']):
        remaining = preview['total_variations'] - len(preview['preview_variations'])
        print(f"  ... and {remaining} more variations")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle utility commands first
    if args.create_examples:
        create_example_templates_files()
        return 0
    
    if args.create_default_config:
        config_manager = ConfigManager()
        config_path = Path(args.config)
        config = config_manager.create_default_config(config_path)
        print(f"Created default configuration: {config_path}")
        return 0
    
    # Require template for main operations unless continuing from existing batch
    if not args.template and not args.validate and not args.continue_from:
        print("Error: --template is required for generation operations")
        print("Use --help for usage information or --create-examples to see example templates")
        return 1
    
    try:
        # Load configuration
        config = load_or_create_config(args.config, args)
        
        # Create orchestrator
        orchestrator = VideoGenerationOrchestrator(config)
        
        # Validate setup
        if args.validate:
            issues = orchestrator.validate_setup()
            if issues:
                print("Setup validation failed:")
                for issue in issues:
                    print(f"  - {issue}")
                return 1
            else:
                print("Setup validation passed ✓")
                if not args.template and not args.continue_from:
                    return 0
        
        # Handle continue from existing batch
        if args.continue_from:
            print(f"Continuing generation from: {args.continue_from}")
            
            batch_results = orchestrator.continue_from_batch(
                batch_directory=args.continue_from,
                new_template=args.template,  # Optional - can be None to use original
                videos_per_variation=args.videos_per_variation,
                max_variations=args.max_variations
            )
            
            print(f"\n🎉 Batch continuation complete!")
            print(f"Results saved to: {batch_results['batch_info']['output_directory']}")
            
            return 0
        
        # Analyze template
        if args.analyze:
            analyze_template(args.template)
            if not args.preview:
                return 0
        
        # Preview batch
        if args.preview:
            preview_batch(orchestrator, args.template)
            return 0
        
        # Run full batch generation
        print(f"Starting batch generation with template: {args.template}")
        
        batch_results = orchestrator.run_full_batch(
            template=args.template,
            batch_name=args.batch_name,
            videos_per_variation=args.videos_per_variation,
            max_variations=args.max_variations
        )
        
        print(f"\n🎉 Batch generation complete!")
        print(f"Results saved to: {batch_results['batch_info']['output_directory']}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Generation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
