#!/usr/bin/env python3
"""
Standalone script for analyzing latent trajectories from stored diffusion data.

This script can be used to analyze latent trajectories after video generation
is complete, providing insights into the geometry of the diffusion latent space.

Usage:
    python analyze_latent_trajectories.py --batch-dir outputs/my_batch_20250804_123456
    python analyze_latent_trajectories.py --latents-dir outputs/my_batch_20250804_123456/latents
    python analyze_latent_trajectories.py --batch-dir outputs/my_batch_20250804_123456 --video-id prompt_001_vid001
"""

import argparse
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.analysis import analyze_latent_trajectories_from_batch, LatentTrajectoryAnalyzer


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze latent trajectories from diffusion generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all videos in a batch directory
  python analyze_latent_trajectories.py --batch-dir outputs/romantic_kiss_20250804_123456
  
  # Analyze specific latents directory
  python analyze_latent_trajectories.py --latents-dir outputs/romantic_kiss_20250804_123456/latents
  
  # Analyze single video
  python analyze_latent_trajectories.py --batch-dir outputs/romantic_kiss_20250804_123456 --video-id prompt_001_vid001
  
  # Compare specific videos
  python analyze_latent_trajectories.py --batch-dir outputs/romantic_kiss_20250804_123456 --compare-videos prompt_001_vid001 prompt_002_vid001
        """
    )
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--batch-dir', type=str,
                      help='Batch directory containing latents subdirectory')
    group.add_argument('--latents-dir', type=str,
                      help='Directory containing stored latent files')
    
    # Analysis options
    parser.add_argument('--video-id', type=str,
                       help='Analyze specific video ID only')
    
    parser.add_argument('--compare-videos', nargs='+',
                       help='Compare specific video IDs')
    
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Skip creating visualization plots')
    
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for analysis results (default: same as latents dir)')
    
    # Comparison metrics
    parser.add_argument('--metrics', nargs='+',
                       choices=['trajectory_linearity', 'total_trajectory_distance', 
                               'trajectory_volume_estimate', 'mean_velocity', 'variance_change'],
                       default=['trajectory_linearity', 'total_trajectory_distance', 'trajectory_volume_estimate'],
                       help='Metrics to use for comparison analysis')
    
    # Utility options
    parser.add_argument('--list-videos', action='store_true',
                       help='List available video IDs and exit')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Determine latents directory
        if args.batch_dir:
            batch_path = Path(args.batch_dir)
            if not batch_path.exists():
                logger.error(f"Batch directory not found: {args.batch_dir}")
                return 1
            
            # Look for latents directory in the batch
            latents_dir = batch_path / "latents"
            if not latents_dir.exists():
                logger.error(f"No latents directory found in batch: {args.batch_dir}")
                logger.info("Make sure the batch was generated with --store-latents flag")
                logger.info("Expected to find 'latents' directory")
                return 1
        else:
            latents_dir = Path(args.latents_dir)
            if not latents_dir.exists():
                logger.error(f"Latents directory not found: {args.latents_dir}")
                return 1
        
        logger.info(f"Using latents directory: {latents_dir}")
        
        # Initialize analyzer
        analyzer = LatentTrajectoryAnalyzer(latents_dir)
        
        # List videos if requested
        if args.list_videos:
            prompt_dirs = analyzer.get_available_prompt_dirs()
            if prompt_dirs:
                print(f"\nFound {len(prompt_dirs)} prompt directories with stored latents:")
                for prompt_dir in sorted(prompt_dirs):
                    print(f"  {prompt_dir}:")
                    
                    # Try to discover individual videos in this prompt directory
                    video_ids = analyzer.discover_videos_in_prompt(prompt_dir)
                    for video_id in video_ids:
                        # Get basic info about the video
                        summary = analyzer.latent_storage.get_video_summary(video_id)
                        if summary:
                            print(f"    {video_id}: {summary['total_stored']} steps stored")
                        else:
                            # Count steps by scanning directory
                            try:
                                latents, _ = analyzer.load_video_trajectory(video_id)
                                print(f"    {video_id}: {len(latents)} steps found")
                            except:
                                print(f"    {video_id}: (could not load)")
                print(f"\nNote: You can analyze by prompt directory (e.g., 'prompt_000') or specific video ID")
            else:
                print("No videos with stored latents found.")
            return 0
        
        # Perform analysis
        if args.video_id:
            # Single video analysis
            logger.info(f"Analyzing single video: {args.video_id}")
            
            result = analyzer.analyze_single_video(
                args.video_id, 
                create_visualizations=not args.no_visualizations
            )
            
            # Save results
            output_file = f"single_video_analysis_{args.video_id}.json"
            analyzer.save_analysis_results(result, output_file)
            
            # Print summary
            print(f"\n=== Analysis Results for {args.video_id} ===")
            print(f"Prompt: {result.prompt}")
            print(f"Trajectory steps: {result.metrics.get('num_steps', 'N/A')}")
            print(f"Total distance: {result.metrics.get('total_trajectory_distance', 'N/A'):.4f}")
            print(f"Linearity: {result.metrics.get('trajectory_linearity', 'N/A'):.4f}")
            print(f"Volume estimate: {result.metrics.get('trajectory_volume_estimate', 'N/A'):.6f}")
            if result.visualization_paths:
                print(f"Visualizations created: {len(result.visualization_paths)}")
                for path in result.visualization_paths:
                    print(f"  - {path}")
        
        elif args.compare_videos:
            # Compare specific videos
            logger.info(f"Comparing videos: {args.compare_videos}")
            
            comparison_results = analyzer.compare_trajectories(
                args.compare_videos, 
                comparison_metrics=args.metrics
            )
            
            # Save results
            analyzer.save_analysis_results(comparison_results, "video_comparison_analysis.json")
            
            # Print comparison summary
            print(f"\n=== Comparison Results for {len(args.compare_videos)} Videos ===")
            for metric in args.metrics:
                if metric in comparison_results['comparison_data']:
                    data = comparison_results['comparison_data'][metric]
                    print(f"\n{metric}:")
                    print(f"  Mean: {data['mean']:.6f}")
                    print(f"  Std:  {data['std']:.6f}")
                    print(f"  Range: {data['min']:.6f} - {data['max']:.6f}")
                    
                    # Show per-video values
                    for i, (video_id, value) in enumerate(zip(args.compare_videos, data['values'])):
                        prompt = data['prompts'][i] if i < len(data['prompts']) else "Unknown"
                        print(f"    {video_id}: {value:.6f} ('{prompt[:50]}...')")
        
        else:
            # Full batch analysis
            logger.info("Performing full batch analysis")
            
            if args.batch_dir:
                results = analyze_latent_trajectories_from_batch(args.batch_dir, args.output_dir)
            else:
                # Get all videos and analyze them
                video_ids = analyzer.get_available_videos()
                if not video_ids:
                    logger.error("No videos with stored latents found")
                    return 1
                
                # Analyze all videos
                all_results = {}
                for video_id in video_ids:
                    try:
                        result = analyzer.analyze_single_video(
                            video_id, 
                            create_visualizations=not args.no_visualizations
                        )
                        all_results[video_id] = result
                    except Exception as e:
                        logger.error(f"Failed to analyze video {video_id}: {e}")
                
                # Compare all videos
                comparison_results = analyzer.compare_trajectories(
                    list(all_results.keys()), 
                    comparison_metrics=args.metrics
                )
                
                # Save results
                results = {
                    'individual_analyses': all_results,
                    'comparison_analysis': comparison_results
                }
                analyzer.save_analysis_results(results, "full_batch_analysis.json")
            
            # Print batch summary
            print(f"\n=== Batch Analysis Complete ===")
            if 'analysis_summary' in results:
                summary = results['analysis_summary']
                print(f"Total videos analyzed: {summary['total_videos']}")
                print(f"Successful analyses: {summary['successful_analyses']}")
                print(f"Failed analyses: {summary['failed_analyses']}")
            
            if 'comparison_analysis' in results:
                comp_data = results['comparison_analysis']['comparison_data']
                print(f"\n=== Key Metrics Summary ===")
                for metric in args.metrics:
                    if metric in comp_data:
                        data = comp_data[metric]
                        print(f"{metric}: {data['mean']:.6f} ± {data['std']:.6f}")
        
        logger.info("Analysis complete!")
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
