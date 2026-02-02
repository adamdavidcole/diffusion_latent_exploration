#!/usr/bin/env python3
"""
Export experiment data for static site deployment.

This script exports experiment data in a format that matches the Flask API responses
exactly, allowing the frontend to work without a backend server.

Usage:
    python scripts/export_static_data.py batch_20251021_144819 batch_20251021_145307
    
    # Or specify output directory
    python scripts/export_static_data.py --output webapp/react-frontend/public batch1 batch2
"""

import sys
import json
import shutil
from pathlib import Path

# Add parent directory to path to import webapp modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from webapp.backend.app import VideoAnalyzer


def write_json(file_path, data):
    """Write JSON file with pretty formatting"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    # Try to show relative path, fall back to absolute if outside cwd
    try:
        display_path = file_path.relative_to(Path.cwd())
    except ValueError:
        display_path = file_path
    print(f"  ✓ Wrote: {display_path}")


def filter_tree_to_batches(tree, batch_names):
    """
    Filter experiment tree to only include specified batches.
    Maintains tree structure but removes unselected experiments.
    """
    if tree['type'] == 'experiment':
        # Check if this experiment is in the batch list
        return tree if tree['name'] in batch_names else None
    
    # Recursively filter children
    filtered_children = []
    for child in tree.get('children', []):
        filtered_child = filter_tree_to_batches(child, batch_names)
        if filtered_child:
            filtered_children.append(filtered_child)
    
    # Only include folders that still have children after filtering
    if filtered_children:
        tree['children'] = filtered_children
        return tree
    
    return None


def copy_analysis_schema(output_path):
    """Copy analysis schema from src to output directory"""
    schema_path = Path(__file__).parent.parent / 'src' / 'analysis' / 'vlm_analysis' / 'vlm_analysis_schema_new.json'
    
    if not schema_path.exists():
        print(f"  ⚠ Warning: Analysis schema not found at {schema_path}")
        return False
    
    shutil.copy(schema_path, output_path)
    # Try to show relative path, fall back to absolute if outside cwd
    try:
        display_path = output_path.relative_to(Path.cwd())
    except ValueError:
        display_path = output_path
    print(f"  ✓ Copied: {display_path}")
    return True


def export_experiments(batch_names, output_dir='webapp/react-frontend/public', r2_base_url=None):
    """
    Export experiment data for static site deployment.
    
    Args:
        batch_names: List of experiment batch names to export
        output_dir: Output directory path (default: webapp/react-frontend/public)
        r2_base_url: R2 bucket base URL for media (optional, for validation)
    
    Reuses VideoAnalyzer methods to match API responses exactly - NO FORKED LOGIC.
    """
    print("\n" + "="*70)
    print("🚀 Static Site Data Export")
    print("="*70)
    
    # Initialize analyzer (same as Flask app)
    outputs_dir = Path('outputs')
    if not outputs_dir.exists():
        print(f"\n❌ Error: outputs directory not found at {outputs_dir.absolute()}")
        sys.exit(1)
    
    print(f"\n📂 Outputs directory: {outputs_dir.absolute()}")
    analyzer = VideoAnalyzer(str(outputs_dir))
    
    # Create output structure
    output_path = Path(output_dir)
    data_dir = output_path / 'data'
    experiments_dir = data_dir / 'experiments'
    
    print(f"📝 Export directory: {data_dir.absolute()}\n")
    
    # Validate batch names exist
    print("🔍 Validating batch names...")
    missing_batches = []
    for batch_name in batch_names:
        batch_path = outputs_dir / batch_name
        if not batch_path.exists():
            missing_batches.append(batch_name)
            print(f"  ❌ Not found: {batch_name}")
        else:
            print(f"  ✓ Found: {batch_name}")
    
    if missing_batches:
        print(f"\n❌ Error: {len(missing_batches)} batch(es) not found. Aborting.")
        sys.exit(1)
    
    # Export each batch
    print(f"\n{'='*70}")
    print(f"📦 Exporting {len(batch_names)} experiment batch(es)")
    print(f"{'='*70}\n")
    
    for i, batch_name in enumerate(batch_names, 1):
        print(f"[{i}/{len(batch_names)}] Processing: {batch_name}")
        print("-" * 70)
        
        batch_dir = experiments_dir / batch_name
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. Get experiment details (reuses backend function)
            print("  → Fetching experiment data...")
            exp_data = analyzer.get_experiment_by_path(batch_name)
            if not exp_data:
                print(f"  ❌ Failed to get experiment data for {batch_name}")
                continue
            write_json(batch_dir / 'experiment.json', exp_data)
            
            # 2. Get analysis (reuses backend function)
            print("  → Fetching VLM analysis...")
            analysis_data = analyzer._load_vlm_analysis(outputs_dir / batch_name)
            write_json(batch_dir / 'analysis.json', analysis_data)
            
            # 3. Get trajectory analysis
            print("  → Fetching trajectory analysis...")
            trajectory_data = analyzer._load_trajectory_analysis(outputs_dir / batch_name)
            write_json(batch_dir / 'trajectory-analysis.json', trajectory_data)
            
            # 4. Get latent + attention videos (combined like API endpoint)
            print("  → Fetching latent and attention videos...")
            latent_data = analyzer._load_latent_videos(outputs_dir / batch_name)
            attention_vids = analyzer._load_attention_videos(outputs_dir / batch_name)
            latent_combined = {**latent_data, **attention_vids}
            write_json(batch_dir / 'latent-videos.json', latent_combined)
            
            # 5. Get attention bending data
            print("  → Fetching attention bending data...")
            bending_data = analyzer._load_attention_bending_data(outputs_dir / batch_name)
            write_json(batch_dir / 'attention-bending.json', bending_data)
            
            print(f"  ✅ Completed: {batch_name}\n")
            
        except Exception as e:
            print(f"  ❌ Error exporting {batch_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Export filtered experiments summary tree
    print(f"{'='*70}")
    print("📊 Generating experiments summary tree...")
    print("-" * 70)
    
    try:
        full_tree = analyzer.scan_outputs(summary_only=True)
        filtered_tree = filter_tree_to_batches(full_tree, batch_names)
        
        if not filtered_tree:
            print("  ⚠ Warning: No experiments matched in tree structure")
            filtered_tree = {"type": "folder", "name": "outputs", "path": "", "children": []}
        
        write_json(data_dir / 'experiments_summary.json', filtered_tree)
        print("  ✅ Summary tree generated\n")
        
    except Exception as e:
        print(f"  ❌ Error generating summary tree: {e}")
        import traceback
        traceback.print_exc()
    
    # Copy analysis schema
    print(f"{'='*70}")
    print("📋 Copying analysis schema...")
    print("-" * 70)
    
    copy_analysis_schema(data_dir / 'analysis_schema.json')
    print()
    
    # Summary report
    print(f"{'='*70}")
    print("✅ EXPORT COMPLETE")
    print(f"{'='*70}")
    print(f"📦 Exported {len(batch_names)} batch(es)")
    print(f"📁 Output location: {data_dir.absolute()}")
    print(f"\n📂 Directory structure:")
    # Try to show relative path, fall back to name if outside cwd
    try:
        display_path = data_dir.relative_to(Path.cwd())
    except ValueError:
        display_path = data_dir.name
    print(f"   {display_path}/")
    print(f"   ├── experiments_summary.json")
    print(f"   ├── analysis_schema.json")
    print(f"   └── experiments/")
    for batch_name in batch_names:
        print(f"       ├── {batch_name}/")
        print(f"       │   ├── experiment.json")
        print(f"       │   ├── analysis.json")
        print(f"       │   ├── trajectory-analysis.json")
        print(f"       │   ├── latent-videos.json")
        print(f"       │   └── attention-bending.json")
    
    if r2_base_url:
        print(f"\n🌐 R2 bucket base URL: {r2_base_url}")
        print(f"\n📌 Next steps:")
        print(f"   1. Upload experiment folders to R2:")
        for batch_name in batch_names:
            print(f"      rclone copy outputs/{batch_name} r2:your-bucket/outputs/{batch_name}")
        print(f"   2. Build frontend with static mode:")
        print(f"      cd webapp/react-frontend")
        print(f"      VITE_STATIC_MODE=true VITE_R2_BASE_URL={r2_base_url} npm run build")
        print(f"   3. Deploy to Netlify (drop dist/ folder)")
    else:
        print(f"\n📌 Next steps:")
        print(f"   1. Upload experiment folders to R2 bucket")
        print(f"   2. Modify frontend api.js for static mode")
        print(f"   3. Build and deploy frontend")
    
    print(f"\n{'='*70}\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Export experiment data for static site deployment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export specific batches
  python scripts/export_static_data.py batch_20251021_144819 batch_20251021_145307
  
  # Specify output directory
  python scripts/export_static_data.py --output webapp/react-frontend/public batch1 batch2
  
  # Include R2 base URL for validation
  python scripts/export_static_data.py --r2-url https://pub-xxx.r2.dev batch1 batch2
        """
    )
    
    parser.add_argument(
        'batches',
        nargs='+',
        help='Experiment batch names to export (e.g., batch_20251021_144819)'
    )
    
    parser.add_argument(
        '--output',
        default='webapp/react-frontend/public',
        help='Output directory path (default: webapp/react-frontend/public)'
    )
    
    parser.add_argument(
        '--r2-url',
        help='R2 bucket base URL (optional, for validation/documentation)'
    )
    
    args = parser.parse_args()
    
    # Run export
    export_experiments(
        batch_names=args.batches,
        output_dir=args.output,
        r2_base_url=args.r2_url
    )


if __name__ == '__main__':
    main()
