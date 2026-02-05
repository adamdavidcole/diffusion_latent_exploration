#!/usr/bin/env python3
"""
Upload experiment batches to Cloudflare R2 bucket using rclone.

This script uploads video files and thumbnails from experiment batches to R2
for static site deployment. By default, it OVERWRITES existing files (--sync mode).

Prerequisites:
    - rclone installed and configured with remote named 'r2'
    - See docs/CLOUDFLARE_R2_SETUP.md for setup instructions

Usage:
    # Upload single batch
    python scripts/upload_to_r2.py batch_20260130_014727
    
    # Upload multiple batches
    python scripts/upload_to_r2.py batch1 batch2 batch3
    
    # Custom bucket and remote name
    python scripts/upload_to_r2.py --bucket my-bucket --remote my-r2 batch1
    
    # Dry run (show what would be uploaded)
    python scripts/upload_to_r2.py --dry-run batch1
    
    # Include all files (not just videos/images)
    python scripts/upload_to_r2.py --include-all batch1
"""

import sys
import subprocess
import argparse
from pathlib import Path


def check_rclone():
    """Check if rclone is installed"""
    try:
        result = subprocess.run(
            ['rclone', 'version'],
            capture_output=True,
            text=True,
            check=True
        )
        version_line = result.stdout.split('\n')[0]
        print(f"✓ Rclone found: {version_line}")
        return True
    except FileNotFoundError:
        print("❌ Error: rclone not found. Please install rclone:")
        print("   Linux/WSL: sudo apt install rclone")
        print("   macOS: brew install rclone")
        print("   Or visit: https://rclone.org/install/")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Error checking rclone: {e}")
        return False


def check_remote(remote_name):
    """Check if rclone remote is configured"""
    try:
        result = subprocess.run(
            ['rclone', 'listremotes'],
            capture_output=True,
            text=True,
            check=True
        )
        remotes = [line.strip().rstrip(':') for line in result.stdout.split('\n') if line.strip()]
        
        if remote_name in remotes:
            print(f"✓ Rclone remote '{remote_name}' found")
            return True
        else:
            print(f"❌ Error: rclone remote '{remote_name}' not configured")
            print(f"   Available remotes: {', '.join(remotes) if remotes else 'none'}")
            print(f"   Configure with: rclone config")
            print(f"   See: docs/CLOUDFLARE_R2_SETUP.md")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Error checking rclone remotes: {e}")
        return False


def check_bucket(remote_name, bucket_name):
    """Check if bucket exists and is accessible"""
    try:
        result = subprocess.run(
            ['rclone', 'lsd', f'{remote_name}:'],
            capture_output=True,
            text=True,
            check=True
        )
        buckets = [line.split()[-1] for line in result.stdout.split('\n') if line.strip()]
        
        if bucket_name in buckets:
            print(f"✓ Bucket '{bucket_name}' found and accessible")
            return True
        else:
            print(f"❌ Error: bucket '{bucket_name}' not found")
            print(f"   Available buckets: {', '.join(buckets) if buckets else 'none'}")
            print(f"   Create bucket in Cloudflare R2 dashboard")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Error checking bucket: {e}")
        print(f"   Error output: {e.stderr if hasattr(e, 'stderr') else 'unknown'}")
        return False


def get_batch_size(batch_path, include_all=False):
    """Calculate total size of files to upload"""
    if include_all:
        patterns = ['**/*']
    else:
        patterns = ['**/*.mp4', '**/*.jpg', '**/*.jpeg', '**/*.png']
    
    total_size = 0
    file_count = 0
    
    for pattern in patterns:
        for file_path in batch_path.glob(pattern):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
    
    return total_size, file_count


def format_size(size_bytes):
    """Format bytes to human-readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def upload_batch(batch_name, remote_name, bucket_name, outputs_dir, 
                 dry_run=False, include_all=False, verbose=False, copy_only=False):
    """
    Upload a single batch to R2 using rclone sync or copy.
    
    Uses 'rclone sync' (default) which:
    - Uploads new/modified files
    - OVERWRITES existing files with same name
    - Deletes files in R2 that don't exist locally (maintains exact copy)
    
    Uses 'rclone copy' (--copy-only flag) which:
    - Uploads new files only
    - OVERWRITES existing files with same name
    - DOES NOT delete remote files (safer for incremental updates)
    
    Args:
        batch_name: Experiment batch name
        remote_name: Rclone remote name (e.g., 'r2')
        bucket_name: R2 bucket name
        outputs_dir: Local outputs directory
        dry_run: If True, show what would be uploaded without actually uploading
        include_all: If True, upload all files. If False, only videos/images
        verbose: If True, show detailed rclone output
        copy_only: If True, use 'copy' instead of 'sync' (safer for adding new files)
    
    Returns:
        True if successful, False otherwise
    """
    batch_path = outputs_dir / batch_name
    
    if not batch_path.exists():
        print(f"  ❌ Batch directory not found: {batch_path}")
        return False
    
    # Calculate upload size
    total_size, file_count = get_batch_size(batch_path, include_all)
    
    if file_count == 0:
        print(f"  ⚠ No files found to upload in {batch_name}")
        return False
    
    print(f"  📦 Files to upload: {file_count} ({format_size(total_size)})")
    
    # Build rclone command
    source = str(batch_path)
    destination = f'{remote_name}:{bucket_name}/{batch_name}'
    
    # Use 'copy' for incremental updates (safer), 'sync' for full mirror
    rclone_cmd = 'copy' if copy_only else 'sync'
    cmd = ['rclone', rclone_cmd, source, destination]
    
    if copy_only:
        print(f"  📋 Using 'copy' mode - will add/update files without deleting remote files")
    else:
        print(f"  🔄 Using 'sync' mode - will mirror local to remote (may delete remote files)")
    
    # Add filters for file types and directories
    if not include_all:
        # Include specific directories that frontend needs
        cmd.extend([
            '--include', 'videos/**',           # Main videos
            '--include', 'attention_videos/**', # Attention visualizations (if present)
            '--include', 'latents_videos/**',   # Latent visualizations (if present)
            '--include', 'configs/video_metadata*.json',  # Metadata files
            '--include', '*.mp4',               # All video files
            '--include', '*.jpg',               # All thumbnails/images
            '--include', '*.jpeg',
            '--include', '*.png',
            '--include', '*.gif',
            '--exclude', '*'  # Exclude everything else
        ])
    
    # Add flags
    cmd.extend([
        '--progress',  # Show progress
        '--transfers', '4',  # Parallel transfers (adjust based on bandwidth)
        '--checkers', '8',  # Parallel checksum operations
    ])
    
    if dry_run:
        cmd.append('--dry-run')
        print(f"  🔍 DRY RUN MODE - No files will be uploaded")
    
    if verbose:
        cmd.append('-v')
    
    # Show command
    print(f"  → Running: {' '.join(cmd)}")
    print()
    
    try:
        # Run rclone with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output
        for line in process.stdout:
            # Indent output for readability
            print(f"    {line.rstrip()}")
        
        process.wait()
        
        if process.returncode == 0:
            if dry_run:
                print(f"  ✓ Dry run completed successfully")
            else:
                print(f"  ✅ Upload completed: {batch_name}")
                print(f"     R2 URL: https://pub-XXXXX.r2.dev/outputs/{batch_name}/")
            return True
        else:
            print(f"  ❌ Upload failed with exit code {process.returncode}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error during upload: {e}")
        return False


def upload_batches(batch_names, remote_name='r2', bucket_name='attention-bender',
                   outputs_dir='outputs', dry_run=False, include_all=False, verbose=False, copy_only=False):
    """Upload multiple batches to R2"""
    
    print("\n" + "="*70)
    print("🚀 R2 Batch Upload")
    print("="*70)
    
    # Validate environment
    print("\n🔍 Validating environment...")
    print("-" * 70)
    
    if not check_rclone():
        sys.exit(1)
    
    if not check_remote(remote_name):
        sys.exit(1)
    
    if not check_bucket(remote_name, bucket_name):
        sys.exit(1)
    
    # Convert outputs_dir to Path
    outputs_path = Path(outputs_dir)
    if not outputs_path.exists():
        print(f"\n❌ Error: outputs directory not found at {outputs_path.absolute()}")
        sys.exit(1)
    
    print(f"✓ Outputs directory: {outputs_path.absolute()}")
    print()
    
    # Validate batch names
    print("🔍 Validating batch names...")
    print("-" * 70)
    
    missing_batches = []
    valid_batches = []
    
    for batch_name in batch_names:
        batch_path = outputs_path / batch_name
        if not batch_path.exists():
            missing_batches.append(batch_name)
            print(f"  ❌ Not found: {batch_name}")
        else:
            valid_batches.append(batch_name)
            print(f"  ✓ Found: {batch_name}")
    
    if missing_batches:
        print(f"\n❌ Error: {len(missing_batches)} batch(es) not found. Aborting.")
        sys.exit(1)
    
    print()
    
    # Upload batches
    print("="*70)
    print(f"📤 Uploading {len(valid_batches)} batch(es) to R2")
    if dry_run:
        print("   (DRY RUN MODE - No actual uploads)")
    print("="*70)
    print()
    
    successful = []
    failed = []
    
    for i, batch_name in enumerate(valid_batches, 1):
        print(f"[{i}/{len(valid_batches)}] Processing: {batch_name}")
        print("-" * 70)
        
        success = upload_batch(
            batch_name=batch_name,
            remote_name=remote_name,
            bucket_name=bucket_name,
            outputs_dir=outputs_path,
            dry_run=dry_run,
            include_all=include_all,
            verbose=verbose,
            copy_only=copy_only
        )
        
        if success:
            successful.append(batch_name)
        else:
            failed.append(batch_name)
        
        print()
    
    # Summary
    print("="*70)
    if dry_run:
        print("✅ DRY RUN COMPLETE")
    else:
        print("✅ UPLOAD COMPLETE")
    print("="*70)
    print(f"✓ Successful: {len(successful)}")
    if failed:
        print(f"✗ Failed: {len(failed)}")
        for batch_name in failed:
            print(f"  - {batch_name}")
    
    print(f"\n📊 Remote location: {remote_name}:{bucket_name}/outputs/")
    
    if not dry_run and successful:
        print(f"\n🌐 Access uploaded files:")
        print(f"   Base URL: https://pub-XXXXX.r2.dev/outputs/")
        print(f"   (Replace XXXXX with your R2 public bucket ID)")
        print()
        print(f"📌 Next steps:")
        print(f"   1. Verify uploads at: https://dash.cloudflare.com → R2 → {bucket_name}")
        print(f"   2. Test file access: https://pub-XXXXX.r2.dev/outputs/{successful[0]}/videos/")
        print(f"   3. Build frontend with R2 URL:")
        print(f"      cd webapp/react-frontend")
        print(f"      VITE_STATIC_MODE=true VITE_R2_BASE_URL=https://pub-XXXXX.r2.dev npm run build")
    
    print("\n" + "="*70 + "\n")
    
    return len(failed) == 0


def main():
    parser = argparse.ArgumentParser(
        description='Upload experiment batches to Cloudflare R2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload single batch
  python scripts/upload_to_r2.py batch_20260130_014727
  
  # Upload multiple batches
  python scripts/upload_to_r2.py batch1 batch2 batch3
  
  # Dry run (preview what would be uploaded)
  python scripts/upload_to_r2.py --dry-run batch1
  
  # Custom bucket and remote
  python scripts/upload_to_r2.py --bucket my-bucket --remote my-r2 batch1
  
  # Include all files (not just videos/images)
  python scripts/upload_to_r2.py --include-all batch1

Setup:
  See docs/CLOUDFLARE_R2_SETUP.md for rclone configuration
        """
    )
    
    parser.add_argument(
        'batches',
        nargs='+',
        help='Experiment batch names to upload'
    )
    
    parser.add_argument(
        '--remote',
        default='r2',
        help='Rclone remote name (default: r2)'
    )
    
    parser.add_argument(
        '--bucket',
        default='attention-bender',
        help='R2 bucket name (default: attention-bender)'
    )
    
    parser.add_argument(
        '--outputs',
        default='outputs',
        help='Local outputs directory (default: outputs)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be uploaded without actually uploading'
    )
    
    parser.add_argument(
        '--copy-only',
        action='store_true',
        help='Use rclone copy instead of sync (adds/updates files without deleting remote files - recommended for incremental updates)'
    )
    
    parser.add_argument(
        '--include-all',
        action='store_true',
        help='Upload all files (default: only videos/, attention_videos/, latents_videos/, and metadata)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output from rclone'
    )
    
    args = parser.parse_args()
    
    success = upload_batches(
        batch_names=args.batches,
        remote_name=args.remote,
        bucket_name=args.bucket,
        outputs_dir=args.outputs,
        dry_run=args.dry_run,
        include_all=args.include_all,
        verbose=args.verbose,
        copy_only=args.copy_only
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
