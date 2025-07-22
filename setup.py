#!/usr/bin/env python3
"""
Setup script for WAN 1.3B Video Generation Project
Initializes the project with all necessary files and configurations.
"""

import os
import sys
from pathlib import Path
import subprocess


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True


def create_directory_structure():
    """Create the complete directory structure."""
    directories = [
        "src/config",
        "src/generators", 
        "src/prompts",
        "src/utils",
        "configs/templates",
        "outputs",
        "logs",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")


def install_dependencies():
    """Install required Python packages."""
    print("\n📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  Failed to install dependencies automatically")
        print("Please run: pip install -r requirements.txt")
        return False


def create_default_configs():
    """Create default configuration if they don't exist."""
    configs_to_check = [
        "configs/default.yaml",
        "configs/fast_test.yaml", 
        "configs/high_quality.yaml"
    ]
    
    for config_path in configs_to_check:
        if not Path(config_path).exists():
            print(f"⚠️  Missing config file: {config_path}")
        else:
            print(f"✅ Config file exists: {config_path}")


def run_validation():
    """Run system validation."""
    print("\n🔍 Running system validation...")
    
    try:
        result = subprocess.run([sys.executable, "main.py", "--validate"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ System validation passed")
            return True
        else:
            print("⚠️  System validation failed:")
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
            return False
            
    except FileNotFoundError:
        print("❌ main.py not found")
        return False


def show_usage_examples():
    """Display usage examples."""
    print(f"\n{'='*60}")
    print("🎯 USAGE EXAMPLES")
    print(f"{'='*60}")
    
    examples = [
        ("Quick Preview", 
         'python main.py --preview --template "a [cute|playful] [cat|dog]"'),
        
        ("Fast Test Generation",
         'python main.py --config configs/fast_test.yaml --template "a [happy|sad] person"'),
        
        ("High Quality Generation", 
         'python main.py --config configs/high_quality.yaml --template "a romantic scene with [two people|a couple]" --videos-per-variation 5'),
        
        ("Analysis Mode",
         'python main.py --analyze --template "a [action] scene in [location]"'),
        
        ("Create Example Templates",
         'python main.py --create-examples'),
        
        ("Custom Batch",
         'python main.py --template "your template here" --batch-name "my_batch" --videos-per-variation 3')
    ]
    
    for title, command in examples:
        print(f"\n{title}:")
        print(f"  {command}")


def main():
    """Main setup function."""
    print("🚀 Setting up WAN 1.3B Video Generation Project")
    print("=" * 50)
    
    # Check requirements
    if not check_python_version():
        sys.exit(1)
    
    # Create directory structure
    print("\n📁 Creating directory structure...")
    create_directory_structure()
    
    # Check configurations
    print("\n⚙️  Checking configurations...")
    create_default_configs()
    
    # Install dependencies (optional for demo)
    install_choice = input("\n📦 Install Python dependencies? (y/N): ").lower().strip()
    if install_choice in ['y', 'yes']:
        install_dependencies()
    else:
        print("⏭️  Skipping dependency installation")
        print("   Run 'pip install -r requirements.txt' later if needed")
    
    # Run validation
    run_validation()
    
    # Show examples
    show_usage_examples()
    
    print(f"\n{'='*60}")
    print("🎉 SETUP COMPLETE!")
    print("Your WAN 1.3B Video Generation project is ready to use!")
    
    print("\nNext steps:")
    print("  1. Run the demo: python demo.py")
    print("  2. Try a preview: python main.py --preview --template 'your template'") 
    print("  3. Generate videos: python main.py --template 'your template'")
    print("  4. Check outputs in the outputs/ directory")
    
    print(f"\n📚 Documentation: README.md")
    print(f"🔧 Configuration: configs/")
    print(f"📁 Templates: configs/templates/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
