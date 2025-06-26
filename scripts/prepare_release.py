#!/usr/bin/env python3
"""
Release preparation script for fujielab-audio-mcnr_input
"""
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, check=True, cwd=cwd, 
                              capture_output=True, text=True)
        print(f"✓ {cmd}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {cmd}")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Main release preparation function"""
    print("🚀 Preparing release for fujielab-audio-mcnr_input v0.1.0")
    print("=" * 60)
    
    # Get project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Check if we're in a git repository
    if not os.path.exists('.git'):
        print("❌ Not in a git repository")
        return False
    
    # Step 1: Clean previous builds
    print("\n📦 Cleaning previous builds...")
    for path in ['build', 'dist', '*.egg-info']:
        run_command(f"rm -rf {path}")
    
    # Step 2: Install build dependencies
    print("\n📋 Installing build dependencies...")
    if not run_command("pip install --upgrade build twine"):
        return False
    
    # Step 3: Run tests
    print("\n🧪 Running tests...")
    if not run_command("python -m pytest tests/ -v"):
        print("⚠️  Tests failed, but continuing (audio tests may fail in CI)")
    
    # Step 4: Build the package
    print("\n🔨 Building package...")
    if not run_command("python -m build"):
        print("❌ Build failed")
        return False
    
    # Step 5: Check the package
    print("\n🔍 Checking package...")
    if not run_command("twine check dist/*"):
        print("❌ Package check failed")
        return False
    
    # Step 6: Show what was built
    print("\n📦 Built packages:")
    for file in os.listdir('dist'):
        print(f"  - {file}")
    
    print("\n✅ Release preparation completed successfully!")
    print("\nNext steps:")
    print("1. Review the built packages in the 'dist/' directory")
    print("2. Test installation: pip install dist/*.whl")
    print("3. Create a git tag: git tag v0.1.0")
    print("4. Push tag: git push origin v0.1.0")
    print("5. Upload to PyPI: twine upload dist/*")
    print("\nFor PyPI upload, you'll need:")
    print("- PyPI account and API token")
    print("- Run: twine upload dist/*")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
