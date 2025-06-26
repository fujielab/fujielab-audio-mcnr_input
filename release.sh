#!/bin/bash
# 
# Release script for fujielab-audio-mcnr_input v0.1.0
# Run this script to create and upload the release
#

set -e  # Exit on any error

echo "🚀 Starting release process for fujielab-audio-mcnr_input v0.1.0"
echo "=================================================================="

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Are you in the project root?"
    exit 1
fi

# Check if git is clean
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️  Warning: You have uncommitted changes. Please commit or stash them first."
    git status --short
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 1: Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/

# Step 2: Build the package
echo "📦 Building package..."
python -m build

# Step 3: Check the package
echo "🔍 Checking package..."
python -m twine check dist/*

echo "✅ Package built and validated successfully!"
echo ""
echo "📦 Built packages:"
ls -la dist/

echo ""
echo "🎯 Next steps:"
echo "1. Test the package locally:"
echo "   pip install dist/fujielab_audio_mcnr_input-0.1.0-py3-none-any.whl"
echo ""
echo "2. Create and push git tag:"
echo "   git tag v0.1.0"
echo "   git push origin v0.1.0"
echo ""
echo "3. Upload to PyPI:"
echo "   python -m twine upload dist/*"
echo ""
echo "4. Create GitHub release at:"
echo "   https://github.com/fujielab/fujielab-audio-mcnr_input/releases/new"
echo ""

read -p "Do you want to upload to PyPI now? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "⬆️  Uploading to PyPI..."
    python -m twine upload dist/*
    echo "✅ Package uploaded to PyPI successfully!"
    echo "🎉 Release completed!"
else
    echo "📝 Package is ready for manual upload when you're ready."
fi

echo ""
echo "🎉 Release process completed!"
echo "View your package at: https://pypi.org/project/fujielab-audio-mcnr_input/"
