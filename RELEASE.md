# Release Guide for fujielab-audio-mcnr_input v0.1.0

This document outlines the steps to release version 0.1.0 to both GitHub and PyPI.

## Prerequisites

1. **GitHub Account**: With push access to the repository
2. **PyPI Account**: Register at https://pypi.org/
3. **PyPI API Token**: Create at https://pypi.org/manage/account/token/
4. **Git**: Properly configured with your credentials
5. **Python Environment**: With required build tools

## Pre-Release Checklist

- [ ] All code changes committed and pushed
- [ ] Version number updated in `pyproject.toml` (currently 0.1.0)
- [ ] `CHANGELOG.md` updated with release notes
- [ ] Tests passing locally
- [ ] Documentation (README.md) is up to date
- [ ] License file is correct (MIT)

## Release Steps

### 1. Prepare the Release Build

```bash
# Navigate to project directory
cd /path/to/fujielab-audio-mcnr_input

# Run the release preparation script
python scripts/prepare_release.py
```

This script will:
- Clean previous builds
- Install build dependencies
- Run tests
- Build the package (wheel and source distribution)
- Validate the package

### 2. Test the Package Locally

```bash
# Create a test environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install the built package
pip install dist/fujielab_audio_mcnr_input-0.1.0-py3-none-any.whl

# Test import
python -c "from fujielab.audio.mcnr_input.core import InputStream; print('Import successful')"

# Deactivate and remove test environment
deactivate
rm -rf test_env
```

### 3. Create GitHub Release

```bash
# Create and push the release tag
git tag v0.1.0
git push origin v0.1.0
```

The GitHub Actions workflow will automatically:
- Run tests on multiple platforms
- Build the package
- Create a GitHub release
- Upload to PyPI (if configured)

### 4. Manual PyPI Upload (if not using GitHub Actions)

```bash
# Install twine if not already installed
pip install twine

# Upload to PyPI
twine upload dist/*
```

You'll be prompted for:
- Username: `__token__`
- Password: Your PyPI API token (starts with `pypi-`)

### 5. Verify the Release

#### GitHub
- Check that the release appears at: https://github.com/fujielab/fujielab-audio-mcnr_input/releases
- Verify release notes and attached files

#### PyPI
- Check that the package appears at: https://pypi.org/project/fujielab-audio-mcnr_input/
- Test installation: `pip install fujielab-audio-mcnr_input==0.1.0`

## Post-Release Tasks

1. **Update Documentation**: Ensure README badges point to correct version
2. **Announce Release**: Social media, mailing lists, etc.
3. **Monitor Issues**: Watch for installation or usage problems
4. **Plan Next Release**: Update project roadmap

## GitHub Actions Setup (Optional)

To enable automatic PyPI publishing via GitHub Actions:

1. **Add PyPI API Token to GitHub Secrets**:
   - Go to your repository settings
   - Navigate to Secrets and variables → Actions
   - Add a new secret named `PYPI_API_TOKEN`
   - Paste your PyPI API token as the value

2. **Configure Repository Environment**:
   - Go to Settings → Environments
   - Create an environment named `release`
   - Add protection rules as needed

## Troubleshooting

### Build Issues
- **Import errors**: Check `PYTHONPATH` and package structure
- **Missing files**: Update `MANIFEST.in` to include necessary files
- **Version conflicts**: Ensure version in `pyproject.toml` is correct

### PyPI Upload Issues
- **Authentication**: Verify API token is correct and has upload permissions
- **Package exists**: You cannot overwrite existing versions on PyPI
- **Package validation**: Run `twine check dist/*` to verify package integrity

### GitHub Actions Issues
- **Secrets**: Ensure `PYPI_API_TOKEN` is correctly set in repository secrets
- **Permissions**: Check that Actions have necessary permissions
- **Workflow syntax**: Validate YAML syntax in workflow files

## Support

For issues with the release process:
1. Check GitHub Actions logs for CI/CD issues
2. Review PyPI upload logs for publication problems
3. Consult PyPI and GitHub documentation
4. Contact repository maintainers

## Version History

- **v0.1.0** (2025-06-26): Initial release
  - Multi-channel audio capture
  - Cross-platform support (macOS/Windows)
  - Real-time processing with callbacks
  - Comprehensive documentation
