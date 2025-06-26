# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-06-26

### Added
- Initial release of fujielab-audio-mcnr_input
- Multi-channel audio capture functionality for both input (microphone) and output (speaker) devices
- Cross-platform support for macOS and Windows
- Real-time audio streaming with callback support
- macOS integration with BlackHole and SwitchAudioSource
- Windows integration with WASAPI loopback capture
- Flexible configuration with CaptureConfig class
- Time synchronization between input and output streams
- Comprehensive documentation and setup instructions
- Example scripts for basic and advanced usage

### Features
- **InputStream**: Main class for managing multi-channel audio capture
- **CaptureConfig**: Configuration class for individual capture devices
- **Platform-specific backends**: Optimized implementations for macOS and Windows
- **Echo cancellation ready**: Designed for noise reduction applications
- **Low-latency processing**: Configurable block sizes and latency settings

### Dependencies
- numpy >= 1.20.0
- sounddevice >= 0.4.0
- soundfile >= 0.10.0
- soundcard >= 0.4.0

### Requirements
- Python 3.8+
- macOS: BlackHole 2ch, SwitchAudioSource, fujielab-output multi-output device
- Windows: Default audio drivers (WASAPI)

[0.1.0]: https://github.com/fujielab/fujielab-audio-mcnr_input/releases/tag/v0.1.0
