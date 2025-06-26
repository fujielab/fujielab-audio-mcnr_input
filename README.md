# fujielab-audio-mcnr_input

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/fujielab-audio-mcnr-input.svg)](https://badge.fury.io/py/fujielab-audio-mcnr-input)

A Python library for multi-channel noise reduction based echo cancelling audio input stream. This library provides real-time audio capture from both input devices (microphones) and output devices (speakers) with cross-platform support for Windows and macOS.

## Features

- **Multi-channel audio capture**: Simultaneously capture from microphone and speaker outputs
- **Cross-platform support**: Works on both macOS and Windows
- **Real-time processing**: Low-latency audio streaming with callback support
- **Echo cancellation ready**: Designed for noise reduction and echo cancellation applications
- **Flexible configuration**: Customizable sample rates, block sizes, and device selection
- **Synchronization**: Automatic time synchronization between input and output streams

## Installation

### Prerequisites

⚠️ **Important**: Platform-specific setup is required before installation.

#### macOS Setup (Required)

On macOS, you need to install BlackHole and SwitchAudioSource, then create a multi-output device:

1. **Install required tools via Homebrew:**
   ```bash
   # Install Homebrew if not already installed
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   # Install BlackHole (virtual audio driver)
   brew install blackhole-2ch
   
   # Install SwitchAudioSource (audio device switcher)
   brew install switchaudio-osx
   ```

2. **Create fujielab-output multi-output device:**
   
   ⚠️ **This step is mandatory for the library to work on macOS**
   
   - Open **System Preferences** → **Sound**
   - Go to the **Output** tab
   - Click the **"+"** button at the bottom
   - Select **"Create Multi-Output Device"**
   - In the multi-output device configuration:
     - Check both your **current speakers/headphones** and **BlackHole 2ch**
     - Name the device exactly **"fujielab-output"**
   - Click **"Done"**

3. **Verify the setup:**
   ```bash
   # Check if BlackHole is installed
   brew list blackhole-2ch
   
   # Check if SwitchAudioSource is available
   which SwitchAudioSource
   
   # List available audio devices
   SwitchAudioSource -a -t output
   ```

#### Windows Setup

On Windows, the library uses the default audio drivers. No additional setup is required.

### Install the Package

```bash
pip install fujielab-audio-mcnr_input
```

## Quick Start

### Basic Usage

```python
from fujielab.audio.mcnr_input.core import InputStream, CaptureConfig
import numpy as np
import time

# Configure capture devices
capture_configs = [
    CaptureConfig(capture_type="Input", channels=1),    # Microphone (mono)
    CaptureConfig(capture_type="Output", channels=2),   # Speaker output (stereo)
]

# Audio data storage
audio_data = []

# Callback function for real-time processing
def audio_callback(data, frames, timestamp, flags):
    """
    Callback function called for each audio block
    
    Args:
        data (numpy.ndarray): Audio data (frames, channels)
        frames (int): Number of frames in this block
        timestamp (float): Timestamp of the audio block
        flags: Status flags for the audio stream
    """
    audio_data.append(data.copy())
    
    # Process audio data here
    # For example: noise reduction, echo cancellation, etc.
    print(f"Received {frames} frames at {timestamp:.3f}s")

# Create and start the input stream
stream = InputStream(
    samplerate=16000,           # 16kHz sampling rate
    blocksize=512,              # 512 frames per block
    captures=capture_configs,   # Capture configuration
    callback=audio_callback,    # Callback function
    debug=True                  # Enable debug output
)

# Start capturing
stream.start()

# Capture for 5 seconds
time.sleep(5)

# Stop capturing
stream.stop()

# Save captured audio
if audio_data:
    import soundfile as sf
    all_audio = np.concatenate(audio_data, axis=0)
    sf.write("captured_audio.wav", all_audio, stream.samplerate)
    print(f"Saved {len(all_audio)} samples to captured_audio.wav")
```

### Advanced Configuration

```python
from fujielab.audio.mcnr_input.core import InputStream, CaptureConfig

# Advanced capture configuration
capture_configs = [
    CaptureConfig(
        capture_type="Input",
        device_name="Built-in Microphone",  # Specific device name
        channels=1,
        offset=0.0,  # Time offset in seconds
        extra_settings={"latency": "low"}
    ),
    CaptureConfig(
        capture_type="Output", 
        device_name="fujielab-output",  # Use the multi-output device
        channels=2,
        offset=0.02,  # Compensate for processing delay
    ),
]

# Create stream with custom settings
stream = InputStream(
    samplerate=44100,           # High quality audio
    blocksize=1024,             # Larger block size for efficiency
    captures=capture_configs,
    dtype='float32',            # 32-bit float precision
    latency='low',              # Low latency mode
    debug=True
)
```

## API Reference

### InputStream

The main class for managing multi-channel audio capture.

#### Constructor Parameters

- `samplerate` (int, default=16000): Sampling rate in Hz
- `blocksize` (int, default=512): Number of frames per audio block
- `captures` (List[CaptureConfig], optional): List of capture configurations
- `callback` (callable, optional): Function called for each audio block
- `dtype` (str, default='float32'): Audio data type
- `latency` (str|float, default='high'): Latency setting
- `extra_settings` (dict, optional): Additional platform-specific settings
- `debug` (bool, default=False): Enable debug output

#### Methods

- `start()`: Start audio capture
- `stop()`: Stop audio capture
- `read()`: Read audio data (blocking, when not using callback)

### CaptureConfig

Configuration class for individual capture devices.

#### Parameters

- `capture_type` (str): "Input" for microphones, "Output" for speakers
- `device_name` (str|int, optional): Device name or index
- `channels` (int, default=2): Number of audio channels
- `offset` (float, default=0.0): Time offset in seconds
- `extra_settings` (dict, optional): Additional device-specific settings

## Examples

See the `examples/` directory for more usage examples:

- `test_input_stream.py`: Basic multi-channel capture
- `test_soundcard_basic.py`: Simple soundcard usage
- `test_platform_separation.py`: Platform-specific implementations

## Platform-Specific Notes

### macOS

- **Requires**: BlackHole 2ch, SwitchAudioSource, and fujielab-output device
- **Audio routing**: Uses multi-output device to capture system audio
- **Permissions**: May require microphone permissions in System Preferences

### Windows

- **Uses**: Default Windows audio drivers (WASAPI)
- **Loopback**: Automatically captures system audio output
- **Compatibility**: Works with most audio interfaces

## Troubleshooting

### macOS Issues

**"fujielab-output device not found"**
- Ensure you've created the multi-output device as described in setup
- Verify the device is named exactly "fujielab-output"
- Check that BlackHole 2ch is included in the multi-output device

**"BlackHole 2ch device not found"**
- Reinstall BlackHole: `brew reinstall blackhole-2ch`
- Restart your system after installation
- Check Audio MIDI Setup for the device

**"SwitchAudioSource command not found"**
- Install via Homebrew: `brew install switchaudio-osx`
- Verify installation: `which SwitchAudioSource`

### General Issues

**Audio dropouts or glitches**
- Increase `blocksize` parameter (try 1024 or 2048)
- Use `latency='high'` for more stable capture
- Close unnecessary applications

**No audio captured**
- Check device permissions (especially microphone on macOS)
- Verify correct device names with `list_devices()` methods
- Ensure audio is actually playing/being recorded

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

- **Shinya Fujie** - *Initial work*

## Acknowledgments

- Built on top of [sounddevice](https://python-sounddevice.readthedocs.io/) and [soundfile](https://pysoundfile.readthedocs.io/)
- Uses [BlackHole](https://github.com/ExistentialAudio/BlackHole) for macOS audio routing
- Inspired by real-time audio processing needs in research environments
