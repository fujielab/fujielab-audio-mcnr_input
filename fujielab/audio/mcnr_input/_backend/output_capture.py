"""
Output Audio Capture Unified Interface Module

This module provides a unified interface for output capture implementations for macOS and Windows.
It hides the details of platform-specific implementations and makes them accessible through a consistent API.

Usage example:
    from ._backend import output_capture

    # Create an instance
    capture = output_capture.create_output_capture_instance(
        sample_rate=16000, channels=2
    )

    # Start capture
    capture.start_audio_capture(device_name="BlackHole 2ch")

    # Read audio data
    audio_data = capture.read_audio_capture()

    # Stop capture
    capture.stop_audio_capture()
"""

import platform
import numpy as np
import time
import sounddevice as sd
import subprocess
from .data import AudioData

# Import platform-specific implementations
if platform.system() == "Darwin":
    from .output_capture_mac import OutputCaptureMac, AudioData
else:
    from .output_capture_win import OutputCaptureWin, AudioData


# Provide platform-appropriate output capture class
def create_output_capture_instance(sample_rate=44100, channels=2, blocksize=None):
    """
    Create an appropriate output capture instance for the platform

    Parameters:
    -----------
    sample_rate : int, optional
        Sampling rate (Hz) (default: 44100Hz)
    channels : int, optional
        Number of channels (default: 2 channels (stereo))
    blocksize : int, optional
        Block size (number of frames) (uses platform default if not specified)

    Returns:
    --------
    OutputCapture
        Platform-appropriate output capture instance
    """
    if blocksize is None:
        blocksize = 1024 if platform.system() == "Darwin" else 512

    if platform.system() == "Darwin":
        return OutputCaptureMac(
            sample_rate=sample_rate,
            channels=channels,
            blocksize=blocksize
        )
    else:
        return OutputCaptureWin(
            sample_rate=sample_rate,
            channels=channels,
            blocksize=blocksize
        )


def list_devices():
    """
    List available audio devices

    Returns:
    --------
    bool
        True if successful, False if failed
    """
    # Call the list display method according to each platform
    if platform.system() == "Darwin":
        return OutputCaptureMac.list_audio_devices()
    else:
        # Use Windows-specific list_audio_devices if available
        if hasattr(OutputCaptureWin, 'list_audio_devices'):
            return OutputCaptureWin.list_audio_devices()
        else:
            print("\nAvailable audio output devices:")
            try:
                devices = sd.query_devices()
                for i, dev in enumerate(devices):
                    if dev['max_output_channels'] > 0:
                        print(f"[{i}] {dev['name']} (Output Channels: {dev['max_output_channels']}, Host: {dev.get('hostapi')})")

                print("\nTo capture output on Windows, use the output device labeled 'WASAPI'.")
                return True
            except Exception as e:
                print(f"Error retrieving device list: {e}")
                return False


def check_fujielab_output_device():
    """
    Check if the fujielab-output composite device (macOS only) is set up correctly

    Returns:
    --------
    bool
        True if set up correctly, False if there is a problem
    """
    if platform.system() == "Darwin":
        return OutputCaptureMac.check_fujielab_output_device()
    else:
        print("This feature is only supported on macOS")
        return False