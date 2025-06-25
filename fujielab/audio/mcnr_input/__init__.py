"""
fujielab.audio.mcnr_input - Multi-channel noise reduction based echo cancelling audio input stream

This package provides audio capture and processing capabilities for
multi-channel noise reduction and echo cancellation.
"""

from .core import CaptureConfig, CallbackFlags, InputStream

__version__ = "0.1.0"
__all__ = [
    "CaptureConfig",
    "CallbackFlags",
    "InputStream",
]