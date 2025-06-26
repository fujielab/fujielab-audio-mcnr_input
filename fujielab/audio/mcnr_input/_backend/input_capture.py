"""
Audio Input Capture Module

This module provides classes for capturing audio from various input sources (e.g., microphones).
Platform-specific implementations are automatically selected.

入力オーディオキャプチャモジュール

このモジュールは、各種オーディオ入力（マイク等）からのオーディオをキャプチャするためのクラスを提供します。
プラットフォーム固有の実装が自動的に選択されます。
"""
import platform

# Import base class for type hints
from .input_capture_base import InputCaptureBase

# Import platform-specific InputCapture classes
if platform.system() == "Darwin":
    from .input_capture_mac import InputCaptureMac as InputCapture
    from .input_capture_mac import create_input_capture_instance, list_devices
else:
    from .input_capture_win import InputCaptureWin as InputCapture
    from .input_capture_win import create_input_capture_instance, list_devices

# Re-export the platform-appropriate class and functions for backward compatibility
__all__ = ['InputCapture', 'InputCaptureBase', 'create_input_capture_instance', 'list_devices']
