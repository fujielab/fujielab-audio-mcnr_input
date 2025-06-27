"""Unified OutputCapture interface selecting platform implementation."""

import platform

from .output_capture_base import OutputCapture as OutputCaptureBase

if platform.system() == "Darwin":
    from .output_capture_mac import (
        OutputCaptureMac as OutputCapture,
        create_output_capture_instance,
        list_devices,
        check_fujielab_output_device,
    )
else:
    from .output_capture_win import (
        OutputCaptureWin as OutputCapture,
        create_output_capture_instance,
        list_devices,
    )

    def check_fujielab_output_device(debug: bool = False) -> bool:
        """Stub for macOS-only feature."""
        print("This feature is only supported on macOS")
        return False


__all__ = [
    "OutputCapture",
    "OutputCaptureBase",
    "create_output_capture_instance",
    "list_devices",
    "check_fujielab_output_device",
]
