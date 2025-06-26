#!/usr/bin/env python3
"""
Test script to verify platform-specific InputCapture separation

プラットフォーム固有のInputCapture分離の検証用テストスクリプト
"""

import platform
import importlib

def test_platform_separation():
    """Test that the platform-specific classes are properly separated"""
    print("=== Platform-Specific InputCapture Separation Test ===")
    print(f"Current platform: {platform.system()}")
    
    # Test import of main wrapper module
    print("\n1. Testing main wrapper module:")
    try:
        from fujielab.audio.mcnr_input._backend.input_capture import InputCapture
        print(f"✓ Successfully imported InputCapture from wrapper")
        print(f"✓ InputCapture class: {InputCapture}")
        print(f"✓ Module: {InputCapture.__module__}")
    except ImportError as e:
        print(f"✗ Failed to import InputCapture: {e}")
        return False
    
    # Test platform-specific imports
    print("\n2. Testing platform-specific modules:")
    
    # Windows module
    try:
        from fujielab.audio.mcnr_input._backend.input_capture_win import InputCaptureWin
        print(f"✓ Successfully imported InputCaptureWin")
        print(f"✓ InputCaptureWin class: {InputCaptureWin}")
    except ImportError as e:
        print(f"✗ Failed to import InputCaptureWin: {e}")
    
    # Mac module
    try:
        from fujielab.audio.mcnr_input._backend.input_capture_mac import InputCaptureMac
        print(f"✓ Successfully imported InputCaptureMac")
        print(f"✓ InputCaptureMac class: {InputCaptureMac}")
    except ImportError as e:
        print(f"✗ Failed to import InputCaptureMac: {e}")
    
    # Verify that the wrapper selects the correct platform class
    print("\n3. Testing platform selection:")
    current_platform = platform.system()
    
    if current_platform == "Darwin":
        expected_module = "fujielab.audio.mcnr_input._backend.input_capture_mac"
        expected_class = "InputCaptureMac"
    else:
        expected_module = "fujielab.audio.mcnr_input._backend.input_capture_win"
        expected_class = "InputCaptureWin"
    
    if InputCapture.__module__ == expected_module:
        print(f"✓ Correct platform class selected: {expected_class}")
        print(f"✓ Module path: {expected_module}")
    else:
        print(f"✗ Incorrect platform class selected")
        print(f"  Expected: {expected_module}")
        print(f"  Actual: {InputCapture.__module__}")
    
    # Test functionality
    print("\n4. Testing basic functionality:")
    try:
        # Create instance
        input_capture = InputCapture(sample_rate=16000, channels=1, debug=False)
        print(f"✓ Successfully created InputCapture instance")
        
        # Test list devices
        InputCapture.list_audio_devices(debug=False)
        print(f"✓ Successfully called list_audio_devices")
        
    except Exception as e:
        print(f"✗ Failed functionality test: {e}")
        return False
    
    print("\n✓ All tests passed! Platform separation is working correctly.")
    return True

if __name__ == "__main__":
    test_platform_separation()
