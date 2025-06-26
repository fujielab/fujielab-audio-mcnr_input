#!/usr/bin/env python3
"""
Test script for soundcard-based InputCapture

soundcardベースのInputCaptureのテストスクリプト
"""

import time
import numpy as np
from fujielab.audio.mcnr_input._backend.input_capture import InputCapture

def test_soundcard_input():
    """Test soundcard input capture"""
    print("=== soundcard InputCapture Test ===")
    
    # List available devices
    print("\n1. Listing available input devices:")
    InputCapture.list_audio_devices(debug=True)
    
    # Create input capture instance
    print("\n2. Creating InputCapture instance:")
    input_capture = InputCapture(sample_rate=16000, channels=1, blocksize=1024, debug=True)
    
    # Start capture
    print("\n3. Starting audio capture:")
    if not input_capture.start_audio_capture():
        print("Failed to start input capture")
        return False
    
    print("Audio capture started successfully!")
    
    # Capture some audio data
    print("\n4. Capturing audio data for 5 seconds:")
    start_time = time.time()
    sample_count = 0
    
    try:
        while time.time() - start_time < 5.0:
            audio_data = input_capture.read_audio_capture()
            if audio_data is not None:
                sample_count += 1
                if sample_count % 10 == 0:  # Print every 10th sample
                    rms = np.sqrt(np.mean(audio_data.data ** 2))
                    print(f"Sample {sample_count}: Time={audio_data.time:.3f}, RMS={rms:.6f}, Shape={audio_data.data.shape}")
            time.sleep(0.01)  # Small delay
            
    except KeyboardInterrupt:
        print("\nCapture interrupted by user")
    
    # Stop capture
    print("\n5. Stopping audio capture:")
    if input_capture.stop_audio_capture():
        print("Audio capture stopped successfully!")
    else:
        print("Failed to stop audio capture")
    
    print(f"\nTotal samples captured: {sample_count}")
    print("Test completed!")
    return True

def test_input_with_specific_device():
    """Test input with specific device selection"""
    print("\n=== Device Selection Test ===")
    
    input_capture = InputCapture(debug=True)
    
    # Try to start with default device
    print("Testing with default device:")
    if input_capture.start_audio_capture():
        time.sleep(1)
        audio_data = input_capture.read_audio_capture()
        if audio_data is not None:
            print(f"Success! Got audio data: {audio_data.data.shape}")
        input_capture.stop_audio_capture()
    
    print("Device selection test completed!")

if __name__ == "__main__":
    try:
        print("Starting soundcard InputCapture tests...")
        test_soundcard_input()
        test_input_with_specific_device()
        print("All tests completed!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
