#!/usr/bin/env python3
"""
Test script to verify abstract base class implementation

抽象基底クラス実装の検証用テストスクリプト
"""

import platform
from abc import ABC

def test_abstract_base_class():
    """Test that the abstract base class is properly implemented"""
    print("=== Abstract Base Class Implementation Test ===")
    print(f"Current platform: {platform.system()}")
    
    # Test import of base class
    print("\n1. Testing abstract base class import:")
    try:
        from fujielab.audio.mcnr_input._backend.input_capture_base import InputCaptureBase
        print(f"✓ Successfully imported InputCaptureBase")
        print(f"✓ InputCaptureBase class: {InputCaptureBase}")
        print(f"✓ Is abstract: {hasattr(InputCaptureBase, '__abstractmethods__')}")
        print(f"✓ Abstract methods: {getattr(InputCaptureBase, '__abstractmethods__', set())}")
    except ImportError as e:
        print(f"✗ Failed to import InputCaptureBase: {e}")
        return False
    
    # Test that base class cannot be instantiated directly
    print("\n2. Testing that base class cannot be instantiated:")
    try:
        instance = InputCaptureBase()
        print(f"✗ Base class should not be instantiable, but got: {instance}")
        return False
    except TypeError as e:
        print(f"✓ Correctly prevented direct instantiation: {e}")
    
    # Test platform-specific class inheritance
    print("\n3. Testing platform-specific class inheritance:")
    try:
        from fujielab.audio.mcnr_input._backend.input_capture import InputCapture
        print(f"✓ Successfully imported InputCapture wrapper")
        print(f"✓ InputCapture class: {InputCapture}")
        print(f"✓ Is subclass of InputCaptureBase: {issubclass(InputCapture, InputCaptureBase)}")
        print(f"✓ Module: {InputCapture.__module__}")
    except ImportError as e:
        print(f"✗ Failed to import InputCapture: {e}")
        return False
    
    # Test concrete implementation can be instantiated
    print("\n4. Testing concrete implementation instantiation:")
    try:
        input_capture = InputCapture(sample_rate=16000, channels=1, debug=False)
        print(f"✓ Successfully created InputCapture instance: {type(input_capture)}")
        print(f"✓ Instance is subclass of base: {isinstance(input_capture, InputCaptureBase)}")
        
        # Test properties from base class
        print(f"✓ Sample rate: {input_capture.sample_rate}")
        print(f"✓ Channels: {input_capture.channels}")
        print(f"✓ Blocksize: {input_capture.blocksize}")
        print(f"✓ Time property: {input_capture.time}")
        
    except Exception as e:
        print(f"✗ Failed to create InputCapture instance: {e}")
        return False
    
    # Test that all abstract methods are implemented
    print("\n5. Testing abstract method implementations:")
    
    # Check for abstract methods
    abstract_methods = getattr(InputCaptureBase, '__abstractmethods__', set())
    for method_name in abstract_methods:
        if hasattr(input_capture, method_name):
            method = getattr(input_capture, method_name)
            print(f"✓ {method_name}: implemented as {method}")
        else:
            print(f"✗ {method_name}: not implemented")
            return False
    
    # Test method calls (without actually starting capture)
    try:
        # Test static method
        InputCapture.list_audio_devices(debug=False)
        print(f"✓ list_audio_devices method works")
        
        # Note: We won't test start_audio_capture, read_audio_capture, stop_audio_capture
        # to avoid side effects, but they should be callable
        print(f"✓ start_audio_capture method exists: {hasattr(input_capture, 'start_audio_capture')}")
        print(f"✓ read_audio_capture method exists: {hasattr(input_capture, 'read_audio_capture')}")
        print(f"✓ stop_audio_capture method exists: {hasattr(input_capture, 'stop_audio_capture')}")
        
    except Exception as e:
        print(f"✗ Failed method calls: {e}")
        return False
    
    print("\n✓ All abstract base class tests passed!")
    return True

if __name__ == "__main__":
    test_abstract_base_class()
