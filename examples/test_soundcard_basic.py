#!/usr/bin/env python3
"""
Simple test for soundcard library functionality
"""

try:
    import soundcard as sc
    print("✓ soundcard library imported successfully")
    
    # Test basic functionality
    print("\n=== Available Speakers ===")
    speakers = sc.all_speakers()
    for i, speaker in enumerate(speakers):
        print(f"{i}: {speaker.name}")
    
    print(f"\nDefault speaker: {sc.default_speaker().name}")
    
    print("\n=== Available Microphones (including loopback) ===")
    microphones = sc.all_microphones(include_loopback=True)
    for i, mic in enumerate(microphones):
        mic_type = "Loopback" if getattr(mic, 'isloopback', False) else "Microphone"
        print(f"{i}: {mic.name} ({mic_type})")
    
    # Test finding loopback devices
    loopback_devices = [mic for mic in microphones if getattr(mic, 'isloopback', False)]
    print(f"\n=== Found {len(loopback_devices)} explicit loopback devices ===")
    for i, device in enumerate(loopback_devices):
        print(f"{i}: {device.name}")
    
    print("\n✓ soundcard basic functionality test completed successfully")

except ImportError as e:
    print(f"✗ Failed to import soundcard: {e}")
except Exception as e:
    print(f"✗ Error during soundcard test: {e}")
    import traceback
    traceback.print_exc()
