#!/usr/bin/env python3
"""
Test script for the new soundcard-based OutputCaptureWin implementation
"""

import time
import numpy as np
from fujielab.audio.mcnr_input._backend.output_capture_win import OutputCaptureWin

def main():
    print("=== Testing soundcard-based OutputCaptureWin ===")
    print("This test will capture audio output from your system for 5 seconds.")
    print("For best results, play some music or video during the test.")
    
    # Create an instance with debug enabled
    capture = OutputCaptureWin(debug=True)
    
    print("\nStarting audio capture...")
    if capture.start_audio_capture():
        print(f"✓ Capture started successfully!")
        print(f"Sample rate: {capture.sample_rate}Hz")
        print(f"Channels: {capture.channels}")
        
        print("\nCollecting data for 5 seconds...")
        print("Tip: Play some audio in another application to see non-zero levels")
        print("Progress: ", end="", flush=True)
        
        try:
            all_audio_data = []
            start_time = time.time()
            block_count = 0
            
            while time.time() - start_time < 5.0:  # 5 seconds
                try:
                    audio_data = capture.read_audio_capture()
                    all_audio_data.append(audio_data.data)
                    block_count += 1
                    
                    # Show progress every 10 blocks
                    if block_count % 10 == 0:
                        print("●", end="", flush=True)
                    else:
                        print(".", end="", flush=True)
                        
                except RuntimeError as e:
                    print(f"\n✗ Error reading audio: {e}")
                    break
                
                time.sleep(0.05)  # Smaller delay for better responsiveness
            
            print(f"\n\n✓ Data collection completed!")
            print(f"Collected {len(all_audio_data)} audio blocks")
            
            if all_audio_data:
                # Calculate some statistics
                all_samples = np.vstack(all_audio_data)
                duration = len(all_samples) / capture.sample_rate
                rms = np.sqrt(np.mean(all_samples**2))
                peak = np.max(np.abs(all_samples))
                
                print(f"\n=== Audio Statistics ===")
                print(f"Total duration: {duration:.2f} seconds")
                print(f"Total samples: {len(all_samples)}")
                print(f"RMS level: {rms:.6f}")
                print(f"Peak level: {peak:.6f}")
                
                if peak > 0.001:
                    print("✓ Audio signal detected!")
                else:
                    print("⚠ No significant audio signal detected.")
                    print("  This might be normal if no audio was playing.")
                
                # Save to file
                try:
                    import soundfile as sf
                    output_file = "soundcard_test_output.wav"
                    sf.write(output_file, all_samples, capture.sample_rate)
                    print(f"✓ Audio saved to: {output_file}")
                except ImportError:
                    print("⚠ soundfile not available, skipping file save")
                except Exception as e:
                    print(f"✗ Error saving file: {e}")
            else:
                print("✗ No audio data collected")
                
        except KeyboardInterrupt:
            print("\n⚠ Stopped by user")
        finally:
            capture.stop_audio_capture()
            print("✓ Audio capture stopped cleanly")
    else:
        print("✗ Failed to start audio capture")
        print("\nPossible solutions:")
        print("1. Run as administrator")
        print("2. Enable 'Stereo Mix' in Windows Sound settings")
        print("3. Check if audio drivers are properly installed")

if __name__ == "__main__":
    main()
