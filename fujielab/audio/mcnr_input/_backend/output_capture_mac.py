"""
Output audio capture module for macOS - BlackHole + SwitchAudioSource version
Implementation using callback functions and queues

Provides the OutputCaptureMac class, which inherits from the OutputCapture abstract class.
"""
import os
import subprocess
import time
import numpy as np
import sounddevice as sd
import shutil
import sys
import queue
import threading
import abc
from .data import AudioData
from .output_capture_base import OutputCapture


class OutputCaptureMac(OutputCapture):
    """
    Output audio capture class for macOS
    Implementation using BlackHole + SwitchAudioSource
    """

    def __init__(self, sample_rate=16000, channels=2, blocksize=1024, debug=False):
        """
        Initialization for macOS output capture

        Parameters:
        -----------
        sample_rate : int, optional
            Sampling rate (Hz) (default: 16000Hz)
        channels : int, optional
            Number of channels (default: 2 channels (stereo))
        blocksize : int, optional
            Block size (number of frames) (default: 1024)
        debug : bool, optional
            Enable debug messages (default: False)
        """
        super().__init__(sample_rate, channels, blocksize, debug)

        # Initialization of instance variables
        self._capture_stream = None
        self._original_device = None
        self._current_device = None
        self._stream_initialized = False
        self._audio_queue = queue.Queue(maxsize=20)
        self._callback_error = None
        self._callback_lock = threading.Lock()
        self._time_offset = 0.0  # offset from system time to audio time

    @staticmethod
    def list_audio_devices(debug=False):
        """
        List available audio devices on the system

        Parameters:
        -----------
        debug : bool, optional
            Enable debug messages (default: False)

        Returns:
        --------
        bool
            True if successful, False if failed
        """
        def _debug_print_local(message):
            if debug:
                print(message)

        # Check if SwitchAudioSource is available
        if not shutil.which('SwitchAudioSource'):
            _debug_print_local("SwitchAudioSource is not installed.")
            _debug_print_local("To install: brew install switchaudio-osx")
            return False

        try:
            # List input devices
            _debug_print_local("Input devices:")
            result = subprocess.run(["SwitchAudioSource", "-a", "-t", "input"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for line in result.stdout.splitlines():
                _debug_print_local(f"  {line}")

            # List output devices
            _debug_print_local("\nOutput devices:")
            result = subprocess.run(["SwitchAudioSource", "-a", "-t", "output"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for line in result.stdout.splitlines():
                _debug_print_local(f"  {line}")

            # Also display the current devices
            current_input = subprocess.run(["SwitchAudioSource", "-c", "-t", "input"],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout.strip()
            current_output = subprocess.run(["SwitchAudioSource", "-c", "-t", "output"],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout.strip()

            _debug_print_local(f"\nCurrent input device: {current_input}")
            _debug_print_local(f"Current output device: {current_output}")
            return True
        except Exception as e:
            _debug_print_local(f"Failed to retrieve device list: {e}")
            return False

    def select_device_by_name(self, device_name):
        """
        Select audio output device by name

        Parameters:
        -----------
        device_name : str
            Name of the audio device to use (e.g., "BlackHole 2ch")

        Returns:
        --------
        bool
            True if successful, False if failed
        """
        if not device_name:
            self._debug_print("Device name is not specified")
            return False

        try:
            # Save the current device
            if self._original_device is None:
                self._original_device = subprocess.run(
                    ["SwitchAudioSource", "-c", "-t", "output"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                ).stdout.strip()
                self._debug_print(f"Original output device: {self._original_device}")

            # Switch to the new device
            subprocess.run(["SwitchAudioSource", "-s", device_name, "-t", "output"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            self._current_device = device_name
            self._debug_print(f"Switched output device to {device_name}")
            return True
        except Exception as e:
            self._debug_print(f"Failed to switch output device: {e}")
            return False

    def restore_original_output(self):
        """
        Restore the original output device

        Returns:
        --------
        bool
            True if successful, False if failed
        """
        if self._original_device and self._original_device != self._current_device:
            try:
                subprocess.run(["SwitchAudioSource", "-s", self._original_device, "-t", "output"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                self._debug_print(f"Restored output device to original: {self._original_device}")
                self._current_device = self._original_device
                return True
            except Exception as e:
                self._debug_print(f"Failed to restore original output device: {e}")
                return False
        return True  # Consider successful if already on the original device

    @staticmethod
    def check_fujielab_output_device(debug=False):
        """
        Check if a multi-output device named fujielab-output exists,
        and if BlackHole 2ch is included and correctly configured

        Parameters:
        -----------
        debug : bool, optional
            Enable debug messages (default: False)

        Returns:
        --------
        bool
            True if correctly configured, False if there are issues
        """
        def _debug_print_local(message):
            if debug:
                print(message)

        try:
            # Get the list of devices
            result = subprocess.run(["SwitchAudioSource", "-a", "-t", "output"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if "fujielab-output" not in result.stdout:
                _debug_print_local("The 'fujielab-output' device does not exist.")
                return False

            # Check if BlackHole is available
            blackhole_found = False
            for line in result.stdout.splitlines():
                if "BlackHole 2ch" in line:
                    blackhole_found = True
                    break

            if not blackhole_found:
                _debug_print_local("BlackHole 2ch device not found. Please install it.")
                return False

            _debug_print_local("fujielab-output device and BlackHole 2ch detected.")

            # Also check if BlackHole 2ch is recognized by sounddevice
            try:
                import sounddevice as sd
                devices = sd.query_devices()
                blackhole_input_found = False

                for i, dev in enumerate(devices):
                    if 'BlackHole 2ch' in dev['name'] and dev['max_input_channels'] > 0:
                        blackhole_input_found = True
                        _debug_print_local(f"sounddevice recognizes BlackHole 2ch input device (ID: {i})")
                        break

                if not blackhole_input_found:
                    _debug_print_local("Warning: sounddevice does not recognize BlackHole 2ch input device")
                    _debug_print_local("- Please ensure BlackHole 2ch is correctly installed")
                    # Issue a warning but do not fail (due to possible internal access restrictions)
            except Exception as sd_err:
                _debug_print_local(f"sounddevice device check error: {sd_err}")
                # Do not consider this a failure

            # Check if fujielab-output can actually be selected
            try:
                # Temporarily switch to fujielab-output
                current = subprocess.run(["SwitchAudioSource", "-c", "-t", "output"],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout.strip()

                if current != "fujielab-output":
                    test_switch = subprocess.run(["SwitchAudioSource", "-s", "fujielab-output", "-t", "output"],
                                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                    # Switch back
                    subprocess.run(["SwitchAudioSource", "-s", current, "-t", "output"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    if test_switch.returncode != 0:
                        _debug_print_local("Warning: Failed to switch to fujielab-output device")
                        _debug_print_local("- Please check if the device is correctly configured")
            except Exception as switch_err:
                _debug_print_local(f"Device switch test error: {switch_err}")

            return True
        except Exception as e:
            _debug_print_local(f"Error occurred while checking devices: {e}")
            return False

    def start_audio_capture(self, device_name=None, sample_rate=None, channels=None, blocksize=None):
        """
        Start audio capture

        Parameters:
        -----------
        device_name : str, optional
            Name of the audio device to use (e.g., "BlackHole 2ch")
            If None, the "fujielab-output" device is used (default: None)
        sample_rate : int, optional
            Sampling rate (Hz) (default: None, uses the rate specified during initialization)
        channels : int, optional
            Number of channels (default: None, uses the channel count specified during initialization)
        blocksize : int, optional
            Block size (number of frames) (default: None, uses the block size specified during initialization)

        Returns:
        --------
        bool
            True if capture started successfully, False if failed
        """
        # Overwrite instance variables if arguments are provided
        if sample_rate is not None:
            self._sample_rate = sample_rate
        if channels is not None:
            self._channels = channels
        if blocksize is not None:
            self._blocksize = blocksize

        # Initially not initialized
        self._stream_initialized = False

        # Check for required tools
        if not self._check_required_tools():
            self._debug_print("Required tools are not installed.")
            self._debug_print("Please run: scripts/install_audio_tools.py")
            return False

        # Check fujielab-output device
        fujielab_device_exists = self.check_fujielab_output_device(debug=self.debug)
        if not fujielab_device_exists:
            print("fujielab-output device not found.")
            print("Please create a multi-output device using the following steps:")
            print("1. Go to System Preferences > Sound")
            print("2. Check BlackHole in the 'Output' tab")
            print("3. Click the '+' button to create a multi-output device")
            print("4. Select both the current speaker and 'BlackHole 2ch'")
            print("5. Name the device 'fujielab-output' and click 'Done'")
            return False

        # Device selection (fujielab-output if not specified)
        target_device = device_name if device_name else "fujielab-output"
        if not self.select_device_by_name(target_device):
            print(f"Failed to switch to {target_device}")
            return False

        # The device is now open and ready to receive data
        # The stream can be started even if no audio is being played

        # Open sound device
        try:
            # Display Available devices
            self._debug_print("\nAvailable audio devices:")
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    self._debug_print(f"  {i}: {dev['name']} (Input channels: {dev['max_input_channels']})")

            # Find BlackHole 2ch device ID (more strict search)
            blackhole_id = None
            blackhole_candidates = []

            for i, dev in enumerate(devices):
                # Check with multiple conditions
                if dev['max_input_channels'] > 0:  # Input devices only
                    if 'BlackHole 2ch' == dev['name']:  # Exact match first
                        blackhole_id = i
                        break
                    elif 'BlackHole' in dev['name']:  # Partial match as a candidate
                        blackhole_candidates.append((i, dev['name']))

            # If no exact match is found, select from candidates
            if blackhole_id is None and blackhole_candidates:
                blackhole_id = blackhole_candidates[0][0]  # Use the first candidate
                print(f"Warning: Exact match for BlackHole 2ch not found, using {blackhole_candidates[0][1]}")

            if blackhole_id is None:
                print("Error: BlackHole 2ch device not found.")
                print("Please ensure BlackHole is installed.")
                print("You can install it with the following command:")
                print("  brew install blackhole-2ch")
                return False

            self._debug_print(f"Detected BlackHole 2ch device (Device ID: {blackhole_id})")

            # Debug output for stream settings
            self._debug_print(f"Stream settings: Sample rate={self._sample_rate}, Channels={self._channels}, Block size={self._blocksize}")

            # For safety, recreate the stream
            if self._capture_stream is not None:
                try:
                    self._capture_stream.stop()
                    self._capture_stream.close()
                    self._capture_stream = None
                    self._debug_print("Cleaned up existing capture stream")
                except Exception as e:
                    self._debug_print(f"Cleanup error for existing stream (ignored): {e}")

            # Allow multiple attempts (rarely, the first attempt may fail)
            max_attempts = 2
            for attempt in range(max_attempts):
                try:
                    # Clear the queue
                    while not self._audio_queue.empty():
                        self._audio_queue.get_nowait()

                    # Reset error state
                    self._callback_error = None

                    # Open stream to record from BlackHole 2ch (using callback function)
                    self._capture_stream = sd.InputStream(
                        device=blackhole_id,
                        samplerate=self._sample_rate,
                        channels=self._channels,
                        blocksize=self._blocksize,
                        dtype='float32',
                        latency='high',  # Prioritize stability with high latency and large buffer
                        callback=self._audio_callback,  # Set the callback function
                        extra_settings=None  # Use system default settings (more stable)
                    )
                    self._capture_stream.start()
                    self._debug_print(f"Started recording from BlackHole 2ch (rate={self._sample_rate}Hz, channels={self._channels}, blocksize={self._blocksize}, using callback)")
                    self._time_offset = time.time() - self._capture_stream.time  # Calculate time offset for accurate timestamps

                    # Wait for the recording stream to stabilize (a bit longer)
                    time.sleep(0.8)

                    # Check if the callback function is working correctly
                    try:
                        # Check for errors
                        if self._callback_error:
                            print(f"Error in callback function: {self._callback_error}")

                        # Check if the stream is actually active
                        if not self._capture_stream.active:
                            print("Warning: Stream is not active")

                        # No problem if all data is 0 - just no audio is being played
                        # Consider successful if the stream can be started
                        # Explicitly record the normal state in the instance variable
                        self._stream_initialized = True
                        self._debug_print("Speaker capture stream (callback method) initialized successfully (_stream_initialized = True)")
                        self._debug_print("- Audio will be captured automatically when played")
                        return True
                    except Exception as callback_err:
                        print(f"Callback initialization error: {callback_err}")

                        if attempt < max_attempts - 1:
                            print(f"Retrying ({attempt+1}/{max_attempts})...")
                            # Reinitialize the stream - close and recreate just in case
                            if self._capture_stream is not None:
                                try:
                                    self._capture_stream.stop()
                                    self._capture_stream.close()
                                    self._capture_stream = None
                                except Exception as e:
                                    print(f"Stream cleanup error: {e}")
                            time.sleep(1.0)  # Wait a bit longer
                            continue
                        else:
                            # Consider successful even if there is an error in the last attempt, as the stream itself is started
                            print("Failed test read, but the stream is considered started")
                            print("- Audio capture will continue to be attempted")
                            # Explicitly record the normal state in the instance variable
                            self._stream_initialized = True
                            print("Test read error occurred, but the stream is initialized (_stream_initialized = True)")
                            return True
                except Exception as stream_err:
                    print(f"Stream startup error: {stream_err}")
                    if attempt < max_attempts - 1:
                        print(f"Retrying ({attempt+1}/{max_attempts})...")
                        time.sleep(1.0)  # Wait a bit longer
                        continue
                    return False
        except Exception as e:
            print(f"Failed to start audio stream: {e}")
            # Display more detailed error information
            import traceback
            traceback.print_exc()
            return False
        finally:
            # If the stream is not initialized, reset the state
            if not self._stream_initialized:
                self._capture_stream = None
                self._debug_print("Stream initialization failed, resetting state")
                self._stream_initialized = False

    @staticmethod
    def _check_required_tools():
        """
        Check if the required tools are installed

        Returns:
        --------
        bool
            True if all are present, False if any are missing
        """
        # Check for Homebrew
        try:
            subprocess.run(["brew", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Homebrew is not installed.")
            return False

        # Check for BlackHole
        try:
            result = subprocess.run(["brew", "list", "blackhole-2ch"],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                print("BlackHole 2ch is not installed.")
                return False
        except Exception:
            print("Failed to check BlackHole 2ch.")
            return False

        # Check for SwitchAudioSource
        if not shutil.which('SwitchAudioSource'):
            print("SwitchAudioSource is not installed.")
            return False

        return True

    def read_audio_capture(self):
        """
        Read the captured audio data

        Returns:
        --------
        AudioData
            Object containing the captured audio data

        Raises:
        -------
        RuntimeError
            If the stream is not initialized or not working correctly
        """
        if not self._stream_initialized:
            raise RuntimeError("The stream is not initialized")

        try:
            # Get data from the queue
            audio_data = self._audio_queue.get(timeout=1.0)
            return audio_data
        except queue.Empty:
            # Return empty AudioData if no data
            empty_data = np.zeros((self._blocksize, self._channels), dtype=np.float32)
            return AudioData(data=empty_data, time=time.time(), overflowed=False)

    def stop_audio_capture(self):
        """
        Stop audio capture
        """
        # Stop and close the stream
        if self._capture_stream is not None:
            self._capture_stream.stop()
            self._capture_stream.close()
            self._capture_stream = None
            self._debug_print("Audio capture stopped")

        # Clear the queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        # Reset error state
        self._callback_error = None

        # Reset initialization state
        self._stream_initialized = False

        # Restore the original output device
        self.restore_original_output()

    def _audio_callback(self, indata, frames, time_info, status):
        """
        Callback function for sounddevice
        Processes the captured audio data and adds it to the queue

        Parameters:
        -----------
        indata : numpy.ndarray
            Input audio data (number of frames x number of channels)
        frames : int
            Number of frames in the current block
        time_info : PaStreamCallbackTimeInfo Struct
            Structure containing timing information
        status : sounddevice.CallbackFlags
            Flags indicating errors, etc.
        """
        try:
            with self._callback_lock:
                # Error detection
                if status:
                    # Log only important errors, excluding warning-level situations
                    important_errors = [flag for flag in dir(status)
                                    if not flag.startswith('_') and
                                    getattr(status, flag) and
                                    flag not in ('input_underflow', 'output_underflow')]

                    if important_errors:
                        error_str = ", ".join(important_errors)
                        self._callback_error = f"Audio callback error: {error_str}"
                        # Output only critical errors
                        if any(err in important_errors for err in ('input_overflow', 'output_overflow')):
                            print(self._callback_error)

                # Data processing
                if indata.size > 0:
                    # Reshape data format
                    if len(indata.shape) == 1:
                        # Reshape 1D array to (samples, 1)
                        data = indata.reshape(-1, 1)
                    elif indata.shape[1] > 1 and self._channels == 1:
                        # Convert stereo to mono (if needed)
                        data = np.mean(indata, axis=1).reshape(-1, 1)
                    else:
                        data = indata

                    # Convert to AudioData object
                    try:
                        timestamp = self._time_offset + time_info.inputBufferAdcTime
                    except AttributeError:
                        timestamp = time.time()

                    # print(f"first 10 bytes of data: {data[:10].flatten()}")  # Print first 10 bytes for debugging
                    audio_data = AudioData(
                        data=data.copy(),
                        time=timestamp,
                        overflowed=(status and hasattr(status, 'input_overflow') and status.input_overflow)
                    )

                    # If the queue is full, discard old data
                    try:
                        if self._audio_queue.full():
                            # Discard one old data
                            self._audio_queue.get_nowait()

                        # Add to queue (non-blocking)
                        self._audio_queue.put_nowait(audio_data)
                        # print(f"Added audio data to queue (Frames: {frames}, Channels: {data.shape[1]})")
                    except queue.Full:
                        pass  # Ignore if the queue is full

        except Exception as e:
            # Record error information
            with self._callback_lock:
                self._callback_error = f"Audio callback exception: {str(e)}"
                print(self._callback_error)
                import traceback
                traceback.print_exc()


# Export necessary classes as a module
__all__ = ['OutputCapture', 'OutputCaptureMac']


if __name__ == "__main__":
    # Test when the module is run directly
    print("=== Audio Capture Module for macOS ===")

    # Create an instance of the macOS speaker capture
    mac_capture = OutputCaptureMac(debug=True)  # Enable debug mode for testing

    if mac_capture.list_audio_devices(debug=True):
        print("\nListed available devices.")
    else:
        print("Failed to retrieve device list.")

    # Check fujielab-output device
    if mac_capture.check_fujielab_output_device(debug=True):
        print("\nfujielab-output device is correctly configured.")
    else:
        print("fujielab-output device is not configured.")

    # Start audio capture (using default settings)
    # Arguments can be specified as needed:
    # - Sample rate: sample_rate=44100
    # - Number of channels: channels=1 (mono)
    # - Block size: blocksize=2048
    mac_capture.start_audio_capture()

    print(f"\nStarted audio capture (rate={mac_capture.sample_rate}Hz, channels={mac_capture.channels}, blocksize={mac_capture.blocksize})")
    print("Please play audio for testing.")
    try:
        # List to accumulate data
        all_audio_data = []
        print("Collecting data... Press Ctrl+C to stop and save to WAV file")

        while True:
            try:
                audio_data = mac_capture.read_audio_capture()
                # Accumulate data
                # print(f"READ: first 10 bytes of data: {audio_data.data[:10].flatten()}")
                all_audio_data.append(audio_data.data)
                # Display progress
                if len(all_audio_data) % 10 == 0:
                    duration = len(all_audio_data) * mac_capture.blocksize / mac_capture.sample_rate
                    print(f"Recording: {duration:.1f} seconds ({len(all_audio_data)} blocks)", end="\r")
            except RuntimeError as e:
                print(f"\nError: {e}")
                break
    except KeyboardInterrupt:
        print("\nEnding recording")
    finally:
        # Stop capture
        mac_capture.stop_audio_capture()
        print("Stopped audio capture.")

        # If there is data, save to WAV file
        if all_audio_data:
            try:
                # Concatenate all data
                all_samples = np.vstack(all_audio_data)
                if len(all_audio_data) > 0:
                    print(f"shape of sample: {all_audio_data[0].shape if hasattr(all_audio_data[0], 'shape') else '-'}")
                print(f"shape of all_samples: {all_samples.shape}")
                # Save as WAV file
                output_file = "output.wav"
                import soundfile as sf
                sf.write(output_file, all_samples, mac_capture.sample_rate)
                duration = len(all_samples) / mac_capture.sample_rate
                print(f"Saved recording data to {output_file} (length: {duration:.2f} seconds)")
            except Exception as e:
                print(f"File save error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("No data to save")
