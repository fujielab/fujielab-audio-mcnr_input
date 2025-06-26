"""
Output audio capture module for Windows - soundcard library version
Implementation using soundcard loopback capture
"""
import soundcard as sc
import numpy as np
import threading
import time
import queue
from .data import AudioData
from .output_capture_base import OutputCapture


class OutputCaptureWin(OutputCapture):
    """
    Output audio capture class for Windows
    Implementation using soundcard library for loopback capture
    """

    def __init__(self, sample_rate=44100, channels=2, blocksize=512, debug=False):
        """
        Initialization for Windows output capture

        Parameters:
        -----------
        sample_rate : int, optional
            Sampling rate (Hz) (default: 44100Hz)
        channels : int, optional
            Number of channels (default: 2 channels (stereo))
        blocksize : int, optional
            Block size (number of frames) (default: 512)
        debug : bool, optional
            Enable debug messages (default: False)
        """
        super().__init__(sample_rate, channels, blocksize, debug)

        # Initialize instance variables
        self._recording_thread = None
        self._audio_queue = queue.Queue(maxsize=20)
        self._stop_recording = threading.Event()
        self._stream_initialized = False
        self._loopback_device = None

    def _find_loopback_device(self):
        """
        Find the best matching loopback device for the current default playback device
        using soundcard.all_microphones(include_loopback=True)

        Returns:
        --------
        soundcard.Microphone
            Loopback device object

        Raises:
        -------
        RuntimeError
            If no suitable loopback device is found
        """
        try:
            # Get default playback device
            default_speaker = sc.default_speaker()
            self._debug_print(f"Default speaker: {default_speaker.name}")

            # Get all microphones including loopback devices
            all_mics = sc.all_microphones(include_loopback=True)

            if not all_mics:
                raise RuntimeError("No microphones or loopback devices found")

            self._debug_print("Available microphones and loopback devices:")
            for i, mic in enumerate(all_mics):
                mic_type = "Loopback" if getattr(mic, 'isloopback', False) else "Microphone"
                self._debug_print(f"  {i}: {mic.name} ({mic_type})")

            # Filter loopback devices only
            loopback_devices = [mic for mic in all_mics if getattr(mic, 'isloopback', False)]

            if not loopback_devices:
                # If no explicit loopback devices found, look for devices with loopback keywords
                self._debug_print("No explicit loopback devices found, searching by name patterns...")
                loopback_keywords = ["loopback", "stereo mix", "ステレオ ミキサー", "what u hear", "mix"]

                for mic in all_mics:
                    mic_name_lower = mic.name.lower()
                    if any(keyword in mic_name_lower for keyword in loopback_keywords):
                        loopback_devices.append(mic)
                        self._debug_print(f"Found potential loopback device by name: {mic.name}")

            if not loopback_devices:
                raise RuntimeError(
                    "No loopback devices found. Please ensure:\n"
                    "1. 'Stereo Mix' or similar loopback device is enabled in Windows Sound settings\n"
                    "2. Your audio drivers support loopback recording\n"
                    "3. You have appropriate audio drivers installed"
                )

            # Find the best matching loopback device for the default speaker
            default_speaker_name = default_speaker.name.lower()
            best_match = None
            best_score = 0

            for loopback_device in loopback_devices:
                loopback_name = loopback_device.name.lower()

                # Calculate similarity score
                score = 0

                # Exact match gets highest score
                if default_speaker_name == loopback_name:
                    score = 100
                # Partial match in either direction
                elif default_speaker_name in loopback_name or loopback_name in default_speaker_name:
                    score = 80
                # Common words match
                else:
                    default_words = set(default_speaker_name.split())
                    loopback_words = set(loopback_name.split())
                    common_words = default_words.intersection(loopback_words)
                    score = len(common_words) * 20

                # Prefer devices with "mix" or "loopback" in the name
                if any(keyword in loopback_name for keyword in ["mix", "loopback", "ミキサー"]):
                    score += 10

                self._debug_print(f"Device: {loopback_device.name}, Score: {score}")

                if score > best_score:
                    best_score = score
                    best_match = loopback_device

            if best_match is None:
                # If no good match found, use the first loopback device
                best_match = loopback_devices[0]
                self._debug_print(f"No good match found, using first loopback device: {best_match.name}")
            else:
                self._debug_print(f"Best matching loopback device: {best_match.name} (score: {best_score})")

            return best_match

        except Exception as e:
            raise RuntimeError(f"Error finding loopback device: {e}")

    def _recording_worker(self):
        """
        Worker thread for continuous audio recording
        """
        try:
            # Initialize COM for this thread (required for Windows audio)
            import pythoncom
            pythoncom.CoInitialize()

            self._debug_print(f"Starting recording with device: {self._loopback_device.name}")
            self._debug_print(f"Block size: {self._blocksize}, Sample rate: {self._sample_rate}")

            # Open the recorder once and keep it open
            with self._loopback_device.recorder(
                samplerate=self._sample_rate,
                channels=self._channels
            ) as recorder:

                while not self._stop_recording.is_set():
                    try:
                        # Record one block of audio
                        data = recorder.record(numframes=self._blocksize)

                        if data.size > 0:
                            # Ensure data has the right shape
                            if len(data.shape) == 1:
                                # Reshape 1D array to (samples, 1)
                                data = data.reshape(-1, 1)
                            elif data.shape[1] > self._channels:
                                # If more channels than needed, take the first N channels
                                data = data[:, :self._channels]
                            elif data.shape[1] < self._channels and self._channels == 2:
                                # If mono but stereo requested, duplicate channel
                                data = np.column_stack([data, data])

                            # Create AudioData object
                            audio_data = AudioData(
                                data=data.astype(np.float32),
                                time=time.time(),
                                overflowed=False
                            )

                            # Add to queue (discard old data if queue is full)
                            try:
                                if self._audio_queue.full():
                                    self._audio_queue.get_nowait()
                                self._audio_queue.put_nowait(audio_data)
                            except queue.Full:
                                pass  # Ignore if queue is still full

                    except Exception as record_error:
                        if not self._stop_recording.is_set():
                            self._debug_print(f"Recording error: {record_error}")
                            time.sleep(0.1)  # Brief pause before retrying

        except ImportError:
            # If pythoncom is not available, try without COM initialization
            self._debug_print("pythoncom not available, trying without COM initialization")
            try:
                self._debug_print(f"Starting recording with device: {self._loopback_device.name}")
                self._debug_print(f"Block size: {self._blocksize}, Sample rate: {self._sample_rate}")

                # Open the recorder once and keep it open
                with self._loopback_device.recorder(
                    samplerate=self._sample_rate,
                    channels=self._channels
                ) as recorder:

                    while not self._stop_recording.is_set():
                        try:
                            # Record one block of audio
                            data = recorder.record(numframes=self._blocksize)

                            if data.size > 0:
                                # Ensure data has the right shape
                                if len(data.shape) == 1:
                                    # Reshape 1D array to (samples, 1)
                                    data = data.reshape(-1, 1)
                                elif data.shape[1] > self._channels:
                                    # If more channels than needed, take the first N channels
                                    data = data[:, :self._channels]
                                elif data.shape[1] < self._channels and self._channels == 2:
                                    # If mono but stereo requested, duplicate channel
                                    data = np.column_stack([data, data])

                                # Create AudioData object
                                audio_data = AudioData(
                                    data=data.astype(np.float32),
                                    time=time.time(),
                                    overflowed=False
                                )

                                # Add to queue (discard old data if queue is full)
                                try:
                                    if self._audio_queue.full():
                                        self._audio_queue.get_nowait()
                                    self._audio_queue.put_nowait(audio_data)
                                except queue.Full:
                                    pass  # Ignore if queue is still full

                        except Exception as record_error:
                            if not self._stop_recording.is_set():
                                self._debug_print(f"Recording error: {record_error}")
                                time.sleep(0.1)  # Brief pause before retrying

            except Exception as e:
                self._debug_print(f"Recording worker error: {e}")
        except Exception as e:
            self._debug_print(f"Recording worker error: {e}")
        finally:
            try:
                pythoncom.CoUninitialize()
            except:
                pass
            self._debug_print("Recording worker stopped")

    def start_audio_capture(self, device_name=None, sample_rate=None, channels=None, blocksize=None):
        """
        Starts audio capture

        Parameters:
        -----------
        device_name : str, optional
            The name of the audio device to use (not used in this implementation)
        sample_rate : int, optional
            Sampling rate (Hz) (default: None, uses the rate specified during initialization)
        channels : int, optional
            Number of channels (default: None, uses the number of channels specified during initialization)
        blocksize : int, optional
            Block size (number of frames) (default: None, uses the block size specified during initialization)

        Returns:
        --------
        bool
            True if capture started successfully, False otherwise
        """
        # Update parameters if provided
        if sample_rate is not None:
            self._sample_rate = sample_rate
        if channels is not None:
            self._channels = channels
        if blocksize is not None:
            self._blocksize = blocksize

        try:
            # Stop any existing recording
            self.stop_audio_capture()

            # Find the best loopback device
            self._debug_print("Searching for loopback device...")
            self._loopback_device = self._find_loopback_device()

            # Clear the queue
            while not self._audio_queue.empty():
                self._audio_queue.get_nowait()

            # Reset stop event
            self._stop_recording.clear()

            # Test the device first
            self._debug_print(f"Testing device: {self._loopback_device.name}")
            try:
                with self._loopback_device.recorder(
                    samplerate=self._sample_rate,
                    channels=self._channels
                ) as test_recorder:
                    # Try to record a small test sample
                    test_data = test_recorder.record(numframes=128)
                    if test_data.size == 0:
                        raise RuntimeError("Device test failed: no data captured")
                    self._debug_print("Device test successful")
            except Exception as test_error:
                raise RuntimeError(f"Device test failed: {test_error}")

            # Start recording thread
            self._recording_thread = threading.Thread(
                target=self._recording_worker,
                daemon=True
            )
            self._recording_thread.start()

            # Wait a moment for the thread to initialize
            time.sleep(0.2)

            # Check if thread is still running
            if not self._recording_thread.is_alive():
                raise RuntimeError("Recording thread failed to start")

            self._stream_initialized = True
            self._debug_print(f"Audio capture started successfully with {self._loopback_device.name}")
            return True

        except Exception as e:
            self._debug_print(f"Failed to start audio capture: {e}")
            self.stop_audio_capture()
            return False

    def read_audio_capture(self):
        """
        Reads the captured audio data
        Retrieves AudioData objects from the queue

        Returns:
        --------
        AudioData
            The object containing the captured audio data

        Raises:
        -------
        RuntimeError
            If the stream is not initialized or not functioning properly
        """
        if not self._stream_initialized:
            raise RuntimeError("Capture stream is not initialized")

        if self._recording_thread is None or not self._recording_thread.is_alive():
            raise RuntimeError("Recording thread is not running")

        try:
            # Retrieve data from the queue (wait up to 1 second)
            audio_data = self._audio_queue.get(block=True, timeout=1.0)
            # self._debug_print(f"Retrieved audio data: {audio_data.data.shape} at {audio_data.time}")
            return audio_data
        except queue.Empty:
            raise RuntimeError("Timeout while waiting for audio data")

    def stop_audio_capture(self):
        """Stops audio capture"""
        # Signal the recording thread to stop
        self._stop_recording.set()

        # Wait for the recording thread to finish
        if self._recording_thread is not None:
            self._recording_thread.join(timeout=2.0)
            if self._recording_thread.is_alive():
                self._debug_print("Warning: Recording thread did not stop cleanly")
            self._recording_thread = None

        # Clear the queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        # Reset state
        self._stream_initialized = False
        self._loopback_device = None
        self._debug_print("Audio capture stopped")


# Export the necessary class as a module
__all__ = ['OutputCaptureWin']
