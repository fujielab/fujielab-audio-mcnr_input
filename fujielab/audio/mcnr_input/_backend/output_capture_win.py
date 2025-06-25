"""
Output audio capture module for Windows - WASAPI Loopback version
Implementation using callback function and queue
"""
import sounddevice as sd
import numpy as np
import threading
import time
import queue
from .data import AudioData
from .output_capture_base import OutputCapture


class OutputCaptureWin(OutputCapture):
    """
    Output audio capture class for Windows
    Implementation using WASAPI Loopback
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
        self._capture_stream = None
        self._audio_queue = queue.Queue(maxsize=20)
        self._callback_error = None
        self._callback_lock = threading.Lock()
        self._stream_initialized = False

    def _audio_callback(self, indata, frames, time_info, status):
        """
        Callback function for sounddevice
        Processes captured audio data and adds it to the queue

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
                    # Log only critical errors, excluding warning-level situations
                    important_errors = [flag for flag in dir(status)
                                     if not flag.startswith('_') and
                                     getattr(status, flag) and
                                     flag not in ('input_underflow', 'output_underflow')]

                    if important_errors:
                        error_str = ", ".join(important_errors)
                        self._callback_error = f"Audio callback error: {error_str}"
                        # Output only critical errors
                        if any(err in important_errors for err in ('input_overflow', 'output_overflow')):
                            self._debug_print(self._callback_error)

                # Data processing
                if indata.size > 0:
                    # Reshape data format
                    if len(indata.shape) == 1:
                        # Reshape 1D array to (samples, 1)
                        data = indata.reshape(-1, 1)
                    elif indata.shape[1] > 1 and self._channels == 1:
                        # Convert from stereo to mono (if necessary)
                        data = np.mean(indata, axis=1).reshape(-1, 1)
                    else:
                        data = indata

                    # Convert to AudioData object
                    try:
                        timestamp = time_info.inputBufferAdcTime
                    except AttributeError:
                        timestamp = time.time()

                    audio_data = AudioData(
                        data=data.copy(),
                        time=timestamp,
                        overflowed=(status and hasattr(status, 'input_overflow') and status.input_overflow)
                    )

                    # Discard old data if the queue is full
                    try:
                        if self._audio_queue.full():
                            # Discard one old data
                            self._audio_queue.get_nowait()

                        # Add to queue (non-blocking)
                        self._audio_queue.put_nowait(audio_data)
                    except queue.Full:
                        pass  # Ignore if the queue is full

        except Exception as e:
            # Log error information
            with self._callback_lock:
                self._callback_error = f"Audio callback exception: {str(e)}"
                # Note: Can't use _debug_print here as it's a static context
                # This error should always be shown as it's critical
                print(self._callback_error)
                import traceback
                traceback.print_exc()

    @staticmethod
    def _find_loopback_device(channels):
        """
        Searches for the loopback device of WASAPI

        Parameters:
        -----------
        channels : int
            Required number of channels

        Returns:
        --------
        int
            Device index

        Raises:
        -------
        RuntimeError
            If no suitable loopback device is found
        """
        for idx, dev in enumerate(sd.query_devices()):
            if (
                "loopback" in dev["name"].lower()
                and dev["max_input_channels"] >= channels
                and sd.query_hostapis()[dev["hostapi"]]["name"] == "Windows WASAPI"
            ):
                return idx
        raise RuntimeError("No suitable WASAPI loopback device found.")

    def start_audio_capture(self, device_name=None, sample_rate=None, channels=None, blocksize=None):
        """
        Starts audio capture

        Parameters:
        -----------
        device_name : str, optional
            The name of the audio device to use (not used in Windows version)
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
        # Overwrite instance variables if arguments are specified
        if sample_rate is not None:
            self._sample_rate = sample_rate
        if channels is not None:
            self._channels = channels
        if blocksize is not None:
            self._blocksize = blocksize

        # Not initialized in the initial state
        self._stream_initialized = False

        try:
            # Get the index of the loopback device
            device_idx = self._find_loopback_device(self._channels)
            self._debug_print(f"Detected suitable WASAPI loopback device (Device ID: {device_idx})")

            # For safety, recreate the stream
            if self._capture_stream is not None:
                try:
                    self._capture_stream.stop()
                    self._capture_stream.close()
                    self._capture_stream = None
                    self._debug_print("Cleaned up existing capture stream")
                except Exception as e:
                    self._debug_print(f"Error cleaning up existing stream (ignored): {e}")

            # Clear the queue
            while not self._audio_queue.empty():
                self._audio_queue.get_nowait()

            # Reset error state
            self._callback_error = None

            # Debug output of stream settings
            print(f"Stream settings: Sample rate={self._sample_rate}, Channels={self._channels}, Block size={self._blocksize}")

            # Open a stream to capture from the loopback device
            self._capture_stream = sd.InputStream(
                device=device_idx,
                samplerate=self._sample_rate,
                channels=self._channels,
                blocksize=self._blocksize,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._capture_stream.start()
            print(f"Started recording from WASAPI loopback")

            # Wait for the stream to stabilize
            time.sleep(0.5)

            # Check if the stream has started successfully
            if not self._capture_stream.active:
                print("Error: Stream is not active")
                return False

            # Successfully initialized
            self._stream_initialized = True
            return True

        except Exception as e:
            print(f"Failed to start audio capture: {e}")
            import traceback
            traceback.print_exc()
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
        # Stream check
        if self._capture_stream is None or not self._stream_initialized:
            raise RuntimeError("Capture stream is not initialized")

        # Check stream status
        try:
            # Raise an exception if the stream is not active
            if not self._capture_stream.active:
                error_msg = "Capture stream is not active"
                print(f"Error: {error_msg}")
                self._stream_initialized = False  # Invalidate stream status
                raise RuntimeError(error_msg)

            # Check error state
            if self._callback_error:
                # Ignore overflow errors and continue
                if not ("input_overflow" in self._callback_error or "output_overflow" in self._callback_error):
                    # Raise an exception for other critical errors
                    raise RuntimeError(f"Audio callback error: {self._callback_error}")

            # Retrieve data from the queue (waits up to 1 second)
            try:
                audio_data = self._audio_queue.get(block=True, timeout=1.0)
                return audio_data
            except queue.Empty:
                raise RuntimeError("Timeout while waiting to retrieve audio data")
            except Exception as get_err:
                error_msg = f"Error retrieving data from queue: {get_err}"
                print(error_msg)
                raise RuntimeError(error_msg) from get_err

        except RuntimeError:
            # Reraise already processed exceptions
            raise
        except Exception as e:
            error_msg = f"Speaker capture error: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()

            # Raise an exception on error
            raise RuntimeError(error_msg) from e

    def stop_audio_capture(self):
        """Stops audio capture"""
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


# Export the necessary class as a module
__all__ = ['OutputCaptureWin']


if __name__ == "__main__":
    # Test when the module is run directly
    print("=== Speaker Audio Capture Module for Windows ===")

    # Create an instance of the Windows speaker capture
    win_capture = OutputCaptureWin()

    print("\nStarting audio capture...")
    if win_capture.start_audio_capture():
        print(f"Capture started successfully (rate={win_capture.sample_rate}Hz, channels={win_capture.channels})")

        print("Collecting data... Press Ctrl+C to stop")
        try:
            # List to accumulate data
            all_audio_data = []

            # Collect data for 10 seconds
            start_time = time.time()
            while time.time() - start_time < 10:
                try:
                    audio_data = win_capture.read_audio_capture()
                    all_audio_data.append(audio_data.data)
                    print(".", end="", flush=True)
                except RuntimeError as e:
                    print(f"\nError: {e}")
                    break

                # Wait briefly
                time.sleep(0.1)

            print("\nData collection for 10 seconds is complete")

        except KeyboardInterrupt:
            print("\nStopping recording")
        finally:
            win_capture.stop_audio_capture()

            # Save to WAV file if there is data
            if all_audio_data:
                try:
                    # Concatenate all data
                    all_samples = np.vstack(all_audio_data)

                    # Save as WAV file
                    output_file = "windows_output.wav"
                    import soundfile as sf
                    sf.write(output_file, all_samples, win_capture.sample_rate)
                    duration = len(all_samples) / win_capture.sample_rate
                    print(f"Recorded data saved to {output_file} (length: {duration:.2f} seconds)")
                except Exception as e:
                    print(f"File save error: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("No data to save")
    else:
        print("Failed to start audio capture")
