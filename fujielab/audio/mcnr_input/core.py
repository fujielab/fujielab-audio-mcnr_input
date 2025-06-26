import threading
import time
import queue
import numpy as np
import platform
import sounddevice as sd
import scipy.signal
from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any
from ._backend.data import AudioData
from ._backend.input_capture import InputCapture, InputCaptureBase

# Import OutputCapture class according to the platform
if platform.system() == "Darwin":
    from ._backend.output_capture_mac import OutputCaptureMac as OutputCapture
else:
    from ._backend.output_capture_win import OutputCaptureWin as OutputCapture


@dataclass
class CaptureConfig:
    """
    Class to manage capture device settings
    キャプチャデバイスの設定を管理するクラス

    Attributes:
    -----------
    capture_type : str
        Capture type ('Input' or 'Output')
        キャプチャタイプ ('Input' または 'Output')
    device_name : str or int, optional
        Device name or index (default: None, auto-select)
        デバイス名またはインデックス (デフォルト: None, 自動選択)
    channels : int, optional
        Number of channels (default: 2)
        チャンネル数 (デフォルト: 2)
    extra_settings : dict, optional
        Additional settings (default: None)
        追加設定 (デフォルト: None)
    """

    capture_type: str
    device_name: Optional[Union[str, int]] = None
    channels: int = 2
    offset: float = 0.0
    extra_settings: Optional[Dict[str, Any]] = None


# sounddevice compatible status flags
class CallbackFlags:
    """
    sounddevice compatible callback flags class
    sounddevice互換のコールバックフラグクラス
    """

    def __init__(self):
        self.INPUT_UNDERFLOW = False
        self.INPUT_OVERFLOW = False
        self.OUTPUT_UNDERFLOW = False
        self.OUTPUT_OVERFLOW = False
        self.PRIMING_OUTPUT = False


class InputStream:
    def __init__(
        self,
        samplerate=16000,
        blocksize=512,
        captures: Optional[List[CaptureConfig]] = None,
        callback: Optional[callable] = None,
        dtype="float32",
        latency="high",
        extra_settings=None,
        debug=False,
    ):
        """
        Manage multiple capture devices (input and output) and provide them as a synchronized
        multi-channel stream.

        複数のキャプチャデバイス（入力と出力）を管理し、それらを同期した
        1つのマルチチャネルストリームとして提供するクラス

        Parameters:
        -----------
        samplerate : int, optional
            Sampling rate (Hz) (default: 16000)
            サンプリングレート (Hz) (デフォルト: 16000)
        blocksize : int, optional
            Block size in frames (default: 512)
            ブロックサイズ、フレーム単位 (デフォルト: 512)
        captures : list of CaptureConfig, optional
            List of capture configurations. If not specified, both input and output are captured by default.
            キャプチャ設定のリスト。指定されない場合は、デフォルトで入力と出力の両方をキャプチャ
        callback : callable, optional
            Callback function called when data is retrieved.
            データ取得時に呼び出されるコールバック関数
            callback(indata, frames, time_info, status)
        dtype : str, optional
            Data type (default: 'float32')
            データ型 (デフォルト: 'float32')
        latency : str or float, optional
            Latency setting (default: 'high')
            レイテンシ設定 (デフォルト: 'high')
        extra_settings : dict, optional
            Additional settings (default: None)
            追加設定 (デフォルト: None)
        debug : bool, optional
            Enable debug messages (default: False)
            デバッグメッセージを有効にする (デフォルト: False)
        """
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.callback = callback
        self.dtype = dtype
        self.latency = latency
        self.extra_settings = extra_settings or {}
        self.debug = debug
        self.running = False

        from .process import MultiChannelNoiseReductionProcessor

        self.processor = MultiChannelNoiseReductionProcessor()

        # Initialize capture settings
        if captures is None:
            # Default settings: Capture both input and output
            captures = [
                CaptureConfig(capture_type="Input", channels=2),
                CaptureConfig(capture_type="Output", channels=2),
            ]

        # Save capture settings
        self.captures = captures

        # Extract channel information
        self.capture_types = [capture.capture_type for capture in captures]
        self.device_names = [capture.device_name for capture in captures]
        self.channels_list = [capture.channels for capture in captures]

        # Calculate total number of channels
        self.total_channels = sum(self.channels_list)

        # Data queue (for read() method) - Increased queue size to support long recordings
        self.data_queue = queue.Queue(maxsize=200)

        # List of capture instances
        self.capture_instances = []

        # Synchronization-related attributes
        self._sync_enabled = False
        self._sync_offsets = [0.0] * len(
            self.captures
        )  # Time offsets for synchronization (in seconds)
        self._sync_lock = threading.Lock()
        self._sync_buffers = [
            [] for _ in range(len(self.captures))
        ]  # Buffers for storing sync data
        self._sync_buffer_size = int(
            self.samplerate * 2.0
        )  # 2 seconds buffer for sync detection

    def _debug_print(self, message):
        """
        Print debug message if debug mode is enabled
        デバッグモードが有効な場合にデバッグメッセージを出力
        """
        if self.debug:
            print(message)

    def _cleanup_captures(self):
        """
        Clean up capture instances when initialization fails
        初期化が失敗した際にキャプチャインスタンスをクリーンアップ
        """
        for capture in self.capture_instances:
            try:
                capture.stop_audio_capture()
            except Exception as e:
                self._debug_print(f"Error cleaning up capture instance: {e}")
        self.capture_instances = []

    @property
    def time(self):
        """
        Get the current time of the stream (sounddevice compatible method)

        ストリームの現在の時間を返す (sounddevice互換メソッド)

        Returns:
        --------
        float
            Current time of the stream (in seconds)
            ストリームの現在の時間（秒単位）
        """
        # Get time from any of the capture instances
        if self.capture_instances:
            for capture in self.capture_instances:
                if hasattr(capture, "get_current_time"):
                    return capture.get_current_time()
        return -1.0  # Return -1.0 if capture is not initialized

    @property
    def sample_rate(self):
        """
        Get the sample rate (alias for samplerate for soundfile compatibility)

        サンプリングレートを取得 (soundfileとの互換性のためのsamplerateのエイリアス)

        Returns:
        --------
        int
            Sample rate in Hz
            サンプリングレート（Hz）
        """
        return self.samplerate

    def start(self):
        """
        Start the stream (sounddevice compatible method)

        ストリームを開始する (sounddevice互換メソッド)

        Returns:
        --------
        self
            Returns self for method chaining
            チェーンメソッドのためにselfを返す
        """
        # Start the stream and check if successful
        result = self._start_stream()
        if not result:
            import platform

            if platform.system() == "Darwin":
                print("\nMacOSでの音声キャプチャに必要な設定が行われていません。")
                print("以下の手順を実行してください:")
                print(
                    "1. scripts/install_audio_tools.py を実行して必要なツールをインストール"
                )
                print(
                    "2. システム環境設定で「fujielab-output」という複数出力デバイスを作成"
                )
                print("   (現在のスピーカーと「BlackHole 2ch」の両方を含める)")
            raise RuntimeError("Failed to start the stream")

        return self  # For sounddevice compatibility, return self

    def stop(self):
        """
        Stop the stream (sounddevice compatible method)

        ストリームを停止する (sounddevice互換メソッド)

        Returns:
        --------
        self
            Returns self for method chaining
            チェーンメソッドのためにselfを返す
        """
        self._stop_stream()
        return self  # チェーンメソッドのためにselfを返す

    def close(self):
        """
        Close the stream (sounddevice compatible method)

        ストリームを閉じる (sounddevice互換メソッド)

        Returns:
        --------
        self
            Returns self for method chaining
            チェーンメソッドのためにselfを返す
        """
        if self.running:
            self.stop()
        return self  # チェーンメソッドのためにselfを返す

    def active(self):
        """
        Check if the stream is active (sounddevice compatible method)

        ストリームがアクティブかどうかを返す (sounddevice互換メソッド)

        Returns:
        --------
        bool
            True if the stream is active, False otherwise
            ストリームがアクティブであればTrue、そうでなければFalse
        """
        return self.running

    # 以下はPyAudio互換のために残す（下位互換性）
    def is_active(self):
        """
        Check if the stream is active (PyAudio compatible method)

        ストリームがアクティブかどうかを返す (PyAudio互換メソッド)

        Returns:
        --------
        bool
            True if the stream is active, False otherwise
            ストリームがアクティブであればTrue、そうでなければFalse
        """
        return self.active()

    def _start_stream(self):
        """内部実装: ストリームを開始する"""
        import platform

        # Clean up existing capture instances
        self.capture_instances = []

        # Create instances according to the specified capture settings
        for i, capture_config in enumerate(self.captures):
            capture_type = capture_config.capture_type
            device_name = capture_config.device_name
            channels = capture_config.channels

            try:
                if capture_type.lower() == "input":
                    # Create InputCapture instance
                    input_instance = InputCapture(
                        sample_rate=self.samplerate,
                        channels=channels,
                        blocksize=self.blocksize,
                        debug=self.debug,
                    )

                    # Start audio capture for input
                    started = input_instance.start_audio_capture(
                        device_name=device_name,
                        sample_rate=self.samplerate,
                        channels=channels,
                        blocksize=self.blocksize,
                    )

                    if started:
                        self._debug_print(
                            f"Input stream started successfully: Device={device_name}, Channels={channels}"
                        )
                        self.capture_instances.append(input_instance)
                    else:
                        self._debug_print(
                            f"Failed to start input stream: Device={device_name}"
                        )
                        self._cleanup_captures()
                        return False

                elif capture_type.lower() == "output":
                    # Create OutputCapture instance (according to the platform)
                    output_instance = OutputCapture(
                        sample_rate=self.samplerate,
                        channels=channels,
                        blocksize=self.blocksize,
                        debug=self.debug,
                    )

                    # Start audio capture for output
                    started = output_instance.start_audio_capture(
                        device_name=device_name,
                        sample_rate=self.samplerate,
                        channels=channels,
                        blocksize=self.blocksize,
                    )

                    if started:
                        self._debug_print(
                            f"Output stream started successfully: Device={device_name}, Channels={channels}"
                        )
                        self.capture_instances.append(output_instance)
                    else:
                        self._debug_print(
                            f"\nFailed to start output capture: Device={device_name}"
                        )
                        self._debug_print(
                            "Please check if the environment is set up correctly."
                        )
                        self._cleanup_captures()
                        return False

                else:
                    self._debug_print(f"Unsupported capture type: {capture_type}")
                    self._cleanup_captures()
                    return False

            except Exception as err:
                self._debug_print(f"Error initializing {capture_type}: {err}")
                self._cleanup_captures()
                return False

        # If all captures are successfully started
        if self.capture_instances:
            # Start processing thread
            self.running = True
            self.thread = threading.Thread(target=self._loop)
            self.thread.daemon = True  # Exit together with
            self.thread.start()
            self._debug_print("Stream started successfully")
            return True  # Return True if the stream started successfully
        else:
            self._debug_print("No capture instances were successfully started")
            return False

    def _stop_stream(self):
        """
        Internal implementation: Stop the stream

        内部実装: ストリームを停止する
        """
        if not self.running:
            return

        self.running = False

        # If thread exists and is running, wait for it to finish
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=2.0)  # Wait for a maximum of 2 seconds

        # Stop all capture instances
        for capture in self.capture_instances:
            try:
                capture.stop_audio_capture()
            except Exception as e:
                self._debug_print(f"Error stopping capture instance: {e}")

        self.capture_instances = []

        # Clear the queue
        def clear_queue(q):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        # Clear processed data queue
        clear_queue(self.data_queue)

        # Explicitly release resources to prevent memory leaks
        import gc

        gc.collect()

    def _loop(self):
        """
        Internal thread: Process audio data and pass it to the callback or queue

        内部スレッド: オーディオデータを処理し、コールバックやキューに渡す
        """
        self._debug_print("Stream processing thread started")

        # Synchronization state
        sync_initialized = False
        overflow_detected = False

        # Buffer to hold previous blocks for sub-block alignment
        previous_blocks = [None] * len(self.capture_instances)

        # Initialize offsets for each capture instance
        offsets = [0] * len(self.capture_instances)

        while self.running:
            # Collect data from all capture instances
            captured_data = []
            timestamps = []

            for i, capture in enumerate(self.capture_instances):
                try:
                    data = capture.read_audio_capture()
                    if data is not None:
                        captured_data.append(data.data)
                        timestamps.append(data.time + self.captures[i].offset)

                        # Store data for sync analysis if sync is being performed
                        with self._sync_lock:
                            if (
                                len(self._sync_buffers[i])
                                < self._sync_buffer_size // self.blocksize
                            ):
                                self._sync_buffers[i].append(data.data.copy())
                            else:
                                # Remove oldest data to maintain buffer size
                                self._sync_buffers[i].pop(0)
                                self._sync_buffers[i].append(data.data.copy())
                    else:
                        self._debug_print("No data returned from capture instance.")
                except Exception as e:
                    self._debug_print(f"Error reading from capture instance: {e}")

            if not captured_data:
                # No data received, wait briefly before retrying
                time.sleep(0.01)
                continue

            if self.debug:
                for i, data in enumerate(captured_data):
                    rms = np.sqrt(np.mean(np.square(data))) if data.size > 0 else 0
                    self._debug_print(
                        f"Capture {i} data: {data.shape} at time {timestamps[i]:.3f}s: RMS={rms:.3f}..."
                    )

            # Synchronization logic
            if not sync_initialized or overflow_detected:
                self._debug_print("Performing synchronization...")

                # Apply chirp-based sync offsets if available
                with self._sync_lock:
                    if self._sync_enabled:
                        self._debug_print("Using chirp-based synchronization offsets")
                        # Apply sync offsets to timestamps
                        for i in range(len(timestamps)):
                            timestamps[i] -= self._sync_offsets[i]

                max_time = max(timestamps)
                base_index = timestamps.index(
                    max_time
                )  # Use the latest timestamp as the base

                self._debug_print(
                    f"Base index: {base_index}, Base time: {max_time:.3f}s"
                )

                # Align all data to the latest timestamp
                for i, capture in enumerate(self.capture_instances):
                    while timestamps[i] < max_time:
                        try:
                            data = capture.read_audio_capture()
                            # self._debug_print(f"Capture {i} data: {data.data.shape} at time {data.time:.3f}s")
                            if data is not None:
                                timestamps[i] = data.time + self.captures[i].offset
                                captured_data[i] = data.data
                            else:
                                break
                            self._debug_print(
                                f"Aligned Capture {i} data: {captured_data[i].shape} at time {timestamps[i]:.3f}s"
                            )
                        except Exception as e:
                            self._debug_print(f"Error during synchronization: {e}")
                            break

                # Calculate and store offsets in samples
                base_time = timestamps[base_index]
                for i, timestamp in enumerate(timestamps):
                    if i != base_index:
                        offsets[i] = int((timestamp - base_time) * self.samplerate)

                sync_initialized = True
                overflow_detected = False
                self._debug_print("Synchronization complete")

            # Adjust data using offsets to ensure continuity
            for i, data in enumerate(captured_data):
                if i == base_index:
                    continue

                offset_samples = offsets[i]
                if previous_blocks[i] is not None:
                    combined_block = np.concatenate((previous_blocks[i], data))
                    start_index = self.blocksize - max(0, offset_samples)
                    # start_index = self.blocksize
                    end_index = start_index + self.blocksize
                    captured_data[i] = combined_block[start_index:end_index]
                else:
                    captured_data[i] = data[: self.blocksize]

                # Store the current block for the next iteration
                previous_blocks[i] = data

            # Validate data sizes before combining
            valid_data = []
            target_size = self.blocksize

            for i, data in enumerate(captured_data):
                if data.size == 0:
                    self._debug_print(
                        f"Warning: Capture {i} returned empty data, skipping this iteration"
                    )
                    continue

                # Ensure data has correct shape and size
                if len(data.shape) == 1:
                    data = data.reshape(-1, 1)

                # Pad or truncate to target size
                if data.shape[0] < target_size:
                    padding = np.zeros(
                        (target_size - data.shape[0], data.shape[1]), dtype=data.dtype
                    )
                    data = np.vstack([data, padding])
                elif data.shape[0] > target_size:
                    data = data[:target_size, :]

                valid_data.append(data)

            # Skip this iteration if no valid data
            if not valid_data:
                self._debug_print("No valid data available, skipping this iteration")
                time.sleep(0.01)
                continue

            # Combine data from all capture instances
            combined_data = (
                np.hstack(valid_data) if len(valid_data) > 1 else valid_data[0]
            )

            # for i, data in enumerate(captured_data):
            #     print(f"Capture {i}: {data.shape} at time {timestamps[i]:.3f} (offset={offsets[i]/self.samplerate:.3f}s, {offsets[i]})")

            processed_data = self.processor.process(combined_data)
            # processed_data = combined_data.copy()

            # Pass data to the callback if provided
            if self.callback:
                try:
                    self.callback(
                        processed_data,
                        len(processed_data),
                        {"current_time": time.time()},
                        CallbackFlags(),
                    )
                except Exception as e:
                    self._debug_print(f"Error in callback: {e}")

            # Add data to the queue for further processing
            try:
                if not self.data_queue.full():
                    self.data_queue.put_nowait(processed_data)
            except Exception as e:
                self._debug_print(f"Error adding data to queue: {e}")

            # Check for overflow in any capture instance
            for capture in self.capture_instances:
                if hasattr(capture, "_callback_error") and capture._callback_error:
                    self._debug_print("Buffer overflow detected. Resynchronizing...")
                    overflow_detected = True
                    break

        self._debug_print("Stream processing thread stopped")

    def synchronize_with_chirp(self, duration=1.0, f0=1000, f1=2000, amplitude=0.3):
        """
        Synchronize multiple capture devices using a chirp signal

        チャープ信号を使用して複数のキャプチャデバイスを同期する

        Parameters:
        -----------
        duration : float, optional
            Duration of the chirp signal in seconds (default: 1.0)
            チャープ信号の長さ（秒）（デフォルト: 1.0）
        f0 : float, optional
            Starting frequency of the chirp in Hz (default: 1000)
            チャープの開始周波数（Hz）（デフォルト: 1000）
        f1 : float, optional
            Ending frequency of the chirp in Hz (default: 2000)
            チャープの終了周波数（Hz）（デフォルト: 2000）
        amplitude : float, optional
            Amplitude of the chirp signal (0.0-1.0) (default: 0.3)
            チャープ信号の振幅（0.0-1.0）（デフォルト: 0.3）

        Returns:
        --------
        bool
            True if synchronization was successful, False otherwise
            同期が成功した場合True、そうでなければFalse
        """
        if not self.running:
            self._debug_print("Stream must be running to perform synchronization")
            return False

        if len(self.capture_instances) < 2:
            self._debug_print(
                "At least 2 capture devices are required for synchronization"
            )
            return False

        self._debug_print("Starting chirp synchronization...")

        try:
            # Generate chirp signal
            chirp_signal = self._generate_chirp_signal(duration, f0, f1, amplitude)

            # Clear sync buffers
            with self._sync_lock:
                self._sync_buffers = [[] for _ in range(len(self.capture_instances))]

            # Start collecting sync data
            sync_start_time = time.time()

            # Play chirp signal
            self._debug_print(
                f"Playing chirp signal (duration: {duration}s, {f0}-{f1}Hz)"
            )
            self._play_chirp_signal(chirp_signal)

            # Wait for chirp to complete and collect additional data
            collection_time = duration + 0.5  # Extra time for signal propagation
            time.sleep(collection_time)

            # Analyze collected data to find chirp onsets
            onset_times = self._detect_chirp_onsets(chirp_signal, f0, f1)

            if len(onset_times) < len(self.capture_instances):
                self._debug_print(
                    f"Could not detect chirp in all devices. Detected in {len(onset_times)}/{len(self.capture_instances)} devices"
                )
                return False

            # Calculate synchronization offsets
            reference_time = min(onset_times)  # Use earliest detection as reference
            with self._sync_lock:
                self._sync_offsets = [
                    onset_time - reference_time for onset_time in onset_times
                ]
                self._sync_enabled = True

            self._debug_print("Synchronization offsets calculated:")
            for i, offset in enumerate(self._sync_offsets):
                self._debug_print(f"  Device {i}: {offset*1000:.2f}ms")

            self._debug_print("Chirp synchronization completed successfully")
            return True

        except Exception as e:
            self._debug_print(f"Error during chirp synchronization: {e}")
            return False

    def _generate_chirp_signal(self, duration, f0, f1, amplitude):
        """
        Generate a linear chirp signal

        線形チャープ信号を生成
        """
        t = np.linspace(0, duration, int(self.samplerate * duration), False)
        chirp = amplitude * scipy.signal.chirp(t, f0, duration, f1, method="linear")

        # Add brief silence at the beginning and end for clear onset detection
        silence_samples = int(0.1 * self.samplerate)  # 100ms silence
        silence = np.zeros(silence_samples)

        full_signal = np.concatenate([silence, chirp, silence])
        return full_signal

    def _play_chirp_signal(self, chirp_signal):
        """
        Play the chirp signal through the default output device

        デフォルト出力デバイスからチャープ信号を再生
        """
        try:
            # Play the signal (blocking call)
            sd.play(chirp_signal, self.samplerate, blocking=True)
        except Exception as e:
            self._debug_print(f"Error playing chirp signal: {e}")
            raise

    def _detect_chirp_onsets(self, reference_chirp, f0, f1):
        """
        Detect chirp onsets in captured audio data using cross-correlation

        相互相関を使用してキャプチャされた音声データからチャープのオンセットを検出
        """
        onset_times = []

        with self._sync_lock:
            sync_buffers = [buffer.copy() for buffer in self._sync_buffers]

        # Prepare reference signal for correlation (just the chirp part without silence)
        ref_start = int(0.1 * self.samplerate)  # Skip initial silence
        ref_end = ref_start + len(reference_chirp) - 2 * ref_start
        reference = reference_chirp[ref_start:ref_end]

        for i, buffer_data in enumerate(sync_buffers):
            if not buffer_data:
                self._debug_print(f"No sync data collected for device {i}")
                onset_times.append(0.0)
                continue

            # Concatenate buffer data
            audio_data = np.concatenate(buffer_data)

            # Convert to mono if stereo
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                audio_data = np.mean(audio_data, axis=1)

            try:
                # Cross-correlation to find chirp
                correlation = scipy.signal.correlate(
                    audio_data, reference, mode="valid"
                )

                # Find peak correlation
                peak_idx = np.argmax(np.abs(correlation))

                # Convert sample index to time
                onset_time = peak_idx / self.samplerate
                onset_times.append(onset_time)

                self._debug_print(
                    f"Device {i}: Chirp detected at {onset_time:.3f}s (correlation peak: {correlation[peak_idx]:.3f})"
                )

            except Exception as e:
                self._debug_print(f"Error detecting onset for device {i}: {e}")
                onset_times.append(0.0)

        return onset_times

    def get_sync_offsets(self):
        """
        Get current synchronization offsets

        現在の同期オフセットを取得

        Returns:
        --------
        list of float
            Synchronization offsets in seconds for each capture device
            各キャプチャデバイスの同期オフセット（秒）
        """
        with self._sync_lock:
            return self._sync_offsets.copy()

    def is_synchronized(self):
        """
        Check if devices are synchronized

        デバイスが同期されているかチェック

        Returns:
        --------
        bool
            True if synchronized, False otherwise
            同期されている場合True、そうでなければFalse
        """
        with self._sync_lock:
            return self._sync_enabled


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test InputStream with audio capture")
    parser.add_argument("--debug", action="store_true", help="Enable debug messages")
    parser.add_argument(
        "--sync", action="store_true", help="Test chirp synchronization"
    )
    args = parser.parse_args()

    audio_buff = None

    def callback(indata, frames, time_info, status):
        global audio_buff
        if audio_buff is None:
            audio_buff = indata.copy()
        else:
            audio_buff = np.vstack((audio_buff, indata))
        if status.INPUT_OVERFLOW:
            print("Input overflow detected!")

    input_stream = InputStream(
        samplerate=16000,
        blocksize=512,
        captures=[
            CaptureConfig(capture_type="Input", device_name=None, channels=1),
            CaptureConfig(
                capture_type="Output", device_name=None, channels=2
            ),  # , offset=0.05),
        ],
        callback=callback,
        debug=args.debug,  # デバッグメッセージを無効にする（必要に応じて True に変更）
    )

    try:
        input_stream.start()
        print("Stream started successfully.")

        if args.sync:
            print("Testing chirp synchronization...")
            time.sleep(1)  # Wait for stream to stabilize

            success = input_stream.synchronize_with_chirp(
                duration=1.0, f0=1000, f1=2000, amplitude=0.3
            )
            if success:
                print("Synchronization successful!")
                offsets = input_stream.get_sync_offsets()
                print("Sync offsets:", [f"{offset*1000:.2f}ms" for offset in offsets])
            else:
                print("Synchronization failed!")

        # Simulate some processing
        time.sleep(5)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        input_stream.close()
        print("Stream closed.")

    import soundfile as sf

    # Save the captured audio data to a file
    sf.write("output.wav", audio_buff, 16000, subtype="PCM_16")
    print("Audio data saved to output.wav")
