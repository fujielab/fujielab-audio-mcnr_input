"""
Common input capture implementation using soundcard

soundcardを使用した共通の入力キャプチャ実装
"""
import numpy as np
import queue
import threading
import time
from .data import AudioData
from .input_capture_base import InputCaptureBase

try:
    import soundcard as sc
    SOUNDCARD_AVAILABLE = True
except ImportError:
    SOUNDCARD_AVAILABLE = False
    sc = None


class SoundcardInputCaptureBase(InputCaptureBase):
    """
    Base class for soundcard-based input capture
    
    soundcardベースの入力キャプチャの基底クラス
    """

    def __init__(self, sample_rate=16000, channels=1, blocksize=1024, debug=False, platform_name=""):
        """
        Initialize soundcard-based input capture

        soundcardベースの入力キャプチャの初期化

        Parameters:
        -----------
        sample_rate : int, optional
            Sampling rate (Hz) (default: 16000Hz)
            サンプリングレート（Hz）（デフォルト: 16000Hz）
        channels : int, optional
            Number of channels (default: 1 channel, mono)
            チャネル数（デフォルト: 1チャネル（モノラル））
        blocksize : int, optional
            Block size (number of frames) (default: 1024)
            ブロックサイズ（フレーム数）（デフォルト: 1024）
        debug : bool, optional
            Enable debug messages (default: False)
            デバッグメッセージを有効にする (デフォルト: False)
        platform_name : str, optional
            Platform name for debug messages
            デバッグメッセージ用のプラットフォーム名
        """
        if not SOUNDCARD_AVAILABLE:
            raise RuntimeError(f"soundcard library is not available. Please install soundcard for {platform_name} input capture.")
            
        # Call parent constructor
        super().__init__(sample_rate, channels, blocksize, debug)

        # Common soundcard-specific variables
        self._recording_thread = None
        self._stop_recording = threading.Event()
        self._microphone_device = None
        self._platform_name = platform_name

        self._debug_print(f"{platform_name} InputCapture initialized with soundcard backend")

    @property
    def time(self):
        """現在の時間（time.time()）"""
        return time.time()

    def _recording_worker(self):
        """
        Worker thread for continuous audio recording using soundcard

        soundcardを使用した連続オーディオ録音のワーカースレッド
        """
        try:
            self._debug_print(f"Starting recording with device: {self._microphone_device.name}")
            self._debug_print(f"Block size: {self._blocksize}, Sample rate: {self._sample_rate}")

            # Open the recorder once and keep it open
            with self._microphone_device.recorder(
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
                                # self._debug_print(f"Audio data added to queue: {audio_data.time:.3f}s")
                            except queue.Full:
                                pass  # Ignore if queue is still full

                    except Exception as record_error:
                        if not self._stop_recording.is_set():
                            self._debug_print(f"Recording error: {record_error}")
                            time.sleep(0.1)  # Brief pause before retrying

        except Exception as e:
            self._debug_print(f"Recording worker error: {e}")
            self._stream_initialized = False

    @staticmethod
    def list_audio_devices_common(debug=False):
        """
        Common implementation for listing audio devices
        
        音声デバイス一覧表示の共通実装
        """
        def _debug_print_local(message):
            if debug:
                print(message)

        try:
            if not SOUNDCARD_AVAILABLE:
                _debug_print_local("soundcard library is not available")
                return False
                
            # soundcard based listing
            _debug_print_local("\nAvailable audio input devices (soundcard):")
            microphones = sc.all_microphones()

            for i, mic in enumerate(microphones):
                _debug_print_local(f"  {i}: {mic.name}")

            # デフォルトの入力デバイスを取得
            try:
                default_mic = sc.default_microphone()
                _debug_print_local(f"Default input device: {default_mic.name}")
            except Exception as e:
                _debug_print_local(f"デフォルトデバイスの取得エラー: {e}")

            return True
        except Exception as e:
            _debug_print_local(f"Failed to list devices: {e}")
            return False

    def start_audio_capture(self, device_name=None, sample_rate=None, channels=None, blocksize=None):
        """
        Start audio capture (common implementation)

        入力キャプチャを開始する（共通実装）

        Parameters:
        -----------
        device_name : str, optional
            Name or index of the input device to use. If None, the system default device is used.
            使用する入力デバイスの名前またはインデックス。
            Noneの場合、システムデフォルトデバイスを使用（デフォルト: None）
        sample_rate : int, optional
            Sampling rate (Hz)
            サンプリングレート（Hz）
        channels : int, optional
            Number of channels
            チャネル数
        blocksize : int, optional
            Block size (number of frames)
            ブロックサイズ（フレーム数）

        Returns:
        --------
        bool
            True if capture started successfully, False otherwise.
            キャプチャ開始に成功した場合はTrue、失敗した場合はFalse
        """
        # パラメータ更新（指定があれば）
        if sample_rate is not None:
            self._sample_rate = sample_rate
        if channels is not None:
            self._channels = channels
        if blocksize is not None:
            self._blocksize = blocksize

        # キューをクリア
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except:
                break

        return self._start_soundcard_capture(device_name)

    def _start_soundcard_capture(self, device_name):
        """
        Start audio capture using soundcard (common implementation)
        
        soundcardを使用してオーディオキャプチャを開始（共通実装）
        """
        try:
            # デバイス情報の取得
            microphones = sc.all_microphones()

            if not microphones:
                self._debug_print("Error: No microphones found")
                self._stream_initialized = False
                return False

            # 指定されたデバイス名またはインデックスを探す
            microphone_device = None
            if device_name is not None:
                # 数値インデックスとして解釈を試みる
                if isinstance(device_name, (int, str)) and str(device_name).isdigit():
                    idx = int(device_name)
                    if 0 <= idx < len(microphones):
                        microphone_device = microphones[idx]
                # デバイス名として検索
                else:
                    for mic in microphones:
                        if device_name.lower() in mic.name.lower():
                            microphone_device = mic
                            break

            # デバイスが見つからない場合はデフォルトを使用
            if microphone_device is None:
                try:
                    microphone_device = sc.default_microphone()
                except Exception as e:
                    self._debug_print(f"デフォルトデバイスの取得エラー: {e}")
                    microphone_device = microphones[0] if microphones else None

            # 有効なデバイスが見つからない場合
            if microphone_device is None:
                self._debug_print("Error: No valid audio input device found")
                self._stream_initialized = False
                return False

            self._debug_print(f"Using audio input device: {microphone_device.name}")
            self._microphone_device = microphone_device

            # 既存の録音スレッドを停止
            if self._recording_thread is not None and self._recording_thread.is_alive():
                self._stop_recording.set()
                self._recording_thread.join(timeout=2.0)
                self._debug_print("Stopped existing recording thread")

            # 新しい録音スレッドを開始
            self._stop_recording.clear()
            self._recording_thread = threading.Thread(target=self._recording_worker, daemon=True)
            self._recording_thread.start()
            
            self._debug_print("Microphone input thread started successfully")

            # テストデータ取得を試行
            time.sleep(0.3)  # 初期化待機
            
            if self._audio_queue.empty():
                self._debug_print("Warning: No data received from the microphone yet")
                self._stream_initialized = True  # とりあえず初期化は成功とみなす
            else:
                self._debug_print("Microphone initialization confirmed: Receiving data")
                self._stream_initialized = True

            return True
        except Exception as e:
            self._debug_print(f"Error creating microphone input capture: {e}")
            self._stream_initialized = False
            return False

    def read_audio_capture(self):
        """
        Read captured audio data (common implementation)

        キャプチャしたオーディオデータを読み取る（共通実装）

        Returns:
        --------
        AudioData
            Captured audio data object. Returns None if no data is available.
            キャプチャしたオーディオデータのオブジェクト。
            データが取得できない場合はNoneを返す

        Raises:
        -------
        RuntimeError
            If the stream is not initialized.
            ストリームが初期化されていない場合
        """
        if not self._stream_initialized:
            raise RuntimeError("マイク入力ストリームが初期化されていません")

        try:
            # タイムアウトを短くして応答性を改善
            audio_data = self._audio_queue.get(timeout=0.5)
            return audio_data  # AudioDataオブジェクトを返す
        except Exception as e:
            # 例外をカウントして頻発する場合のみ警告
            self._error_count += 1
            if self._error_count % 100 == 0:
                self._debug_print(f"Error retrieving microphone input data: {e} (last 100 attempts)")
                self._error_count = 0

            return None

    def stop_audio_capture(self):
        """
        Stop audio capture (common implementation)

        入力オーディオキャプチャを停止する（共通実装）

        Returns:
        --------
        bool
            True if stopped successfully.
            停止に成功した場合はTrue
        """
        return self._stop_soundcard_capture()

    def _stop_soundcard_capture(self):
        """
        Stop soundcard capture (common implementation)
        
        soundcardキャプチャを停止（共通実装）
        """
        try:
            # 録音スレッドを停止
            if self._recording_thread is not None and self._recording_thread.is_alive():
                self._stop_recording.set()
                self._recording_thread.join(timeout=2.0)
                self._recording_thread = None
                self._debug_print("Recording thread stopped")
            
            self._stream_initialized = False
            return True
        except Exception as e:
            self._debug_print(f"入力ストリーム停止エラー: {e}")
            return False
