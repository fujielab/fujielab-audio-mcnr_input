"""
Audio Input Capture Module for Windows

This module provides classes for capturing audio from various input sources (e.g., microphones) on Windows using soundcard.

Windows用入力オーディオキャプチャモジュール

このモジュールは、soundcardを使用してWindows上で各種オーディオ入力（マイク等）からのオーディオをキャプチャするためのクラスを提供します。
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


class InputCaptureWin(InputCaptureBase):
    """
    Audio Input Capture Class for Windows

    This class provides an interface for capturing audio from input sources on Windows using soundcard library.

    Windows用入力オーディオキャプチャクラス

    このクラスは、soundcardライブラリを使用してWindows上でオーディオ入力からのキャプチャ機能を提供します。
    """

    def __init__(self, sample_rate=16000, channels=1, blocksize=1024, debug=False):
        """
        Initialize audio input capture for Windows

        Windows用入力キャプチャの初期化

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
        """
        if not SOUNDCARD_AVAILABLE:
            raise RuntimeError("soundcard library is not available. Please install soundcard for Windows input capture.")
            
        # Call parent constructor
        super().__init__(sample_rate, channels, blocksize, debug)
        
        # soundcard specific variables
        self._recording = False
        self._microphone = None
        self._capture_thread = None
        self._stop_event = None

        self._debug_print("Windows InputCapture initialized with soundcard backend")

    @property
    def time(self):
        """現在の時間（time.time()）"""
        return time.time()

    def _soundcard_capture_thread(self):
        """
        soundcard recording thread
        
        soundcard録音用スレッド
        """
        try:
            self._debug_print(f"Starting soundcard recording thread with sample rate: {self._sample_rate}Hz")
            
            with self._microphone.recorder(samplerate=self._sample_rate, channels=self._channels) as recorder:
                while not self._stop_event.is_set():
                    try:
                        # Record a block of audio data
                        data = recorder.record(numframes=self._blocksize)
                        
                        if data is not None and len(data) > 0:
                            # Create AudioData with current timestamp
                            current_time = time.time()
                            audio_data = AudioData(
                                data=data.copy(),
                                time=current_time,
                                overflowed=False  # soundcard doesn't provide overflow info
                            )
                            
                            # Add to queue, removing old data if queue is full
                            try:
                                if self._audio_queue.full():
                                    try:
                                        self._audio_queue.get_nowait()
                                    except queue.Empty:
                                        pass
                                self._audio_queue.put_nowait(audio_data)
                                # self._debug_print(f"Audio data added to queue: {audio_data.time:.3f}s")
                            except queue.Full:
                                pass  # Queue is still full, skip this data
                                
                    except Exception as e:
                        if not self._stop_event.is_set():
                            self._debug_print(f"Error in recording loop: {e}")
                        break
                        
        except Exception as e:
            self._debug_print(f"Recording thread error: {e}")
        finally:
            self._debug_print("Recording thread finished")

    @staticmethod
    def list_audio_devices(debug=False):
        """
        List available audio input devices on Windows

        Windows上の利用可能なオーディオ入力デバイスを一覧表示する

        Parameters:
        -----------
        debug : bool, optional
            Enable debug messages (default: False)
            デバッグメッセージを有効にする (デフォルト: False)

        Returns:
        --------
        bool
            True if successful, False otherwise

            成功した場合はTrue、失敗した場合はFalse
        """
        def _debug_print_local(message):
            if debug:
                print(message)

        try:
            if not SOUNDCARD_AVAILABLE:
                _debug_print_local("soundcard library is not available")
                return False
                
            # soundcard based listing for Windows
            _debug_print_local("\nAvailable audio input devices (soundcard):")
            
            try:
                microphones = sc.all_microphones(include_loopback=False)
                for i, mic in enumerate(microphones):
                    _debug_print_local(f"  {i}: {mic.name}")
                    
                # Show default device
                try:
                    default_mic = sc.default_microphone()
                    _debug_print_local(f"Default microphone: {default_mic.name}")
                except Exception as e:
                    _debug_print_local(f"Could not get default microphone: {e}")
                    
                # Show preferred device for Windows
                preferred_device = InputCaptureWin._find_preferred_windows_device_soundcard(debug=debug)
                if preferred_device is not None:
                    _debug_print_local(f"Recommended Windows device: {preferred_device.name}")
                    
            except Exception as e:
                _debug_print_local(f"Error listing soundcard devices: {e}")
                return False

            return True
        except Exception as e:
            _debug_print_local(f"Failed to list devices: {e}")
            return False

    def start_audio_capture(self, device_name=None, sample_rate=None, channels=None, blocksize=None):
        """
        Start audio capture

        入力キャプチャを開始する

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

    def _start_soundcard_capture(self, device_name=None):
        """
        Start audio capture using soundcard (Windows)
        
        soundcardを使用してオーディオキャプチャを開始（Windows）
        """
        try:
            # Stop existing recording
            if self._recording:
                self.stop_audio_capture()

            # Select microphone
            if device_name is not None:
                # Find microphone by name or index
                microphones = sc.all_microphones(include_loopback=False)
                selected_mic = None
                
                # Try as index first
                if isinstance(device_name, (int, str)) and str(device_name).isdigit():
                    idx = int(device_name)
                    if 0 <= idx < len(microphones):
                        selected_mic = microphones[idx]
                
                # Try as name
                if selected_mic is None:
                    for mic in microphones:
                        if device_name.lower() in mic.name.lower():
                            selected_mic = mic
                            break
                
                if selected_mic is None:
                    self._debug_print(f"Device '{device_name}' not found, using default")
                    self._microphone = sc.default_microphone()
                else:
                    self._microphone = selected_mic
            else:
                # Try to get preferred Windows device
                preferred_mic = self._find_preferred_windows_device_soundcard(debug=self.debug)
                if preferred_mic is not None:
                    self._microphone = preferred_mic
                    self._debug_print(f"Using preferred Windows microphone: {self._microphone.name}")
                else:
                    # Use default microphone
                    self._microphone = sc.default_microphone()

            self._debug_print(f"Using microphone: {self._microphone.name}")
            
            # Test sample rate compatibility
            if not self._test_soundcard_sample_rate(self._microphone, self._sample_rate):
                # Try common rates
                for rate in [48000, 44100, 22050, 16000, 8000]:
                    if self._test_soundcard_sample_rate(self._microphone, rate):
                        self._debug_print(f"Adjusting sample rate from {self._sample_rate}Hz to {rate}Hz")
                        self._sample_rate = rate
                        break
                else:
                    self._debug_print("Warning: Could not find compatible sample rate")
            
            # Start recording thread
            self._stop_event = threading.Event()
            self._capture_thread = threading.Thread(target=self._soundcard_capture_thread)
            self._recording = True
            self._stream_initialized = True
            self._capture_thread.start()
            
            # Wait a bit and test
            time.sleep(0.3)
            if self._audio_queue.empty():
                self._debug_print("Warning: No data received from the microphone yet")
            else:
                self._debug_print("Microphone initialization confirmed: Receiving data")
            
            return True
            
        except Exception as e:
            self._debug_print(f"Error starting soundcard capture: {e}")
            self._recording = False
            self._stream_initialized = False
            return False

    def _test_soundcard_sample_rate(self, microphone, sample_rate):
        """
        Test if sample rate is supported by soundcard microphone
        
        soundcardマイクでサンプリングレートがサポートされているかテスト
        """
        try:
            # Try to create a recorder with the given sample rate
            with microphone.recorder(samplerate=sample_rate, channels=1) as recorder:
                # If we can create it without exception, it's supported
                return True
        except Exception as e:
            self._debug_print(f"Sample rate {sample_rate}Hz not supported: {e}")
            return False

    @staticmethod
    def _find_preferred_windows_device_soundcard(debug=False):
        """
        Find preferred input device for Windows using soundcard
        
        soundcardを使用してWindows用の優先入力デバイスを探す
        
        Parameters:
        -----------
        debug : bool, optional
            Enable debug messages (default: False)
            
        Returns:
        --------
        soundcard.Microphone or None
            Microphone object if found, None otherwise
        """
        def _debug_print_local(message):
            if debug:
                print(message)
                
        try:
            if not SOUNDCARD_AVAILABLE:
                return None
                
            # Get default microphone
            try:
                default_mic = sc.default_microphone()
                default_name = default_mic.name.lower()
                _debug_print_local(f"Default microphone: {default_mic.name}")
            except Exception as e:
                _debug_print_local(f"Could not get default microphone: {e}")
                return None
            
            # Get all microphones
            microphones = sc.all_microphones(include_loopback=False)
            
            # Find best match
            best_mic = None
            best_score = 0
            
            for mic in microphones:
                mic_name_lower = mic.name.lower()
                score = 0
                
                # Exact match gets highest score
                if mic_name_lower == default_name:
                    score = 100
                # Partial match
                elif default_name in mic_name_lower or mic_name_lower in default_name:
                    score = 80
                # Word-based matching
                else:
                    default_words = set(default_name.split())
                    mic_words = set(mic_name_lower.split())
                    common_words = default_words.intersection(mic_words)
                    score = len(common_words) * 20
                
                # Prefer devices with 'microphone' or 'mic' in the name
                if any(keyword in mic_name_lower for keyword in ['microphone', 'mic', 'マイク']):
                    score += 5
                
                _debug_print_local(f"Microphone: {mic.name} (Score: {score})")
                
                if score > best_score:
                    best_score = score
                    best_mic = mic
            
            if best_mic:
                _debug_print_local(f"Selected microphone: {best_mic.name} (Score: {best_score})")
                return best_mic
            else:
                _debug_print_local("No suitable microphone found")
                return None
                
        except Exception as e:
            _debug_print_local(f"Error in Windows device selection: {e}")
            return None

    def read_audio_capture(self):
        """
        Read captured audio data

        キャプチャしたオーディオデータを読み取る

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
        Stop audio capture

        入力オーディオキャプチャを停止する

        Returns:
        --------
        bool
            True if stopped successfully.
            停止に成功した場合はTrue
        """
        return self._stop_soundcard_capture()

    def _stop_soundcard_capture(self):
        """
        Stop soundcard capture
        
        soundcardキャプチャを停止
        """
        try:
            if self._recording:
                self._recording = False
                if self._stop_event:
                    self._stop_event.set()
                if self._capture_thread and self._capture_thread.is_alive():
                    self._capture_thread.join(timeout=2.0)
                self._stream_initialized = False
                self._debug_print("Soundcard capture stopped")
            return True
        except Exception as e:
            self._debug_print(f"Error stopping soundcard capture: {e}")
            return False


# モジュールレベルでの関数
def create_input_capture_instance(sample_rate=16000, channels=1, blocksize=1024):
    """
    Create an input capture instance for Windows

    Windows用入力キャプチャインスタンスを作成する

    Parameters:
    -----------
    sample_rate : int, optional
        サンプリングレート（Hz）（デフォルト: 16000Hz）
    channels : int, optional
        チャネル数（デフォルト: 1チャネル（モノラル））
    blocksize : int, optional
        ブロックサイズ（フレーム数）（デフォルト: 1024）

    Returns:
    --------
    InputCaptureWin
        Windows用入力キャプチャインスタンス
    """
    return InputCaptureWin(sample_rate=sample_rate, channels=channels, blocksize=blocksize)


def list_devices():
    """
    List available audio input devices on Windows

    Windows上の利用可能なオーディオ入力デバイスを一覧表示

    Returns:
    --------
    bool
        成功した場合はTrue、失敗した場合はFalse
    """
    return InputCaptureWin.list_audio_devices()
