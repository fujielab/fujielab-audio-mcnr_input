"""
Audio Input Capture Module for Mac

This module provides classes for capturing audio from various input sources (e.g., microphones) on Mac using sounddevice.

Mac用入力オーディオキャプチャモジュール

このモジュールは、sounddeviceを使用してMac上で各種オーディオ入力（マイク等）からのオーディオをキャプチャするためのクラスを提供します。
"""
import numpy as np
import queue
import threading
import time
from .data import AudioData
from .input_capture_base import InputCaptureBase

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    sd = None


class InputCaptureMac(InputCaptureBase):
    """
    Audio Input Capture Class for Mac

    This class provides an interface for capturing audio from input sources on Mac using sounddevice library.

    Mac用入力オーディオキャプチャクラス

    このクラスは、sounddeviceライブラリを使用してMac上でオーディオ入力からのキャプチャ機能を提供します。
    """

    def __init__(self, sample_rate=16000, channels=1, blocksize=1024, debug=False):
        """
        Initialize audio input capture for Mac

        Mac用入力キャプチャの初期化

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
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError("sounddevice library is not available. Please install sounddevice for Mac input capture.")
            
        # Call parent constructor
        super().__init__(sample_rate, channels, blocksize, debug)

        # Additional Mac-specific variables
        self._callback_error = None
        self._callback_lock = threading.Lock()

        # sounddevice specific variables
        self._stream = None
        self._time_offset = 0.0

        self._debug_print("Mac InputCapture initialized with sounddevice backend")

    @property
    def time(self):
        """現在の時間（time.time()）"""
        return time.time()

    def _audio_callback(self, indata, frames, time_info, status):
        """
        sounddevice callback function (for Mac)

        Processes captured audio data and adds it to the queue.

        sounddeviceのコールバック関数（Mac用）

        キャプチャしたオーディオデータを処理してキューに追加する

        Parameters:
        -----------
        indata : numpy.ndarray
            入力オーディオデータ（フレーム数 x チャンネル数）
        frames : int
            現在のブロック内のフレーム数
        time_info : PaStreamCallbackTimeInfo Struct
            タイミング情報を含む構造体
        status : sounddevice.CallbackFlags
            エラーなどを示すフラグ
        """
        try:
            with self._callback_lock:
                # エラーの検出
                is_overflow = False
                if status and status.input_overflow:
                    self._debug_print("Microphone input overflow detected")
                    is_overflow = True

                # Use time.time() for timestamp (as requested)
                current_time = time.time()

                # AudioDataオブジェクトとして時間情報付きでキューに保存
                audio_data = AudioData(
                    data=indata.copy(),
                    time=current_time,
                    overflowed=is_overflow
                )

                # キューがいっぱいの場合は古いデータを捨てる
                try:
                    if self._audio_queue.full():
                        try:
                            self._audio_queue.get_nowait()
                        except:
                            pass
                    self._audio_queue.put_nowait(audio_data)
                    self._debug_print(f"Audio data added to queue: {audio_data.time:.3f}s (Overflow: {is_overflow})")
                except:
                    # キューへの追加に失敗した場合
                    pass
        except Exception as e:
            self._debug_print(f"Microphone callback error: {e}")

    @staticmethod
    def list_audio_devices(debug=False):
        """
        List available audio input devices on Mac

        Mac上の利用可能なオーディオ入力デバイスを一覧表示する

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
            if not SOUNDDEVICE_AVAILABLE:
                _debug_print_local("sounddevice library is not available")
                return False
                
            # sounddevice based listing for Mac
            _debug_print_local("\nAvailable audio input devices (sounddevice):")
            devices = sd.query_devices()
            host_apis = sd.query_hostapis()
            input_devices = []

            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    host_api_name = host_apis[dev['hostapi']]['name']
                    _debug_print_local(f"  {i}: {dev['name']} (入力チャンネル: {dev['max_input_channels']}, Host API: {host_api_name})")
                    input_devices.append(i)

            # デフォルトの入力デバイスを取得
            try:
                default_device = sd.query_devices(kind='input')
                default_index = sd.default.device[0]
                _debug_print_local(f"Default input device: {default_device['name']} (Index {default_index})")
            except Exception as e:
                _debug_print_local(f"デフォルトデバイスの取得エラー: {e}")

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

        # マイクは通常1チャンネル（モノラル）のみサポート
        mic_channels = 1

        # キューをクリア
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except:
                break

        return self._start_sounddevice_capture(device_name, mic_channels)

    def _start_sounddevice_capture(self, device_name, mic_channels):
        """
        Start audio capture using sounddevice (Mac)
        
        sounddeviceを使用してオーディオキャプチャを開始（Mac）
        """
        try:
            # デバイス情報の取得
            devices = sd.query_devices()
            input_devices = []

            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    input_devices.append(i)

            # 指定されたデバイス名またはインデックスを探す
            device_index = None
            if device_name is not None:
                # 数値インデックスとして解釈を試みる
                if isinstance(device_name, (int, str)) and str(device_name).isdigit():
                    idx = int(device_name)
                    if 0 <= idx < len(devices) and devices[idx]['max_input_channels'] > 0:
                        device_index = idx
                # デバイス名として検索
                else:
                    for i, dev in enumerate(devices):
                        if dev['max_input_channels'] > 0 and device_name.lower() in dev['name'].lower():
                            device_index = i
                            break

            # デバイスが見つからない場合はデフォルトを使用
            if device_index is None:
                try:
                    default_device = sd.query_devices(kind='input')
                    default_index = sd.default.device[0]
                    if default_index in input_devices:
                        device_index = default_index
                    else:
                        device_index = input_devices[0] if input_devices else None
                except Exception as e:
                    self._debug_print(f"デフォルトデバイスの取得エラー: {e}")
                    device_index = input_devices[0] if input_devices else None

            # 有効なデバイスが見つからない場合
            if device_index is None:
                self._debug_print("Error: No valid audio input device found")
                self._stream_initialized = False
                return False

            self._debug_print(f"Using audio input device: {devices[device_index]['name']} (Index {device_index})")

            # 既存のストリームをクリーンアップ
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                    self._debug_print("Cleaned up existing microphone stream")
                except Exception as e:
                    self._debug_print(f"Stream cleanup error: {e}")

            # ストリーム作成とエラーハンドリング
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=mic_channels,
                blocksize=self._blocksize,
                callback=self._audio_callback,
                device=device_index
            )
            self._stream.start()
            self._debug_print("Microphone input stream started successfully")

            # time.time() と InputStream.time のオフセットを計算（使用しないが互換性のため）
            self._time_offset = time.time() - self._stream.time

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
            self._debug_print(f"Error creating microphone input stream: {e}")
            self._stream_initialized = False
            return False

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
        return self._stop_sounddevice_capture()

    def _stop_sounddevice_capture(self):
        """
        Stop sounddevice capture
        
        sounddeviceキャプチャを停止
        """
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
                self._stream = None
                self._stream_initialized = False
                self._debug_print("Input stream stopped")
                return True
            except Exception as e:
                self._debug_print(f"入力ストリーム停止エラー: {e}")
                return False
        return True  # 既に停止している場合も成功とみなす


# モジュールレベルでの関数
def create_input_capture_instance(sample_rate=16000, channels=1, blocksize=1024):
    """
    Create an input capture instance for Mac

    Mac用入力キャプチャインスタンスを作成する

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
    InputCaptureMac
        Mac用入力キャプチャインスタンス
    """
    return InputCaptureMac(sample_rate=sample_rate, channels=channels, blocksize=blocksize)


def list_devices():
    """
    List available audio input devices on Mac

    Mac上の利用可能なオーディオ入力デバイスを一覧表示

    Returns:
    --------
    bool
        成功した場合はTrue、失敗した場合はFalse
    """
    return InputCaptureMac.list_audio_devices()
