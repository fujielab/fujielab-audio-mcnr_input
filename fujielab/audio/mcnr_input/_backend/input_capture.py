"""
Audio Input Capture Module

This module provides classes for capturing audio from various input sources (e.g., microphones).

入力オーディオキャプチャモジュール

このモジュールは、各種オーディオ入力（マイク等）からのオーディオをキャプチャするためのクラスを提供します。
"""
import sounddevice as sd
import numpy as np
import queue
import threading
import time
from .data import AudioData


class InputCapture:
    """
    Audio Input Capture Class

    This class provides an interface similar to OutputCapture and captures audio from various input sources (e.g., microphones).

    入力オーディオキャプチャクラス

    このクラスは、OutputCaptureと類似のインターフェースを提供し、
    各種オーディオ入力（マイク等）からのオーディオをキャプチャします。
    """

    def __init__(self, sample_rate=16000, channels=1, blocksize=1024, debug=False):
        """
        Initialize audio input capture

        入力キャプチャの初期化

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
        self._sample_rate = sample_rate
        self._channels = channels
        self._blocksize = blocksize
        self.debug = debug

        # インスタンス変数の初期化
        self._audio_queue = queue.Queue(maxsize=20)
        self._callback_error = None
        self._callback_lock = threading.Lock()
        self._stream = None
        self._error_count = 0
        self._stream_initialized = False

    def _debug_print(self, message):
        """
        Print debug message if debug mode is enabled
        デバッグモードが有効な場合にデバッグメッセージを出力
        """
        if self.debug:
            print(message)

    @property
    def sample_rate(self):
        """サンプリングレート（Hz）"""
        return self._sample_rate

    @property
    def channels(self):
        """チャネル数"""
        return self._channels

    @property
    def blocksize(self):
        """ブロックサイズ（フレーム数）"""
        return self._blocksize

    @property
    def time(self):
        """Sream上の現在の時間"""
        return self._stream.time

    def _audio_callback(self, indata, frames, time_info, status):
        """
        sounddevice callback function

        Processes captured audio data and adds it to the queue.

        sounddeviceのコールバック関数

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

                # AudioDataオブジェクトとして時間情報付きでキューに保存
                audio_data = AudioData(
                    data=indata.copy(),
                    time=time_info.inputBufferAdcTime if hasattr(time_info, 'inputBufferAdcTime') else time.time(),
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
                except:
                    # キューへの追加に失敗した場合
                    pass
        except Exception as e:
            self._debug_print(f"Microphone callback error: {e}")

    @staticmethod
    def list_audio_devices(debug=False):
        """
        List available audio input devices

        システム上の利用可能なオーディオ入力デバイスを一覧表示する

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
            _debug_print_local("\nAvailable audio input devices:")
            devices = sd.query_devices()
            input_devices = []

            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    _debug_print_local(f"  {i}: {dev['name']} (入力チャンネル: {dev['max_input_channels']})")
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

        # デバイスが見つからない場合はデフォルトまたは最初の入力デバイスを使用
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

        try:
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

            # データがない場合は無音データを返すことも可能
            # silent_data = np.zeros((self._blocksize, 1), dtype=np.float32)
            # return AudioData(data=silent_data, time=time.time(), overflowed=False)

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
    Create an input capture instance

    入力キャプチャインスタンスを作成する

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
    InputCapture
        入力キャプチャインスタンス
    """
    return InputCapture(sample_rate=sample_rate, channels=channels, blocksize=blocksize)


def list_devices():
    """
    List available audio input devices

    利用可能なオーディオ入力デバイスを一覧表示

    Returns:
    --------
    bool
        成功した場合はTrue、失敗した場合はFalse
    """
    return InputCapture.list_audio_devices()

# staticmethodとしてのユーティリティ関数
