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

        self._time_offset = 0.0 # time.time() と InputStream.time のオフセット

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

                # Stream時間からUNIX時間への調整
                if hasattr(time_info, 'inputBufferAdcTime') and time_info.inputBufferAdcTime > 0:
                    adjusted_time = time_info.inputBufferAdcTime + self._time_offset
                    self._debug_print(f"Using inputBufferAdcTime: {time_info.inputBufferAdcTime:.3f}s")
                else:
                    adjusted_time = time.time()
                    if hasattr(time_info, 'inputBufferAdcTime'):
                        self._debug_print(f"inputBufferAdcTime is 0 or invalid, using time.time()")
                    else:
                        self._debug_print(f"inputBufferAdcTime not available, using time.time()")

                # AudioDataオブジェクトとして時間情報付きでキューに保存
                audio_data = AudioData(
                    data=indata.copy(),
                    time=adjusted_time,
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

            # Windows環境での推奨デバイス表示
            import platform
            if platform.system() == 'Windows':
                preferred_device = InputCapture._find_preferred_windows_device(debug=debug)
                if preferred_device is not None:
                    _debug_print_local(f"Recommended Windows device: {devices[preferred_device]['name']} (Index {preferred_device})")

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
                # Windows環境でのWASAPIデバイス優先選択
                import platform
                if platform.system() == 'Windows':
                    preferred_device = self._find_preferred_windows_device(debug=self.debug)
                    if preferred_device is not None:
                        device_index = preferred_device
                        self._debug_print(f"Using preferred Windows WASAPI device: {devices[device_index]['name']}")
                    
                # WASAPIデバイスが見つからない、またはWindows以外の場合はデフォルトを使用
                if device_index is None:
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

        # Windows環境でWASAPIデバイスの場合、サポートされているサンプリングレートをテスト
        import platform
        host_apis = sd.query_hostapis()
        if (platform.system() == 'Windows' and 
            host_apis[devices[device_index]['hostapi']]['name'] == 'Windows WASAPI'):
            
            supported_rates = self._test_device_sample_rates(device_index, debug=self.debug)
            if supported_rates and self._sample_rate not in supported_rates:
                # Prefer higher quality rates if available
                preferred_order = [48000, 44100, 22050, 16000, 8000]
                new_rate = None
                for rate in preferred_order:
                    if rate in supported_rates:
                        new_rate = rate
                        break
                
                if new_rate:
                    self._debug_print(f"Adjusting sample rate from {self._sample_rate}Hz to {new_rate}Hz for WASAPI compatibility")
                    self._sample_rate = new_rate
                else:
                    self._debug_print(f"Warning: No suitable sample rate found, using first available: {supported_rates[0]}Hz")
                    self._sample_rate = supported_rates[0]

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

            # time.time() と InputStream.time のオフセットを計算
            self._time_offset = time.time() - self._stream.time

            # テストデータ取得を試行
            time.sleep(0.3)  # 初期化待機
            
            # Windows環境でタイムスタンプの品質をテスト
            import platform
            if platform.system() == 'Windows':
                self._test_timestamp_quality()
            
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

    def _test_timestamp_quality(self):
        """
        Test the quality of timestamp information from the selected device
        選択されたデバイスからのタイムスタンプ情報の品質をテスト
        """
        try:
            # Wait for some audio data and check timestamp quality
            time.sleep(0.5)
            valid_timestamps = 0
            zero_timestamps = 0
            test_samples = 0
            
            # Check a few samples
            for _ in range(5):
                try:
                    audio_data = self._audio_queue.get(timeout=0.2)
                    test_samples += 1
                    # Check if this data was captured with valid timestamp
                    # This is a simple heuristic - in practice you'd need more sophisticated checking
                    if hasattr(audio_data, 'time') and audio_data.time > 0:
                        # Check if the timestamp seems reasonable (not just time.time())
                        current_time = time.time()
                        if abs(audio_data.time - current_time) < 10:  # Within 10 seconds
                            valid_timestamps += 1
                        else:
                            zero_timestamps += 1
                    else:
                        zero_timestamps += 1
                except queue.Empty:
                    break
            
            if test_samples > 0:
                quality_ratio = valid_timestamps / test_samples
                self._debug_print(f"Timestamp quality test: {valid_timestamps}/{test_samples} valid ({quality_ratio:.2%})")
                
                if quality_ratio < 0.5:  # Less than 50% valid timestamps
                    self._debug_print("Warning: Low timestamp quality detected. Consider using a different input device.")
            
        except Exception as e:
            self._debug_print(f"Timestamp quality test failed: {e}")

    def _test_device_sample_rates(self, device_index, debug=False):
        """
        Test which sample rates are supported by the device
        デバイスでサポートされているサンプリングレートをテスト
        
        Parameters:
        -----------
        device_index : int
            Device index to test
        debug : bool, optional
            Enable debug messages
            
        Returns:
        --------
        list
            List of supported sample rates
        """
        def _debug_print_local(message):
            if debug:
                print(message)
                
        common_rates = [8000, 16000, 22050, 44100, 48000, 96000]
        supported_rates = []
        
        _debug_print_local(f"Testing sample rates for device {device_index}...")
        
        for rate in common_rates:
            try:
                # Try to create a test stream with minimal blocksize
                test_stream = sd.InputStream(
                    device=device_index,
                    samplerate=rate,
                    channels=1,
                    blocksize=128,
                    dtype='float32'
                )
                test_stream.close()  # Close immediately if successful
                supported_rates.append(rate)
                _debug_print_local(f"  ✓ {rate}Hz supported")
            except Exception as e:
                _debug_print_local(f"  ✗ {rate}Hz not supported: {e}")
        
        return supported_rates
    
    @staticmethod
    def _find_preferred_windows_device(debug=False):
        """
        Find preferred input device for Windows that provides proper timing information
        
        Windows環境で適切なタイミング情報を提供する優先入力デバイスを探す
        
        Parameters:
        -----------
        debug : bool, optional
            Enable debug messages (default: False)
            
        Returns:
        --------
        int or None
            Device index if found, None otherwise
        """
        def _debug_print_local(message):
            if debug:
                print(message)
                
        try:
            import platform
            if platform.system() != 'Windows':
                return None
                
            devices = sd.query_devices()
            host_apis = sd.query_hostapis()
            
            # Get default input device
            try:
                default_device = sd.query_devices(kind='input')
                default_index = sd.default.device[0]
                default_device_name = default_device['name'].lower()
                _debug_print_local(f"Default input device: {default_device['name']} (Index: {default_index})")
            except Exception as e:
                _debug_print_local(f"Could not get default input device: {e}")
                return None
            
            # Find WASAPI devices that match or are similar to the default device
            wasapi_candidates = []
            all_input_devices = []
            
            for idx, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    all_input_devices.append((idx, dev))
                    host_api_name = host_apis[dev['hostapi']]['name']
                    
                    if host_api_name == 'Windows WASAPI':
                        dev_name_lower = dev['name'].lower()
                        
                        # Calculate similarity score with default device
                        score = 0
                        
                        # Exact match gets highest score
                        if dev_name_lower == default_device_name:
                            score = 100
                        # Partial match
                        elif default_device_name in dev_name_lower or dev_name_lower in default_device_name:
                            score = 80
                        # Word-based matching
                        else:
                            default_words = set(default_device_name.split())
                            dev_words = set(dev_name_lower.split())
                            common_words = default_words.intersection(dev_words)
                            score = len(common_words) * 20
                        
                        # Prefer devices without certain keywords that might indicate loopback
                        if not any(keyword in dev_name_lower for keyword in ['loopback', 'mix', 'ミキサー', 'stereo mix']):
                            score += 10
                            
                        # Prefer devices with 'microphone' or 'mic' in the name
                        if any(keyword in dev_name_lower for keyword in ['microphone', 'mic', 'マイク']):
                            score += 5
                        
                        wasapi_candidates.append((idx, dev, score))
                        _debug_print_local(f"WASAPI device: {dev['name']} (Score: {score})")
            
            # Sort candidates by score (highest first)
            wasapi_candidates.sort(key=lambda x: x[2], reverse=True)
            
            if wasapi_candidates:
                best_device = wasapi_candidates[0]
                _debug_print_local(f"Selected WASAPI device: {best_device[1]['name']} (Score: {best_device[2]})")
                return best_device[0]
            else:
                _debug_print_local("No suitable WASAPI devices found")
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
