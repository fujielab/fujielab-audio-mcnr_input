"""
Abstract Base Class for Audio Input Capture

This module provides the abstract base class for capturing audio from various input sources.

入力オーディオキャプチャ抽象基底クラス

このモジュールは、各種オーディオ入力からのキャプチャのための抽象基底クラスを提供します。
"""
from abc import ABC, abstractmethod
from typing import Optional, Union
import queue
from .data import AudioData


class InputCaptureBase(ABC):
    """
    Abstract Base Class for Audio Input Capture

    This abstract class defines the common interface for all audio input capture implementations.

    入力オーディオキャプチャ抽象基底クラス

    この抽象クラスは、すべてのオーディオ入力キャプチャ実装の共通インターフェースを定義します。
    """

    def __init__(self, sample_rate: int = 16000, channels: int = 1, blocksize: int = 1024, debug: bool = False):
        """
        Initialize audio input capture

        オーディオ入力キャプチャの初期化

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

        # Common instance variables
        self._audio_queue = queue.Queue(maxsize=20)
        self._stream_initialized = False
        self._error_count = 0

    def _debug_print(self, message: str) -> None:
        """
        Print debug message if debug mode is enabled
        デバッグモードが有効な場合にデバッグメッセージを出力

        Parameters:
        -----------
        message : str
            Debug message to print
            出力するデバッグメッセージ
        """
        if self.debug:
            print(message)

    @property
    def sample_rate(self) -> int:
        """
        Get sampling rate (Hz)
        サンプリングレート（Hz）を取得

        Returns:
        --------
        int
            Sampling rate in Hz
            サンプリングレート（Hz）
        """
        return self._sample_rate

    @property
    def channels(self) -> int:
        """
        Get number of channels
        チャネル数を取得

        Returns:
        --------
        int
            Number of channels
            チャネル数
        """
        return self._channels

    @property
    def blocksize(self) -> int:
        """
        Get block size (number of frames)
        ブロックサイズ（フレーム数）を取得

        Returns:
        --------
        int
            Block size in frames
            ブロックサイズ（フレーム数）
        """
        return self._blocksize

    @property
    @abstractmethod
    def time(self) -> float:
        """
        Get current time
        現在時刻を取得

        Returns:
        --------
        float
            Current time in seconds
            現在時刻（秒）
        """
        pass

    @staticmethod
    @abstractmethod
    def list_audio_devices(debug: bool = False) -> None:
        """
        List available audio input devices
        利用可能なオーディオ入力デバイスを一覧表示

        Parameters:
        -----------
        debug : bool, optional
            Enable debug output (default: False)
            デバッグ出力を有効にする（デフォルト: False）
        """
        pass

    @abstractmethod
    def start_audio_capture(self, device_name: Optional[Union[str, int]] = None, 
                          sample_rate: Optional[int] = None, 
                          channels: Optional[int] = None, 
                          blocksize: Optional[int] = None) -> bool:
        """
        Start audio capture
        オーディオキャプチャを開始

        Parameters:
        -----------
        device_name : str or int, optional
            Device name or index (default: None, auto-select)
            デバイス名またはインデックス（デフォルト: None、自動選択）
        sample_rate : int, optional
            Sampling rate (Hz) (default: None, use initialization value)
            サンプリングレート（Hz）（デフォルト: None、初期化値を使用）
        channels : int, optional
            Number of channels (default: None, use initialization value)
            チャネル数（デフォルト: None、初期化値を使用）
        blocksize : int, optional
            Block size (default: None, use initialization value)
            ブロックサイズ（デフォルト: None、初期化値を使用）

        Returns:
        --------
        bool
            True if capture started successfully, False otherwise
            キャプチャが正常に開始された場合True、そうでなければFalse
        """
        pass

    @abstractmethod
    def read_audio_capture(self) -> Optional[AudioData]:
        """
        Read audio data from capture
        キャプチャからオーディオデータを読み取り

        Returns:
        --------
        AudioData or None
            Audio data if available, None otherwise
            利用可能な場合はオーディオデータ、そうでなければNone
        """
        pass

    @abstractmethod
    def stop_audio_capture(self) -> None:
        """
        Stop audio capture
        オーディオキャプチャを停止
        """
        pass
