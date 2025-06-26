"""
Audio Input Capture Module for Mac

This module provides classes for capturing audio from various input sources (e.g., microphones) on Mac using soundcard.

Mac用入力オーディオキャプチャモジュール

このモジュールは、soundcardを使用してMac上で各種オーディオ入力（マイク等）からのオーディオをキャプチャするためのクラスを提供します。
"""
from .input_capture_soundcard_base import SoundcardInputCaptureBase


class InputCaptureMac(SoundcardInputCaptureBase):
    """
    Audio Input Capture Class for Mac

    This class provides an interface for capturing audio from input sources on Mac using soundcard library.

    Mac用入力オーディオキャプチャクラス

    このクラスは、soundcardライブラリを使用してMac上でオーディオ入力からのキャプチャ機能を提供します。
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
        super().__init__(sample_rate, channels, blocksize, debug, platform_name="Mac")

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
        return SoundcardInputCaptureBase.list_audio_devices_common(debug)


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
