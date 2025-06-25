import abc

class OutputCapture(abc.ABC):
    """
    Output audio capture abstract base class

    This class defines a common interface independent of specific platforms
    (MacOS, Windows, etc.). It is used for audio capture from output devices
    such as speakers.
    """

    def __init__(self, sample_rate=16000, channels=2, blocksize=1024, debug=False):
        """
        Initialize output capture

        Parameters:
        -----------
        sample_rate : int, optional
            Sampling rate (Hz) (default: 16000Hz)
        channels : int, optional
            Number of channels (default: 2 channels (stereo))
        blocksize : int, optional
            Block size (number of frames) (default: 1024)
        debug : bool, optional
            Enable debug messages (default: False)
        """
        self._sample_rate = sample_rate
        self._channels = channels
        self._blocksize = blocksize
        self.debug = debug

    def _debug_print(self, message):
        """
        Print debug message if debug mode is enabled
        """
        if self.debug:
            print(message)

    @property
    def sample_rate(self):
        """Sampling rate (Hz)"""
        return self._sample_rate

    @property
    def channels(self):
        """Number of channels"""
        return self._channels

    @property
    def blocksize(self):
        """Block size (number of frames)"""
        return self._blocksize

    @abc.abstractmethod
    def start_audio_capture(self, device_name=None, sample_rate=None, channels=None, blocksize=None):
        """
        Start audio capture

        Parameters:
        -----------
        device_name : str, optional
            Name of the audio device to use
        sample_rate : int, optional
            Sampling rate (Hz)
        channels : int, optional
            Number of channels
        blocksize : int, optional
            Block size (number of frames)

        Returns:
        --------
        bool
            True if capture started successfully, False otherwise
        """
        pass

    @abc.abstractmethod
    def read_audio_capture(self):
        """
        Read captured audio data

        Returns:
        --------
        AudioData
            Object containing the captured audio data

        Raises:
        -------
        RuntimeError
            If the stream is not initialized or not functioning properly
        """
        pass

    @abc.abstractmethod
    def stop_audio_capture(self):
        """Stop audio capture"""
        pass
