from typing import Tuple, Optional
import numpy as np
from abc import ABC, abstractmethod


class Processor(ABC):
    """
    Abstract base class for audio data processors.

    This class defines a common interface for processing multi-channel audio data.
    Subclasses should implement the `process` method to define specific processing logic.
    """

    @abstractmethod
    def process(self, input_chunk: np.ndarray) -> np.ndarray:
        """
        Process the input audio data chunk.

        Parameters:
            input_chunk : np.ndarray
                Input audio data to be processed. Shape is (blocksize, total_channels).

        Returns:
            np.ndarray
                Processed audio data. Shape is the same as input.
        """
        pass


class NullProcessor(Processor):
    """
    A dummy processor that does not perform any processing.

    This class is used when no specific processing is required.
    It simply returns the input data as is.
    """

    def process(self, input_chunk: np.ndarray) -> np.ndarray:
        """
        Process function that returns the input data unchanged.

        Parameters:
            input_chunk : np.ndarray
                Input audio data to be processed. Shape is (blocksize, total_channels).

        Returns:
            np.ndarray
                The same input audio data without any changes.
        """
        return input_chunk.copy()


class MultiChannelNoiseReductionProcessor(Processor):
    """
    Processor for multi-channel noise reduction.

    This class processes multi-channel audio data from multiple capture sources (microphones, speakers, etc.)
    and returns the output. It is a simple dummy implementation that returns the input data as is.
    """

    def __init__(self):
        """
        Initialize the multi-channel noise reduction processor.

        This constructor can be extended to include any initialization logic if needed.
        """
        super().__init__()

        self.input_audio_buffer = None  # Placeholder for audio buffer if needed in future
        self.output_audio_buffer = None  # Placeholder for output audio buffer if needed in future
        self.window_size = 512
        self.hop_size = self.window_size // 4  # 75% overlap
        self.chunk_size = None

    def process(self, input_chunk: np.ndarray) -> np.ndarray:
        """
        Process the input audio data chunk.
        """
        if self.input_audio_buffer is None:
            # Initialize the audio buffer if it is not already set
            self.chunk_size = input_chunk.shape[0]
            self.input_audio_buffer = input_chunk.copy()

        self.input_audio_buffer = np.vstack((self.input_audio_buffer, input_chunk.copy()))

        # Precompute normalization factor for overlap-add
        normalization_window = np.hanning(self.window_size)
        normalization_factor = np.sum(normalization_window ** 2) / self.hop_size

        while self.input_audio_buffer.shape[0] > self.window_size:
            chunk_to_process = self.input_audio_buffer[:self.window_size]
            self.input_audio_buffer = self.input_audio_buffer[self.hop_size:]

            # Apply windowing
            window = np.hanning(self.window_size).reshape(-1, 1)
            chunk_to_process = chunk_to_process * window

            # STFT
            stft_result = np.fft.rfft(chunk_to_process, axis=0)

            # Power spectrum (dummy processing for now)
            power_spectrum = np.abs(stft_result) ** 2

            # Maximum power channel index for each band (dummy processing for now)
            max_power_channel = np.argmax(power_spectrum, axis=1)

            # Create a mask to keep only the maximum power channel (dummy processing for now)
            mask = np.zeros_like(stft_result, dtype=bool)
            mask[np.arange(stft_result.shape[0]), max_power_channel] = True

            # Apply the mask to the STFT result (commented out for now)
            stft_result[~mask] = 0

            # ISTFT
            chunk_to_process = np.fft.irfft(stft_result, axis=0)
            chunk_to_process = chunk_to_process * window

            # Overlap-add with normalization
            if self.output_audio_buffer is None:
                self.output_audio_buffer = chunk_to_process / normalization_factor
            else:
                required_size = self.window_size - self.hop_size
                if self.output_audio_buffer.shape[0] < required_size:
                    # Pad the output buffer if it's too small
                    padding = np.zeros((required_size - self.output_audio_buffer.shape[0], self.output_audio_buffer.shape[1]))
                    self.output_audio_buffer = np.vstack((self.output_audio_buffer, padding))

                overlap_start = self.output_audio_buffer.shape[0] - required_size
                overlap_end = self.output_audio_buffer.shape[0]
                self.output_audio_buffer[overlap_start:overlap_end] += chunk_to_process[:required_size] / normalization_factor
                self.output_audio_buffer = np.vstack([self.output_audio_buffer, chunk_to_process[required_size:] / normalization_factor])

        if self.output_audio_buffer is None or len(self.output_audio_buffer) < self.chunk_size + (self.window_size - self.hop_size):
            # Ensure the output audio buffer has enough data to return
            return np.zeros_like(input_chunk)
        else:
            # Return the output chunk of the specified size
            output_chunk = self.output_audio_buffer[:self.chunk_size].copy()
            self.output_audio_buffer = self.output_audio_buffer[self.chunk_size:]

            return output_chunk
