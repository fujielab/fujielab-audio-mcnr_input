import numpy as np
from dataclasses import dataclass

@dataclass
class AudioData:
    """
    A class to represent audio data.

    Attributes:
        data (np.ndarray): The audio data as a NumPy array.
        time (float): The timestamp when the data was captured.
        overflowed (bool): Flag indicating if the data capture overflowed.
    """
    data: np.ndarray
    time: float
    overflowed: bool = False

