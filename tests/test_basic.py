"""
Basic tests for fujielab-audio-mcnr_input package
"""
import pytest
import numpy as np
from fujielab.audio.mcnr_input.core import InputStream, CaptureConfig
from fujielab.audio.mcnr_input._backend.data import AudioData


class TestCaptureConfig:
    """Test CaptureConfig class"""

    def test_capture_config_creation(self):
        """Test basic CaptureConfig creation"""
        config = CaptureConfig(capture_type="Input")
        assert config.capture_type == "Input"
        assert config.channels == 2  # default
        assert config.offset == 0.0  # default
        assert config.device_name is None  # default

    def test_capture_config_with_parameters(self):
        """Test CaptureConfig with custom parameters"""
        config = CaptureConfig(
            capture_type="Output",
            device_name="Test Device",
            channels=1,
            offset=0.05,
            extra_settings={"test": "value"}
        )
        assert config.capture_type == "Output"
        assert config.device_name == "Test Device"
        assert config.channels == 1
        assert config.offset == 0.05
        assert config.extra_settings == {"test": "value"}

    def test_invalid_capture_type(self):
        """Test that invalid capture types are handled gracefully"""
        # The class doesn't validate capture_type, so this should work
        # but might fail during actual stream creation
        config = CaptureConfig(capture_type="Invalid")
        assert config.capture_type == "Invalid"


class TestAudioData:
    """Test AudioData class"""

    def test_audio_data_creation(self):
        """Test AudioData creation"""
        data = np.random.random((1024, 2)).astype(np.float32)
        timestamp = 123.456
        audio_data = AudioData(data=data, time=timestamp, overflowed=False)
        
        assert np.array_equal(audio_data.data, data)
        assert audio_data.time == timestamp
        assert audio_data.overflowed is False

    def test_audio_data_with_overflow(self):
        """Test AudioData with overflow flag"""
        data = np.zeros((512, 1), dtype=np.float32)
        audio_data = AudioData(data=data, time=0.0, overflowed=True)
        
        assert audio_data.overflowed is True
        assert audio_data.data.shape == (512, 1)


class TestInputStream:
    """Test InputStream class (basic functionality only)"""

    def test_input_stream_creation(self):
        """Test basic InputStream creation without starting"""
        # Test with minimal configuration
        stream = InputStream(
            samplerate=16000,
            blocksize=512,
            debug=False
        )
        
        assert stream.samplerate == 16000
        assert stream.blocksize == 512
        assert stream.debug is False

    def test_input_stream_with_captures(self):
        """Test InputStream creation with capture configs"""
        captures = [
            CaptureConfig(capture_type="Input", channels=1),
            CaptureConfig(capture_type="Output", channels=2)
        ]
        
        stream = InputStream(
            samplerate=44100,
            blocksize=1024,
            captures=captures,
            debug=True
        )
        
        assert stream.samplerate == 44100
        assert stream.blocksize == 1024
        assert len(stream.captures) == 2
        assert stream.debug is True

    def test_input_stream_callback_signature(self):
        """Test that callback function can be set"""
        def test_callback(data, frames, timestamp, flags):
            pass
        
        stream = InputStream(
            callback=test_callback,
            debug=False
        )
        
        assert stream.callback is test_callback

    @pytest.mark.skip(reason="Requires audio hardware and system setup")
    def test_input_stream_start_stop(self):
        """Test stream start/stop (skipped in CI)"""
        # This test would require proper audio hardware setup
        # and is skipped in automated testing
        pass


class TestPlatformDetection:
    """Test platform-specific imports"""

    def test_output_capture_import(self):
        """Test that OutputCapture imports correctly based on platform"""
        import platform
        
        if platform.system() == "Darwin":
            from fujielab.audio.mcnr_input._backend.output_capture_mac import OutputCaptureMac
            # Test that the class exists
            assert OutputCaptureMac is not None
        else:
            from fujielab.audio.mcnr_input._backend.output_capture_win import OutputCaptureWin
            # Test that the class exists
            assert OutputCaptureWin is not None


if __name__ == "__main__":
    pytest.main([__file__])
