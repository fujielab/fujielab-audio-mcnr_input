from fujielab.audio.mcnr_input.core import InputStream, CaptureConfig
import numpy as np
import time
import soundfile as sf

capture_configs = [
    CaptureConfig(capture_type="Input", channels=1),
    CaptureConfig(capture_type="Output", channels=2),
]

audio_list = []
def callback(data, frames, timestamp, flags):
    audio_list.append(data)

input_stream = InputStream(debug=True, captures=capture_configs, callback=callback)
input_stream.start()
time.sleep(5)
input_stream.stop()

if audio_list:
    audio_data = np.concatenate(audio_list, axis=0)
    print(f"Captured {len(audio_data)} samples from {len(capture_configs)} captures")
    sf.write("test_output.wav", audio_data, input_stream.sample_rate)
