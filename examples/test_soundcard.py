import soundcard as sc

mic = sc.all_microphones(include_loopback=True)[3]
print(f"Microphone: {mic.name}")

with mic.recorder(samplerate=16000) as mic_recorder:
    print("Recording...")
    audio = mic_recorder.record(numframes=16000 * 10)
    print("Recording complete.")

import soundfile as sf
sf.write("test_soundcard.wav", audio, 16000)