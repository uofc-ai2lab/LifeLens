import pyaudio
import wave
from config.audio_settings import AUDIO_DIR

OUTPUT = AUDIO_DIR / "test1.wav"

def record_audio(duration=5, output_filename=OUTPUT, rate=16000, channels=6, width=2, index=2, chunk=1024):
    """
    Records audio from the ReSpeaker microphone and saves it to a WAV file.

    Parameters:
    - duration: Recording duration in seconds (default: 5)
    - output_filename: Output WAV file name (default: "output.wav")
    - rate: Sample rate (default: 16000)
    - channels: Number of channels (default: 6)
    - width: Sample width in bytes (default: 2)
    - index: Input device index (default: 2)
    - chunk: Chunk size for reading (default: 1024)
    """
    p = pyaudio.PyAudio()

    stream = p.open(
        rate=rate,
        format=p.get_format_from_width(width),
        channels=channels,
        input=True,
        input_device_index=index,
    )

    print("* recording")

    frames = []

    for i in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(p.get_format_from_width(width)))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    return output_filename

if __name__ == "__main__":
    # Default behavior when run as a script
    record_audio()