import queue
import wave
import numpy as np
import sounddevice as sd

from config import (
    SAMPLE_RATE,
    CHANNELS,
    DTYPE,
    BLOCK_DURATION,
    SILENCE_SECONDS,
    RMS_THRESHOLD,
)


def record_until_silence(output_wav: str) -> str:
    """
After the user starts speaking,
If it is silent for 3 seconds in a row, end the recording and save it with WAV.
    """
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    recorded_frames = []

    speech_started = False
    silence_duration = 0.0
    block_size = int(SAMPLE_RATE * BLOCK_DURATION)

    def callback(indata, frames, time_info, status):
        if status:
            print("[Audio status]", status)
        audio_queue.put(indata.copy())

    print("Waiting to record... Start speaking.")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=block_size,
        callback=callback,
    ):
        while True:
            chunk = audio_queue.get()
            recorded_frames.append(chunk)

            rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))

            if rms >= RMS_THRESHOLD:
                if not speech_started:
                    print("Voice detected. Starting recording.")
                speech_started = True
                silence_duration = 0.0
            else:
                if speech_started:
                    silence_duration += BLOCK_DURATION
                    if silence_duration >= SILENCE_SECONDS:
                        print(f"{SILENCE_SECONDS:.1f} seconds of silence detected. Stopping recording.")
                        break

    audio = np.concatenate(recorded_frames, axis=0)

    with wave.open(output_wav, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())

    print(f"WAV saved successfully: {output_wav}")
    return output_wav