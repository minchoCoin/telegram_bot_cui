import time
import threading
from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd
from tensorflow.lite.python.interpreter import Interpreter

from config import (
    CHANNELS,
    DTYPE,
    WAKEWORD_TFLITE,
    WAKE_BLOCK_SECONDS,
    WAKE_COOLDOWN_SECONDS,
    WAKE_FMAX,
    WAKE_FMIN,
    WAKE_HOP_LENGTH,
    WAKE_MIN_RMS,
    WAKE_N_FFT,
    WAKE_N_MELS,
    WAKE_SAMPLE_RATE,
    WAKE_WIN_LENGTH,
    WAKEWORD_THRESHOLD,
)


class WakewordDetector:
    def __init__(self, model_path: str = WAKEWORD_TFLITE):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Wakeword TFLite file not found: {model_path}")

        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()

        self.input_info = self.interpreter.get_input_details()[0]
        self.output_info = self.interpreter.get_output_details()[0]

        self.input_shape = list(self.input_info["shape"])
        self.input_dtype = self.input_info["dtype"]
        self.last_detect_time = 0.0

        print("[Wakeword] model input shape:", self.input_shape)
        print("[Wakeword] model input dtype:", self.input_dtype)
        print("[Wakeword] model output shape:", self.output_info["shape"])

    def compute_rms(self, audio_int16: np.ndarray) -> float:
        audio = audio_int16.astype(np.float32).reshape(-1)
        return float(np.sqrt(np.mean(audio ** 2)))

    def extract_log_mel(self, audio_int16: np.ndarray) -> np.ndarray:
        audio = audio_int16.astype(np.float32).reshape(-1) / 32768.0

        log_mel = librosa.feature.melspectrogram(
            y=audio,
            sr=WAKE_SAMPLE_RATE,
            n_fft=WAKE_N_FFT,
            hop_length=WAKE_HOP_LENGTH,
            win_length=WAKE_WIN_LENGTH,
            n_mels=WAKE_N_MELS,
            fmin=WAKE_FMIN,
            fmax=WAKE_FMAX,
            power=2.0,
            center=False,
        )
        log_mel = librosa.power_to_db(log_mel, ref=np.max).T

        mean = log_mel.mean()
        std = log_mel.std() + 1e-6
        log_mel = (log_mel - mean) / std
        return log_mel.astype(np.float32)

    def preprocess_audio(self, audio_int16: np.ndarray) -> np.ndarray:
        features = self.extract_log_mel(audio_int16)
        shape = [d for d in self.input_shape if d > 0]

        if len(shape) == 2:
            expected_t, expected_f = shape[0], shape[1]
        elif len(shape) == 3:
            expected_t, expected_f = shape[1], shape[2]
        elif len(shape) == 4:
            expected_t, expected_f = shape[1], shape[2]
        else:
            raise ValueError(f"Unsupported wakeword input shape: {self.input_shape}")

        if features.shape[1] > expected_f:
            features = features[:, :expected_f]
        elif features.shape[1] < expected_f:
            pad_f = np.zeros((features.shape[0], expected_f - features.shape[1]), dtype=np.float32)
            features = np.concatenate([features, pad_f], axis=1)

        if features.shape[0] > expected_t:
            features = features[:expected_t, :]
        elif features.shape[0] < expected_t:
            pad_t = np.zeros((expected_t - features.shape[0], expected_f), dtype=np.float32)
            features = np.concatenate([features, pad_t], axis=0)

        if len(self.input_shape) == 2:
            x = features
        elif len(self.input_shape) == 3:
            x = features[np.newaxis, :, :]
        elif len(self.input_shape) == 4:
            x = features[np.newaxis, :, :, np.newaxis]
        else:
            raise ValueError(f"Unsupported wakeword input shape: {self.input_shape}")

        return x.astype(self.input_dtype)

    def predict(self, audio_int16: np.ndarray):
        x = self.preprocess_audio(audio_int16)

        self.interpreter.set_tensor(self.input_info["index"], x)
        self.interpreter.invoke()

        y = np.array(self.interpreter.get_tensor(self.output_info["index"])).reshape(-1)

        if len(y) == 1:
            score = float(y[0])
            pred = int(score >= WAKEWORD_THRESHOLD)
            return pred, score

        pred = int(np.argmax(y))
        score = float(y[1])
        return pred, score

    def listen_once(self):
        frames = int(WAKE_SAMPLE_RATE * WAKE_BLOCK_SECONDS)

        audio = sd.rec(
            frames,
            samplerate=WAKE_SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocking=True,
        )

        mono = audio.reshape(-1)
        rms = self.compute_rms(mono)

        if rms < WAKE_MIN_RMS:
            return False

        pred, score = self.predict(mono)
        print(f"[Wakeword] rms={rms:.1f}, pred={pred}, marvin_prob={score:.4f}")

        now = time.time()
        if pred == 1 and (now - self.last_detect_time) >= WAKE_COOLDOWN_SECONDS:
            self.last_detect_time = now
            return True

        return False

    def wait_for_wakeword(self):
        print("Waiting for wakeword...")
        while True:
            if self.listen_once():
                print("Wakeword detected. Starting recording.")
                return True