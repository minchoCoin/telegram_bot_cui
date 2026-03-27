import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import librosa

from config import (
    SAMPLE_RATE,
    CHANNELS,
    DTYPE,
    WAKEWORD_TFLITE,
    WAKE_BLOCK_SECONDS,
    WAKE_THRESHOLD,
    WAKE_COOLDOWN_SECONDS,
    WAKE_MIN_RMS,
    MFCC_N_MFCC,
    MFCC_N_FFT,
    MFCC_HOP_LENGTH,
    MFCC_WIN_LENGTH,
)

from tensorflow.lite.python.interpreter import Interpreter


class WakewordDetector:
    def __init__(self, model_path: str = WAKEWORD_TFLITE):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"TFLite model not found: {model_path}")

        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_info = self.input_details[0]
        self.output_info = self.output_details[0]

        self.input_shape = list(self.input_info["shape"])
        self.input_dtype = self.input_info["dtype"]

        self.last_detect_time = 0.0

        print("[Wakeword] model input shape:", self.input_shape)
        print("[Wakeword] model input dtype:", self.input_dtype)
        print("[Wakeword] model output shape:", self.output_info["shape"])

    def compute_rms(self, audio_int16: np.ndarray) -> float:
        audio = audio_int16.astype(np.float32).reshape(-1)
        return float(np.sqrt(np.mean(audio ** 2)))
    def _expected_mfcc_shape(self):
        """
        shape:
        [1, T, F]
        [1, T, F, 1]
        [T, F]
        [1, F, T]  <- 이런 경우는 직접 수정 필요할 수 있음
        """
        shape = [d for d in self.input_shape if d > 0]

        if len(shape) == 2:
            return shape[0], shape[1]   # T, F
        if len(shape) == 3:
            # [1, T, F] 라고 가정
            return shape[1], shape[2]
        if len(shape) == 4:
            # [1, T, F, 1] 라고 가정
            return shape[1], shape[2]

        raise ValueError(f"지원하지 않는 MFCC 입력 shape: {self.input_shape}")

    def extract_mfcc(self, audio_int16: np.ndarray) -> np.ndarray:
        audio = audio_int16.astype(np.float32).reshape(-1) / 32768.0

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=SAMPLE_RATE,
            n_mfcc=MFCC_N_MFCC,
            n_fft=MFCC_N_FFT,
            hop_length=MFCC_HOP_LENGTH,
            win_length=MFCC_WIN_LENGTH,
        ).T  # (frames, n_mfcc)

        # 간단 정규화
        mfcc = (mfcc - np.mean(mfcc, axis=0, keepdims=True)) / (
            np.std(mfcc, axis=0, keepdims=True) + 1e-6
        )
        return mfcc.astype(np.float32)

    def preprocess_audio(self, audio_int16: np.ndarray) -> np.ndarray:
        mfcc = self.extract_mfcc(audio_int16)  # (T, F)

        expected_t, expected_f = self._expected_mfcc_shape()

        if mfcc.shape[1] != expected_f:
            if mfcc.shape[1] > expected_f:
                mfcc = mfcc[:, :expected_f]
            else:
                pad_f = np.zeros((mfcc.shape[0], expected_f - mfcc.shape[1]), dtype=np.float32)
                mfcc = np.concatenate([mfcc, pad_f], axis=1)

        if mfcc.shape[0] < expected_t:
            pad_t = np.zeros((expected_t - mfcc.shape[0], expected_f), dtype=np.float32)
            mfcc = np.concatenate([mfcc, pad_t], axis=0)
        else:
            mfcc = mfcc[:expected_t, :]

        shape = self.input_shape

        if len(shape) == 2:
            x = mfcc
        elif len(shape) == 3:
            x = mfcc[np.newaxis, :, :]          # [1, T, F]
        elif len(shape) == 4:
            x = mfcc[np.newaxis, :, :, np.newaxis]  # [1, T, F, 1]
        else:
            raise ValueError(f"지원하지 않는 입력 shape: {shape}")

        return x.astype(self.input_dtype)

    def predict(self, audio_int16: np.ndarray):
        x = self.preprocess_audio(audio_int16)

        self.interpreter.set_tensor(self.input_info["index"], x)
        self.interpreter.invoke()

        y = self.interpreter.get_tensor(self.output_info["index"])
        y = np.array(y).reshape(-1)

        # sigmoid 출력 [p]
        if len(y) == 1:
            score = float(y[0])
            pred = 1 if score >= WAKE_THRESHOLD else 0
            return pred, score

        # softmax 출력 [p0, p1]
        pred = int(np.argmax(y))
        score = float(y[pred])
        return pred, score

    def wait_for_wakeword(self):
        frames = int(SAMPLE_RATE * WAKE_BLOCK_SECONDS)

        print("Waiting for wakeword...")

        while True:
            audio = sd.rec(
                frames,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocking=True,
            )

            mono = audio.reshape(-1)

            rms = self.compute_rms(mono)

            # 너무 작은 소리는 wakeword 추론 스킵
            if rms < WAKE_MIN_RMS:
                print(f"[Wakeword] skipped (rms={rms:.1f})")
                continue

            pred, score = self.predict(mono)
            print(f"[Wakeword] rms={rms:.1f}, pred={pred}, score={score:.4f}")

            now = time.time()
            if pred == 1 and (now - self.last_detect_time) >= WAKE_COOLDOWN_SECONDS:
                self.last_detect_time = now
                print("Wakeword detected. Starting main recording.")
                return True