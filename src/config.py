import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Telegram
API_ID = int(os.getenv("api_id", "0"))
API_HASH = os.getenv("api_hash", "")
BOT_USERNAME = os.getenv("bot_username", "")

# whisper.cpp
WHISPER_MAIN = "./whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "./whisper.cpp/models/ggml-small.bin"

# Audio files
#BASE_DIR = Path(__file__).resolve().parent
BASE_DIR=Path('.')
RECORD_WAV = str(BASE_DIR / "input_record.wav")
TTS_MP3 = str(BASE_DIR / "reply.mp3")
TTS_WAV = str(BASE_DIR / "reply.wav")

# Recording settings
SAMPLE_RATE = 48000
CHANNELS = 1
DTYPE = "int16"
BLOCK_DURATION = 0.1
SILENCE_SECONDS = 3.0
RMS_THRESHOLD = 700  # 환경에 따라 조절

# Telegram wait settings
TELEGRAM_IDLE_SECONDS = 5.0
TELEGRAM_TIMEOUT = 120.0

# Edge TTS voices
VOICE_KO = "ko-KR-SunHiNeural"
VOICE_EN = "en-US-JennyNeural"

# wakeword tflite
WAKEWORD_TFLITE = "./models/wakeword.tflite"

# Wakeword settings
WAKE_BLOCK_SECONDS = 1.0      # wakeword detection input length (seconds)
WAKE_THRESHOLD = 0.5          # threshold for sigmoid output of 1
WAKE_COOLDOWN_SECONDS = 1.0   # cooldown period to prevent consecutive detections

# MFCC settings
MFCC_N_MFCC = 13
MFCC_N_FFT = 512
MFCC_HOP_LENGTH = 160         # 10ms @ 16kHz
MFCC_WIN_LENGTH = 400         # 25ms @ 16kHz

# Wakeword RMS skip threshold
WAKE_MIN_RMS = 300