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
WHISPER_MODEL = "./whisper.cpp/models/ggml-base.bin"

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
VOICE_JP = "ja-JP-NanamiNeural"
VOICE_ZH = "zh-CN-XiaoxiaoNeural"

# wakeword tflite
WAKEWORD_TFLITE = "./kws2_float32.tflite"

# Wakeword settings
WAKE_BLOCK_SECONDS = 1.0      # wakeword detection input length (seconds)
WAKE_COOLDOWN_SECONDS = 1.0   # cooldown period to prevent consecutive detections

# Wakeword audio / log-mel settings
WAKE_SAMPLE_RATE = 16000
WAKE_N_MELS = 64
WAKE_N_FFT = 512
WAKE_HOP_LENGTH = 160         # 10ms @ 16kHz
WAKE_WIN_LENGTH = 400         # 25ms @ 16kHz
WAKE_FMIN = 20
WAKE_FMAX = WAKE_SAMPLE_RATE // 2

# Wakeword RMS skip threshold
WAKE_MIN_RMS = 300
