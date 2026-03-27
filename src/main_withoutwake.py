import asyncio

from config import RECORD_WAV, TTS_MP3, TTS_WAV
from audio_recorder import record_until_silence
from stt_whisper import transcribe_with_whisper_cpp
from telegram_bridge import send_to_telegram_and_get_reply
from tts_player import synthesize_to_wav, play_wav
from wakeword_detector import WakewordDetector

async def async_main():
    #detector = WakewordDetector()

    # 0. 기동어 대기
    #detector.wait_for_wakeword()

    # 1. 녹음
    record_until_silence(RECORD_WAV)

    # 2. STT
    user_text = transcribe_with_whisper_cpp(RECORD_WAV)
    print("[STT result]", user_text)

    # 3. Telegram 전송 및 답변 수신
    reply_text = await send_to_telegram_and_get_reply(user_text)
    print("[Telegram reply]", reply_text)

    # 4. TTS -> WAV
    await synthesize_to_wav(reply_text, TTS_MP3, TTS_WAV)

    # 5. 재생
    play_wav(TTS_WAV)


def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nprogram interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()