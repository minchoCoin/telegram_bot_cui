import asyncio

from config import RECORD_WAV, TTS_MP3, TTS_WAV
from audio_recorder import record_until_silence
from stt_whisper import transcribe_with_whisper_cpp
from telegram_bridge import send_to_telegram_and_get_reply
from tts_player import synthesize_to_wav, play_wav

from wakeword_detector import WakewordDetector
async def async_main():
    detector = WakewordDetector()

    while True:
        print("\n=== 기동어 대기 ===")
        detector.wait_for_wakeword()

        record_until_silence(RECORD_WAV)

        user_text = transcribe_with_whisper_cpp(RECORD_WAV)
        print("[STT 결과]", user_text)
        if user_text=="":
            continue
        if user_text.lower() in ["exit", "quit", "종료"]:
            print("프로그램 종료")
            break

        reply_text = await send_to_telegram_and_get_reply(user_text)
        print("[텔레그램 답변]", reply_text)

        await synthesize_to_wav(reply_text, TTS_MP3, TTS_WAV)
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