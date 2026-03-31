import asyncio
import time

from gpiozero import LED

from config import RECORD_WAV, TTS_MP3, TTS_WAV
from audio_recorder import record_until_silence
from stt_whisper import transcribe_with_whisper_cpp
from telegram_bridge import send_to_telegram_and_get_reply
from tts_player import synthesize_to_wav, AudioPlayer
from wakeword_detector import WakewordDetector


led = LED(17)  # GPIO17


async def async_main():
    detector = WakewordDetector()
    player = AudioPlayer()

    # True면 다음 루프에서 wait_for_wakeword()를 건너뛰고
    # 바로 record_until_silence()로 들어감
    skip_wakeword = False

    while True:
        try:
            # 1) wakeword 대기
            if not skip_wakeword:
                led.off()
                print("\n=== wakeword waiting ===")
                detector.wait_for_wakeword()
            else:
                print("\n=== wakeword already captured during TTS -> start recording immediately ===")

            skip_wakeword = False

            # 2) 사용자 발화 녹음
            led.on()
            print("[Main] recording user speech...")
            record_until_silence(RECORD_WAV)

            # 3) STT
            user_text = transcribe_with_whisper_cpp(RECORD_WAV)
            print("[STT result]", user_text)

            if user_text == "":
                print("[Main] empty STT result")
                continue

            if user_text.lower() in ["exit", "quit", "종료"]:
                print("Program terminated")
                break

            # 4) 응답 생성
            reply_text = await send_to_telegram_and_get_reply(user_text)
            print("[Telegram reply]", reply_text)

            if not reply_text or not reply_text.strip():
                print("[Main] empty reply text")
                continue

            # 5) TTS 생성
            await synthesize_to_wav(reply_text, TTS_MP3, TTS_WAV)

            # 6) TTS 재생
            print("[Main] playing TTS...")
            player.play(TTS_WAV)

            # 7) 재생 중 wakeword 감시
            #    별도 watcher thread 없이 메인 루프에서만 감시
            interrupted = False

            while player.is_playing():
                detected = detector.listen_once()
                if detected:
                    print("[Main] wakeword detected during TTS -> stop playback")
                    player.stop()
                    interrupted = True
                    break

            # 재생 프로세스 정리
            player.wait()

            # 8) 인터럽트되었으면 wakeword를 다시 요구하지 않고 즉시 녹음 단계로
            if interrupted:
                print("[Main] TTS interrupted -> next loop skips wakeword wait")
                skip_wakeword = True

                # 사용자가 wakeword 직후 바로 본문을 말하는 경우를 조금 배려
                time.sleep(0.15)

        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[Main] loop error: {e}")
            try:
                player.stop()
            except Exception:
                pass
            led.off()
            time.sleep(0.2)

    try:
        player.stop()
    except Exception:
        pass

    led.off()


def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nprogram interrupted by user.")
        led.off()
    except Exception as e:
        print(f"\nFatal Error: {e}")
        led.off()


if __name__ == "__main__":
    main()