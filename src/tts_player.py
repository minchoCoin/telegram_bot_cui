import re
import shutil
import subprocess
import edge_tts

from config import VOICE_KO, VOICE_EN


def detect_tts_voice(text: str) -> str:
    hangul_count = len(re.findall(r"[가-힣]", text))
    latin_count = len(re.findall(r"[A-Za-z]", text))
    return VOICE_KO if hangul_count >= latin_count else VOICE_EN


def check_audio_tools() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("you need to install ffmpeg")
    if not shutil.which("aplay") and not shutil.which("ffplay"):
        raise RuntimeError("you need to install either aplay or ffplay")


async def synthesize_to_wav(text: str, mp3_path: str, wav_path: str) -> str:
    check_audio_tools()

    voice = detect_tts_voice(text)
    print(f"TTS voice: {voice}")

    tts = edge_tts.Communicate(text, voice)
    await tts.save(mp3_path)
    print(f"MP3 saved successfully: {mp3_path}")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", mp3_path,
            "-ar", "24000",
            "-ac", "1",
            wav_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print(f"WAV conversion completed: {wav_path}")
    return wav_path


def play_wav(path: str) -> None:

    subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
            check=True,
    )