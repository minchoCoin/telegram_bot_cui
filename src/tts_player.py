import re
import shutil
import subprocess
import edge_tts
    
from config import VOICE_BY_LANG

import fasttext

model = fasttext.load_model("lid.176.ftz")

def get_voice_by_lang(lang: str) -> str:
    lang = (lang or "").lower()
    return VOICE_BY_LANG.get(lang, VOICE_BY_LANG["default"])


def check_audio_tools() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("you need to install ffmpeg")
    if not shutil.which("aplay") and not shutil.which("ffplay"):
        raise RuntimeError("you need to install either aplay or ffplay")


async def synthesize_to_wav(text: str, mp3_path: str, wav_path: str) -> str:
    check_audio_tools()

    lang = model.predict(text)[0][0].split("__")[-1]
    voice = get_voice_by_lang(lang)
    print(f"TTS voice: {voice}")

    tts = edge_tts.Communicate(text, voice)
    await tts.save(mp3_path)

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

    return wav_path


def play_wav(path: str) -> None:

    subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
            check=True,
    )