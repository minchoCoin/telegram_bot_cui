import re
import subprocess
from pathlib import Path

from config import WHISPER_MAIN, WHISPER_MODEL


def clean_whisper_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def check_whisper_files() -> None:
    if not Path(WHISPER_MAIN).exists():
        raise FileNotFoundError(
            f"whisper.cli not found: {WHISPER_MAIN}"
        )
    if not Path(WHISPER_MODEL).exists():
        raise FileNotFoundError(
            f"whisper.cpp model file not found: {WHISPER_MODEL}"
        )


def transcribe_with_whisper_cpp(wav_path: str) -> str:
    """
    Converting WAV files to text using the whisper.cpp CLI.
    """
    check_whisper_files()

    cmd = [
        WHISPER_MAIN,
        "-m", WHISPER_MODEL,
        "-f", wav_path,
        "-l", "auto",
        "--no-prints",
        "--no-timestamps"
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    text = clean_whisper_text(result.stdout)
    if not text:
        return ""
    text = "This text is transcribed text. if followed text is english, then speak in english, and if korean, then speak in korean. ANSWER AS MUCH AS SIMPLE: \n" + text
    return text