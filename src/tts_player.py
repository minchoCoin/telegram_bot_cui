import shutil
import subprocess
import threading
import edge_tts
import fasttext

from config import VOICE_BY_LANG

model = fasttext.load_model("lid.176.ftz")


def get_voice_by_lang(lang: str) -> str:
    lang = (lang or "").lower()
    return VOICE_BY_LANG.get(lang, VOICE_BY_LANG["default"])


def check_audio_tools() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("you need to install ffmpeg")
    if not shutil.which("ffplay"):
        raise RuntimeError("you need to install ffplay")


async def synthesize_to_wav(text: str, mp3_path: str, wav_path: str) -> str:
    check_audio_tools()

    lang = model.predict(text.split('\n')[0])[0][0].split("__")[-1]
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

class AudioPlayer:
    def __init__(self):
        self.proc = None
        self._lock = threading.Lock()

    def play(self, path: str) -> None:
        # lock 밖에서 stop 호출
        self.stop()

        proc = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        with self._lock:
            self.proc = proc

    def is_playing(self) -> bool:
        with self._lock:
            return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        with self._lock:
            proc = self.proc
            self.proc = None

        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def wait(self) -> None:
        with self._lock:
            proc = self.proc

        if proc is not None:
            proc.wait()

        with self._lock:
            if self.proc == proc:
                self.proc = None