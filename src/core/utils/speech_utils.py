from openai import OpenAI
import os
import shutil
import subprocess
import soundfile as sf

from src.core.core_constants import TRANSCRIPTION_MODEL, TTS_MODEL
from web.app_constants import ROOT_TEMP_DATA, FILENAME_AUDIO_INPUT, FILENAME_AUDIO_OUTPUT

os.makedirs(ROOT_TEMP_DATA, exist_ok=True)


class ExitProgram(Exception):
    def __init__(self, message: str = "User wants to exit program"):
        super().__init__(message)


def _which(cmd: str) -> str | None:
    return shutil.which(cmd)


def _record_with_arecord(out_wav: str, rate: int = 44100, channels: int = 1):
    # arecord writes wav natively
    # -f S16_LE ensures 16-bit PCM
    cmd = ["arecord", "-q", "-f", "S16_LE", "-r", str(rate), "-c", str(channels), out_wav]
    # user stops with Ctrl+C
    subprocess.run(cmd, check=True)


def _record_with_ffmpeg(out_wav: str, rate: int = 44100, channels: int = 1):
    # tries default ALSA input
    # user stops with Ctrl+C
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "alsa", "-i", "default",
        "-ac", str(channels), "-ar", str(rate),
        "-y", out_wav
    ]
    subprocess.run(cmd, check=True)


def record_audio() -> int:
    out_path = f"{ROOT_TEMP_DATA}{FILENAME_AUDIO_INPUT}"

    print("Press ENTER to start recording, Ctrl+C to stop, or type 'q' + ENTER to quit.")
    cmd = input("> ").strip().lower()
    if cmd == "q":
        return 1

    rec = _which("arecord") or _which("ffmpeg")
    if rec is None:
        print("No recorder found (arecord/ffmpeg). Falling back to text input.")
        return 2  # special: no audio

    print("Recording... (Ctrl+C to stop)")
    try:
        if rec.endswith("arecord"):
            _record_with_arecord(out_path)
        else:
            _record_with_ffmpeg(out_path)
    except KeyboardInterrupt:
        print("\nRecording stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Audio recording failed ({e}). Falling back to text input.")
        return 2

    return 0


async def speech_to_text() -> str:
    status = record_audio()
    if status == 1:
        raise ExitProgram()
    if status == 2:
        # fallback
        return input("You (typed): ").strip()

    client = OpenAI()
    wav_path = f"{ROOT_TEMP_DATA}{FILENAME_AUDIO_INPUT}"
    with open(wav_path, "rb") as audiofile:
        transcription = client.audio.transcriptions.create(
            model=TRANSCRIPTION_MODEL,
            file=audiofile
        )
    print("transcription:", transcription.text)
    return transcription.text


def wav_to_16_bit(filepath: str) -> None:
    data, samplerate = sf.read(filepath)
    sf.write(file=filepath, data=data, samplerate=samplerate, subtype="PCM_16")


def play_wav(filepath: str) -> None:
    """
    Play WAV using system tools (no Python audio deps).
    Tries: aplay -> ffplay -> mpv
    """
    wav_to_16_bit(filepath)

    for cmd in (
        ["aplay", "-q", filepath],
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", filepath],
        ["mpv", "--really-quiet", filepath],
    ):
        if _which(cmd[0]) is None:
            continue
        try:
            subprocess.run(cmd, check=True)
            return
        except Exception:
            pass

    print(f"(Couldn't auto-play audio. File saved at: {filepath})")


def text_to_speech(text: str) -> str:
    client = OpenAI()
    file_path = f"{ROOT_TEMP_DATA}{FILENAME_AUDIO_OUTPUT}"
    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice="onyx",
        input=text,
    )
    response.stream_to_file(file_path)
    return FILENAME_AUDIO_OUTPUT