from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
AUDIO_DIR = DATA_DIR / "audio"
SUBTITLE_DIR = DATA_DIR / "subtitles"
OUTPUT_DIR = DATA_DIR / "output"
METADATA_DIR = OUTPUT_DIR / "metadata"

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

WHISPER_MODEL_NAME = "base"
WHISPER_LANGUAGE = "ko"

AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_FORMAT = "wav"

VIDEO_OUTPUT_FORMAT = "mp4"
SRT_ENCODING = "utf-8-sig"