import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
AUDIO_DIR = DATA_DIR / "audio"
SUBTITLE_DIR = DATA_DIR / "subtitles"
OUTPUT_DIR = DATA_DIR / "output"
METADATA_DIR = OUTPUT_DIR / "metadata"

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

# 기존 openai-whisper용 이름 대신 HF Whisper 기준으로 사용
BASE_MODEL_ID = "openai/whisper-large-v3"
WHISPER_LANGUAGE = "ko"
WHISPER_TASK = "transcribe"
DEFAULT_DOMAIN = "general"

AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_FORMAT = "wav"

VIDEO_OUTPUT_FORMAT = "mp4"
SRT_ENCODING = "utf-8-sig"

# 도메인별 adapter 경로
LORA_REGISTRY = {
    "social_news": BASE_DIR / "ai/data/results/social_news_lora/adapter",
    "ent": BASE_DIR / "ai/data/results/ent_lora/adapter",
    "vacation": BASE_DIR / "ai/data/results/vacation_lora/adapter",
    "politics": BASE_DIR / "ai/data/results/politics_lora/adapter",
}

ENABLED_LORA_DOMAINS = {
    "social_news",
    "ent",
    "vacation",
    "politics",
}

FALLBACK_TO_BASE = True

# 외부 LLM 후처리 서비스
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "https://5596-61-34-253-239.ngrok-free.app")
LLM_SERVICE_TIMEOUT_SECONDS = int(os.getenv("LLM_SERVICE_TIMEOUT_SECONDS", "60"))

OPENCV_FONT_PATH = os.getenv("OPENCV_FONT_PATH", r"C:\Windows\Fonts\malgun.ttf")