from pathlib import Path
from uuid import uuid4

from backend.app.config import AUDIO_CHANNELS, AUDIO_DIR, AUDIO_FORMAT, AUDIO_SAMPLE_RATE
from backend.app.utils.ffmpeg_utils import run_ffmpeg_command


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_audio_from_video(video_path: str) -> dict:
    ensure_directory(AUDIO_DIR)

    input_path = Path(video_path)

    if not input_path.exists():
        raise FileNotFoundError(f"입력 영상 파일이 존재하지 않습니다: {video_path}")

    output_filename = f"{uuid4().hex}.{AUDIO_FORMAT}"
    output_path = AUDIO_DIR / output_filename

    command = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-ac", str(AUDIO_CHANNELS),
        str(output_path),
    ]

    try:
        run_ffmpeg_command(command)
    except RuntimeError as e:
        raise RuntimeError(f"오디오 추출 실패: {str(e)}")

    return {
        "video_path": str(input_path.resolve()),
        "audio_filename": output_filename,
        "audio_path": str(output_path.resolve()),
        "sample_rate": AUDIO_SAMPLE_RATE,
        "channels": AUDIO_CHANNELS,
        "format": AUDIO_FORMAT,
    }