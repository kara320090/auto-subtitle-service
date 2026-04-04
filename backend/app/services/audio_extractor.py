from pathlib import Path
from uuid import uuid4
import subprocess


BASE_DIR = Path(__file__).resolve().parents[3]
AUDIO_DIR = BASE_DIR / "data" / "audio"


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_audio_from_video(video_path: str) -> dict:
    """
    영상 파일에서 WAV 오디오를 추출한다.
    """
    ensure_directory(AUDIO_DIR)

    input_path = Path(video_path)

    if not input_path.exists():
        raise FileNotFoundError(f"입력 영상 파일이 존재하지 않습니다: {video_path}")

    output_filename = f"{uuid4().hex}.wav"
    output_path = AUDIO_DIR / output_filename

    command = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError("FFmpeg가 설치되어 있지 않거나 PATH에 등록되지 않았습니다.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"오디오 추출 실패: {e.stderr.strip() if e.stderr else '알 수 없는 오류'}"
        )

    return {
        "video_path": str(input_path),
        "audio_filename": output_filename,
        "audio_path": str(output_path),
        "sample_rate": 16000,
        "channels": 1,
        "format": "wav",
    }