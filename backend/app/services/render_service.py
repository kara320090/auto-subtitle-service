from pathlib import Path
from uuid import uuid4

from backend.app.config import OUTPUT_DIR, VIDEO_OUTPUT_FORMAT
from backend.app.services.opencv_render_service import render_video_with_subtitle_opencv
from backend.app.utils.ffmpeg_utils import run_ffmpeg_command


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_ffmpeg_subtitle_path(path: Path) -> str:
    resolved = path.resolve()
    path_str = str(resolved).replace("\\", "/")

    if len(path_str) >= 2 and path_str[1] == ":":
        drive = path_str[0]
        rest = path_str[2:]
        path_str = f"{drive}\\:{rest}"

    return path_str


def render_video_with_subtitle(video_path: str, subtitle_path: str) -> dict:
    ensure_directory(OUTPUT_DIR)

    input_video = Path(video_path)
    input_subtitle = Path(subtitle_path)

    if not input_video.exists():
        raise FileNotFoundError(f"입력 영상 파일이 존재하지 않습니다: {video_path}")

    if not input_subtitle.exists():
        raise FileNotFoundError(f"입력 자막 파일이 존재하지 않습니다: {subtitle_path}")

    output_filename = f"{uuid4().hex}.{VIDEO_OUTPUT_FORMAT}"
    output_path = OUTPUT_DIR / output_filename

    return render_video_with_subtitle_opencv(
        video_path=video_path,
        subtitle_path=subtitle_path,
        output_path=output_path,
    )