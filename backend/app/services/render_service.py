from pathlib import Path
from uuid import uuid4

from backend.app.config import OUTPUT_DIR, VIDEO_OUTPUT_FORMAT
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

    subtitle_filter_path = to_ffmpeg_subtitle_path(input_subtitle)

    command = [
        "ffmpeg",
        "-y",
        "-i", str(input_video),
        "-vf", f"subtitles='{subtitle_filter_path}'",
        "-c:a", "copy",
        str(output_path),
    ]

    try:
        run_ffmpeg_command(command)
    except RuntimeError as e:
        raise RuntimeError(f"자막 삽입 영상 생성 실패: {str(e)}")

    return {
        "video_path": str(input_video.resolve()),
        "subtitle_path": str(input_subtitle.resolve()),
        "output_filename": output_filename,
        "output_path": str(output_path.resolve()),
        "format": VIDEO_OUTPUT_FORMAT,
    }