from pathlib import Path
from uuid import uuid4
import subprocess


BASE_DIR = Path(__file__).resolve().parents[3]
OUTPUT_DIR = BASE_DIR / "data" / "output"


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_ffmpeg_subtitle_path(path: Path) -> str:
    """
    Windows 경로를 ffmpeg subtitles 필터에 넣을 수 있게 변환한다.
    예:
    C:\test\abc.srt
    -> C\\:/test/abc.srt
    """
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

    output_filename = f"{uuid4().hex}.mp4"
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
        subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError("FFmpeg가 설치되어 있지 않거나 PATH에 등록되지 않았습니다.")
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else "알 수 없는 오류"
        raise RuntimeError(f"자막 삽입 영상 생성 실패: {error_msg}")

    return {
        "video_path": str(input_video.resolve()),
        "subtitle_path": str(input_subtitle.resolve()),
        "output_filename": output_filename,
        "output_path": str(output_path.resolve()),
        "format": "mp4",
    }