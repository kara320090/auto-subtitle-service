from importlib.util import find_spec
import subprocess
from typing import Sequence


def run_ffmpeg_command(command: Sequence[str]) -> subprocess.CompletedProcess:
    """
    ffmpeg 명령어를 실행하고, 실패 시 RuntimeError를 발생시킨다.
    """
    try:
        result = subprocess.run(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return result
    except FileNotFoundError:
        raise RuntimeError("FFmpeg가 설치되어 있지 않거나 PATH에 등록되지 않았습니다.")
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else "알 수 없는 FFmpeg 오류"
        raise RuntimeError(error_msg)


def check_ffmpeg_available() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return True
    except Exception:
        return False


def check_opencv_available() -> bool:
    try:
        return find_spec("cv2") is not None
    except Exception:
        return False