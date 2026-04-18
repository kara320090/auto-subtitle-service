from fastapi import APIRouter

from backend.app.utils.ffmpeg_utils import check_ffmpeg_available, check_opencv_available

router = APIRouter()


@router.get("/")
def health_check():
    ffmpeg_available = check_ffmpeg_available()
    opencv_available = check_opencv_available()

    return {
        "status": "ok",
        "service": "auto-subtitle-backend",
        "checks": {
            "ffmpeg_available": ffmpeg_available,
            "opencv_available": opencv_available,
        },
    }