from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[3]
SUBTITLE_DIR = BASE_DIR / "data" / "subtitles"
OUTPUT_DIR = BASE_DIR / "data" / "output"


@router.get("/subtitle/{filename}")
def download_subtitle(filename: str):
    file_path = SUBTITLE_DIR / filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"SRT 파일을 찾을 수 없습니다: {filename}")

    return FileResponse(
        path=str(file_path),
        media_type="application/x-subrip",
        filename=filename,
    )


@router.get("/video/{filename}")
def download_video(filename: str):
    file_path = OUTPUT_DIR / filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"영상 파일을 찾을 수 없습니다: {filename}")

    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        filename=filename,
    )