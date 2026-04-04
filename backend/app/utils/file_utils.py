from pathlib import Path
from uuid import uuid4
from fastapi import UploadFile


# 프로젝트 루트 기준 data/input 경로
BASE_DIR = Path(__file__).resolve().parents[3]
INPUT_DIR = BASE_DIR / "data" / "input"

# 허용할 영상 확장자
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def validate_video_file(filename: str) -> None:
    ext = get_file_extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"지원하지 않는 파일 형식입니다: {ext}. "
            f"허용 형식: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )


def generate_saved_filename(original_filename: str) -> str:
    ext = get_file_extension(original_filename)
    unique_id = uuid4().hex
    return f"{unique_id}{ext}"


async def save_upload_file(upload_file: UploadFile) -> dict:
    ensure_directory(INPUT_DIR)

    original_filename = upload_file.filename or "unknown.mp4"
    validate_video_file(original_filename)

    saved_filename = generate_saved_filename(original_filename)
    saved_path = INPUT_DIR / saved_filename

    content = await upload_file.read()
    saved_path.write_bytes(content)

    return {
        "original_filename": original_filename,
        "saved_filename": saved_filename,
        "saved_path": str(saved_path),
        "size_bytes": len(content),
    }