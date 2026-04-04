from pathlib import Path
from uuid import uuid4


BASE_DIR = Path(__file__).resolve().parents[3]
SUBTITLE_DIR = BASE_DIR / "data" / "subtitles"


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def format_srt_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0

    total_milliseconds = int(round(seconds * 1000))
    hours = total_milliseconds // 3_600_000
    remainder = total_milliseconds % 3_600_000
    minutes = remainder // 60_000
    remainder = remainder % 60_000
    secs = remainder // 1000
    milliseconds = remainder % 1000

    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def build_srt_content(segments: list[dict]) -> str:
    lines = []

    for idx, seg in enumerate(segments, start=1):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = str(seg.get("text", "")).strip()

        if not text:
            continue

        lines.append(str(idx))
        lines.append(f"{format_srt_time(start)} --> {format_srt_time(end)}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def save_srt_file(segments: list[dict], base_name: str | None = None) -> dict:
    ensure_directory(SUBTITLE_DIR)

    if not isinstance(segments, list) or len(segments) == 0:
        raise ValueError("segments가 비어 있거나 올바른 형식이 아닙니다.")

    srt_content = build_srt_content(segments)

    if not base_name:
        base_name = uuid4().hex

    file_name = f"{base_name}.srt"
    file_path = SUBTITLE_DIR / file_name

    file_path.write_text(srt_content, encoding="utf-8-sig")

    return {
        "subtitle_filename": file_name,
        "subtitle_path": str(file_path),
        "content": srt_content,
        "segment_count": len(segments),
    }