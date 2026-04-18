from pathlib import Path
from uuid import uuid4
import math
import re

from backend.app.config import SRT_ENCODING, SUBTITLE_DIR


MAX_SUBTITLE_CHARS = 28
MAX_SUBTITLE_DURATION = 4.0


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


def normalize_text(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def split_text_by_chars(text: str, target_count: int) -> list[str]:
    text = normalize_text(text)

    if not text:
        return []

    if target_count <= 1 and len(text) <= MAX_SUBTITLE_CHARS:
        return [text]

    # 1차: 문장부호 기준 분리
    candidates = re.split(r"(?<=[.!?。！？…])\s*", text)
    candidates = [c.strip() for c in candidates if c.strip()]

    # 문장부호 기준으로 잘 안 나뉘면 공백 기준 사용
    if len(candidates) <= 1:
        words = text.split(" ")
        chunks = []
        current = ""

        for word in words:
            if not current:
                current = word
            elif len(current) + 1 + len(word) <= MAX_SUBTITLE_CHARS:
                current += " " + word
            else:
                chunks.append(current)
                current = word

        if current:
            chunks.append(current)

        candidates = chunks

    # 그래도 너무 긴 덩어리는 글자 수 기준으로 강제 분리
    final_chunks = []

    for chunk in candidates:
        chunk = chunk.strip()

        if len(chunk) <= MAX_SUBTITLE_CHARS:
            final_chunks.append(chunk)
            continue

        start = 0
        while start < len(chunk):
            final_chunks.append(chunk[start:start + MAX_SUBTITLE_CHARS].strip())
            start += MAX_SUBTITLE_CHARS

    # 시간 길이에 비해 조각 수가 부족하면 목표 개수에 맞춰 다시 분리
    if len(final_chunks) < target_count:
        joined = "".join(final_chunks)
        avg_len = max(1, math.ceil(len(joined) / target_count))

        rebuilt = []
        start = 0
        while start < len(joined):
            rebuilt.append(joined[start:start + avg_len].strip())
            start += avg_len

        final_chunks = [c for c in rebuilt if c]

    return final_chunks


def split_long_segment(seg: dict) -> list[dict]:
    start = float(seg.get("start", 0.0))
    end = float(seg.get("end", start))
    text = normalize_text(seg.get("text", ""))

    if not text:
        return []

    if end < start:
        end = start

    duration = end - start

    # timestamp가 0초로 깨져 들어온 경우 최소 표시 시간 부여
    if duration <= 0:
        duration = min(3.0, max(1.5, len(text) / 10.0))
        end = start + duration

    count_by_duration = max(1, math.ceil(duration / MAX_SUBTITLE_DURATION))
    count_by_chars = max(1, math.ceil(len(text) / MAX_SUBTITLE_CHARS))
    split_count = max(count_by_duration, count_by_chars)

    parts = split_text_by_chars(text, split_count)

    if len(parts) <= 1:
        return [
            {
                "start": start,
                "end": end,
                "text": text,
            }
        ]

    total_chars = sum(len(p) for p in parts) or 1
    current_time = start
    result = []

    for idx, part in enumerate(parts):
        if idx == len(parts) - 1:
            part_end = end
        else:
            ratio = len(part) / total_chars
            part_duration = max(0.8, duration * ratio)
            part_end = min(end, current_time + part_duration)

        result.append(
            {
                "start": current_time,
                "end": part_end,
                "text": part,
            }
        )

        current_time = part_end

    return result


def normalize_segments_for_srt(segments: list[dict]) -> list[dict]:
    normalized = []

    for seg in segments:
        normalized.extend(split_long_segment(seg))

    # 혹시 시간이 겹치거나 역전되는 경우를 최소 보정
    fixed = []
    previous_end = 0.0

    for seg in normalized:
        start = float(seg["start"])
        end = float(seg["end"])
        text = seg["text"]

        if start < previous_end:
            start = previous_end

        if end <= start:
            end = start + 1.5

        fixed.append(
            {
                "start": start,
                "end": end,
                "text": text,
            }
        )

        previous_end = end

    return fixed


def build_srt_content(segments: list[dict]) -> str:
    srt_segments = normalize_segments_for_srt(segments)

    lines = []

    for idx, seg in enumerate(srt_segments, start=1):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = normalize_text(seg.get("text", ""))

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
    file_path.write_text(srt_content, encoding=SRT_ENCODING)

    return {
        "subtitle_filename": file_name,
        "subtitle_path": str(file_path.resolve()),
        "content": srt_content,
        "segment_count": len(normalize_segments_for_srt(segments)),
    }