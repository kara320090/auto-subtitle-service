from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from backend.app.config import OPENCV_FONT_PATH, OUTPUT_DIR, VIDEO_OUTPUT_FORMAT
from backend.app.utils.ffmpeg_utils import run_ffmpeg_command


@dataclass
class SubtitleItem:
    start_sec: float
    end_sec: float
    text: str


@dataclass
class DisplaySubtitleItem:
    start_sec: float
    end_sec: float
    text: str
    original_start: float
    original_end: float


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


def parse_srt_time(time_text: str) -> float:
    hours_part, minutes_part, second_part = time_text.split(":")
    seconds_part, milliseconds_part = second_part.split(",")
    return (
        int(hours_part) * 3600
        + int(minutes_part) * 60
        + int(seconds_part)
        + int(milliseconds_part) / 1000.0
    )


def load_subtitles_from_srt(srt_path: str) -> List[SubtitleItem]:
    path = Path(srt_path)
    if not path.exists():
        raise FileNotFoundError(f"SRT 파일을 찾을 수 없습니다: {srt_path}")

    content = path.read_text(encoding="utf-8-sig").strip()
    if not content:
        raise ValueError("SRT 파일이 비어 있습니다.")

    subtitles: List[SubtitleItem] = []
    for raw_block in content.split("\n\n"):
        lines = [line.strip() for line in raw_block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        time_line_index = next((idx for idx, line in enumerate(lines) if "-->" in line), None)
        if time_line_index is None:
            continue

        time_line = lines[time_line_index]
        text_lines = lines[time_line_index + 1 :]
        if not text_lines:
            continue

        try:
            start_text, end_text = [part.strip() for part in time_line.split("-->", 1)]
            start_sec = parse_srt_time(start_text)
            end_sec = parse_srt_time(end_text)
        except Exception:
            continue

        text = "\n".join(text_lines).strip()
        if not text:
            continue

        if end_sec <= start_sec:
            continue

        subtitles.append(
            SubtitleItem(
                start_sec=start_sec,
                end_sec=end_sec,
                text=text,
            )
        )

    if not subtitles:
        raise ValueError("사용 가능한 SRT 자막이 없습니다.")

    subtitles.sort(key=lambda item: item.start_sec)
    return subtitles


def estimate_reading_duration(
    text: str,
    min_duration: float = 1.6,
    chars_per_sec: float = 10.0,
    max_duration: float = 5.5,
) -> float:
    normalized = "".join(text.split())
    char_count = max(1, len(normalized))
    estimated = char_count / chars_per_sec
    return max(min_duration, min(estimated, max_duration))


def build_readable_timeline(
    subtitles: List[SubtitleItem],
    min_duration: float = 1.6,
    chars_per_sec: float = 10.0,
    max_duration: float = 5.5,
    extra_hold: float = 0.2,
    max_overlap_after_next_start: float = 1.0,
) -> List[DisplaySubtitleItem]:
    result: List[DisplaySubtitleItem] = []

    for index, subtitle in enumerate(subtitles):
        readable_duration = estimate_reading_duration(
            subtitle.text,
            min_duration=min_duration,
            chars_per_sec=chars_per_sec,
            max_duration=max_duration,
        )
        original_duration = max(0.0, subtitle.end_sec - subtitle.start_sec)
        final_duration = max(original_duration, readable_duration) + extra_hold
        desired_end = subtitle.start_sec + final_duration

        if index + 1 < len(subtitles):
            next_start = subtitles[index + 1].start_sec
            desired_end = min(desired_end, next_start + max_overlap_after_next_start)

        result.append(
            DisplaySubtitleItem(
                start_sec=subtitle.start_sec,
                end_sec=desired_end,
                text=subtitle.text,
                original_start=subtitle.start_sec,
                original_end=subtitle.end_sec,
            )
        )

    return result


def get_active_subtitles(
    current_sec: float,
    subtitles: List[DisplaySubtitleItem],
    activation_epsilon: float = 0.0,
) -> List[DisplaySubtitleItem]:
    active = [
        item
        for item in subtitles
        if (item.start_sec - activation_epsilon) <= current_sec < (item.end_sec + activation_epsilon)
    ]
    active.sort(key=lambda item: item.start_sec)
    return active


def draw_korean_subtitle_stack(
    frame: np.ndarray,
    items: List[DisplaySubtitleItem],
    font_path: str,
    font_size: int = 36,
    margin_bottom: int = 60,
    item_gap: int = 16,
    line_spacing: int = 8,
    max_visible_items: int = 2,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    bg_alpha: float = 0.45,
    stroke_width: int = 2,
    stroke_fill: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    if not items:
        return frame

    items = items[-max_visible_items:]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).convert("RGBA")
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, font_size)

    img_w, img_h = pil_img.size

    prepared = []
    for item in items:
        lines = item.text.split("\n")
        line_sizes = []
        block_w = 0
        block_h = 0

        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            line_sizes.append((width, height))
            block_w = max(block_w, width)
            block_h += height

        if len(lines) > 1:
            block_h += line_spacing * (len(lines) - 1)

        prepared.append(
            {
                "lines": lines,
                "line_sizes": line_sizes,
                "block_w": block_w,
                "block_h": block_h,
            }
        )

    padding_x = 20
    padding_y = 12

    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    current_bottom = img_h - margin_bottom

    for item in reversed(prepared):
        block_w = item["block_w"]
        block_h = item["block_h"]

        x = (img_w - block_w) // 2
        y = current_bottom - block_h

        rect_x1 = x - padding_x
        rect_y1 = y - padding_y
        rect_x2 = x + block_w + padding_x
        rect_y2 = y + block_h + padding_y

        item["x"] = x
        item["y"] = y

        overlay_draw.rounded_rectangle(
            [rect_x1, rect_y1, rect_x2, rect_y2],
            radius=12,
            fill=(*bg_color, int(255 * bg_alpha)),
        )

        current_bottom = rect_y1 - item_gap

    pil_img = Image.alpha_composite(pil_img, overlay)
    draw = ImageDraw.Draw(pil_img)

    for item in reversed(prepared):
        current_y = item["y"]

        for line, (width, height) in zip(item["lines"], item["line_sizes"]):
            line_x = (img_w - width) // 2
            draw.text(
                (line_x, current_y),
                line,
                font=font,
                fill=text_color,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
            )
            current_y += height + line_spacing

    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def mux_video_with_original_audio(
    subtitled_video_path: str,
    original_video_path: str,
    output_video_path: str,
    ffmpeg_bin: str = "ffmpeg",
) -> None:
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        subtitled_video_path,
        "-i",
        original_video_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        output_video_path,
    ]

    run_ffmpeg_command(cmd)


def render_video_with_subtitle_opencv(
    video_path: str,
    subtitle_path: str,
    output_path: Path,
    font_path: str = OPENCV_FONT_PATH,
    font_size: int = 36,
    min_duration: float = 1.6,
    chars_per_sec: float = 10.0,
    max_duration: float = 5.5,
    extra_hold: float = 0.2,
    max_overlap_after_next_start: float = 1.0,
    margin_bottom: int = 60,
    item_gap: int = 16,
    line_spacing: int = 8,
    max_visible_items: int = 2,
    activation_epsilon: float | None = None,
) -> dict:
    ensure_directory(output_path.parent)

    input_video = Path(video_path)
    input_subtitle = Path(subtitle_path)

    if not input_video.exists():
        raise FileNotFoundError(f"입력 영상 파일이 존재하지 않습니다: {video_path}")

    if not input_subtitle.exists():
        raise FileNotFoundError(f"입력 자막 파일이 존재하지 않습니다: {subtitle_path}")

    subtitles = load_subtitles_from_srt(str(input_subtitle))
    display_timeline = build_readable_timeline(
        subtitles=subtitles,
        min_duration=min_duration,
        chars_per_sec=chars_per_sec,
        max_duration=max_duration,
        extra_hold=extra_hold,
        max_overlap_after_next_start=max_overlap_after_next_start,
    )

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"입력 영상을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("입력 영상의 해상도를 읽을 수 없습니다.")

    if activation_epsilon is None:
        activation_epsilon = 0.5 / fps

    temp_video_path = output_path.parent / f"{output_path.stem}_temp_noaudio.mp4"
    writer = cv2.VideoWriter(str(temp_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"임시 비디오를 생성할 수 없습니다: {temp_video_path}")

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_sec = frame_idx / fps
            active_items = get_active_subtitles(
                current_sec=current_sec,
                subtitles=display_timeline,
                activation_epsilon=activation_epsilon,
            )

            if active_items:
                frame = draw_korean_subtitle_stack(
                    frame=frame,
                    items=active_items,
                    font_path=font_path,
                    font_size=font_size,
                    margin_bottom=margin_bottom,
                    item_gap=item_gap,
                    line_spacing=line_spacing,
                    max_visible_items=max_visible_items,
                )

            writer.write(frame)
            frame_idx += 1

            if frame_idx % 100 == 0:
                print(f"진행률: {frame_idx}/{frame_count} 프레임")
    finally:
        cap.release()
        writer.release()

    mux_video_with_original_audio(
        subtitled_video_path=str(temp_video_path),
        original_video_path=str(input_video),
        output_video_path=str(output_path),
    )

    if temp_video_path.exists():
        temp_video_path.unlink()

    return {
        "video_path": str(input_video.resolve()),
        "subtitle_path": str(input_subtitle.resolve()),
        "output_filename": output_path.name,
        "output_path": str(output_path.resolve()),
        "format": VIDEO_OUTPUT_FORMAT,
        "engine": "opencv",
    }