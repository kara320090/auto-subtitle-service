from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from backend.app.config import OUTPUT_DIR, VIDEO_OUTPUT_FORMAT
from backend.app.utils.ffmpeg_utils import run_ffmpeg_command


@dataclass
class SRTItem:
    start_sec: float
    end_sec: float
    text: str


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_srt_time(value: str) -> float:
    # Format: HH:MM:SS,mmm
    match = re.match(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})$", value.strip())
    if not match:
        raise ValueError(f"Invalid SRT time format: {value}")

    hh, mm, ss, ms = map(int, match.groups())
    return hh * 3600 + mm * 60 + ss + (ms / 1000.0)


def _load_srt_items(subtitle_path: Path) -> list[SRTItem]:
    content = subtitle_path.read_text(encoding="utf-8-sig")
    blocks = [b.strip() for b in re.split(r"\r?\n\r?\n", content) if b.strip()]

    items: list[SRTItem] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        # Block can start with index line or directly with time line.
        if "-->" in lines[0]:
            time_line = lines[0]
            text_lines = lines[1:]
        else:
            if len(lines) < 3 or "-->" not in lines[1]:
                continue
            time_line = lines[1]
            text_lines = lines[2:]

        start_raw, end_raw = [part.strip() for part in time_line.split("-->", 1)]
        start_sec = _parse_srt_time(start_raw)
        end_sec = _parse_srt_time(end_raw)
        if end_sec <= start_sec:
            continue

        text = "\n".join(text_lines).strip()
        if not text:
            continue

        items.append(SRTItem(start_sec=start_sec, end_sec=end_sec, text=text))

    return items


def _resolve_font(font_path: str | None, font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates: list[Path] = []
    if font_path:
        candidates.append(Path(font_path))

    # Common Windows/Korean-friendly fonts first.
    candidates.extend(
        [
            Path("C:/Windows/Fonts/malgun.ttf"),
            Path("C:/Windows/Fonts/gulim.ttc"),
            Path("C:/Windows/Fonts/batang.ttc"),
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), font_size)

    return ImageFont.load_default()


def _draw_subtitle_on_frame(
    frame: np.ndarray,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    margin_bottom: int = 48,
    stroke_width: int = 2,
) -> np.ndarray:
    if not text:
        return frame

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).convert("RGBA")

    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(pil_img)

    lines = text.split("\n")
    spacing = 8

    max_width = 0
    total_height = 0
    line_sizes: list[tuple[int, int]] = []

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        line_sizes.append((width, height))
        max_width = max(max_width, width)
        total_height += height

    total_height += spacing * (len(lines) - 1)

    img_w, img_h = pil_img.size
    x = (img_w - max_width) // 2
    y = img_h - total_height - margin_bottom

    pad_x = 20
    pad_y = 10
    overlay_draw.rounded_rectangle(
        [x - pad_x, y - pad_y, x + max_width + pad_x, y + total_height + pad_y],
        radius=10,
        fill=(0, 0, 0, 140),
    )

    pil_img = Image.alpha_composite(pil_img, overlay)
    draw = ImageDraw.Draw(pil_img)

    cy = y
    for line, (w, h) in zip(lines, line_sizes):
        lx = (img_w - w) // 2
        draw.text(
            (lx, cy),
            line,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=stroke_width,
            stroke_fill=(0, 0, 0, 255),
        )
        cy += h + spacing

    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def render_video_with_subtitle_opencv(
    video_path: str,
    subtitle_path: str,
    font_path: str | None = None,
    font_size: int = 36,
) -> dict:
    ensure_directory(OUTPUT_DIR)

    input_video = Path(video_path)
    input_subtitle = Path(subtitle_path)

    if not input_video.exists():
        raise FileNotFoundError(f"Input video file does not exist: {video_path}")
    if not input_subtitle.exists():
        raise FileNotFoundError(f"Input subtitle file does not exist: {subtitle_path}")

    srt_items = _load_srt_items(input_subtitle)

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open input video: {video_path}")

    temp_video_only_path: Path | None = None

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_filename = f"{uuid4().hex}.{VIDEO_OUTPUT_FORMAT}"
        output_path = OUTPUT_DIR / output_filename
        temp_video_only_path = OUTPUT_DIR / f"{output_path.stem}_video_only.mp4"

        # mp4v is broadly available in OpenCV builds; ffmpeg pipeline can re-encode later if needed.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp_video_only_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create temporary output video: {temp_video_only_path}")

        font = _resolve_font(font_path=font_path, font_size=font_size)

        subtitle_idx = 0
        frame_index = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            current_sec = frame_index / fps

            while subtitle_idx < len(srt_items) and srt_items[subtitle_idx].end_sec < current_sec:
                subtitle_idx += 1

            text = ""
            if subtitle_idx < len(srt_items):
                item = srt_items[subtitle_idx]
                if item.start_sec <= current_sec <= item.end_sec:
                    text = item.text

            if text:
                frame = _draw_subtitle_on_frame(frame=frame, text=text, font=font)

            writer.write(frame)
            frame_index += 1

        writer.release()

        # Keep original audio stream (if present) while replacing the video stream
        # with OpenCV-rendered frames.
        run_ffmpeg_command(
            [
                "ffmpeg",
                "-y",
                "-i", str(temp_video_only_path),
                "-i", str(input_video),
                "-map", "0:v:0",
                "-map", "1:a?",
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                str(output_path),
            ]
        )

    finally:
        capture.release()
        if temp_video_only_path and temp_video_only_path.exists():
            temp_video_only_path.unlink()

    return {
        "video_path": str(input_video.resolve()),
        "subtitle_path": str(input_subtitle.resolve()),
        "output_filename": output_filename,
        "output_path": str(output_path.resolve()),
        "format": VIDEO_OUTPUT_FORMAT,
    }
