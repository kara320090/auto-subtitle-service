import cv2
import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


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


def parse_srt_timestamp(ts: str) -> float:
    match = re.match(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", ts.strip())
    if not match:
        raise ValueError(f"잘못된 SRT 타임스탬프 형식: {ts}")

    hh, mm, ss, ms = map(int, match.groups())
    return hh * 3600 + mm * 60 + ss + ms / 1000.0


def load_srt(srt_path: str) -> List[SubtitleItem]:
    with open(srt_path, "r", encoding="utf-8-sig") as f:
        content = f.read().strip()

    blocks = re.split(r"\n\s*\n", content)
    subtitles: List[SubtitleItem] = []

    for block in blocks:
        lines = [line.strip("\ufeff") for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        if re.match(r"^\d+$", lines[0]):
            time_line = lines[1]
            text_lines = lines[2:]
        else:
            time_line = lines[0]
            text_lines = lines[1:]

        time_match = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2},\d{3})",
            time_line
        )
        if not time_match:
            continue

        start_ts, end_ts = time_match.groups()
        text = "\n".join(text_lines).strip()

        subtitles.append(
            SubtitleItem(
                start_sec=parse_srt_timestamp(start_ts),
                end_sec=parse_srt_timestamp(end_ts),
                text=text
            )
        )

    subtitles.sort(key=lambda x: x.start_sec)
    return subtitles


def estimate_reading_duration(
    text: str,
    min_duration: float = 1.6,
    chars_per_sec: float = 10.0,
    max_duration: float = 5.5
) -> float:
    """
    자막 길이에 따라 읽기 시간을 대략 추정.
    공백/개행은 제외해서 계산.
    """
    normalized = re.sub(r"\s+", "", text)
    char_count = max(1, len(normalized))
    estimated = char_count / chars_per_sec
    return max(min_duration, min(estimated, max_duration))


def build_readable_timeline(
    subtitles: List[SubtitleItem],
    min_duration: float = 1.6,
    chars_per_sec: float = 10.0,
    max_duration: float = 5.5,
    extra_hold: float = 0.2
) -> List[DisplaySubtitleItem]:
    """
    SRT 원본 시간을 그대로 쓰지 않고,
    가독성을 위해 표시 종료 시점을 조금 늘린 타임라인 생성.

    포인트:
    - 최소 표시 시간 보장
    - 글자 수에 따른 읽기 시간 반영
    - 살짝 더 남겨서 화면 전환이 덜 급하게 보이게 처리
    - 다음 자막 시작과 겹쳐도 자르지 않음
      -> 겹치는 시간에는 여러 줄로 함께 표시
    """
    result: List[DisplaySubtitleItem] = []

    for s in subtitles:
        readable_duration = estimate_reading_duration(
            s.text,
            min_duration=min_duration,
            chars_per_sec=chars_per_sec,
            max_duration=max_duration
        )
        original_duration = max(0.0, s.end_sec - s.start_sec)
        final_duration = max(original_duration, readable_duration) + extra_hold

        result.append(
            DisplaySubtitleItem(
                start_sec=s.start_sec,
                end_sec=s.start_sec + final_duration,
                text=s.text,
                original_start=s.start_sec,
                original_end=s.end_sec
            )
        )

    return result


def get_active_subtitles(
    current_sec: float,
    subtitles: List[DisplaySubtitleItem]
) -> List[DisplaySubtitleItem]:
    """
    현재 시점에 화면에 보여야 하는 자막들을 모두 반환.
    겹치는 경우 2개 이상 반환될 수 있음.
    """
    active = [
        item for item in subtitles
        if item.start_sec <= current_sec <= item.end_sec
    ]
    active.sort(key=lambda x: x.start_sec)
    return active


def compose_multiline_text(active_items: List[DisplaySubtitleItem]) -> str:
    """
    active 자막들을 하나의 멀티라인 텍스트로 합침.
    먼저 시작한 자막이 위,
    나중에 시작한 자막이 아래.
    """
    if not active_items:
        return ""

    merged_lines: List[str] = []
    for item in active_items:
        lines = item.text.split("\n")
        merged_lines.extend(lines)

    return "\n".join(merged_lines)


def draw_korean_subtitle(
    frame: np.ndarray,
    text: str,
    font_path: str,
    font_size: int = 36,
    margin_bottom: int = 60,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    bg_alpha: float = 0.5,
    stroke_width: int = 2,
    stroke_fill: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    if not text:
        return frame

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)

    font = ImageFont.truetype(font_path, font_size)

    lines = text.split("\n")
    spacing = 8

    line_sizes = []
    max_width = 0
    total_height = 0

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        line_sizes.append((w, h))
        max_width = max(max_width, w)
        total_height += h

    total_height += spacing * (len(lines) - 1)

    img_w, img_h = pil_img.size
    x = (img_w - max_width) // 2
    y = img_h - total_height - margin_bottom

    padding_x = 20
    padding_y = 12
    rect_x1 = x - padding_x
    rect_y1 = y - padding_y
    rect_x2 = x + max_width + padding_x
    rect_y2 = y + total_height + padding_y

    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle(
        [rect_x1, rect_y1, rect_x2, rect_y2],
        radius=12,
        fill=(*bg_color, int(255 * bg_alpha))
    )

    pil_img = pil_img.convert("RGBA")
    pil_img = Image.alpha_composite(pil_img, overlay)
    draw = ImageDraw.Draw(pil_img)

    current_y = y
    for line, (w, h) in zip(lines, line_sizes):
        line_x = (img_w - w) // 2
        draw.text(
            (line_x, current_y),
            line,
            font=font,
            fill=text_color,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill
        )
        current_y += h + spacing

    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def add_subtitles_to_video(
    input_video_path: str,
    output_video_path: str,
    subtitles: List[SubtitleItem],
    font_path: str,
    font_size: int = 36,
    min_duration: float = 1.6,
    chars_per_sec: float = 10.0,
    max_duration: float = 5.5,
    extra_hold: float = 0.2
) -> None:
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"비디오를 열 수 없습니다: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("FPS 정보를 읽을 수 없습니다.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"출력 비디오를 생성할 수 없습니다: {output_video_path}")

    display_timeline = build_readable_timeline(
        subtitles=subtitles,
        min_duration=min_duration,
        chars_per_sec=chars_per_sec,
        max_duration=max_duration,
        extra_hold=extra_hold
    )

    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_sec = frame_idx / fps
            active_items = get_active_subtitles(current_sec, display_timeline)

            if active_items:
                merged_text = compose_multiline_text(active_items)
                frame = draw_korean_subtitle(
                    frame=frame,
                    text=merged_text,
                    font_path=font_path,
                    font_size=font_size
                )

            writer.write(frame)
            frame_idx += 1

            if frame_idx % 100 == 0:
                print(f"진행률: {frame_idx}/{total_frames} 프레임")

    finally:
        cap.release()
        writer.release()

    print(f"완료: {output_video_path}")


if __name__ == "__main__":
    input_video = "input.mp4"
    output_video = "output_with_readable_subtitles.mp4"
    srt_path = "subtitle.srt"

    font_path = "C:/Windows/Fonts/malgun.ttf"
    # macOS 예시: "/System/Library/Fonts/AppleSDGothicNeo.ttc"
    # Linux 예시: "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

    subtitles = load_srt(srt_path)

    add_subtitles_to_video(
        input_video_path=input_video,
        output_video_path=output_video,
        subtitles=subtitles,
        font_path=font_path,
        font_size=36,
        min_duration=1.8,     # 자막 최소 유지 시간
        chars_per_sec=9.5,    # 숫자가 낮을수록 더 오래 보여줌
        max_duration=6.0,     # 너무 오래 남는 것 방지
        extra_hold=0.25       # 살짝 여유
    )