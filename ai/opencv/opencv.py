import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
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


def load_pairs_from_jsonl_line(
    jsonl_path: str,
    line_index: int = 0,
    text_field: str = "after_text",
) -> List[SubtitleItem]:
    """
    JSONL 파일의 특정 한 줄을 읽고,
    그 안의 pairs를 SubtitleItem 리스트로 변환한다.

    text_field:
    - "before_text" 사용 가능
    - "after_text" 사용 가능
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL 파일을 찾을 수 없습니다: {jsonl_path}")

    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError("JSONL 파일이 비어 있습니다.")

    if line_index < 0 or line_index >= len(lines):
        raise IndexError(f"line_index 범위를 벗어났습니다: {line_index} / 전체 {len(lines)}줄")

    row = json.loads(lines[line_index])

    pairs = row.get("pairs", [])
    if not pairs:
        raise ValueError("선택한 JSONL 줄에 pairs가 없습니다.")

    subtitles: List[SubtitleItem] = []
    for item in pairs:
        start = float(item.get("start", 0.0))
        end = float(item.get("end", 0.0))
        text = str(item.get(text_field, "")).strip()

        if end <= start:
            continue
        if not text:
            continue

        subtitles.append(
            SubtitleItem(
                start_sec=start,
                end_sec=end,
                text=text,
            )
        )

    if not subtitles:
        raise ValueError("사용 가능한 pair 자막이 없습니다.")

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
    normalized = "".join(text.split())
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
    원본 segment 시간을 그대로 쓰지 않고,
    가독성을 위해 표시 종료 시점을 조금 늘린 타임라인 생성.
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
    active = [
        item for item in subtitles
        if item.start_sec <= current_sec <= item.end_sec
    ]
    active.sort(key=lambda x: x.start_sec)
    return active


def compose_multiline_text(active_items: List[DisplaySubtitleItem]) -> str:
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


def check_ffmpeg(ffmpeg_bin: str = "ffmpeg"):
    if shutil.which(ffmpeg_bin) is None:
        raise EnvironmentError(f"ffmpeg를 찾지 못했습니다. PATH 확인 필요: {ffmpeg_bin}")


def create_temp_black_subtitle_video(
    subtitles: List[SubtitleItem],
    temp_video_path: str,
    font_path: str,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    font_size: int = 36,
    min_duration: float = 1.6,
    chars_per_sec: float = 10.0,
    max_duration: float = 5.5,
    extra_hold: float = 0.2
) -> None:
    """
    OpenCV로 임시 검은 배경 자막 영상을 생성한다.
    이 영상은 나중에 ffmpeg로 H.264 MP4로 후처리된다.
    """
    display_timeline = build_readable_timeline(
        subtitles=subtitles,
        min_duration=min_duration,
        chars_per_sec=chars_per_sec,
        max_duration=max_duration,
        extra_hold=extra_hold
    )

    total_duration = max(item.end_sec for item in display_timeline) + 1.0
    total_frames = int(total_duration * fps)

    temp_path = Path(temp_video_path)
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    # OpenCV 기본 저장
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"임시 비디오를 생성할 수 없습니다: {temp_video_path}")

    try:
        for frame_idx in range(total_frames):
            current_sec = frame_idx / fps

            # 검은 배경
            frame = np.zeros((height, width, 3), dtype=np.uint8)

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

            if frame_idx % 100 == 0:
                print(f"진행률: {frame_idx}/{total_frames} 프레임")

    finally:
        writer.release()

    print(f"[INFO] 임시 영상 생성 완료: {temp_path}")


def reencode_with_ffmpeg(
    input_video_path: str,
    output_video_path: str,
    ffmpeg_bin: str = "ffmpeg"
) -> None:
    """
    ffmpeg로 H.264 + yuv420p로 재인코딩하여
    플레이어 호환성을 높인다.
    """
    check_ffmpeg(ffmpeg_bin)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", input_video_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_video_path,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 후처리 실패\n"
            f"입력: {input_video_path}\n"
            f"출력: {output_video_path}\n"
            f"stderr:\n{result.stderr}"
        )

    print(f"[INFO] 최종 영상 생성 완료: {output_video_path}")


def create_black_subtitle_preview_from_pairs_jsonl(
    jsonl_path: str,
    output_video_path: str,
    font_path: str,
    line_index: int = 0,
    text_field: str = "after_text",
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    font_size: int = 36,
    min_duration: float = 1.6,
    chars_per_sec: float = 10.0,
    max_duration: float = 5.5,
    extra_hold: float = 0.2,
    ffmpeg_bin: str = "ffmpeg"
) -> None:
    """
    JSONL 파일의 특정 한 줄을 읽어서
    검은 배경 자막 preview 영상을 생성하고,
    ffmpeg로 최종 mp4를 후처리한다.
    """
    subtitles = load_pairs_from_jsonl_line(
        jsonl_path=jsonl_path,
        line_index=line_index,
        text_field=text_field,
    )

    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_video_path = output_path.parent / f"{output_path.stem}_temp.mp4"

    create_temp_black_subtitle_video(
        subtitles=subtitles,
        temp_video_path=str(temp_video_path),
        font_path=font_path,
        width=width,
        height=height,
        fps=fps,
        font_size=font_size,
        min_duration=min_duration,
        chars_per_sec=chars_per_sec,
        max_duration=max_duration,
        extra_hold=extra_hold,
    )

    reencode_with_ffmpeg(
        input_video_path=str(temp_video_path),
        output_video_path=str(output_path),
        ffmpeg_bin=ffmpeg_bin,
    )

    # 임시 파일 삭제
    if temp_video_path.exists():
        temp_video_path.unlink()
        print(f"[INFO] 임시 파일 삭제: {temp_video_path}")


if __name__ == "__main__":
    # =========================
    # 입력 JSONL 경로
    # =========================
    jsonl_path = r"C:\auto-subtitle-service\ai\opencv\refine_before_after.jsonl"

    # 몇 번째 줄을 볼지
    line_index = 0

    # before_text / after_text 중 선택
    text_field = "after_text"

    # =========================
    # 출력 경로
    # =========================
    output_dir = Path(r"C:\auto-subtitle-service\data\subtitles\sample")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_video = output_dir / f"black_preview_line_{line_index:04d}_{text_field}.mp4"

    # =========================
    # 폰트 / ffmpeg
    # =========================
    font_path = r"C:/Windows/Fonts/malgun.ttf"
    ffmpeg_bin = "ffmpeg"

    create_black_subtitle_preview_from_pairs_jsonl(
        jsonl_path=jsonl_path,
        output_video_path=str(output_video),
        font_path=font_path,
        line_index=line_index,
        text_field=text_field,
        width=1280,
        height=720,
        fps=30,
        font_size=36,
        min_duration=1.8,
        chars_per_sec=9.5,
        max_duration=6.0,
        extra_hold=0.25,
        ffmpeg_bin=ffmpeg_bin,
    )