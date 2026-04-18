import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

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
    - "before_text"
    - "after_text"
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
        start = max(0.0, float(item.get("start", 0.0)))
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
    max_duration: float = 5.5,
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
    extra_hold: float = 0.2,
    max_overlap_after_next_start: float = 1.0,
) -> List[DisplaySubtitleItem]:
    """
    원본 segment 시간을 그대로 쓰지 않고,
    가독성을 위해 표시 종료 시점을 조금 늘린 타임라인 생성.

    추가로, 이전 자막이 다음 자막 시작 이후로 너무 오래 남지 않도록
    겹침 시간을 제한한다.
    """
    result: List[DisplaySubtitleItem] = []

    for i, s in enumerate(subtitles):
        readable_duration = estimate_reading_duration(
            s.text,
            min_duration=min_duration,
            chars_per_sec=chars_per_sec,
            max_duration=max_duration,
        )
        original_duration = max(0.0, s.end_sec - s.start_sec)
        final_duration = max(original_duration, readable_duration) + extra_hold
        desired_end = s.start_sec + final_duration

        if i + 1 < len(subtitles):
            next_start = subtitles[i + 1].start_sec
            desired_end = min(desired_end, next_start + max_overlap_after_next_start)

        result.append(
            DisplaySubtitleItem(
                start_sec=s.start_sec,
                end_sec=desired_end,
                text=s.text,
                original_start=s.start_sec,
                original_end=s.end_sec,
            )
        )

    return result


def get_active_subtitles(
    current_sec: float,
    subtitles: List[DisplaySubtitleItem],
    activation_epsilon: float = 0.0,
) -> List[DisplaySubtitleItem]:
    """
    0초대/프레임 경계 부근 자막이 덜 튀도록 epsilon을 둔다.
    종료 판정은 half-open interval [start, end) 로 처리한다.
    """
    active = [
        item for item in subtitles
        if (item.start_sec - activation_epsilon) <= current_sec < (item.end_sec + activation_epsilon)
    ]
    active.sort(key=lambda x: x.start_sec)
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
    bg_alpha: float = 0.5,
    stroke_width: int = 2,
    stroke_fill: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    여러 active subtitle을 하나의 multiline 문자열로 합치지 않고,
    자막 아이템 단위로 독립 박스를 쌓아 올려서 그린다.

    newest item = 가장 아래
    older item  = 그 위
    """
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
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            line_sizes.append((w, h))
            block_w = max(block_w, w)
            block_h += h

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

    # 1차: 박스 배경 위치 계산
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    current_bottom = img_h - margin_bottom

    # newest -> bottom slot
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
        item["rect_y1"] = rect_y1

        overlay_draw.rounded_rectangle(
            [rect_x1, rect_y1, rect_x2, rect_y2],
            radius=12,
            fill=(*bg_color, int(255 * bg_alpha)),
        )

        current_bottom = rect_y1 - item_gap

    pil_img = Image.alpha_composite(pil_img, overlay)
    draw = ImageDraw.Draw(pil_img)

    # 2차: 텍스트 그리기
    for item in reversed(prepared):
        current_y = item["y"]

        for line, (w, h) in zip(item["lines"], item["line_sizes"]):
            line_x = (img_w - w) // 2
            draw.text(
                (line_x, current_y),
                line,
                font=font,
                fill=text_color,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
            )
            current_y += h + line_spacing

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
    extra_hold: float = 0.2,
    max_overlap_after_next_start: float = 1.0,
    margin_bottom: int = 60,
    item_gap: int = 16,
    line_spacing: int = 8,
    max_visible_items: int = 2,
    activation_epsilon: Optional[float] = None,
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
        extra_hold=extra_hold,
        max_overlap_after_next_start=max_overlap_after_next_start,
    )

    total_duration = max(item.end_sec for item in display_timeline) + 1.0
    total_frames = int(math.ceil(total_duration * fps))

    if activation_epsilon is None:
        activation_epsilon = 0.5 / fps

    temp_path = Path(temp_video_path)
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"임시 비디오를 생성할 수 없습니다: {temp_video_path}")

    try:
        for frame_idx in range(total_frames):
            current_sec = frame_idx / fps

            frame = np.zeros((height, width, 3), dtype=np.uint8)

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

            if frame_idx % 100 == 0:
                print(f"진행률: {frame_idx}/{total_frames} 프레임")

    finally:
        writer.release()

    print(f"[INFO] 임시 영상 생성 완료: {temp_path}")


def reencode_with_ffmpeg(
    input_video_path: str,
    output_video_path: str,
    ffmpeg_bin: str = "ffmpeg",
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
    max_overlap_after_next_start: float = 1.0,
    margin_bottom: int = 60,
    item_gap: int = 16,
    line_spacing: int = 8,
    max_visible_items: int = 2,
    activation_epsilon: Optional[float] = None,
    ffmpeg_bin: str = "ffmpeg",
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
        max_overlap_after_next_start=max_overlap_after_next_start,
        margin_bottom=margin_bottom,
        item_gap=item_gap,
        line_spacing=line_spacing,
        max_visible_items=max_visible_items,
        activation_epsilon=activation_epsilon,
    )

    reencode_with_ffmpeg(
        input_video_path=str(temp_video_path),
        output_video_path=str(output_path),
        ffmpeg_bin=ffmpeg_bin,
    )

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
        max_overlap_after_next_start=1.0,   # 다음 자막 시작 후 최대 1초까지만 겹침
        margin_bottom=120,                  # 자막을 더 위로 올리고 싶으면 이 값 증가
        item_gap=16,                        # 자막 박스 간 간격
        line_spacing=8,                     # 한 자막 내부 줄 간격
        max_visible_items=2,                # 동시에 최대 몇 개 자막 박스를 보일지
        activation_epsilon=1 / 60,          # 0초대 자막 안정화용
        ffmpeg_bin=ffmpeg_bin,
    )