
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# 원본 자막 데이터 구조
@dataclass
class SubtitleItem:
    start_sec: float
    end_sec: float
    text: str


# 화면 표시용 자막 데이터 구조
@dataclass
class DisplaySubtitleItem:
    start_sec: float
    end_sec: float
    text: str
    original_start: float
    original_end: float


# JSONL 자막 로드
def load_pairs_from_jsonl_line(
    jsonl_path: str,
    line_index: int = 0,
    text_field: str = "after_text",
) -> List[SubtitleItem]:
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


# 읽기 시간 계산
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


# 표시용 자막 타임라인 생성
def build_readable_timeline(
    subtitles: List[SubtitleItem],
    min_duration: float = 1.6,
    chars_per_sec: float = 10.0,
    max_duration: float = 5.5,
    extra_hold: float = 0.2,
    max_overlap_after_next_start: float = 1.0,
) -> List[DisplaySubtitleItem]:
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


# 현재 활성 자막 추출
def get_active_subtitles(
    current_sec: float,
    subtitles: List[DisplaySubtitleItem],
    activation_epsilon: float = 0.0,
) -> List[DisplaySubtitleItem]:
    active = [
        item for item in subtitles
        if (item.start_sec - activation_epsilon) <= current_sec < (item.end_sec + activation_epsilon)
    ]
    active.sort(key=lambda x: x.start_sec)
    return active


# 자막 박스 렌더링
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
        item["rect_y1"] = rect_y1

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


# ffmpeg 환경 확인
def check_ffmpeg(ffmpeg_bin: str = "ffmpeg"):
    if shutil.which(ffmpeg_bin) is None:
        raise EnvironmentError(f"ffmpeg를 찾지 못했습니다. PATH 확인 필요: {ffmpeg_bin}")


# 임시 자막 영상 생성
def create_temp_subtitled_video_from_source(
    input_video_path: str,
    subtitles: List[SubtitleItem],
    temp_video_path: str,
    font_path: str,
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
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"입력 영상을 열 수 없습니다: {input_video_path}")

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

    display_timeline = build_readable_timeline(
        subtitles=subtitles,
        min_duration=min_duration,
        chars_per_sec=chars_per_sec,
        max_duration=max_duration,
        extra_hold=extra_hold,
        max_overlap_after_next_start=max_overlap_after_next_start,
    )

    temp_path = Path(temp_video_path)
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))

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

            if frame_idx % 100 == 0:
                print(f"진행률: {frame_idx}/{frame_count} 프레임")

            frame_idx += 1

    finally:
        cap.release()
        writer.release()

    print(f"[INFO] 임시 자막 영상 생성 완료: {temp_video_path}")


# 원본 오디오 합성
def mux_video_with_original_audio(
    subtitled_video_path: str,
    original_video_path: str,
    output_video_path: str,
    ffmpeg_bin: str = "ffmpeg",
) -> None:
    check_ffmpeg(ffmpeg_bin)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", subtitled_video_path,
        "-i", original_video_path,
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
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
            f"ffmpeg mux 실패\n"
            f"자막 영상: {subtitled_video_path}\n"
            f"원본 영상: {original_video_path}\n"
            f"출력: {output_video_path}\n"
            f"stderr:\n{result.stderr}"
        )

    print(f"[INFO] 최종 영상 생성 완료: {output_video_path}")


# 최종 자막 영상 생성
def create_subtitled_video_from_pairs_jsonl(
    input_video_path: str,
    jsonl_path: str,
    output_video_path: str,
    font_path: str,
    line_index: int = 0,
    text_field: str = "after_text",
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
    subtitles = load_pairs_from_jsonl_line(
        jsonl_path=jsonl_path,
        line_index=line_index,
        text_field=text_field,
    )

    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_video_path = output_path.parent / f"{output_path.stem}_temp_noaudio.mp4"

    create_temp_subtitled_video_from_source(
        input_video_path=input_video_path,
        subtitles=subtitles,
        temp_video_path=str(temp_video_path),
        font_path=font_path,
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

    mux_video_with_original_audio(
        subtitled_video_path=str(temp_video_path),
        original_video_path=input_video_path,
        output_video_path=str(output_path),
        ffmpeg_bin=ffmpeg_bin,
    )

    if temp_video_path.exists():
        temp_video_path.unlink()
        print(f"[INFO] 임시 파일 삭제: {temp_video_path}")


# 로컬 실행 예시
if __name__ == "__main__":
    input_video_path = r"C:\auto-subtitle-service\data\videos\sample.mp4"
    jsonl_path = r"C:\auto-subtitle-service\ai\opencv\refine_before_after.jsonl"

    line_index = 0
    text_field = "after_text"

    output_dir = Path(r"C:\auto-subtitle-service\data\subtitles\sample")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_video = output_dir / f"subtitled_line_{line_index:04d}_{text_field}.mp4"

    font_path = r"C:/Windows/Fonts/malgun.ttf"
    ffmpeg_bin = "ffmpeg"

    create_subtitled_video_from_pairs_jsonl(
        input_video_path=input_video_path,                 # 입력 영상 경로
        jsonl_path=jsonl_path,                             # 자막 JSONL 경로
        output_video_path=str(output_video),               # 출력 영상 경로
        font_path=font_path,                               # 자막 폰트 경로

        line_index=line_index,                             # JSONL 선택 줄 번호
        text_field=text_field,                             # before_text / after_text 선택

        font_size=36,                                      # 자막 글자 크기
        margin_bottom=120,                                 # 자막 세로 위치
        max_visible_items=2,                               # 동시 표시 자막 수

        min_duration=1.8,                                  # 최소 자막 표시 시간
        chars_per_sec=9.5,                                 # 글자당 읽기 속도 기준
        max_duration=6.0,                                  # 최대 자막 표시 시간
        extra_hold=0.25,                                   # 자막 추가 유지 시간
        max_overlap_after_next_start=1.0,                  # 다음 자막 시작 후 이전 자막 유지 시간

        item_gap=16,                                       # 자막 박스 간 간격
        line_spacing=8,                                    # 자막 내부 줄 간격
        activation_epsilon=1 / 60,                         # 0초대 자막 보정값

        ffmpeg_bin=ffmpeg_bin,                             # ffmpeg 실행 파일명
    )