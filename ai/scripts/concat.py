# 파일명 예시: make_game_concat_chunks.py

import subprocess
from pathlib import Path
import math
import shutil
import sys

# =========================
# 설정
# =========================
INPUT_DIR = Path("/home/data/aihub_71699/game_dataSet/VS_02_game")
OUTPUT_DIR = Path("/home/user/SWPJ3/auto-subtitle-service/ai/data/results/concat")

TARGETS = [
    {"name": "10min", "seconds": 10 * 60},
    {"name": "20min", "seconds": 20 * 60},
]

SILENCE_DURATION = 1  # 초

# 해상도 / fps는 첫 번째 mp4를 기준으로 맞춤
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30

# wav 추출 설정
WAV_SAMPLE_RATE = 16000
WAV_CHANNELS = 1

# =========================
# 유틸
# =========================
def run_cmd(cmd):
    print("[RUN]", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)

def ffprobe_duration(file_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())

def get_video_info(file_path: Path):
    """
    첫 번째 비디오의 width, height, fps를 가져옵니다.
    실패 시 기본값 사용.
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(file_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = [x.strip() for x in result.stdout.strip().splitlines() if x.strip()]
        width = int(lines[0])
        height = int(lines[1])

        fps_raw = lines[2]  # 예: 30000/1001
        if "/" in fps_raw:
            a, b = fps_raw.split("/")
            fps = float(a) / float(b)
        else:
            fps = float(fps_raw)

        return width, height, fps
    except Exception:
        print("[WARN] 비디오 정보를 읽지 못해 기본값을 사용합니다.")
        return DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS

def make_silence_video(output_path: Path, width: int, height: int, fps: float, duration: int):
    """
    검은 화면 + 무음 1초짜리 mp4 생성
    """
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=black:s={width}x{height}:r={fps}",
        "-f", "lavfi",
        "-i", "anullsrc=r=16000:cl=mono",
        "-t", str(duration),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        str(output_path)
    ]
    run_cmd(cmd)

def normalize_video(input_path: Path, output_path: Path, width: int, height: int, fps: float):
    """
    concat 안정성을 위해 비디오/오디오 포맷을 통일
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", f"scale={width}:{height},fps={fps}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-ar", "16000",
        "-ac", "1",
        str(output_path)
    ]
    run_cmd(cmd)

def build_concat_list(files, concat_txt_path: Path):
    with concat_txt_path.open("w", encoding="utf-8") as f:
        for fp in files:
            f.write(f"file '{fp.resolve()}'\n")

def concat_mp4s(concat_txt_path: Path, output_mp4: Path):
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_txt_path),
        "-c", "copy",
        str(output_mp4)
    ]
    run_cmd(cmd)

def extract_wav(input_mp4: Path, output_wav: Path):
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_mp4),
        "-vn",
        "-ac", str(WAV_CHANNELS),
        "-ar", str(WAV_SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        str(output_wav)
    ]
    run_cmd(cmd)

def pick_files_for_target(mp4_files, target_seconds, silence_duration):
    """
    이름순으로 파일을 더해 가면서
    파일 사이에 silence_duration 초를 넣었을 때
    target_seconds를 넘기지 않는 최대 묶음을 선택합니다.
    """
    selected = []
    total = 0.0

    for i, mp4 in enumerate(mp4_files):
        dur = ffprobe_duration(mp4)
        add_time = dur
        if len(selected) > 0:
            add_time += silence_duration

        if total + add_time > target_seconds:
            break

        selected.append(mp4)
        total += add_time

    return selected, total

# =========================
# 메인
# =========================
def main():
    if not INPUT_DIR.exists():
        print(f"[ERROR] 입력 폴더가 없습니다: {INPUT_DIR}")
        sys.exit(1)

    if shutil.which("ffmpeg") is None:
        print("[ERROR] ffmpeg를 찾을 수 없습니다.")
        sys.exit(1)

    if shutil.which("ffprobe") is None:
        print("[ERROR] ffprobe를 찾을 수 없습니다.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mp4_files = sorted(INPUT_DIR.glob("*.mp4"))
    if not mp4_files:
        print(f"[ERROR] mp4 파일이 없습니다: {INPUT_DIR}")
        sys.exit(1)

    print(f"[INFO] 발견한 mp4 개수: {len(mp4_files)}")

    width, height, fps = get_video_info(mp4_files[0])
    print(f"[INFO] 기준 영상 정보: {width}x{height}, fps={fps:.3f}")

    silence_mp4 = OUTPUT_DIR / "silence_1s.mp4"
    make_silence_video(silence_mp4, width, height, fps, SILENCE_DURATION)

    normalized_dir = OUTPUT_DIR / "normalized"
    normalized_dir.mkdir(exist_ok=True)

    normalized_map = {}
    print("[INFO] mp4 정규화 시작")
    for i, src in enumerate(mp4_files, start=1):
        dst = normalized_dir / f"{src.stem}_norm.mp4"
        normalize_video(src, dst, width, height, fps)
        normalized_map[src] = dst
        print(f"[{i}/{len(mp4_files)}] 정규화 완료: {src.name}")

    for target in TARGETS:
        target_name = target["name"]
        target_seconds = target["seconds"]

        selected_srcs, total_sec = pick_files_for_target(mp4_files, target_seconds, SILENCE_DURATION)

        if not selected_srcs:
            print(f"[WARN] {target_name}: 선택된 파일이 없습니다.")
            continue

        print(f"[INFO] {target_name}: 선택 파일 수 = {len(selected_srcs)}, 총 길이 = {total_sec:.2f}초")

        concat_sequence = []
        for idx, src in enumerate(selected_srcs):
            concat_sequence.append(normalized_map[src])
            if idx < len(selected_srcs) - 1:
                concat_sequence.append(silence_mp4)

        concat_txt = OUTPUT_DIR / f"concat_list_{target_name}.txt"
        output_mp4 = OUTPUT_DIR / f"game_concat_{target_name}.mp4"
        output_wav = OUTPUT_DIR / f"game_concat_{target_name}.wav"

        build_concat_list(concat_sequence, concat_txt)
        concat_mp4s(concat_txt, output_mp4)
        extract_wav(output_mp4, output_wav)

        print(f"[DONE] mp4 생성 완료: {output_mp4}")
        print(f"[DONE] wav 추출 완료: {output_wav}")

    print("[ALL DONE] 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()