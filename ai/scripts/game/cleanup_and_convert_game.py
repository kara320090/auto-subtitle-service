#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================
# 파일명: cleanup_and_convert_game.py
#
# 역할:
# - 지정한 ~game 폴더 내부를 재귀적으로 탐색한다.
# - 폴더 안의 모든 .mp4 파일과 .json 파일을 찾는다.
# - mp4 파일과 이름(stem)이 같은 json 파일만 남긴다.
# - mp4 파일과 이름이 맞지 않는 json 파일은 삭제한다.
# - 남아 있는 모든 mp4 파일을 wav 파일로 변환한다.
# - wav 파일은 같은 폴더에 저장하거나, 별도 폴더에 저장할 수 있다.
# - 실행 후 전체 mp4 수, json 수, 삭제된 json 수, wav 변환 결과를 출력한다.
#
# 주의:
# - 이 스크립트는 mp4와 매칭되지 않는 json 파일을 실제로 삭제한다.
# - 원본 mp4 파일은 삭제하지 않는다.
# - ffmpeg가 설치되어 있어야 한다.
#
# 기본 처리 대상 경로:
# - /home/user/game_dataSet/TS_02_game
#
# wav 저장 방식:
# - SAVE_WAV_IN_SAME_FOLDER = True  -> mp4와 같은 위치에 wav 저장
# - SAVE_WAV_IN_SAME_FOLDER = False -> 별도 WAV_OUTPUT_DIR에 저장
# ============================================

import shutil
import subprocess
from pathlib import Path

# ==========================================
# 1) 사용자 설정
# ==========================================
GAME_DIR = Path("/home/user/game_dataSet/VS_02_game")
FFMPEG_BIN = "ffmpeg"

SAVE_WAV_IN_SAME_FOLDER = True
WAV_OUTPUT_DIR = Path("/home/user/game_dataSet/VS_02_game")

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
WAV_CODEC = "pcm_s16le"

OVERWRITE_WAV = False


def safe_print(msg: str):
    try:
        print(msg)
    except Exception:
        print(str(msg))


def check_paths():
    if not GAME_DIR.exists():
        raise FileNotFoundError(f"폴더가 존재하지 않습니다: {GAME_DIR}")

    if shutil.which(FFMPEG_BIN) is None:
        raise EnvironmentError(
            f"ffmpeg를 찾을 수 없습니다. 설치 후 PATH에 등록하세요: {FFMPEG_BIN}"
        )


def collect_files_by_stem(root: Path, ext: str):
    result = {}

    for f in root.rglob(f"*{ext}"):
        if f.is_file():
            result.setdefault(f.stem, []).append(f)

    return result


def report_duplicates(file_map: dict, ext_name: str):
    dup_count = 0

    for stem, paths in file_map.items():
        if len(paths) > 1:
            dup_count += 1
            safe_print(f"[경고] 같은 이름의 {ext_name} 파일이 여러 개 있습니다: {stem}")
            for p in paths:
                safe_print(f"       - {p}")

    return dup_count


def delete_unmatched_jsons(json_map: dict, valid_mp4_stems: set):
    deleted = 0
    failed = 0

    for stem, json_paths in json_map.items():
        if stem not in valid_mp4_stems:
            for json_path in json_paths:
                try:
                    json_path.unlink()
                    deleted += 1
                    safe_print(f"[삭제] {json_path}")
                except Exception as e:
                    failed += 1
                    safe_print(f"[실패] json 삭제 실패: {json_path} / {e}")

    return deleted, failed


def get_wav_output_path(mp4_path: Path) -> Path:
    if SAVE_WAV_IN_SAME_FOLDER:
        return mp4_path.with_suffix(".wav")

    WAV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return WAV_OUTPUT_DIR / f"{mp4_path.stem}.wav"


def convert_mp4_to_wav(mp4_path: Path, wav_path: Path):
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    if wav_path.exists() and not OVERWRITE_WAV:
        return "skip"

    cmd = [
        FFMPEG_BIN,
        "-hide_banner",
        "-loglevel", "error",
        "-y" if OVERWRITE_WAV else "-n",
        "-i", str(mp4_path),
        "-vn",
        "-ac", str(TARGET_CHANNELS),
        "-ar", str(TARGET_SAMPLE_RATE),
        "-c:a", WAV_CODEC,
        str(wav_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )

    if result.returncode != 0:
        stderr_text = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"ffmpeg 변환 실패\n"
            f"입력: {mp4_path}\n"
            f"출력: {wav_path}\n"
            f"stderr:\n{stderr_text}"
        )

    return "converted"


def main():
    safe_print("=" * 70)
    safe_print("mp4-json 정리 및 mp4->wav 변환 시작")
    safe_print("=" * 70)

    check_paths()

    mp4_map = collect_files_by_stem(GAME_DIR, ".mp4")
    json_map = collect_files_by_stem(GAME_DIR, ".json")

    total_mp4 = sum(len(v) for v in mp4_map.values())
    total_json = sum(len(v) for v in json_map.values())

    safe_print(f"[개수] mp4 파일 수  : {total_mp4}")
    safe_print(f"[개수] json 파일 수 : {total_json}")

    mp4_dup = report_duplicates(mp4_map, "mp4")
    json_dup = report_duplicates(json_map, "json")

    if mp4_dup > 0:
        safe_print(f"[주의] 같은 stem의 mp4가 {mp4_dup}건 있습니다.")
        safe_print("       이런 경우 어떤 json이 대응되는지 애매할 수 있습니다.")

    if json_dup > 0:
        safe_print(f"[주의] 같은 stem의 json이 {json_dup}건 있습니다.")
        safe_print("       mp4 없는 json 정리 시 모두 삭제되거나 모두 유지될 수 있습니다.")

    valid_mp4_stems = set(mp4_map.keys())
    deleted_json, failed_delete = delete_unmatched_jsons(json_map, valid_mp4_stems)

    converted = 0
    skipped = 0
    failed_convert = 0

    for _, mp4_paths in mp4_map.items():
        for mp4_path in mp4_paths:
            wav_path = get_wav_output_path(mp4_path)

            try:
                status = convert_mp4_to_wav(mp4_path, wav_path)

                if status == "converted":
                    converted += 1
                    safe_print(f"[변환] {mp4_path} -> {wav_path}")
                elif status == "skip":
                    skipped += 1
                    safe_print(f"[건너뜀] 이미 존재: {wav_path}")

            except Exception as e:
                failed_convert += 1
                safe_print(f"[실패] wav 변환 실패: {mp4_path}")
                safe_print(str(e))

    safe_print("")
    safe_print("=" * 70)
    safe_print("작업 완료")
    safe_print("=" * 70)
    safe_print(f"전체 mp4 수              : {total_mp4}")
    safe_print(f"전체 json 수             : {total_json}")
    safe_print(f"삭제된 json 수           : {deleted_json}")
    safe_print(f"json 삭제 실패 수        : {failed_delete}")
    safe_print(f"wav 변환 성공 수         : {converted}")
    safe_print(f"wav 이미 존재하여 skip 수: {skipped}")
    safe_print(f"wav 변환 실패 수         : {failed_convert}")


if __name__ == "__main__":
    main()