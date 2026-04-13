#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================
# 파일명: move_game_json_by_wav_split.py
#
# 역할:
# - 이미 분리되어 있는 game wav 파일(train/test/validation)을 기준으로
#   대응되는 json 파일을 각각 train/test/validation 라벨 폴더로 이동한다.
# - train/test wav 기준 json 원본은 TL_02_game 폴더에서 찾는다.
# - validation wav 기준 json 원본은 VL_02_game 폴더에서 찾는다.
# - wav 파일과 json 파일은 파일명에서 확장자를 제외한 stem이 같으면 대응된다고 본다.
# - 예:
#   - raw/game/train/MYB000123.wav  -> label/game/train/MYB000123.json
#   - raw/game/test/MYB000456.wav   -> label/game/test/MYB000456.json
#   - raw/game/validation/MYB000789.wav -> label/game/validation/MYB000789.json
#
# 주의:
# - 이 스크립트는 json 파일을 "복사"가 아니라 "이동"한다.
# - 원본 json을 보존하고 싶으면 shutil.move() 대신 shutil.copy2()로 바꾸면 된다.
# - 대응되는 json 파일이 없으면 경고만 출력하고 계속 진행한다.
# ============================================

import shutil
from pathlib import Path

# ==========================================
# 1) 사용자 설정
# ==========================================
# wav split 결과 폴더
RAW_TRAIN_DIR = Path("/home/user/SWPJ3/auto-subtitle-service/ai/data/raw/game/train")
RAW_TEST_DIR = Path("/home/user/SWPJ3/auto-subtitle-service/ai/data/raw/game/test")
RAW_VALID_DIR = Path("/home/user/SWPJ3/auto-subtitle-service/ai/data/raw/game/validation")

# json 원본 폴더
TL_JSON_DIR = Path("/home/user/game_dataSet/TL_02_game")
VL_JSON_DIR = Path("/home/user/game_dataSet/VL_02_game")

# json 목적지 폴더
LABEL_TRAIN_DIR = Path("/home/user/SWPJ3/auto-subtitle-service/ai/data/label/game/train")
LABEL_TEST_DIR = Path("/home/user/SWPJ3/auto-subtitle-service/ai/data/label/game/test")
LABEL_VALID_DIR = Path("/home/user/SWPJ3/auto-subtitle-service/ai/data/label/game/validation")


def safe_print(msg: str):
    try:
        print(msg)
    except Exception:
        print(str(msg))


def check_paths():
    """
    필수 경로 존재 여부를 확인하고,
    목적지 라벨 폴더가 없으면 생성합니다.
    """
    required_dirs = [
        RAW_TRAIN_DIR, RAW_TEST_DIR, RAW_VALID_DIR,
        TL_JSON_DIR, VL_JSON_DIR,
    ]

    for path in required_dirs:
        if not path.exists():
            raise FileNotFoundError(f"필수 경로가 존재하지 않습니다: {path}")

    LABEL_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_TEST_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_VALID_DIR.mkdir(parents=True, exist_ok=True)


def collect_wav_stems(wav_dir: Path):
    """
    지정 폴더 안의 모든 wav 파일 stem을 수집합니다.
    """
    return {
        p.stem
        for p in wav_dir.rglob("*")
        if p.is_file() and p.suffix.lower() == ".wav"
    }


def build_json_index(json_dir: Path):
    """
    json 원본 폴더에서 stem -> json 경로 딕셔너리를 만듭니다.
    같은 stem이 여러 개면 첫 번째만 쓰지 않고 경고를 위해 리스트로 모읍니다.
    """
    index = {}

    for json_path in json_dir.rglob("*.json"):
        index.setdefault(json_path.stem, []).append(json_path)

    return index


def move_file_safe(src: Path, dst_dir: Path):
    """
    파일을 목적지 폴더로 이동합니다.
    같은 이름 파일이 이미 있으면 __dupN을 붙여 저장합니다.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst_path = dst_dir / src.name

    if not dst_path.exists():
        shutil.copy2(str(src), str(dst_path))
        return dst_path

    stem = src.stem
    suffix = src.suffix
    counter = 1

    while True:
        candidate = dst_dir / f"{stem}__dup{counter}{suffix}"
        if not candidate.exists():
            shutil.move(str(src), str(candidate))
            return candidate
        counter += 1


def move_jsons_by_stems(stems: set, json_index: dict, dst_dir: Path, split_name: str):
    """
    wav stem 집합을 기준으로 대응 json을 목적지 폴더로 이동합니다.
    """
    moved = 0
    missing = 0
    duplicated = 0
    failed = 0

    for stem in sorted(stems):
        json_candidates = json_index.get(stem, [])

        if not json_candidates:
            missing += 1
            safe_print(f"[{split_name}] 대응 json 없음: {stem}")
            continue

        if len(json_candidates) > 1:
            duplicated += 1
            safe_print(f"[{split_name}] 같은 stem의 json이 여러 개 있습니다: {stem}")
            for p in json_candidates:
                safe_print(f"    - {p}")

        # 여러 개여도 첫 번째 파일 하나만 이동
        src_json = json_candidates[0]

        try:
            new_path = move_file_safe(src_json, dst_dir)
            moved += 1
            safe_print(f"[{split_name}] 이동: {src_json} -> {new_path}")
        except Exception as e:
            failed += 1
            safe_print(f"[{split_name}] 이동 실패: {src_json} / {e}")

    return moved, missing, duplicated, failed


def main():
    safe_print("=" * 70)
    safe_print("game json train/test/validation 분리 시작")
    safe_print("=" * 70)

    check_paths()

    # wav 기준 stem 수집
    train_stems = collect_wav_stems(RAW_TRAIN_DIR)
    test_stems = collect_wav_stems(RAW_TEST_DIR)
    valid_stems = collect_wav_stems(RAW_VALID_DIR)

    safe_print(f"[wav 개수] train={len(train_stems)} / test={len(test_stems)} / validation={len(valid_stems)}")

    # json 원본 인덱스 생성
    tl_json_index = build_json_index(TL_JSON_DIR)
    vl_json_index = build_json_index(VL_JSON_DIR)

    # train/test는 TL 원본에서 이동
    train_result = move_jsons_by_stems(train_stems, tl_json_index, LABEL_TRAIN_DIR, "TRAIN")
    test_result = move_jsons_by_stems(test_stems, tl_json_index, LABEL_TEST_DIR, "TEST")

    # validation은 VL 원본에서 이동
    valid_result = move_jsons_by_stems(valid_stems, vl_json_index, LABEL_VALID_DIR, "VALIDATION")

    safe_print("")
    safe_print("=" * 70)
    safe_print("작업 완료")
    safe_print("=" * 70)

    safe_print("[TRAIN]")
    safe_print(f"이동 성공 수        : {train_result[0]}")
    safe_print(f"대응 json 없음 수   : {train_result[1]}")
    safe_print(f"중복 json stem 수   : {train_result[2]}")
    safe_print(f"이동 실패 수        : {train_result[3]}")

    safe_print("")
    safe_print("[TEST]")
    safe_print(f"이동 성공 수        : {test_result[0]}")
    safe_print(f"대응 json 없음 수   : {test_result[1]}")
    safe_print(f"중복 json stem 수   : {test_result[2]}")
    safe_print(f"이동 실패 수        : {test_result[3]}")

    safe_print("")
    safe_print("[VALIDATION]")
    safe_print(f"이동 성공 수        : {valid_result[0]}")
    safe_print(f"대응 json 없음 수   : {valid_result[1]}")
    safe_print(f"중복 json stem 수   : {valid_result[2]}")
    safe_print(f"이동 실패 수        : {valid_result[3]}")

    safe_print("")
    safe_print("[출력 경로]")
    safe_print(f"train json      : {LABEL_TRAIN_DIR}")
    safe_print(f"test json       : {LABEL_TEST_DIR}")
    safe_print(f"validation json : {LABEL_VALID_DIR}")


if __name__ == "__main__":
    main()