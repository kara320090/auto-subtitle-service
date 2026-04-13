#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================
# 파일명: wav_train_move_test.py
#
# 역할:
# - /home/user/game_dataSet/TS_02_game 폴더 안의 모든 .wav 파일을 수집한다.
# - 전체 wav 파일을 random_state=42 기준으로 섞는다.
# - 그중 10%는 test 데이터셋으로 분리한다.
# - 나머지 90%는 train 데이터셋으로 분리한다.
# - test wav 파일은
#   /home/user/SWPJ3/auto-subtitle-service/ai/data/raw/game/test
#   경로로 이동한다.
# - train wav 파일은
#   /home/user/SWPJ3/auto-subtitle-service/ai/data/raw/game/train
#   경로로 이동한다.
#
# 주의:
# - 이 스크립트는 wav 파일을 "복사"가 아니라 "이동"한다.
# - 실행 후 원본 폴더(/home/user/game_dataSet/TS_02_game)의 wav는 사라진다.
# - 원본을 보존하고 싶다면 shutil.move()를 shutil.copy2()로 바꾸면 된다.
# ============================================

import random
import shutil
from pathlib import Path

# ==========================================
# 1) 사용자 설정
# ==========================================
SOURCE_DIR = Path("/home/user/game_dataSet/TS_02_game")

TRAIN_DIR = Path("/home/user/SWPJ3/auto-subtitle-service/ai/data/raw/game/train")
TEST_DIR = Path("/home/user/SWPJ3/auto-subtitle-service/ai/data/raw/game/test")

TEST_RATIO = 0.10
SEED = 42


def safe_print(msg: str):
    try:
        print(msg)
    except Exception:
        print(str(msg))


def check_paths():
    """
    필수 경로를 확인하고, 출력 폴더가 없으면 생성한다.
    """
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"원본 폴더가 존재하지 않습니다: {SOURCE_DIR}")

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)


def collect_wav_files(root: Path):
    """
    원본 폴더 내부의 모든 wav 파일을 재귀적으로 수집한다.
    """
    wav_files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"]
    return sorted(wav_files)


def split_files(wav_files, test_ratio: float, seed: int):
    """
    wav 파일 목록을 랜덤 시드 고정으로 섞고
    test / train 으로 분리한다.
    """
    if not wav_files:
        raise ValueError("분리할 wav 파일이 없습니다.")

    rng = random.Random(seed)
    shuffled = wav_files[:]
    rng.shuffle(shuffled)

    test_count = int(len(shuffled) * test_ratio)
    if test_count == 0 and len(shuffled) > 0:
        test_count = 1

    test_files = shuffled[:test_count]
    train_files = shuffled[test_count:]

    return train_files, test_files


def move_file_safe(src: Path, dst_dir: Path):
    """
    파일을 목적지 폴더로 이동한다.
    같은 이름 파일이 이미 있으면 뒤에 __dupN 을 붙여 저장한다.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst_path = dst_dir / src.name

    if not dst_path.exists():
        shutil.move(str(src), str(dst_path))
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


def move_split_files(train_files, test_files):
    """
    train/test로 나뉜 wav 파일들을 각각 지정 경로로 이동한다.
    """
    moved_train = 0
    moved_test = 0
    failed = 0

    for wav_path in train_files:
        try:
            new_path = move_file_safe(wav_path, TRAIN_DIR)
            moved_train += 1
            safe_print(f"[TRAIN 이동] {wav_path} -> {new_path}")
        except Exception as e:
            failed += 1
            safe_print(f"[실패] TRAIN 이동 실패: {wav_path} / {e}")

    for wav_path in test_files:
        try:
            new_path = move_file_safe(wav_path, TEST_DIR)
            moved_test += 1
            safe_print(f"[TEST 이동] {wav_path} -> {new_path}")
        except Exception as e:
            failed += 1
            safe_print(f"[실패] TEST 이동 실패: {wav_path} / {e}")

    return moved_train, moved_test, failed


def main():
    safe_print("=" * 70)
    safe_print("game wav train/test 분리 시작")
    safe_print("=" * 70)

    check_paths()

    wav_files = collect_wav_files(SOURCE_DIR)
    total_count = len(wav_files)

    safe_print(f"[전체 wav 수] {total_count}")

    train_files, test_files = split_files(wav_files, TEST_RATIO, SEED)

    safe_print(f"[분리 결과] train={len(train_files)} / test={len(test_files)}")

    moved_train, moved_test, failed = move_split_files(train_files, test_files)

    safe_print("")
    safe_print("=" * 70)
    safe_print("작업 완료")
    safe_print("=" * 70)
    safe_print(f"전체 wav 수        : {total_count}")
    safe_print(f"최종 train 수      : {len(train_files)}")
    safe_print(f"최종 test 수       : {len(test_files)}")
    safe_print(f"train 이동 성공 수 : {moved_train}")
    safe_print(f"test 이동 성공 수  : {moved_test}")
    safe_print(f"이동 실패 수       : {failed}")


if __name__ == "__main__":
    main()