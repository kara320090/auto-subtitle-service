import random
import shutil
from pathlib import Path
from collections import defaultdict

# =========================
# 1. 경로 설정
# =========================
SRC_DIR = Path("/home/user/SWPJ3/auto-subtitle-service/data/raw/vacation/train").expanduser()
TRAIN_DIR = Path("/home/user/SWPJ3/auto-subtitle-service/ai/data/raw/vacation/train")
TEST_DIR = Path("/home/user/SWPJ3/auto-subtitle-service/ai/data/raw/vacation/test")

# =========================
# 2. 옵션 설정
# =========================
TEST_RATIO = 0.1
SEED = 42

# 옮길 파일 확장자
EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".wav", ".mp3", ".flac", ".m4a",
    ".txt", ".json",
    ".mp4", ".avi", ".mov", ".mkv"
}

# =========================
# 3. 원천데이터 -> train 이동
# =========================
def move_source_to_train():
    if not SRC_DIR.exists():
        print(f"[오류] 원본 경로가 없습니다: {SRC_DIR}")
        raise SystemExit

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    files = [
        f for f in SRC_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() in EXTENSIONS
    ]

    if not files:
        print("[오류] 원천데이터 폴더에 옮길 파일이 없습니다.")
        raise SystemExit

    print(f"[1단계] 원천데이터 -> train 이동 시작")
    print(f"원본 파일 수: {len(files)}개")

    moved = 0
    for src in files:
        rel_path = src.relative_to(SRC_DIR)
        dst = TRAIN_DIR / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved += 1

    print(f"[완료] train으로 {moved}개 파일 이동 완료\n")


# =========================
# 4. train -> test 10% 분리
#    같은 stem 파일끼리 함께 이동
# =========================
def split_train_to_test():
    if not TRAIN_DIR.exists():
        print(f"[오류] train 경로가 없습니다: {TRAIN_DIR}")
        raise SystemExit

    TEST_DIR.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)

    for f in TRAIN_DIR.rglob("*"):
        if f.is_file() and f.suffix.lower() in EXTENSIONS:
            rel = f.relative_to(TRAIN_DIR)
            key = str(rel.with_suffix(""))  # 확장자 제거한 상대경로
            groups[key].append(f)

    group_keys = list(groups.keys())

    if not group_keys:
        print("[오류] train 폴더에 분리할 파일이 없습니다.")
        raise SystemExit

    random.seed(SEED)
    test_group_count = int(len(group_keys) * TEST_RATIO)

    if test_group_count == 0:
        print("[오류] test로 보낼 묶음 수가 0개입니다. 데이터 수를 확인하세요.")
        raise SystemExit

    selected_keys = random.sample(group_keys, test_group_count)

    print(f"[2단계] train -> test 분리 시작")
    print(f"전체 묶음 수: {len(group_keys)}개")
    print(f"test로 이동할 묶음 수: {test_group_count}개")

    moved_files = 0
    for key in selected_keys:
        for src in groups[key]:
            rel_path = src.relative_to(TRAIN_DIR)
            dst = TEST_DIR / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            moved_files += 1

    print(f"[완료] test로 총 {moved_files}개 파일 이동 완료\n")


# =========================
# 5. 최종 실행
# =========================
if __name__ == "__main__":
    move_source_to_train()
    split_train_to_test()
    print("[전체 완료] 원천데이터 이동 + test 분리 완료")