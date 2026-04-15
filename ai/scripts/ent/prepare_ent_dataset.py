# ============================================
# 파일명: prepare_ent_dataset.py
#
# 역할:
# - 엔터테인먼트 데이터셋의 train 원천(wav) / 라벨(json) 파일을 수집한다.
# - 같은 파일명 stem 기준으로 source/label을 매칭한다.
# - 매칭된 train 데이터만 train / test로 분리하여 복사한다.
# - validation 데이터는 별도 분리 없이 그대로 validation에 복사한다.
# - 결과를 raw/ent/train, test, validation 및 label 하위 폴더에 저장한다.
# ============================================

from pathlib import Path
import shutil
import random

# =========================
# 경로 설정
# =========================
BASE_DST = Path("/home/user/SWPJ3/auto-subtitle-service/ai/data/raw/ent")

TRAIN_SRC_DIR = Path("/home/user/ent_dateSet/TS_06_ent")
TRAIN_LABEL_DIR = Path("/home/user/ent_dateSet/TL_06_ent")
VAL_SRC_DIR = Path("/home/user/ent_dateSet/VS_06_ent")
VAL_LABEL_DIR = Path("/home/user/ent_dateSet/VL_06_ent")

# =========================
# 옵션 설정
# =========================
TEST_RATIO = 0.1
RANDOM_SEED = 42
OVERWRITE = False

# =========================
# 목적지 폴더
# =========================
TRAIN_DST = BASE_DST / "train"
TEST_DST = BASE_DST / "test"
VAL_DST = BASE_DST / "validation"

LABEL_TRAIN_DST = BASE_DST / "label" / "train"
LABEL_TEST_DST = BASE_DST / "label" / "test"
LABEL_VAL_DST = BASE_DST / "label" / "validation"


def ensure_dirs():
    for d in [
        TRAIN_DST, TEST_DST, VAL_DST,
        LABEL_TRAIN_DST, LABEL_TEST_DST, LABEL_VAL_DST
    ]:
        d.mkdir(parents=True, exist_ok=True)


def collect_files(root: Path, suffix: str | None = None):
    """
    root 아래의 파일을 stem 기준으로 수집
    suffix가 있으면 해당 확장자만 수집
    예: abc.wav -> key: abc
    """
    file_map = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue

        if suffix is not None and p.suffix.lower() != suffix.lower():
            continue

        file_map.setdefault(p.stem, p)

    return file_map


def copy_file(src: Path, dst_dir: Path):
    dst = dst_dir / src.name
    if dst.exists() and not OVERWRITE:
        return False
    shutil.copy2(src, dst)
    return True


def split_train_test(common_stems, test_ratio=0.1, seed=42):
    stems = sorted(list(common_stems))
    random.seed(seed)
    random.shuffle(stems)

    test_count = max(1, int(len(stems) * test_ratio)) if stems else 0
    test_stems = set(stems[:test_count])
    train_stems = set(stems[test_count:])
    return train_stems, test_stems


def main():
    ensure_dirs()

    # -------------------------
    # 1) train 데이터 수집
    # -------------------------
    train_src_map = collect_files(TRAIN_SRC_DIR, suffix=".wav")
    train_label_map = collect_files(TRAIN_LABEL_DIR, suffix=".json")

    train_common_stems = set(train_src_map.keys()) & set(train_label_map.keys())
    train_only_src = set(train_src_map.keys()) - set(train_label_map.keys())
    train_only_label = set(train_label_map.keys()) - set(train_src_map.keys())

    print(f"[INFO] TRAIN source(wav) 파일 수 : {len(train_src_map)}")
    print(f"[INFO] TRAIN label(json) 파일 수: {len(train_label_map)}")
    print(f"[INFO] TRAIN 매칭 파일 수      : {len(train_common_stems)}")
    print(f"[WARN] TRAIN wav만 있는 파일 수 : {len(train_only_src)}")
    print(f"[WARN] TRAIN json만 있는 파일 수: {len(train_only_label)}")

    train_stems, test_stems = split_train_test(
        train_common_stems,
        test_ratio=TEST_RATIO,
        seed=RANDOM_SEED
    )

    print(f"[INFO] 최종 TRAIN 개수: {len(train_stems)}")
    print(f"[INFO] 최종 TEST 개수 : {len(test_stems)}")

    copied_train_src = 0
    copied_train_label = 0
    for stem in sorted(train_stems):
        if copy_file(train_src_map[stem], TRAIN_DST):
            copied_train_src += 1
        if copy_file(train_label_map[stem], LABEL_TRAIN_DST):
            copied_train_label += 1

    copied_test_src = 0
    copied_test_label = 0
    for stem in sorted(test_stems):
        if copy_file(train_src_map[stem], TEST_DST):
            copied_test_src += 1
        if copy_file(train_label_map[stem], LABEL_TEST_DST):
            copied_test_label += 1

    # -------------------------
    # 2) validation 데이터 복사
    # -------------------------
    val_src_map = collect_files(VAL_SRC_DIR, suffix=".wav")
    val_label_map = collect_files(VAL_LABEL_DIR, suffix=".json")

    val_common_stems = set(val_src_map.keys()) & set(val_label_map.keys())
    val_only_src = set(val_src_map.keys()) - set(val_label_map.keys())
    val_only_label = set(val_label_map.keys()) - set(val_src_map.keys())

    print(f"[INFO] VAL source(wav) 파일 수   : {len(val_src_map)}")
    print(f"[INFO] VAL label(json) 파일 수  : {len(val_label_map)}")
    print(f"[INFO] VAL 매칭 파일 수        : {len(val_common_stems)}")
    print(f"[WARN] VAL wav만 있는 파일 수   : {len(val_only_src)}")
    print(f"[WARN] VAL json만 있는 파일 수  : {len(val_only_label)}")

    copied_val_src = 0
    copied_val_label = 0
    for stem in sorted(val_common_stems):
        if copy_file(val_src_map[stem], VAL_DST):
            copied_val_src += 1
        if copy_file(val_label_map[stem], LABEL_VAL_DST):
            copied_val_label += 1

    print("\n===== 복사 완료 =====")
    print(f"TRAIN wav 복사 수       : {copied_train_src}")
    print(f"TRAIN json 복사 수      : {copied_train_label}")
    print(f"TEST wav 복사 수        : {copied_test_src}")
    print(f"TEST json 복사 수       : {copied_test_label}")
    print(f"VALIDATION wav 복사 수  : {copied_val_src}")
    print(f"VALIDATION json 복사 수 : {copied_val_label}")


if __name__ == "__main__":
    main()