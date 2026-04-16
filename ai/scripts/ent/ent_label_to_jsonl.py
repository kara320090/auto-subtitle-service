# ============================================
# 파일명: prepare_ent_dataset_by_metadata.py
#
# 역할:
# - TL_06_ent의 JSON을 기준으로 metadata.filename을 읽는다.
# - TS_06_ent에서 같은 이름의 wav를 찾아 1:1 매칭한다.
# - 매칭된 train 데이터만 train / test로 분할하여 복사한다.
# - validation 데이터는 VL_06_ent / VS_06_ent에서 그대로 매칭 복사한다.
# - 결과를 raw/ent/train, test, validation 및 label 하위 폴더에 저장한다.
# ============================================

from pathlib import Path
import json
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

TRAIN_DST = BASE_DST / "train"
TEST_DST = BASE_DST / "test"
VAL_DST = BASE_DST / "validation"

LABEL_TRAIN_DST = BASE_DST / "label" / "train"
LABEL_TEST_DST = BASE_DST / "label" / "test"
LABEL_VAL_DST = BASE_DST / "label" / "validation"

TEST_RATIO = 0.1
RANDOM_SEED = 42
OVERWRITE = True


def ensure_dirs():
    for d in [
        TRAIN_DST, TEST_DST, VAL_DST,
        LABEL_TRAIN_DST, LABEL_TEST_DST, LABEL_VAL_DST
    ]:
        d.mkdir(parents=True, exist_ok=True)


def clear_dest_dirs():
    for d in [
        TRAIN_DST, TEST_DST, VAL_DST,
        LABEL_TRAIN_DST, LABEL_TEST_DST, LABEL_VAL_DST
    ]:
        for p in d.iterdir():
            if p.is_file():
                p.unlink()


def copy_file(src: Path, dst_dir: Path):
    dst = dst_dir / src.name
    if dst.exists() and not OVERWRITE:
        return False
    shutil.copy2(src, dst)
    return True


def find_wav_by_filename(root: Path, base_filename: str):
    candidates = list(root.rglob(f"{base_filename}.wav"))
    if candidates:
        return candidates[0]
    return None


def build_pairs_from_label_dir(label_dir: Path, wav_dir: Path):
    pairs = []
    missing_filename = 0
    missing_wav = 0
    invalid_json = 0

    json_files = sorted(label_dir.rglob("*.json"))

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            invalid_json += 1
            continue

        base_filename = data.get("metadata", {}).get("filename")
        if not base_filename:
            missing_filename += 1
            continue

        wav_path = find_wav_by_filename(wav_dir, base_filename)
        if wav_path is None:
            missing_wav += 1
            continue

        pairs.append((base_filename, json_path, wav_path))

    return pairs, missing_filename, missing_wav, invalid_json


def split_pairs(pairs, test_ratio=0.1, seed=42):
    pairs = pairs[:]
    random.seed(seed)
    random.shuffle(pairs)

    test_count = max(1, int(len(pairs) * test_ratio)) if pairs else 0
    test_pairs = pairs[:test_count]
    train_pairs = pairs[test_count:]
    return train_pairs, test_pairs


def main():
    ensure_dirs()
    clear_dest_dirs()

    # -------------------------
    # train/test 분할용 쌍 구성
    # -------------------------
    train_pairs, train_missing_filename, train_missing_wav, train_invalid_json = build_pairs_from_label_dir(
        TRAIN_LABEL_DIR, TRAIN_SRC_DIR
    )

    print(f"[INFO] train 매칭 쌍 수        : {len(train_pairs)}")
    print(f"[WARN] train filename 없음    : {train_missing_filename}")
    print(f"[WARN] train wav 없음         : {train_missing_wav}")
    print(f"[WARN] train invalid json     : {train_invalid_json}")

    final_train_pairs, final_test_pairs = split_pairs(
        train_pairs,
        test_ratio=TEST_RATIO,
        seed=RANDOM_SEED
    )

    print(f"[INFO] 최종 TRAIN 쌍 수       : {len(final_train_pairs)}")
    print(f"[INFO] 최종 TEST 쌍 수        : {len(final_test_pairs)}")

    copied_train_wav = 0
    copied_train_json = 0
    for _, json_path, wav_path in final_train_pairs:
        if copy_file(wav_path, TRAIN_DST):
            copied_train_wav += 1
        if copy_file(json_path, LABEL_TRAIN_DST):
            copied_train_json += 1

    copied_test_wav = 0
    copied_test_json = 0
    for _, json_path, wav_path in final_test_pairs:
        if copy_file(wav_path, TEST_DST):
            copied_test_wav += 1
        if copy_file(json_path, LABEL_TEST_DST):
            copied_test_json += 1

    # -------------------------
    # validation 그대로 복사
    # -------------------------
    val_pairs, val_missing_filename, val_missing_wav, val_invalid_json = build_pairs_from_label_dir(
        VAL_LABEL_DIR, VAL_SRC_DIR
    )

    print(f"[INFO] val 매칭 쌍 수          : {len(val_pairs)}")
    print(f"[WARN] val filename 없음      : {val_missing_filename}")
    print(f"[WARN] val wav 없음           : {val_missing_wav}")
    print(f"[WARN] val invalid json       : {val_invalid_json}")

    copied_val_wav = 0
    copied_val_json = 0
    for _, json_path, wav_path in val_pairs:
        if copy_file(wav_path, VAL_DST):
            copied_val_wav += 1
        if copy_file(json_path, LABEL_VAL_DST):
            copied_val_json += 1

    print("\n===== 복사 완료 =====")
    print(f"TRAIN wav 복사 수       : {copied_train_wav}")
    print(f"TRAIN json 복사 수      : {copied_train_json}")
    print(f"TEST wav 복사 수        : {copied_test_wav}")
    print(f"TEST json 복사 수       : {copied_test_json}")
    print(f"VALIDATION wav 복사 수  : {copied_val_wav}")
    print(f"VALIDATION json 복사 수 : {copied_val_json}")


if __name__ == "__main__":
    main()