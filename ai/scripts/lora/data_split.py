import random
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# ==========================================
# 1) 사용자 설정
# ==========================================
TRAIN_RAW_DIR = Path(
    r"D:\data\017.한국어 텍스트-비디오-사운드 데이터\3.개방데이터\1.데이터\Training\01.원천데이터"
)
TRAIN_LABEL_DIR = Path(
    r"D:\data\017.한국어 텍스트-비디오-사운드 데이터\3.개방데이터\1.데이터\Training\02.라벨링데이터"
)

VAL_RAW_DIR = Path(
    r"D:\data\017.한국어 텍스트-비디오-사운드 데이터\3.개방데이터\1.데이터\Validation\01.원천데이터"
)
VAL_LABEL_DIR = Path(
    r"D:\data\017.한국어 텍스트-비디오-사운드 데이터\3.개방데이터\1.데이터\Validation\02.라벨링데이터"
)

# 결과 저장 루트
DEST_ROOT = Path(r"D:\data")

# train/test 비율
TEST_RATIO = 0.2
SEED = 42

# ffmpeg 실행 파일명
FFMPEG_BIN = "ffmpeg"

# 오디오 변환 설정 (Whisper 학습용으로 무난한 값)
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
WAV_CODEC = "pcm_s16le"

# 덮어쓰기 여부
OVERWRITE_WAV = False
OVERWRITE_JSON = False

# 실행 전에 출력 폴더(train/test/validation) 초기화할지 여부
RESET_OUTPUT_DIRS = False

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
JSON_EXTENSIONS = {".json"}


# ==========================================
# 2) 유틸
# ==========================================
def check_paths():
    required = [TRAIN_RAW_DIR, TRAIN_LABEL_DIR, VAL_RAW_DIR, VAL_LABEL_DIR]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"경로가 존재하지 않습니다: {p}")

    if shutil.which(FFMPEG_BIN) is None:
        raise EnvironmentError(
            f"ffmpeg를 찾지 못했습니다. PATH에 등록되어 있는지 확인하세요. (현재 설정: {FFMPEG_BIN})"
        )


def reset_output_dirs():
    if not RESET_OUTPUT_DIRS:
        return

    for split in ["train", "test", "validation"]:
        split_dir = DEST_ROOT / split
        if split_dir.exists():
            print(f"[초기화] 삭제: {split_dir}")
            shutil.rmtree(split_dir)


def ensure_dest_dirs():
    for split in ["train", "test", "validation"]:
        (DEST_ROOT / split / "wav").mkdir(parents=True, exist_ok=True)
        (DEST_ROOT / split / "json").mkdir(parents=True, exist_ok=True)


def cleanup_jpgs_in_dest():
    deleted = 0
    for split in ["train", "test", "validation"]:
        split_dir = DEST_ROOT / split
        if not split_dir.exists():
            continue
        for f in split_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                f.unlink()
                deleted += 1
    print(f"[정리] 대상 폴더의 jpg/png 삭제 완료: {deleted}개")


def find_files_by_stem(root: Path, extensions: set[str]) -> Dict[str, Path]:
    """
    stem -> file_path 맵 생성.
    stem 중복이 있으면 오류를 내서 데이터 충돌을 조기에 확인.
    """
    mapping: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = {}

    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() in extensions:
            stem = f.stem
            if stem in mapping:
                duplicates.setdefault(stem, [mapping[stem]]).append(f)
            else:
                mapping[stem] = f

    if duplicates:
        msg_lines = ["중복 stem이 발견되었습니다. stem 기준 매칭이 불가능합니다."]
        for stem, paths in duplicates.items():
            msg_lines.append(f"- {stem}")
            for p in paths:
                msg_lines.append(f"    {p}")
        raise ValueError("\n".join(msg_lines))

    return mapping


def build_pairs(raw_dir: Path, label_dir: Path) -> List[Tuple[str, Path, Path]]:
    """
    mp4/json 짝이 있는 샘플만 추출
    return: [(stem, video_path, json_path), ...]
    """
    raw_map = find_files_by_stem(raw_dir, VIDEO_EXTENSIONS)
    label_map = find_files_by_stem(label_dir, JSON_EXTENSIONS)

    raw_stems = set(raw_map.keys())
    label_stems = set(label_map.keys())

    common_stems = sorted(raw_stems & label_stems)
    raw_only = sorted(raw_stems - label_stems)
    label_only = sorted(label_stems - raw_stems)

    print(f"[매칭] raw(video) 개수: {len(raw_stems)}")
    print(f"[매칭] label(json) 개수: {len(label_stems)}")
    print(f"[매칭] 공통 stem 개수: {len(common_stems)}")

    if raw_only:
        print(f"[경고] video만 있고 json 없는 샘플: {len(raw_only)}개")
        print("       예시:", raw_only[:5])

    if label_only:
        print(f"[경고] json만 있고 video 없는 샘플: {len(label_only)}개")
        print("       예시:", label_only[:5])

    pairs = [(stem, raw_map[stem], label_map[stem]) for stem in common_stems]
    return pairs


def split_train_test(
    pairs: List[Tuple[str, Path, Path]],
    test_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[str, Path, Path]], List[Tuple[str, Path, Path]]]:
    if not pairs:
        raise ValueError("분할할 training 쌍이 없습니다.")

    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)

    test_count = int(len(shuffled) * test_ratio)
    if test_count <= 0:
        raise ValueError(
            f"test로 분할되는 개수가 0입니다. 전체 개수={len(shuffled)}, TEST_RATIO={test_ratio}"
        )

    test_pairs = shuffled[:test_count]
    train_pairs = shuffled[test_count:]

    return train_pairs, test_pairs


def convert_video_to_wav(src_video: Path, dst_wav: Path):
    dst_wav.parent.mkdir(parents=True, exist_ok=True)

    if dst_wav.exists() and not OVERWRITE_WAV:
        return

    cmd = [
        FFMPEG_BIN,
        "-y" if OVERWRITE_WAV else "-n",
        "-i", str(src_video),
        "-vn",
        "-ac", str(TARGET_CHANNELS),
        "-ar", str(TARGET_SAMPLE_RATE),
        "-c:a", WAV_CODEC,
        str(dst_wav),
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 변환 실패\n"
            f"입력: {src_video}\n"
            f"출력: {dst_wav}\n"
            f"stderr:\n{result.stderr}"
        )


def copy_json(src_json: Path, dst_json: Path):
    dst_json.parent.mkdir(parents=True, exist_ok=True)

    if dst_json.exists() and not OVERWRITE_JSON:
        return

    shutil.copy2(src_json, dst_json)


def process_split(split_name: str, pairs: List[Tuple[str, Path, Path]]):
    wav_dir = DEST_ROOT / split_name / "wav"
    json_dir = DEST_ROOT / split_name / "json"

    wav_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    total = len(pairs)
    print(f"\n[{split_name}] 처리 시작 - 총 {total}개")

    success = 0
    failed = 0

    for idx, (stem, video_path, json_path) in enumerate(pairs, start=1):
        dst_wav = wav_dir / f"{stem}.wav"
        dst_json = json_dir / json_path.name

        try:
            convert_video_to_wav(video_path, dst_wav)
            copy_json(json_path, dst_json)
            success += 1
        except Exception as e:
            failed += 1
            print(f"[실패] {split_name} / {stem}")
            print(e)

        if idx % 100 == 0 or idx == total:
            print(f"[{split_name}] 진행률: {idx}/{total} | 성공={success} 실패={failed}")

    print(f"[{split_name}] 완료 | 성공={success} 실패={failed}")


# ==========================================
# 3) 메인 실행
# ==========================================
def main():
    print("[시작] 데이터셋 준비 작업 시작")
    check_paths()
    reset_output_dirs()
    ensure_dest_dirs()

    # 1) Training에서 pair 구성 후 train/test 분할
    train_pairs_all = build_pairs(TRAIN_RAW_DIR, TRAIN_LABEL_DIR)
    train_pairs, test_pairs = split_train_test(train_pairs_all, TEST_RATIO, SEED)

    print(f"\n[분할 결과] train={len(train_pairs)} / test={len(test_pairs)}")

    # 2) Validation은 전체를 validation으로
    val_pairs = build_pairs(VAL_RAW_DIR, VAL_LABEL_DIR)
    print(f"[validation] 전체 개수={len(val_pairs)}")

    # 3) 실제 처리
    process_split("train", train_pairs)
    process_split("test", test_pairs)
    process_split("validation", val_pairs)

    # 4) 혹시 남아 있을 이미지 파일 정리
    cleanup_jpgs_in_dest()

    print("\n[전체 완료]")
    print(f"출력 루트: {DEST_ROOT}")
    print(f"- {DEST_ROOT / 'train' / 'wav'}")
    print(f"- {DEST_ROOT / 'train' / 'json'}")
    print(f"- {DEST_ROOT / 'test' / 'wav'}")
    print(f"- {DEST_ROOT / 'test' / 'json'}")
    print(f"- {DEST_ROOT / 'validation' / 'wav'}")
    print(f"- {DEST_ROOT / 'validation' / 'json'}")


if __name__ == "__main__":
    main()