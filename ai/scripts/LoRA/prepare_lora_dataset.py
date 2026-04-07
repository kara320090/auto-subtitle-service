import json
from pathlib import Path
from tqdm import tqdm


# =========================
# 경로 설정
# =========================
DATA_ROOT = Path(r"D:\data")

TRAIN_WAV_DIR = DATA_ROOT / "train" / "wav"
TRAIN_JSON_DIR = DATA_ROOT / "train" / "json"

VAL_WAV_DIR = DATA_ROOT / "val" / "wav"
VAL_JSON_DIR = DATA_ROOT / "val" / "json"

OUTPUT_DIR = Path(r"C:\auto-subtitle-service\ai\data\lora_dataset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUTPUT = OUTPUT_DIR / "train.jsonl"
VAL_OUTPUT = OUTPUT_DIR / "val.jsonl"


# =========================
# 정답지 추출
# =========================
def extract_ground_truth_text(json_path: Path) -> str:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    terms = data.get("video", {}).get("term", [])
    terms = sorted(terms, key=lambda x: x.get("start", 0))

    texts = []
    for item in terms:
        text = item.get("transcription", "").strip()
        if text:
            texts.append(text)

    return " ".join(texts).strip()


# =========================
# split별 manifest 생성
# =========================
def build_manifest(wav_dir: Path, json_dir: Path, output_file: Path, split_name: str):
    if not wav_dir.exists():
        raise FileNotFoundError(f"{split_name} wav 폴더가 없습니다: {wav_dir}")
    if not json_dir.exists():
        raise FileNotFoundError(f"{split_name} json 폴더가 없습니다: {json_dir}")

    wav_files = sorted(wav_dir.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"{split_name} wav 파일이 없습니다: {wav_dir}")

    rows = []
    missing_json = []
    empty_text = []

    for wav_path in tqdm(wav_files, desc=f"Preparing {split_name}"):
        stem = wav_path.stem
        json_path = json_dir / f"{stem}.json"

        if not json_path.exists():
            missing_json.append(stem)
            continue

        try:
            ground_truth_text = extract_ground_truth_text(json_path)
        except Exception as e:
            print(f"[JSON 읽기 실패] {json_path}: {e}")
            continue

        if not ground_truth_text:
            empty_text.append(stem)
            continue

        row = {
            "audio_path": str(wav_path),
            "text": ground_truth_text,
        }
        rows.append(row)

    with open(output_file, "w", encoding="utf-8-sig") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n[{split_name}] 저장 완료: {output_file}")
    print(f"[{split_name}] 총 샘플 수: {len(rows)}")

    if missing_json:
        print(f"\n[{split_name}] 매칭 실패 JSON ({len(missing_json)}개)")
        for name in missing_json[:20]:
            print(f"- {name}")
        if len(missing_json) > 20:
            print(f"... 외 {len(missing_json) - 20}개")

    if empty_text:
        print(f"\n[{split_name}] 빈 정답 텍스트 ({len(empty_text)}개)")
        for name in empty_text[:20]:
            print(f"- {name}")
        if len(empty_text) > 20:
            print(f"... 외 {len(empty_text) - 20}개")


# =========================
# 실행
# =========================
if __name__ == "__main__":
    build_manifest(TRAIN_WAV_DIR, TRAIN_JSON_DIR, TRAIN_OUTPUT, "train")
    build_manifest(VAL_WAV_DIR, VAL_JSON_DIR, VAL_OUTPUT, "val")

    print("\n전체 dataset manifest 생성 완료")
    print(f"- train: {TRAIN_OUTPUT}")
    print(f"- val:   {VAL_OUTPUT}")