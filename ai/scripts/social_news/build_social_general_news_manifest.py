import json
from pathlib import Path
from typing import Dict, List

# =========================
# 1. 경로 설정
# =========================
DATA_ROOT = Path(r"D:\data\social_general_news_data")

TRAIN_WAV_DIR = DATA_ROOT / "train" / "wav"
TRAIN_JSON_DIR = DATA_ROOT / "train" / "json"

VAL_WAV_DIR = DATA_ROOT / "validation" / "wav"
VAL_JSON_DIR = DATA_ROOT / "validation" / "json"

TEST_WAV_DIR = DATA_ROOT / "test" / "wav"
TEST_JSON_DIR = DATA_ROOT / "test" / "json"

# 프로젝트 폴더 안 jsonl 저장 위치
MANIFEST_DIR = Path(r"C:\auto-subtitle-service\ai\data\social_general_news_data")

DOMAIN_NAME = "사회일반뉴스"


def load_text_from_json(json_path: Path) -> Dict[str, str]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    terms = data.get("video", {}).get("term", [])
    terms = sorted(terms, key=lambda x: x.get("start", 0.0))

    transcriptions: List[str] = []
    for term in terms:
        text = str(term.get("transcription", "")).strip()
        if text:
            transcriptions.append(text)

    full_text = " ".join(transcriptions).strip()
    category = data.get("metadata", {}).get("category", "")
    summary = data.get("summary", "")

    return {
        "text": full_text,
        "category": category,
        "summary": summary,
    }


def write_jsonl(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_rows(split_name: str, wav_dir: Path, json_dir: Path) -> List[dict]:
    if not wav_dir.exists():
        raise FileNotFoundError(f"[{split_name}] WAV 폴더 없음: {wav_dir}")
    if not json_dir.exists():
        raise FileNotFoundError(f"[{split_name}] JSON 폴더 없음: {json_dir}")

    wav_files = sorted(wav_dir.glob("*.wav"))
    rows: List[dict] = []
    missing_json: List[str] = []
    empty_text_count = 0

    for wav_path in wav_files:
        json_path = json_dir / f"{wav_path.stem}.json"

        if not json_path.exists():
            missing_json.append(wav_path.name)
            continue

        label_info = load_text_from_json(json_path)

        if not label_info["text"]:
            empty_text_count += 1
            print(f"[WARN] [{split_name}] 전사문 비어있음: {json_path.name}")
            continue

        rows.append({
            "id": wav_path.stem,
            "audio": str(wav_path),
            "text": label_info["text"],
            "domain": DOMAIN_NAME,
            "category": label_info["category"] or DOMAIN_NAME,
            "summary": label_info["summary"],
            "split": split_name,
            "json_path": str(json_path),
        })

    if missing_json:
        print(f"[WARN] [{split_name}] 대응 json 없는 wav: {len(missing_json)}개")
        for name in missing_json[:20]:
            print(" -", name)
        if len(missing_json) > 20:
            print(f" ... 외 {len(missing_json) - 20}개")

    print(f"[INFO] [{split_name}] wav 개수: {len(wav_files)}")
    print(f"[INFO] [{split_name}] usable rows: {len(rows)}")
    print(f"[INFO] [{split_name}] empty text skipped: {empty_text_count}")

    return rows


def main():
    train_rows = build_rows("train", TRAIN_WAV_DIR, TRAIN_JSON_DIR)
    val_rows = build_rows("validation", VAL_WAV_DIR, VAL_JSON_DIR)
    test_rows = build_rows("test", TEST_WAV_DIR, TEST_JSON_DIR)

    if not train_rows:
        raise ValueError("train 데이터가 비어 있습니다.")
    if not val_rows:
        raise ValueError("validation 데이터가 비어 있습니다.")
    if not test_rows:
        raise ValueError("test 데이터가 비어 있습니다.")

    write_jsonl(MANIFEST_DIR / "train.jsonl", train_rows)
    write_jsonl(MANIFEST_DIR / "val.jsonl", val_rows)
    write_jsonl(MANIFEST_DIR / "test.jsonl", test_rows)

    print("\n[전체 완료]")
    print(f"[INFO] train: {len(train_rows)}")
    print(f"[INFO] val:   {len(val_rows)}")
    print(f"[INFO] test:  {len(test_rows)}")
    print(f"[INFO] 저장 위치: {MANIFEST_DIR}")


if __name__ == "__main__":
    main()