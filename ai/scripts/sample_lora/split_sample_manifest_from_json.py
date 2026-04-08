import json
import random
from pathlib import Path

WAV_DIR = Path(r"C:\auto-subtitle-service\ai\data\raw\Sample\wav")
JSON_DIR = Path(r"C:\auto-subtitle-service\ai\data\raw\Sample\json")
MANIFEST_DIR = Path(r"C:\auto-subtitle-service\ai\data\processed\sample_lora")

SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def load_text_from_json(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    terms = data.get("video", {}).get("term", [])
    terms = sorted(terms, key=lambda x: x.get("start", 0.0))

    transcriptions = []
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


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    if not WAV_DIR.exists():
        raise FileNotFoundError(f"WAV 폴더 없음: {WAV_DIR}")
    if not JSON_DIR.exists():
        raise FileNotFoundError(f"JSON 폴더 없음: {JSON_DIR}")

    wav_files = sorted(WAV_DIR.glob("*.wav"))
    rows = []
    missing_json = []

    for wav_path in wav_files:
        json_path = JSON_DIR / f"{wav_path.stem}.json"
        if not json_path.exists():
            missing_json.append(wav_path.name)
            continue

        label_info = load_text_from_json(json_path)

        if not label_info["text"]:
            print(f"[WARN] 전사문 비어있음: {json_path.name}")
            continue

        rows.append({
            "id": wav_path.stem,
            "audio": str(wav_path),
            "text": label_info["text"],
            "category": label_info["category"],
            "summary": label_info["summary"],
            "json_path": str(json_path),
        })

    if missing_json:
        print("[WARN] 대응 json 없는 wav:")
        for name in missing_json:
            print(" -", name)

    if len(rows) < 3:
        raise ValueError("학습에 사용할 데이터가 너무 적습니다.")

    random.seed(SEED)
    random.shuffle(rows)

    n = len(rows)
    n_train = max(1, int(n * TRAIN_RATIO))
    n_val = max(1, int(n * VAL_RATIO))
    n_test = n - n_train - n_val

    if n_test < 1:
        n_test = 1
        n_train = max(1, n_train - 1)

    train_rows = rows[:n_train]
    val_rows = rows[n_train:n_train + n_val]
    test_rows = rows[n_train + n_val:]

    write_jsonl(MANIFEST_DIR / "train.jsonl", train_rows)
    write_jsonl(MANIFEST_DIR / "val.jsonl", val_rows)
    write_jsonl(MANIFEST_DIR / "test.jsonl", test_rows)

    print(f"[INFO] total: {len(rows)}")
    print(f"[INFO] train: {len(train_rows)}")
    print(f"[INFO] val:   {len(val_rows)}")
    print(f"[INFO] test:  {len(test_rows)}")
    print(f"[INFO] 저장 위치: {MANIFEST_DIR}")


if __name__ == "__main__":
    main()