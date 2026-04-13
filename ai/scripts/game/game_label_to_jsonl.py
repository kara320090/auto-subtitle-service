from pathlib import Path
import json

# =========================
# 경로 설정
# =========================
PROJECT_ROOT = Path("/home/user/SWPJ3/auto-subtitle-service")

LABEL_ROOT = PROJECT_ROOT / "ai/data/raw/game/label"
AUDIO_ROOT = PROJECT_ROOT / "ai/data/raw/game"
OUTPUT_DIR = PROJECT_ROOT / "ai/data/processed/game"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "validation", "test"]


def find_audio_path(split: str, base_filename: str) -> str | None:
    """
    base_filename 예: MYB_00679
    대응되는 wav 파일을 raw/game 아래에서 찾는다.
    우선 split 하위에서 찾고, 없으면 전체 raw/game에서 다시 찾는다.
    """
    candidates = list((AUDIO_ROOT / split).rglob(f"{base_filename}.wav"))
    if candidates:
        return str(candidates[0])

    candidates = list(AUDIO_ROOT.rglob(f"{base_filename}.wav"))
    if candidates:
        return str(candidates[0])

    return None


def convert_split(split: str):
    input_dir = LABEL_ROOT / split
    output_file = OUTPUT_DIR / f"{split}.jsonl"

    if not input_dir.exists():
        print(f"[WARN] 폴더 없음: {input_dir}")
        return

    json_files = sorted(input_dir.rglob("*.json"))

    total_json_files = 0
    total_terms = 0
    written_rows = 0
    skipped_rows = 0
    missing_audio = 0

    with output_file.open("w", encoding="utf-8") as fout:
        for json_path in json_files:
            total_json_files += 1

            try:
                with json_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[ERROR] JSON 읽기 실패: {json_path} | {e}")
                continue

            metadata = data.get("metadata", {})
            video = data.get("video", {})
            terms = video.get("term", [])

            base_filename = metadata.get("filename")
            if not base_filename:
                print(f"[SKIP] metadata.filename 없음: {json_path}")
                skipped_rows += 1
                continue

            audio_path = find_audio_path(split, base_filename)
            if audio_path is None:
                print(f"[WARN] audio 없음: {base_filename} ({json_path})")
                missing_audio += 1
                continue

            if not isinstance(terms, list):
                print(f"[SKIP] video.term 형식 이상: {json_path}")
                skipped_rows += 1
                continue

            for idx, term in enumerate(terms):
                total_terms += 1

                text = str(term.get("transcription", "")).strip()
                start = term.get("start", None)
                end = term.get("end", None)
                speaker_id = term.get("speaker_id", None)

                if not text:
                    skipped_rows += 1
                    continue

                row = {
                    "audio": audio_path,
                    "text": text,
                    "start": start,
                    "end": end,
                    "speaker_id": speaker_id,
                    "source_json": str(json_path),
                    "utterance_id": f"{base_filename}_{idx:04d}",
                    "file_id": base_filename,
                    "split": split,
                }

                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                written_rows += 1

    print(f"\n===== {split} =====")
    print(f"[INFO] input_dir       = {input_dir}")
    print(f"[INFO] output_file     = {output_file}")
    print(f"[INFO] total_json      = {total_json_files}")
    print(f"[INFO] total_terms     = {total_terms}")
    print(f"[INFO] written_rows    = {written_rows}")
    print(f"[INFO] skipped_rows    = {skipped_rows}")
    print(f"[INFO] missing_audio   = {missing_audio}")


def main():
    for split in SPLITS:
        convert_split(split)

    print("\n[INFO] 모든 jsonl 생성 완료")
    print(f"[INFO] 저장 경로: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()