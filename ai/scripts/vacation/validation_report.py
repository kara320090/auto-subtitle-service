import csv
import json
from pathlib import Path

from jiwer import wer, cer

PROJECT_ROOT = Path("/home/user/SWPJ3/auto-subtitle-service")
VALID_JSONL = PROJECT_ROOT / "ai/data/processed/vacation/validation.jsonl"
PRED_DIR = PROJECT_ROOT / "ai/data/results/vacation_lora/vacation_lora_validation_inference"
OUT_CSV = PROJECT_ROOT / "ai/data/results/vacation_lora/validation_report.csv"

reference_map = {}

with VALID_JSONL.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        stem = Path(row["audio"]).stem
        reference_map[stem] = {
            "audio": row["audio"],
            "reference": row["text"],
        }

pred_files = sorted(PRED_DIR.glob("*_prediction_with_timestamps.json"))
if not pred_files:
    raise FileNotFoundError(f"no prediction json files found in: {PRED_DIR}")

rows = []
wers = []
cers = []

for pred_file in pred_files:
    with pred_file.open("r", encoding="utf-8") as f:
        pred = json.load(f)

    audio_path = pred.get("audio", "")
    stem = Path(audio_path).stem if audio_path else pred_file.stem.replace("_prediction_with_timestamps", "")
    prediction = pred.get("prediction", "")
    segments = pred.get("segments", [])
    segments_count = len(segments)

    ref_info = reference_map.get(stem, {})
    reference = ref_info.get("reference", "")

    row_wer = ""
    row_cer = ""

    if reference and prediction:
        try:
            row_wer = wer(reference, prediction)
            row_cer = cer(reference, prediction)
            wers.append(row_wer)
            cers.append(row_cer)
        except Exception:
            pass

    rows.append({
        "file_name": f"{stem}.wav",
        "audio": audio_path,
        "reference": reference,
        "prediction": prediction,
        "segments_count": segments_count,
        "wer": row_wer,
        "cer": row_cer,
    })

with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file_name",
            "audio",
            "reference",
            "prediction",
            "segments_count",
            "wer",
            "cer",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"[INFO] saved: {OUT_CSV}")
print(f"[INFO] rows: {len(rows)}")

if wers:
    print(f"[INFO] average WER: {sum(wers) / len(wers):.6f}")
if cers:
    print(f"[INFO] average CER: {sum(cers) / len(cers):.6f}")