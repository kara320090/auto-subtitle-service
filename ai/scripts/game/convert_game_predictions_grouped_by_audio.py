# ============================================
# 파일명: convert_game_predictions_grouped_by_audio.py
#
# 역할:
# - game_validation_predictions.json 파일을 읽는다.
# - 같은 audio 파일명(stem) 기준으로 항목들을 묶는다.
# - 각 항목의 segments를 하나로 합친다.
# - start/end는 소수점 둘째 자리까지 반올림한다.
# - audio별 JSON 파일 1개씩 저장한다.
# ============================================

import json
from pathlib import Path
from collections import defaultdict

SOURCE_FILE = Path(
    "/home/user/SWPJ3/auto-subtitle-service/ai/data/results/game_lora/val_json/game_validation_predictions.json"
)

OUTPUT_DIR = Path(
    "/home/user/SWPJ3/auto-subtitle-service/ai/data/results/game_lora/validation"
)

if not SOURCE_FILE.exists():
    raise FileNotFoundError(f"source file not found: {SOURCE_FILE}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with SOURCE_FILE.open("r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, list):
    raise TypeError("expected top-level JSON to be a list")

# audio 파일명 기준으로 그룹화
grouped = {}

for item in data:
    audio_path = item.get("audio")
    if not audio_path:
        print("[WARN] skipped item because audio field is missing")
        continue

    sample_name = Path(audio_path).stem
    prediction = item.get("prediction", "")
    active_adapters = item.get("active_adapters", [])
    segments = item.get("segments", [])

    if sample_name not in grouped:
        grouped[sample_name] = {
            "audio": audio_path,
            "prediction": prediction,
            "active_adapters": active_adapters,
            "segments": []
        }

    # segments 누적
    for seg in segments:
        grouped[sample_name]["segments"].append({
            "start": round(float(seg.get("start", 0.0)), 2),
            "end": round(float(seg.get("end", 0.0)), 2),
            "text": seg.get("text", "")
        })

# 각 audio별로 segments 정렬 및 저장
saved_count = 0

for sample_name, item in grouped.items():
    # start, end 기준 정렬
    item["segments"].sort(key=lambda x: (x["start"], x["end"]))

    output_path = OUTPUT_DIR / f"{sample_name}.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(item, f, ensure_ascii=False, indent=2)

    saved_count += 1

print(f"[INFO] source file: {SOURCE_FILE}")
print(f"[INFO] output dir : {OUTPUT_DIR}")
print(f"[INFO] saved json files: {saved_count}")