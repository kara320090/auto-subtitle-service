# ============================================
# 파일명: round_validation_segment_times.py
#
# 역할:
# - validation predictions json 파일 1개를 읽는다.
# - 각 항목의 segments 안에 있는 start, end를
#   소수점 둘째 자리까지 반올림한다.
# - 원본 파일에 그대로 덮어쓴다.
# ============================================

import json
from pathlib import Path

TARGET_FILE = Path("/home/user/SWPJ3/auto-subtitle-service/ai/data/results/game_lora/val_json/game_validation_predictions.json")

if not TARGET_FILE.exists():
    raise FileNotFoundError(f"file not found: {TARGET_FILE}")

with TARGET_FILE.open("r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, list):
    raise TypeError("expected top-level JSON to be a list")

updated_file_count = 0
updated_segment_count = 0

for item in data:
    segments = item.get("segments", [])
    item_changed = False

    for seg in segments:
        seg_changed = False

        if "start" in seg:
            new_start = round(float(seg["start"]), 2)
            if seg["start"] != new_start:
                seg["start"] = new_start
                seg_changed = True

        if "end" in seg:
            new_end = round(float(seg["end"]), 2)
            if seg["end"] != new_end:
                seg["end"] = new_end
                seg_changed = True

        if seg_changed:
            updated_segment_count += 1
            item_changed = True

    if item_changed:
        updated_file_count += 1

with TARGET_FILE.open("w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"[INFO] updated items: {updated_file_count}")
print(f"[INFO] updated segments: {updated_segment_count}")
print(f"[INFO] target file: {TARGET_FILE}")