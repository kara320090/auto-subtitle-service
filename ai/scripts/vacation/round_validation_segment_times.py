# ============================================
# 파일명: round_validation_segment_times.py
#
# 역할:
# - 이미 저장된 validation inference json 파일들을 순회한다.
# - segments의 start, end를 소수점 둘째 자리까지 반올림한다.
# - 원본 파일에 그대로 덮어쓴다.
# ============================================

import json
from pathlib import Path

TARGET_DIR = Path("/home/user/SWPJ3/auto-subtitle-service/ai/data/results/vacation_lora/vacation_lora_validation_inference")

if not TARGET_DIR.exists():
    raise FileNotFoundError(f"directory not found: {TARGET_DIR}")

json_files = sorted(TARGET_DIR.glob("*.json"))

if not json_files:
    raise FileNotFoundError(f"no json files found in: {TARGET_DIR}")

updated_count = 0

for json_path in json_files:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    changed = False

    for seg in segments:
        if "start" in seg:
            new_start = round(float(seg["start"]), 2)
            if seg["start"] != new_start:
                seg["start"] = new_start
                changed = True

        if "end" in seg:
            new_end = round(float(seg["end"]), 2)
            if seg["end"] != new_end:
                seg["end"] = new_end
                changed = True

    if changed:
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        updated_count += 1

print(f"[INFO] processed files: {len(json_files)}")
print(f"[INFO] updated files: {updated_count}")
print(f"[INFO] target dir: {TARGET_DIR}")