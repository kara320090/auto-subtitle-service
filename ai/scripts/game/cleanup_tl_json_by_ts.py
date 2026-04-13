#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================
# 파일명: cleanup_tl_json_by_ts.py
#
# 역할:
# - TS_02_game 폴더 안의 wav 파일 stem 목록을 수집한다.
# - TL_02_game 폴더 안의 json 파일을 확인한다.
# - json과 같은 이름의 wav가 TS_02_game에 있으면 json을 유지한다.
# - json과 같은 이름의 wav가 TS_02_game에 없으면 json을 삭제한다.
#
# 기준 예시:
# - MYB000999.json 이 있고 MYB000999.wav 도 있으면 유지
# - MYB000888.json 은 있는데 MYB000888.wav 가 없으면 삭제
#
# 주의:
# - 이 스크립트는 실제로 json 파일을 삭제한다.
# ============================================

from pathlib import Path

TL_DIR = Path("/home/user/game_dataSet/TL_02_game")
TS_DIR = Path("/home/user/game_dataSet/TS_02_game")

def main():
    # TS 폴더에서 wav 파일 stem 수집
    wav_stems = {
        p.stem
        for p in TS_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() == ".wav"
    }

    keep_count = 0
    delete_count = 0

    # TL 폴더의 json 파일 검사
    for json_path in TL_DIR.rglob("*.json"):
        if json_path.stem in wav_stems:
            keep_count += 1
            print(f"[유지] {json_path}")
        else:
            json_path.unlink()
            delete_count += 1
            print(f"[삭제] {json_path}")

    print("=" * 60)
    print(f"유지한 json 수: {keep_count}")
    print(f"삭제한 json 수: {delete_count}")

if __name__ == "__main__":
    main()