# ============================================
# 파일명: make_manifest.py
#
# 역할:
# - vacation/{split} 폴더 안의 wav 파일들을 찾는다.
# - 같은 이름의 JSON 라벨 파일을 split에 맞는 라벨링데이터 폴더에서 찾는다.
# - JSON 안의 video.term[].transcription 값을 순서대로 이어 붙여
#   학습/검증/테스트용 text 문장으로 만든다.
# - 최종적으로 {"audio": "...", "text": "..."} 형태의
#   {split}.jsonl manifest 파일을 생성한다.
#
# 입력:
# - /home/user/SWPJ3/auto-subtitle-service/ai/data/raw/vacation/{split}/*.wav
# - ~/017.한국어_텍스트-비디오-사운드_데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터/*.json
# - ~/017.한국어_텍스트-비디오-사운드_데이터/3.개방데이터/1.데이터/Validation/02.라벨링데이터/*.json
#
# 출력:
# - /home/user/SWPJ3/auto-subtitle-service/ai/data/processed/vacation/train.jsonl
# - /home/user/SWPJ3/auto-subtitle-service/ai/data/processed/vacation/validation.jsonl
# - /home/user/SWPJ3/auto-subtitle-service/ai/data/processed/vacation/test.jsonl
#
# 목적:
# - LoRA 학습 코드에서 바로 읽을 수 있는 manifest를 만든다.
# - wav 파일 자체를 변환하는 것이 아니라,
#   wav 경로와 정답 전사문을 묶은 목록 파일을 만든다.
#
# 참고:
# - 현재 라벨 JSON의 전사문은 summary가 아니라
#   video -> term -> transcription 에 들어 있다.
# - 같은 파일명 기준으로 wav와 json을 매칭한다.
#   예: MYJ_00046.wav <-> MYJ_00046.json
# - JSON이 없거나 전사문이 비어 있으면 해당 파일은 건너뛴다.
# - train, test는 Training 라벨 폴더를 사용한다.
# - validation은 Validation 라벨 폴더를 사용한다.
# ============================================

from pathlib import Path
import json

# =========================
# 기본 경로 설정
# =========================

project_root = Path("/home/user/SWPJ3/auto-subtitle-service")

train_label_dir = Path(
    "~/017.한국어_텍스트-비디오-사운드_데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터"
).expanduser()

validation_label_dir = Path(
    "~/017.한국어_텍스트-비디오-사운드_데이터/3.개방데이터/1.데이터/Validation/02.라벨링데이터"
).expanduser()

splits = ["train", "validation", "test"]

# =========================
# JSON에서 전사문 추출
# =========================
def extract_text(label_data):
    """
    라벨 JSON에서 video.term 리스트를 읽고,
    각 항목의 transcription 문장을 순서대로 이어 붙여
    하나의 학습용 text로 만든다.
    """
    terms = label_data["video"]["term"]
    texts = []

    for item in terms:
        t = item.get("transcription", "").strip()
        if t:
            texts.append(t)

    return " ".join(texts).strip()

# =========================
# split별 manifest 생성
# =========================
for split in splits:
    print(f"\n===== {split} 시작 =====")

    audio_dir = project_root / "ai" / "data" / "raw" / "vacation" / split
    output_path = project_root / "ai" / "data" / "processed" / "vacation" / f"{split}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # split에 따라 라벨 폴더 선택
    if split == "validation":
        label_dir = validation_label_dir
    else:
        label_dir = train_label_dir

    wav_files = sorted(audio_dir.glob("*.wav"))

    count = 0
    missing_json = []
    errors = []

    with output_path.open("w", encoding="utf-8") as f:
        for wav_path in wav_files:
            json_path = label_dir / f"{wav_path.stem}.json"

            if not json_path.exists():
                missing_json.append(wav_path.name)
                continue

            try:
                label_data = json.loads(json_path.read_text(encoding="utf-8"))
                text = extract_text(label_data)

                if not text:
                    errors.append((wav_path.name, "빈 전사문"))
                    continue

                item = {
                    "audio": str(wav_path),
                    "text": text
                }

                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1

            except Exception as e:
                errors.append((wav_path.name, str(e)))

    print(f"[INFO] 저장 완료: {output_path}")
    print(f"[INFO] 총 {count}개 샘플 저장")

    if missing_json:
        print(f"[WARN] 대응 JSON 없는 파일 수: {len(missing_json)}")
        for name in missing_json[:10]:
            print(" -", name)

    if errors:
        print(f"[WARN] 처리 실패 파일 수: {len(errors)}")
        for name, msg in errors[:10]:
            print(f" - {name}: {msg}")