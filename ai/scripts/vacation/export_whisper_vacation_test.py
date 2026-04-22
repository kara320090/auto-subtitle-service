# ============================================
# 파일명: export_whisper_vacation_test.py
#
# 역할:
# - vacation 도메인용으로 학습된 LoRA adapter를 다시 불러온다.
# - whisper-large-v3 base 모델에 vacation adapter를 붙인다.
# - raw/vacation/test 안의 모든 .wav 파일을 순회하며 추론한다.
# - wav 1개당 json 1개씩 저장한다.
#
# 저장 경로:
# - /home/user/SWPJ3/auto-subtitle-service/ai/data/results/vacation_lora/vacation_lora_test_inference/
# ============================================

import json
from pathlib import Path

import librosa
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration

# =========================
# 경로
# =========================
PROJECT_ROOT = Path("/home/user/SWPJ3/auto-subtitle-service")
RAW_TEST_DIR = PROJECT_ROOT / "ai/data/raw/vacation/test"
RESULT_DIR = PROJECT_ROOT / "ai/data/results/vacation_lora"
ADAPTER_DIR = RESULT_DIR / "adapter"
PRED_DIR = RESULT_DIR / "vacation_lora_test_inference"

MODEL_ID = "openai/whisper-large-v3"
ADAPTER_NAME = "default"
LANGUAGE = "korean"
TASK = "transcribe"

# =========================
# 디바이스 / dtype
# =========================
device = "cuda:0" if torch.cuda.is_available() else "cpu"

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
load_dtype = torch.bfloat16 if use_bf16 else (
    torch.float16 if torch.cuda.is_available() else torch.float32
)

print(f"[INFO] device = {device}")
print(f"[INFO] use_bf16 = {use_bf16}")
print(f"[INFO] dtype  = {load_dtype}")

# =========================
# test 폴더 안 wav 전부 수집
# =========================
if not RAW_TEST_DIR.exists():
    raise FileNotFoundError(f"test directory not found: {RAW_TEST_DIR}")

audio_files = sorted(RAW_TEST_DIR.glob("*.wav"))

if not audio_files:
    raise FileNotFoundError(f"no wav files found in: {RAW_TEST_DIR}")

print(f"[INFO] found {len(audio_files)} wav files")

# =========================
# processor / base model 로드
# =========================
print("[INFO] loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("[INFO] loading base model...")
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=load_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
).to(device)

model.generation_config.language = LANGUAGE
model.generation_config.task = TASK
model.generation_config.forced_decoder_ids = None
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# =========================
# adapter 로드
# =========================
print(f"[INFO] loading adapter from: {ADAPTER_DIR}")
model.load_adapter(str(ADAPTER_DIR), adapter_name=ADAPTER_NAME)
model.set_adapter(ADAPTER_NAME)
model.eval()

print(f"[INFO] active adapters: {model.active_adapters()}")

# =========================
# 출력 폴더 생성
# =========================
PRED_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 전체 파일 순회
# =========================
for idx, audio_path in enumerate(audio_files, start=1):
    print("\n" + "=" * 60)
    print(f"[INFO] ({idx}/{len(audio_files)}) processing: {audio_path.name}")

    # 오디오 로드
    audio_array, _ = librosa.load(str(audio_path), sr=16000, mono=True)

    inputs = processor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt",
    )

    input_features = inputs["input_features"].to(device)

    if torch.cuda.is_available():
        input_features = input_features.to(dtype=load_dtype)

    # 추론
    with torch.no_grad():
        outputs = model.generate(
            input_features=input_features,
            language=LANGUAGE,
            task=TASK,
            return_timestamps=True,
            return_dict_in_generate=True,
            max_new_tokens=256,
        )

    # 전체 텍스트
    full_text = processor.batch_decode(
        outputs["sequences"],
        skip_special_tokens=True,
    )[0].strip()

    # segment 정리
    segments_out = []
    segments = outputs.get("segments", [])

    if segments and len(segments) > 0:
        for seg in segments[0]:
            seg_tokens = seg.get("tokens", [])
            seg_text = processor.tokenizer.decode(
                seg_tokens,
                skip_special_tokens=True,
            ).strip()

            segments_out.append(
                {
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0)),
                    "text": seg_text,
                }
            )

    payload = {
        "audio": str(audio_path),
        "prediction": full_text,
        "active_adapters": model.active_adapters(),
        "segments": segments_out,
    }

    # wav 1개당 json 1개 저장
    out_path = PRED_DIR / f"{audio_path.stem}_prediction_with_timestamps.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n===== PREDICTION =====")
    print(full_text)
    print("\n===== SEGMENTS =====")
    for seg in segments_out:
        print(seg)

    print(f"\n[INFO] saved to: {out_path}")