# ============================================
# 파일명: test_adapter_reload_with_timestamps_vacation.py
#
# 역할:
# - vacation 도메인용으로 학습된 LoRA adapter를 다시 불러온다.
# - whisper-large-v3 base 모델에 vacation adapter를 붙인다.
# - vacation/test.jsonl 안의 샘플 1개를 읽어 실제로 추론을 수행한다.
# - 전사 결과(text)와 함께 segment 단위 타임스탬프(start, end, text)만 저장한다.
#
# 입력:
# - /home/user/SWPJ3/auto-subtitle-service/ai/data/processed/vacation/test.jsonl
# - /home/user/SWPJ3/auto-subtitle-service/ai/data/results/vacation_lora/adapter
#
# 출력:
# - /home/user/SWPJ3/auto-subtitle-service/ai/data/results/vacation_lora/predictions/vacation_prediction_with_timestamps.json
#
# 목적:
# - vacation adapter가 실제로 다시 로드되는지 확인
# - base 모델 + vacation adapter 조합으로 추론이 가능한지 확인
# - 타임스탬프 포함 결과 구조를 확인
#
# 참고:
# - segments = 문장/구간 단위
# - token_offsets = 사용하지 않음
# - 따라서 JSON에는 segments만 저장한다.
# - 이 파일은 "학습 코드"가 아니라 "추론/검증 코드"다.
# - 4090 환경에서는 fp16 또는 bf16으로 빠르게 추론 가능하다.
# ============================================

import json
from pathlib import Path

import librosa
import torch
from datasets import load_dataset
from transformers import AutoProcessor, WhisperForConditionalGeneration

# =========================
# 경로
# =========================
PROJECT_ROOT = Path("/home/user/SWPJ3/auto-subtitle-service")
MANIFEST_DIR = PROJECT_ROOT / "ai/data/processed/vacation"
RESULT_DIR = PROJECT_ROOT / "ai/data/results/vacation_lora"

TEST_JSONL = str(MANIFEST_DIR / "test.jsonl")
ADAPTER_DIR = RESULT_DIR / "adapter"
PRED_DIR = RESULT_DIR / "predictions"

MODEL_ID = "openai/whisper-large-v3"
ADAPTER_NAME = "default"
LANGUAGE = "korean"
TASK = "transcribe"

# =========================
# 디바이스 / dtype
# =========================
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 4090에서는 bf16 지원 시 bf16 우선, 아니면 fp16
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
load_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)

print(f"[INFO] device = {device}")
print(f"[INFO] use_bf16 = {use_bf16}")
print(f"[INFO] dtype  = {load_dtype}")

# =========================
# test 샘플 1개 읽기
# =========================
dataset = load_dataset("json", data_files={"test": TEST_JSONL})
sample = dataset["test"][0]

audio_path = sample["audio"]
ref_text = sample["text"]

# Windows 경로로 저장된 경우 Ubuntu 경로로 자동 변환
if audio_path.startswith("C:\\auto-subtitle-service\\"):
    audio_path = audio_path.replace(
        "C:\\auto-subtitle-service\\",
        str(PROJECT_ROOT) + "/",
    )
    audio_path = audio_path.replace("\\", "/")

audio_path = str(Path(audio_path))

if not Path(audio_path).exists():
    raise FileNotFoundError(f"audio file not found: {audio_path}")

print(f"[INFO] audio = {audio_path}")

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
# 오디오 로드
# =========================
audio_array, _ = librosa.load(audio_path, sr=16000, mono=True)

inputs = processor(
    audio_array,
    sampling_rate=16000,
    return_tensors="pt",
)

input_features = inputs["input_features"].to(device)

if torch.cuda.is_available():
    input_features = input_features.to(dtype=load_dtype)

# =========================
# generate with segment timestamps only
# =========================
with torch.no_grad():
    outputs = model.generate(
        input_features=input_features,
        language=LANGUAGE,
        task=TASK,
        return_timestamps=True,
        return_dict_in_generate=True,
        max_new_tokens=256,
    )

# =========================
# 전체 텍스트
# =========================
full_text = processor.batch_decode(
    outputs["sequences"],
    skip_special_tokens=True,
)[0].strip()

# =========================
# segment 정리
# =========================
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

# =========================
# 저장
# =========================
PRED_DIR.mkdir(parents=True, exist_ok=True)
out_path = PRED_DIR / "vacation_prediction_with_timestamps.json"

payload = {
    "audio": audio_path,
    "reference": ref_text,
    "prediction": full_text,
    "active_adapters": model.active_adapters(),
    "segments": segments_out,
}

with out_path.open("w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print("\n===== REFERENCE =====")
print(ref_text)
print("\n===== PREDICTION =====")
print(full_text)
print("\n===== SEGMENTS =====")
for seg in segments_out:
    print(seg)

print(f"\n[INFO] saved to: {out_path}")