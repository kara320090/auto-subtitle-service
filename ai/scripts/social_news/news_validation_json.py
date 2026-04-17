# ============================================
# 파일명: export_whisper_social_news_validation.py
#
# 역할:
# - social_news 도메인용으로 학습된 LoRA adapter를 다시 불러온다.
# - whisper-large-v3 base 모델에 social_news adapter를 붙인다.
# - validation 안의 모든 .wav 파일을 순회하며 추론한다.
# - wav 1개당 json 1개씩 저장한다.
# - 시작 시각 / 종료 시각 / 총 소요 시간을 출력한다.
#
# 입력:
# - D:\data\social_general_news_data\validation\wav
# - C:\auto-subtitle-service\ai\data\results\social_news_lora\adapter
#
# 출력:
# - C:\auto-subtitle-service\ai\data\results\social_news_lora\validation_json\*.json
#
# 목적:
# - validation 세트 전체에 대해 adapter가 실제로 잘 동작하는지 확인
# - 파일 단위 예측 결과 JSON을 개별적으로 저장
#
# 참고:
# - torchcodec 문제를 피하기 위해 pipeline 대신 librosa + model.generate() 사용
# - max_new_tokens/max_length 관련 반복 경고를 줄이기 위해
#   transformers 로깅 레벨을 낮추고 max_length를 비움
# ============================================

import json
import time
from datetime import datetime
from pathlib import Path

import librosa
import torch
from tqdm import tqdm
from transformers import AutoProcessor, WhisperForConditionalGeneration
from transformers.utils import logging as hf_logging


def main():
    # transformers 경고 로그 최소화
    hf_logging.set_verbosity_error()

    # =========================
    # 경로 설정 (내 컴퓨터 기준)
    # =========================
    RAW_VALIDATION_DIR = Path(r"D:\data\social_general_news_data\validation\wav")

    RESULT_DIR = Path(r"C:\auto-subtitle-service\ai\data\results\social_news_lora")
    ADAPTER_DIR = RESULT_DIR / "adapter"
    PRED_DIR = RESULT_DIR / "validation_json"

    MODEL_ID = "openai/whisper-large-v3"
    ADAPTER_NAME = "news_adapter"
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
    print(f"[INFO] dtype = {load_dtype}")

    # =========================
    # validation wav 확인
    # =========================
    if not RAW_VALIDATION_DIR.exists():
        raise FileNotFoundError(f"validation directory not found: {RAW_VALIDATION_DIR}")

    audio_files = sorted(RAW_VALIDATION_DIR.glob("*.wav"))

    if not audio_files:
        raise FileNotFoundError(f"no wav files found in: {RAW_VALIDATION_DIR}")

    print(f"[INFO] found {len(audio_files)} wav files")

    # =========================
    # processor / model 로드
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

    # max_new_tokens만 사용하도록 정리
    model.generation_config.max_length = None
    model.config.max_length = None

    # =========================
    # adapter 로드
    # =========================
    print(f"[INFO] loading adapter from: {ADAPTER_DIR}")
    model.load_adapter(str(ADAPTER_DIR), adapter_name=ADAPTER_NAME)
    model.set_adapter(ADAPTER_NAME)
    model.eval()

    print(f"[INFO] active adapters: {model.active_adapters()}")

    # =========================
    # 결과 저장 폴더 생성
    # =========================
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    start_dt = datetime.now()
    print(f"[INFO] validation inference started at: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    # =========================
    # validation 전체 추론
    # =========================
    for idx, audio_path in enumerate(tqdm(audio_files, desc="social_news validation inference"), start=1):
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

        # 세그먼트 정리
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
                        "start": round(float(seg.get("start", 0.0)), 2),
                        "end": round(float(seg.get("end", 0.0)), 2),
                        "text": seg_text,
                    }
                )

        # 저장 payload
        payload = {
            "audio": str(audio_path),
            "prediction": full_text,
            "active_adapters": model.active_adapters(),
            "segments": segments_out,
        }

        # 파일별 json 저장
        out_path = PRED_DIR / f"{audio_path.stem}_prediction_with_timestamps.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"[INFO] saved to: {out_path}")

    end_time = time.time()
    end_dt = datetime.now()
    elapsed = end_time - start_time

    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60

    print("\n" + "=" * 60)
    print(f"[INFO] validation inference finished at: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] total elapsed time: {hours}h {minutes}m {seconds:.2f}s")


if __name__ == "__main__":
    main()