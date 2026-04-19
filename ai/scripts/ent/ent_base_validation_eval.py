# ============================================
# 파일명: ent_base_validation_eval.py
#
# 역할:
# - whisper-large-v3 base 모델만 사용한다. (LoRA adapter 사용 안 함)
# - validation.jsonl 전체를 대상으로 추론을 수행한다.
# - 샘플별 예측 결과(JSON)와 전체 평가 지표(JSON)를 저장한다.
#
# 입력:
# - /home/user/SWPJ3/auto-subtitle-service/ai/data/processed/ent/validation.jsonl
#
# 출력:
# - /home/user/SWPJ3/auto-subtitle-service/ai/data/results/ent_lora/val_json/ent_validation_predictions.json
# - /home/user/SWPJ3/auto-subtitle-service/ai/data/results/ent_lora/val/ent_base_validation_metrics.json
#
# 평가 지표:
# - avg_val_loss         : validation 전체 평균 loss
# - wer                  : 단어 기준 오류율
# - cer                  : 문자 기준 오류율
# - loss_evaluated_count : loss 계산에 실제로 사용된 샘플 수
# - loss_skipped_count   : label 길이 초과로 loss 계산을 건너뛴 샘플 수
# ============================================

import json
import re
from pathlib import Path
from typing import Dict, Any, List

import librosa
import torch
from datasets import load_dataset
from jiwer import wer, cer
from transformers import AutoProcessor, WhisperForConditionalGeneration
from transformers.utils import logging as hf_logging
from tqdm import tqdm


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def main():
    hf_logging.set_verbosity_error()

    # =========================
    # 도메인 / 경로 설정
    # =========================
    DOMAIN = "ent"

    PROJECT_ROOT = Path("/home/user/SWPJ3/auto-subtitle-service")
    MANIFEST_DIR = PROJECT_ROOT / f"ai/data/processed/{DOMAIN}"
    RESULT_DIR = PROJECT_ROOT / f"ai/data/results/{DOMAIN}_lora"

    VAL_JSONL = str(MANIFEST_DIR / "validation.jsonl")
    VAL_JSON_DIR = RESULT_DIR / "val_json"
    VAL_METRIC_DIR = RESULT_DIR / "val"

    MODEL_ID = "openai/whisper-large-v3"
    LANGUAGE = "korean"
    TASK = "transcribe"

    # =========================
    # 입력 파일 존재 여부 확인
    # =========================
    if not Path(VAL_JSONL).exists():
        raise FileNotFoundError(f"validation.jsonl 파일이 없습니다: {VAL_JSONL}")

    # =========================
    # 디바이스 / dtype 설정
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
    # validation 전체 로드
    # =========================
    dataset = load_dataset("json", data_files={"validation": VAL_JSONL})
    val_dataset = dataset["validation"]

    print(f"[INFO] validation samples = {len(val_dataset)}")

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
    model.config.use_cache = False

    model.generation_config.max_length = None
    model.config.max_length = None

    model.eval()

    max_label_length = model.config.max_target_positions
    print(f"[INFO] max_label_length = {max_label_length}")
    print("[INFO] running base model only (no LoRA adapter)")

    # =========================
    # validation 전체 추론 + loss 계산
    # =========================
    results: List[Dict[str, Any]] = []
    references_raw: List[str] = []
    predictions_raw: List[str] = []

    loss_sum = 0.0
    loss_count = 0
    loss_skipped_count = 0
    audio_missing_count = 0
    invalid_range_count = 0
    empty_segment_count = 0

    for idx, sample in enumerate(tqdm(val_dataset, desc="Validation inference")):
        audio_path = str(Path(sample["audio"]))
        ref_text = sample["text"]

        start = float(sample.get("start", 0.0))
        end = float(sample.get("end", 0.0))

        if not Path(audio_path).exists():
            print(f"[WARN] audio not found: {audio_path}")
            audio_missing_count += 1
            continue

        if end <= start:
            print(f"[WARN] invalid segment range: idx={idx}, start={start}, end={end}, file={audio_path}")
            invalid_range_count += 1
            continue

        # -------------------------
        # 오디오 로드 및 구간 자르기
        # -------------------------
        audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)

        start_sample = int(start * sr)
        end_sample = int(end * sr)
        audio_segment = audio_array[start_sample:end_sample]

        if audio_segment.size == 0:
            print(f"[WARN] empty audio segment: idx={idx}, start={start}, end={end}, file={audio_path}")
            empty_segment_count += 1
            continue

        # -------------------------
        # Whisper 입력 feature 생성
        # -------------------------
        inputs = processor(
            audio_segment,
            sampling_rate=16000,
            return_tensors="pt",
        )

        input_features = inputs["input_features"].to(device)
        if torch.cuda.is_available():
            input_features = input_features.to(dtype=load_dtype)

        # -------------------------
        # 정답 라벨 생성
        # -------------------------
        label_ids = processor.tokenizer(ref_text).input_ids

        # -------------------------
        # forward loss 계산
        # -------------------------
        sample_loss = None
        loss_skipped = False

        if len(label_ids) <= max_label_length:
            labels = torch.tensor([label_ids], device=device)

            bos_token_id = processor.tokenizer.bos_token_id
            if bos_token_id is not None and labels.shape[1] > 0:
                if (labels[:, 0] == bos_token_id).all():
                    labels = labels[:, 1:]

            if labels.shape[1] <= max_label_length:
                with torch.no_grad():
                    loss_outputs = model(
                        input_features=input_features,
                        labels=labels,
                    )
                    sample_loss = float(loss_outputs.loss.item())

                loss_sum += sample_loss
                loss_count += 1
            else:
                loss_skipped = True
        else:
            loss_skipped = True

        if loss_skipped:
            loss_skipped_count += 1
            print(
                f"[WARN] label too long for loss, skip loss only: "
                f"idx={idx}, len={len(label_ids)}"
            )

        # -------------------------
        # generate 추론
        # -------------------------
        with torch.no_grad():
            outputs = model.generate(
                input_features=input_features,
                language=LANGUAGE,
                task=TASK,
                return_timestamps=True,
                return_dict_in_generate=True,
                max_new_tokens=256,
            )

        full_text = processor.batch_decode(
            outputs["sequences"],
            skip_special_tokens=True,
        )[0].strip()

        # -------------------------
        # segment 정리
        # -------------------------
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

        # -------------------------
        # 텍스트 정규화
        # -------------------------
        ref_norm = normalize_text(ref_text)
        pred_norm = normalize_text(full_text)

        references_raw.append(ref_norm)
        predictions_raw.append(pred_norm)

        # -------------------------
        # 샘플별 결과 저장
        # -------------------------
        results.append(
            {
                "index": idx,
                "domain": DOMAIN,
                "split": "validation",
                "audio": audio_path,
                "reference": ref_text,
                "prediction": full_text,
                "reference_normalized": ref_norm,
                "prediction_normalized": pred_norm,
                "sample_loss": sample_loss,
                "loss_skipped": loss_skipped,
                "label_token_length": len(label_ids),
                "segment_start": start,
                "segment_end": end,
                "segments": segments_out,
                "model_type": "base_only",
            }
        )

    # =========================
    # 전체 metrics 계산
    # =========================
    avg_val_loss = loss_sum / loss_count if loss_count > 0 else None
    total_wer = wer(references_raw, predictions_raw) if references_raw else None
    total_cer = cer(references_raw, predictions_raw) if references_raw else None

    metrics = {
        "domain": DOMAIN,
        "split": "validation",
        "num_samples": len(results),
        "avg_val_loss": avg_val_loss,
        "wer": total_wer,
        "cer": total_cer,
        "loss_evaluated_count": loss_count,
        "loss_skipped_count": loss_skipped_count,
        "audio_missing_count": audio_missing_count,
        "invalid_range_count": invalid_range_count,
        "empty_segment_count": empty_segment_count,
        "max_label_length": max_label_length,
        "model_id": MODEL_ID,
        "model_type": "base_only",
    }

    # =========================
    # 결과 저장
    # =========================
    VAL_JSON_DIR.mkdir(parents=True, exist_ok=True)
    VAL_METRIC_DIR.mkdir(parents=True, exist_ok=True)

    pred_out_path = VAL_JSON_DIR / f"{DOMAIN}_validation_predictions.json"
    metric_out_path = VAL_METRIC_DIR / "ent_base_validation_metrics.json"

    with pred_out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with metric_out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[INFO] predictions saved to: {pred_out_path}")
    print(f"[INFO] metrics saved to: {metric_out_path}")
    print("\n===== METRICS =====")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()