# ============================================
# 파일명: vacation_base_validation_eval_from_label.py
# ============================================

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import librosa
import torch
from jiwer import wer, cer
from tqdm import tqdm
from transformers import AutoProcessor, WhisperForConditionalGeneration
from transformers.utils import logging as hf_logging


def normalize_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_last_5digits(name: str) -> Optional[str]:
    stem = Path(name).stem
    match = re.search(r"(\d{5})$", stem)
    if match:
        return match.group(1)
    return None


def label_to_audio_key(label_key_full: str) -> str:
    """
    예:
    17002 -> 00002
    16990 -> 00990
    """
    return label_key_full[-3:].zfill(5)


def extract_text_from_json(obj: Any) -> Optional[str]:
    priority_keys = [
        "text", "sentence", "script", "transcript",
        "transcription", "label", "utterance", "stt", "answer"
    ]

    if isinstance(obj, dict):
        for key in priority_keys:
            if key in obj and isinstance(obj[key], str):
                candidate = normalize_text(obj[key])
                if candidate:
                    return candidate

        for value in obj.values():
            found = extract_text_from_json(value)
            if found:
                return found

    elif isinstance(obj, list):
        collected = []
        for item in obj:
            found = extract_text_from_json(item)
            if found:
                collected.append(found)
        if collected:
            return normalize_text(" ".join(collected))

    elif isinstance(obj, str):
        candidate = normalize_text(obj)
        if candidate:
            return candidate

    return None


def read_label_text(label_path: Path) -> Optional[str]:
    try:
        with label_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except UnicodeDecodeError:
        with label_path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
    return extract_text_from_json(data)


def build_pairs(label_dir: Path, audio_dir: Path) -> Tuple[List[Dict[str, str]], int, int]:
    matched_pairs: List[Dict[str, str]] = []
    audio_missing_count = 0
    duplicate_audio_key_count = 0

    audio_map: Dict[str, Path] = {}
    duplicate_keys = set()

    for audio_path in sorted(audio_dir.glob("*.wav")):
        key = extract_last_5digits(audio_path.name)
        if key is None:
            print(f"[WARN] audio key parse failed: {audio_path.name}")
            continue

        if key in audio_map:
            duplicate_keys.add(key)
        else:
            audio_map[key] = audio_path

    if duplicate_keys:
        duplicate_audio_key_count = len(duplicate_keys)
        print(f"[WARN] duplicated audio keys found: {sorted(duplicate_keys)}")

    for label_path in sorted(label_dir.glob("*.json")):
        label_key_full = extract_last_5digits(label_path.name)

        if label_key_full is None:
            print(f"[WARN] label key parse failed: {label_path.name}")
            continue

        match_key = label_to_audio_key(label_key_full)

        if match_key not in audio_map:
            audio_missing_count += 1
            print(
                f"[WARN] matching audio not found: "
                f"label={label_path.name}, label_key={label_key_full}, match_key={match_key}"
            )
            continue

        matched_pairs.append(
            {
                "label_name": label_path.name,
                "audio_name": audio_map[match_key].name,
                "label_key": label_key_full,
                "match_key": match_key,
                "label_path": str(label_path),
                "audio_path": str(audio_map[match_key]),
            }
        )

    return matched_pairs, audio_missing_count, duplicate_audio_key_count


def main():
    hf_logging.set_verbosity_error()

    DOMAIN = "vacation"
    PROJECT_ROOT = Path("/home/user/SWPJ3/auto-subtitle-service")

    LABEL_DIR = PROJECT_ROOT / "ai/data/raw/vacation/label/val"
    AUDIO_DIR = PROJECT_ROOT / "ai/data/raw/vacation/validation"

    RESULT_DIR = PROJECT_ROOT / "ai/data/results/vacation_base"
    VAL_JSON_DIR = RESULT_DIR / "val_json"
    VAL_METRIC_DIR = RESULT_DIR / "val"

    MODEL_ID = "openai/whisper-large-v3"
    LANGUAGE = "korean"
    TASK = "transcribe"

    if not LABEL_DIR.exists():
        raise FileNotFoundError(f"label 폴더가 없습니다: {LABEL_DIR}")
    if not AUDIO_DIR.exists():
        raise FileNotFoundError(f"audio 폴더가 없습니다: {AUDIO_DIR}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    load_dtype = torch.bfloat16 if use_bf16 else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )

    print(f"[INFO] device = {device}")
    print(f"[INFO] use_bf16 = {use_bf16}")
    print(f"[INFO] dtype = {load_dtype}")

    matched_pairs, audio_missing_count, duplicate_audio_key_count = build_pairs(
        LABEL_DIR, AUDIO_DIR
    )

    if not matched_pairs:
        raise RuntimeError("매칭된 label-audio 쌍이 없습니다. 파일명 규칙을 다시 확인하세요.")

    print(f"[INFO] matched pairs = {len(matched_pairs)}")
    print(f"[INFO] audio missing count = {audio_missing_count}")
    print(f"[INFO] duplicate audio key count = {duplicate_audio_key_count}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
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

    max_label_length = model.config.max_target_positions
    model.eval()

    results: List[Dict[str, Any]] = []
    references_raw: List[str] = []
    predictions_raw: List[str] = []

    loss_sum = 0.0
    loss_count = 0
    loss_skipped_count = 0
    label_missing_text_count = 0
    decode_error_count = 0

    for idx, pair in enumerate(tqdm(matched_pairs, desc="Vacation base validation")):
        label_path = Path(pair["label_path"])
        audio_path = Path(pair["audio_path"])

        try:
            ref_text = read_label_text(label_path)
        except Exception as e:
            print(f"[WARN] label read failed: {label_path} | {e}")
            label_missing_text_count += 1
            continue

        if not ref_text:
            print(f"[WARN] reference text not found in label: {label_path}")
            label_missing_text_count += 1
            continue

        ref_text = normalize_text(ref_text)

        try:
            audio_array, _ = librosa.load(audio_path, sr=16000, mono=True)
        except Exception as e:
            print(f"[WARN] audio load failed: {audio_path} | {e}")
            decode_error_count += 1
            continue

        if audio_array is None or len(audio_array) == 0:
            print(f"[WARN] empty audio: {audio_path}")
            decode_error_count += 1
            continue

        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(device)
        if torch.cuda.is_available():
            input_features = input_features.to(dtype=load_dtype)

        label_ids = processor.tokenizer(ref_text).input_ids

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
                    loss_outputs = model(input_features=input_features, labels=labels)
                    sample_loss = float(loss_outputs.loss.item())
                loss_sum += sample_loss
                loss_count += 1
            else:
                loss_skipped = True
        else:
            loss_skipped = True

        if loss_skipped:
            loss_skipped_count += 1

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

        pred_norm = normalize_text(full_text)

        references_raw.append(ref_text)
        predictions_raw.append(pred_norm)

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

        results.append(
            {
                "index": idx,
                "domain": DOMAIN,
                "split": "validation",
                "label_name": pair["label_name"],
                "audio_name": pair["audio_name"],
                "label_key": pair["label_key"],
                "match_key": pair["match_key"],
                "audio": str(audio_path),
                "label_path": str(label_path),
                "reference": ref_text,
                "prediction": full_text,
                "reference_normalized": ref_text,
                "prediction_normalized": pred_norm,
                "sample_loss": sample_loss,
                "loss_skipped": loss_skipped,
                "label_token_length": len(label_ids),
                "model_type": "base_only",
                "segments": segments_out,
            }
        )

    avg_val_loss = loss_sum / loss_count if loss_count > 0 else None
    total_wer = wer(references_raw, predictions_raw) if references_raw else None
    total_cer = cer(references_raw, predictions_raw) if references_raw else None

    metrics = {
        "domain": DOMAIN,
        "split": "validation",
        "num_samples": len(results),
        "matched_count": len(matched_pairs),
        "avg_val_loss": avg_val_loss,
        "wer": total_wer,
        "cer": total_cer,
        "loss_evaluated_count": loss_count,
        "loss_skipped_count": loss_skipped_count,
        "label_missing_text_count": label_missing_text_count,
        "audio_missing_count": audio_missing_count,
        "decode_error_count": decode_error_count,
        "duplicate_audio_key_count": duplicate_audio_key_count,
        "max_label_length": max_label_length,
        "model_type": "base_only",
        "model_id": MODEL_ID,
        "label_dir": str(LABEL_DIR),
        "audio_dir": str(AUDIO_DIR),
    }

    VAL_JSON_DIR.mkdir(parents=True, exist_ok=True)
    VAL_METRIC_DIR.mkdir(parents=True, exist_ok=True)

    pred_out_path = VAL_JSON_DIR / "vacation_base_validation_predictions.json"
    metric_out_path = VAL_METRIC_DIR / "vacation_base_validation_metrics.json"

    with pred_out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with metric_out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[INFO] predictions saved to: {pred_out_path}")
    print(f"[INFO] metrics saved to: {metric_out_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()