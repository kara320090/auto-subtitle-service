from pathlib import Path
from threading import Lock
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoProcessor, WhisperForConditionalGeneration, pipeline

from backend.app.config import (
    BASE_MODEL_ID,
    DEFAULT_DOMAIN,
    FALLBACK_TO_BASE,
    WHISPER_LANGUAGE,
    WHISPER_TASK,
)
from backend.app.services.domain_router import choose_domain
from backend.app.services.lora_registry import get_adapter_path


_PROCESSOR = None
_GENERAL_PIPELINE = None
_DOMAIN_PIPELINE_CACHE: dict[str, Any] = {}
_PIPELINE_LOCK = Lock()


def _get_device_index() -> int:
    return 0 if torch.cuda.is_available() else -1


def _get_device_str() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _get_torch_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def get_processor():
    global _PROCESSOR
    if _PROCESSOR is None:
        _PROCESSOR = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    return _PROCESSOR


def _configure_model(model: WhisperForConditionalGeneration) -> WhisperForConditionalGeneration:
    # ko를 쓸지 korean을 쓸지는 tokenizer/processor 조합에 따라 다를 수 있습니다.
    # 현재 config의 WHISPER_LANGUAGE 값을 그대로 사용합니다.
    model.generation_config.language = WHISPER_LANGUAGE
    model.generation_config.task = WHISPER_TASK
    model.generation_config.forced_decoder_ids = None

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = True

    return model


def _load_base_model_instance() -> WhisperForConditionalGeneration:
    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=_get_torch_dtype(),
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    model = _configure_model(model)
    model = model.to(_get_device_str())
    model.eval()
    return model


def _build_pipeline_from_model(model: WhisperForConditionalGeneration):
    processor = get_processor()

    return pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=_get_device_index(),
        torch_dtype=_get_torch_dtype(),
        chunk_length_s=30,
        batch_size=1,
    )


def _build_general_pipeline():
    model = _load_base_model_instance()
    return _build_pipeline_from_model(model)


def _build_domain_pipeline(domain: str):
    adapter_path = get_adapter_path(domain)
    if adapter_path is None:
        raise ValueError(f"adapter 경로를 찾을 수 없습니다: domain={domain}")

    if not adapter_path.exists():
        raise FileNotFoundError(f"adapter 디렉터리가 존재하지 않습니다: {adapter_path}")

    base_model = _load_base_model_instance()

    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        is_trainable=False,
    )
    model = model.to(_get_device_str())
    model.eval()

    return _build_pipeline_from_model(model)


def get_asr_pipeline(domain: str):
    global _GENERAL_PIPELINE

    with _PIPELINE_LOCK:
        if domain == DEFAULT_DOMAIN:
            if _GENERAL_PIPELINE is None:
                _GENERAL_PIPELINE = _build_general_pipeline()
            return _GENERAL_PIPELINE

        if domain not in _DOMAIN_PIPELINE_CACHE:
            _DOMAIN_PIPELINE_CACHE[domain] = _build_domain_pipeline(domain)

        return _DOMAIN_PIPELINE_CACHE[domain]


def clear_domain_pipeline_cache(domain: str | None = None) -> None:
    global _GENERAL_PIPELINE

    with _PIPELINE_LOCK:
        if domain is None:
            _DOMAIN_PIPELINE_CACHE.clear()
            _GENERAL_PIPELINE = None
            return

        normalized = str(domain).strip().lower()
        if normalized == DEFAULT_DOMAIN:
            _GENERAL_PIPELINE = None
        else:
            _DOMAIN_PIPELINE_CACHE.pop(normalized, None)


def _safe_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _convert_chunks_to_segments(chunks: list[dict]) -> list[dict]:
    segments: list[dict] = []

    for idx, chunk in enumerate(chunks):
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue

        timestamp = chunk.get("timestamp")
        if not timestamp or len(timestamp) != 2:
            # timestamp가 없더라도 텍스트는 살려두되 시간은 0으로 둡니다.
            start = 0.0
            end = 0.0
        else:
            start = _safe_float(timestamp[0], 0.0)
            end = _safe_float(timestamp[1], start)
            if end < start:
                end = start

        segments.append(
            {
                "id": idx,
                "start": start,
                "end": end,
                "text": text,
            }
        )

    return segments


def _run_asr(asr_pipeline, audio_path: str) -> dict:
    result = asr_pipeline(
        audio_path,
        return_timestamps=True,
        generate_kwargs={
            "language": WHISPER_LANGUAGE,
            "task": WHISPER_TASK,
        },
    )

    full_text = str(result.get("text", "")).strip()
    chunks = result.get("chunks", []) or []
    segments = _convert_chunks_to_segments(chunks)

    return {
        "full_text": full_text,
        "segments": segments,
        "segment_count": len(segments),
    }


def transcribe_audio(audio_path: str, domain: str | None = None) -> dict:
    input_path = Path(audio_path)

    if not input_path.exists():
        raise FileNotFoundError(f"입력 오디오 파일이 존재하지 않습니다: {audio_path}")

    requested_domain = domain
    applied_domain = choose_domain(domain)
    used_adapter = None
    fallback_used = False
    fallback_reason = None

    try:
        asr_pipeline = get_asr_pipeline(applied_domain)
        if applied_domain != DEFAULT_DOMAIN:
            used_adapter = applied_domain
    except Exception as e:
        if not FALLBACK_TO_BASE:
            raise

        fallback_used = True
        fallback_reason = str(e)
        applied_domain = DEFAULT_DOMAIN
        used_adapter = None
        asr_pipeline = get_asr_pipeline(DEFAULT_DOMAIN)

    result = _run_asr(asr_pipeline, str(input_path))

    return {
        "audio_path": str(input_path.resolve()),
        "language": WHISPER_LANGUAGE,
        "full_text": result["full_text"],
        "segments": result["segments"],
        "segment_count": result["segment_count"],
        "model_name": BASE_MODEL_ID,
        "requested_domain": requested_domain,
        "applied_domain": applied_domain,
        "used_adapter": used_adapter,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
    }