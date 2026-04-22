from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from backend.app.config import LLM_SERVICE_TIMEOUT_SECONDS, LLM_SERVICE_URL


_CANDIDATE_PATHS = (
    "/api/subtitle/refine",
)


def _build_prompt(domain: str | None) -> str:
    domain_text = domain or "general"
    return (
        "You are a Korean subtitle correction assistant. "
        "Fix transcription errors, improve punctuation, and keep the meaning faithful. "
        f"Target domain: {domain_text}. "
        "Return only valid JSON with this structure: "
        '{"full_text": "...", "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "..."}]}. '
        "Keep the timestamps unchanged. If you cannot improve a segment, keep the original text."
    )


def _clone_segments(segments: list[dict]) -> list[dict]:
    return [deepcopy(segment) for segment in segments]


def _normalize_segment(segment: dict[str, Any], fallback_segment: dict[str, Any] | None = None, index: int = 0) -> dict:
    source = fallback_segment or {}

    start = segment.get("start", source.get("start", 0.0))
    end = segment.get("end", source.get("end", start))
    text = segment.get("text", source.get("text", ""))

    try:
        start_value = float(start)
    except Exception:
        start_value = float(source.get("start", 0.0))

    try:
        end_value = float(end)
    except Exception:
        end_value = float(source.get("end", start_value))

    if end_value < start_value:
        end_value = start_value

    return {
        "id": int(segment.get("id", source.get("id", index))),
        "start": start_value,
        "end": end_value,
        "text": str(text).strip(),
    }


def _extract_json_payload(data: Any) -> dict[str, Any]:
    if isinstance(data, dict):
        # /api/subtitle/refine 응답 형식: { status, message, refinement: {...} }
        refinement = data.get("refinement")
        if isinstance(refinement, dict):
            return refinement

        if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
            first_choice = data["choices"][0] or {}
            message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
            content = message.get("content") if isinstance(message, dict) else None
            if content is None and isinstance(first_choice, dict):
                content = first_choice.get("text")

            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    return {"full_text": content}

        return data

    if isinstance(data, list):
        return {"segments": data}

    if isinstance(data, str):
        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"segments": parsed}
        except Exception:
            return {"full_text": data}

    return {}


def _request_json(url: str, payload: dict[str, Any]) -> Any:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=LLM_SERVICE_TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8")
            try:
                return json.loads(body)
            except Exception:
                return body
    except HTTPError as exc:
        error_body = ""
        try:
            raw = exc.read()
            if raw:
                error_body = raw.decode("utf-8", errors="replace").strip()
        except Exception:
            error_body = ""

        message = f"HTTP {exc.code} calling {url}"
        if error_body:
            message = f"{message} | body={error_body[:500]}"
        raise RuntimeError(message) from exc


def _build_request_payload(transcription_result: dict, domain: str | None) -> dict[str, Any]:
    raw_segments = transcription_result.get("segments", []) or []

    segments = []
    for index, segment in enumerate(raw_segments):
        if not isinstance(segment, dict):
            continue
        segments.append(
            {
                "id": segment.get("id", index),
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", segment.get("start", 0.0))),
                "text": str(segment.get("text", "")).strip(),
                "token_confidence": segment.get("token_confidence", []) or [],
            }
        )

    # 현재 연동되는 LLM 서버는 /api/subtitle/refine 스키마를 사용합니다.
    return {
        "segments": segments,
        "top_k": 5,
        "max_masks_per_segment": 1,
        "use_llm_postprocess": True,
        "llm_context_window": 8,
        "llm_model": None,
        "llm_temperature": 0.2,
        "llm_max_tokens": 256,
    }


def _candidate_urls() -> list[str]:
    raw_base_url = (LLM_SERVICE_URL or "").strip()
    if not raw_base_url:
        return []

    if not raw_base_url.startswith(("http://", "https://")):
        raw_base_url = f"https://{raw_base_url}"

    base_url = raw_base_url.rstrip("/")

    candidates: list[str] = []

    # base가 이미 /api로 끝나면 /subtitle/refine, 아니면 /api/subtitle/refine를 우선 사용
    if base_url.endswith("/api"):
        candidates.append(f"{base_url}/subtitle/refine")
    else:
        candidates.append(f"{base_url}/api/subtitle/refine")

    # 혹시 모를 배포 경로 차이를 위해 기존 경로도 보조로 시도
    for path in _CANDIDATE_PATHS:
        if path == "/":
            candidates.append(base_url)
        else:
            candidates.append(f"{base_url}{path}")

    unique: list[str] = []
    for url in candidates:
        if url not in unique:
            unique.append(url)

    return unique


def _merge_refined_segments(original_segments: list[dict], refined_segments: list[dict]) -> list[dict]:
    if not refined_segments:
        return _clone_segments(original_segments)

    merged: list[dict] = []
    for index, original_segment in enumerate(original_segments):
        if index < len(refined_segments) and isinstance(refined_segments[index], dict):
            merged.append(_normalize_segment(refined_segments[index], original_segment, index))
        else:
            merged.append(_normalize_segment(original_segment, original_segment, index))

    if len(refined_segments) > len(original_segments):
        for index in range(len(original_segments), len(refined_segments)):
            if isinstance(refined_segments[index], dict):
                merged.append(_normalize_segment(refined_segments[index], index=index))

    return merged


def _pick_refined_segments(parsed: dict[str, Any]) -> list[dict]:
    for key in ("refined_segments", "segments", "corrected_segments", "items", "chunks"):
        value = parsed.get(key)
        if isinstance(value, list) and value:
            return [segment for segment in value if isinstance(segment, dict)]

    # before_after_pairs만 온 경우를 대비해 최소 segments로 변환합니다.
    before_after_pairs = parsed.get("before_after_pairs")
    if isinstance(before_after_pairs, list) and before_after_pairs:
        converted: list[dict] = []
        for index, pair in enumerate(before_after_pairs):
            if not isinstance(pair, dict):
                continue
            converted.append(
                {
                    "id": pair.get("segment_id", index),
                    "start": pair.get("start", 0.0),
                    "end": pair.get("end", 0.0),
                    "text": pair.get("after_text", ""),
                }
            )
        if converted:
            return converted

    return []


def _pick_refined_text(parsed: dict[str, Any], fallback_text: str) -> str:
    for key in ("full_text", "corrected_text", "text", "content", "result", "output"):
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    refined_segments = _pick_refined_segments(parsed)
    if refined_segments:
        joined = " ".join(str(segment.get("text", "")).strip() for segment in refined_segments).strip()
        if joined:
            return joined

    return fallback_text


async def refine_transcription_with_llm(transcription_result: dict, domain: str | None = None) -> dict:
    original_segments = transcription_result.get("segments", []) or []
    original_full_text = str(transcription_result.get("full_text", "")).strip()

    if not original_segments and not original_full_text:
        updated = dict(transcription_result)
        updated.update(
            {
                "llm_used": False,
                "llm_service_url": None,
                "llm_fallback_used": True,
                "llm_fallback_reason": "전사 결과가 비어 있어 LLM 후처리를 건너뜁니다.",
            }
        )
        return updated

    request_payload = _build_request_payload(transcription_result, domain)
    last_error: str | None = None

    for url in _candidate_urls():
        try:
            response_data = await asyncio.to_thread(_request_json, url, request_payload)
            parsed = _extract_json_payload(response_data)
            refined_segments = _pick_refined_segments(parsed)
            refined_text = _pick_refined_text(parsed, original_full_text)

            merged_segments = _merge_refined_segments(original_segments, refined_segments)
            merged_text = refined_text or original_full_text
            if not merged_text and merged_segments:
                merged_text = " ".join(str(segment.get("text", "")).strip() for segment in merged_segments).strip()

            updated = dict(transcription_result)
            updated.update(
                {
                    "full_text": merged_text,
                    "segments": merged_segments,
                    "segment_count": len(merged_segments),
                    "llm_used": True,
                    "llm_service_url": url,
                    "llm_fallback_used": False,
                    "llm_fallback_reason": None,
                }
            )
            return updated
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError, OSError) as exc:
            last_error = str(exc)
            continue
        except Exception as exc:
            last_error = str(exc)
            continue

    updated = dict(transcription_result)
    updated.update(
        {
            "llm_used": False,
            "llm_service_url": None,
            "llm_fallback_used": True,
            "llm_fallback_reason": last_error or "LLM 후처리에 실패했습니다.",
        }
    )
    return updated