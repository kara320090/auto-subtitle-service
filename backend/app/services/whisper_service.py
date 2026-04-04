from pathlib import Path

import whisper

from backend.app.config import WHISPER_LANGUAGE, WHISPER_MODEL_NAME


_MODEL = None


def get_whisper_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = whisper.load_model(WHISPER_MODEL_NAME)
    return _MODEL


def transcribe_audio(audio_path: str) -> dict:
    input_path = Path(audio_path)

    if not input_path.exists():
        raise FileNotFoundError(f"입력 오디오 파일이 존재하지 않습니다: {audio_path}")

    model = get_whisper_model()

    result = model.transcribe(
        str(input_path),
        language=WHISPER_LANGUAGE,
        task="transcribe",
        verbose=False,
    )

    segments = []
    for seg in result.get("segments", []):
        text = str(seg.get("text", "")).strip()
        if not text:
            continue

        segments.append(
            {
                "id": seg.get("id"),
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": text,
            }
        )

    return {
        "audio_path": str(input_path.resolve()),
        "language": result.get("language", WHISPER_LANGUAGE),
        "full_text": result.get("text", "").strip(),
        "segments": segments,
        "segment_count": len(segments),
        "model_name": WHISPER_MODEL_NAME,
    }