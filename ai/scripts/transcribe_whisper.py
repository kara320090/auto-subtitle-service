import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


# =========================
# 경로 설정
# =========================
model_path = Path(r"C:\auto-subtitle-service\ai\models\whisper-large-v3")
wav_dir = Path(r"C:\auto-subtitle-service\ai\data\raw\Sample\wav")
ref_json_dir = Path(r"C:\auto-subtitle-service\ai\data\raw\Sample\json")
output_dir = Path(r"C:\auto-subtitle-service\ai\data\processed\sample")
output_dir.mkdir(parents=True, exist_ok=True)


# =========================
# 입력 파일 수집
# =========================
wav_files = sorted(wav_dir.glob("*.wav"))
if not wav_files:
    raise FileNotFoundError(f"WAV 파일이 없습니다: {wav_dir}")


# =========================
# 디바이스 설정
# =========================
use_cuda = torch.cuda.is_available()
device = 0 if use_cuda else -1
dtype = torch.float16 if use_cuda else torch.float32

print(f"Using device: {'cuda' if use_cuda else 'cpu'}")
if use_cuda:
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# =========================
# 모델 / 프로세서 로드
# =========================
processor = AutoProcessor.from_pretrained(str(model_path))
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    str(model_path),
    dtype=dtype,
    low_cpu_mem_usage=True,
)

if use_cuda:
    model = model.to("cuda")

asr_pipe = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    dtype=dtype,
    device=device,
)


# =========================
# 헬퍼 함수
# =========================
def extract_reference_info(ref_json_path: Path):
    if not ref_json_path.exists():
        return {
            "exists": False,
            "metadata": {},
            "summary": "",
            "speakers_info": [],
            "ground_truth_text": "",
        }

    with open(ref_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    terms = data.get("video", {}).get("term", [])
    terms = sorted(terms, key=lambda x: x.get("start", 0))

    texts = []
    for item in terms:
        text = item.get("transcription", "").strip()
        if text:
            texts.append(text)

    ground_truth_text = " ".join(texts).strip()

    return {
        "exists": True,
        "metadata": data.get("metadata", {}),
        "summary": data.get("summary", ""),
        "speakers_info": data.get("video", {}).get("speakers_info", []),
        "ground_truth_text": ground_truth_text,
    }


def convert_prediction_chunks(chunks):
    converted = []
    for chunk in chunks:
        converted.append(
            {
                "timestamp": list(chunk.get("timestamp", (None, None))),
                "text": chunk.get("text", "").strip(),
            }
        )
    return converted


# =========================
# 전사 수행
# =========================
for wav_path in tqdm(wav_files, desc="Transcribing WAV files"):
    try:
        stem = wav_path.stem
        ref_json_path = ref_json_dir / f"{stem}.json"

        audio, sr = sf.read(str(wav_path))

        # stereo -> mono
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        audio = audio.astype(np.float32)

        reference_info = extract_reference_info(ref_json_path)

        prediction = asr_pipe(
            {
                "array": audio,
                "sampling_rate": sr,
            },
            return_timestamps=True,
            generate_kwargs={
                "language": "ko",
                "task": "transcribe",
            },
        )

        whisper_text = prediction.get("text", "").strip()
        whisper_chunks = convert_prediction_chunks(prediction.get("chunks", []))

        result = {
            "filename": stem,
            "paths": {
                "wav": str(wav_path),
                "reference_json": str(ref_json_path) if ref_json_path.exists() else None,
            },
            "metadata": reference_info.get("metadata", {}),
            "summary": reference_info.get("summary", ""),
            "speakers_info": reference_info.get("speakers_info", []),
            "ground_truth_text": reference_info.get("ground_truth_text", ""),
            "whisper_text": whisper_text,
            "whisper_chunks": whisper_chunks,
            "model_info": {
                "model_name": "openai/whisper-large-v3",
                "language": "ko",
                "task": "transcribe",
                "sampling_rate": int(sr),
            },
        }

        output_json_path = output_dir / f"{stem}.json"

        with open(output_json_path, "w", encoding="utf-8-sig") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[완료] {wav_path.name}")
        print(f"       JSON -> {output_json_path}")

    except Exception as e:
        print(f"[실패] {wav_path.name}: {e}")

print("전체 전사 완료")