# ============================================
# 파일명: train_lora_vacation.py
#
# 역할:
# - Hugging Face에서 whisper-large-v3 base 모델을 불러온다.
# - vacation 도메인용 LoRA adapter를 모델에 붙인다.
# - train.jsonl / validation.jsonl 데이터를 읽어
#   여행 도메인 음성 데이터로 LoRA 학습을 수행한다.
# - 학습이 끝나면 adapter 파일을 저장한다.
#
# 입력:
# - /home/user/SWPJ3/auto-subtitle-service/ai/data/processed/vacation/train.jsonl
# - /home/user/SWPJ3/auto-subtitle-service/ai/data/processed/vacation/validation.jsonl
#
# 출력:
# - /home/user/SWPJ3/auto-subtitle-service/ai/data/results/vacation_lora/adapter/adapter_config.json
# - /home/user/SWPJ3/auto-subtitle-service/ai/data/results/vacation_lora/adapter/adapter_model.safetensors
# - /home/user/SWPJ3/auto-subtitle-service/ai/data/results/vacation_lora/logs/*
#
# 목적:
# - 여행(vacation) 도메인에 맞는 Whisper LoRA adapter를 학습한다.
# - 구어체, 자연스러운 화자 말투, 여행 브이로그/설명체 표현에 더 잘 맞도록
#   도메인 적응(domain adaptation)을 수행한다.
#
# 참고:
# - 이 코드는 "base 모델 전체 재학습"이 아니라
#   "LoRA adapter만 학습"하는 코드다.
# - 따라서 full fine-tuning보다 VRAM 사용량과 학습 비용이 낮다.
# - text는 manifest 안에 있는 전사문을 그대로 사용하며,
#   구어체/화자 말투를 유지하는 방향으로 학습한다.
# - RTX 4090 24GB 기준의 1차 설정값으로 작성했다.
# - Whisper는 label 최대 길이 제한이 있으므로,
#   너무 긴 샘플은 학습 전에 필터링해서 제외한다.
# ============================================

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import librosa
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
)

# =========================
# 경로
# =========================
PROJECT_ROOT = Path("/home/user/SWPJ3/auto-subtitle-service")

MANIFEST_DIR = PROJECT_ROOT / "ai/data/processed/vacation"
RESULT_DIR = PROJECT_ROOT / "ai/data/results/vacation_lora"

TRAIN_JSONL = str(MANIFEST_DIR / "train.jsonl")
VAL_JSONL = str(MANIFEST_DIR / "validation.jsonl")

ADAPTER_DIR = RESULT_DIR / "adapter"
LOG_DIR = RESULT_DIR / "logs"

MODEL_ID = "openai/whisper-large-v3"
ADAPTER_NAME = "default"
LANGUAGE = "korean"
TASK = "transcribe"

# =========================
# 디바이스 / dtype
# =========================
device = "cuda:0" if torch.cuda.is_available() else "cpu"
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

# 4090에서는 bf16 우선, 안 되면 fp16, CPU면 fp32
load_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)

print(f"[INFO] device = {device}")
print(f"[INFO] use_bf16 = {use_bf16}")
print(f"[INFO] load_dtype = {load_dtype}")

# =========================
# processor / model
# =========================
print("[INFO] loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("[INFO] loading base model...")
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=load_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)

# Whisper 설정
model.generation_config.language = LANGUAGE
model.generation_config.task = TASK
model.generation_config.forced_decoder_ids = None
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False

# Whisper가 허용하는 최대 label 길이
max_label_length = model.config.max_target_positions
print(f"[INFO] max_label_length = {max_label_length}")

# =========================
# LoRA 설정
# =========================
# 4090 24GB 기준 1차안
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)

model.add_adapter(lora_config, adapter_name=ADAPTER_NAME)
model.set_adapter(ADAPTER_NAME)

if torch.cuda.is_available():
    model = model.to(device)

def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0

    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    pct = 100 * trainable_params / all_params
    print(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_params:,d} || "
        f"trainable%: {pct:.4f}"
    )

print_trainable_parameters(model)

# =========================
# 데이터셋 로드
# =========================
print("[INFO] loading dataset...")
dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_JSONL,
        "validation": VAL_JSONL,
    },
)

def get_label_length(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    text를 토큰화해서 label 길이를 계산한다.
    Whisper 허용 길이(예: 448)를 초과하는 샘플을
    미리 걸러내기 위한 단계다.
    """
    input_ids = processor.tokenizer(batch["text"]).input_ids
    batch["label_length"] = len(input_ids)
    return batch

print("[INFO] calculating label lengths...")
dataset = dataset.map(get_label_length)

train_before = len(dataset["train"])
val_before = len(dataset["validation"])

print("[INFO] filtering long-label samples...")
dataset = dataset.filter(lambda x: x["label_length"] <= max_label_length)

train_after = len(dataset["train"])
val_after = len(dataset["validation"])

print(f"[INFO] train kept: {train_after}/{train_before}")
print(f"[INFO] validation kept: {val_after}/{val_before}")
print(f"[INFO] train removed: {train_before - train_after}")
print(f"[INFO] validation removed: {val_before - val_after}")

def prepare_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    각 샘플에서
    - audio 경로를 읽어 16kHz mono waveform으로 로드하고
    - Whisper 입력 feature를 만든 뒤
    - text를 tokenizer로 변환하여 labels를 구성한다.
    """
    audio_path = batch["audio"]

    # Windows 경로가 섞여 들어오는 경우를 대비한 보정
    if isinstance(audio_path, str) and audio_path.startswith("C:\\auto-subtitle-service\\"):
        audio_path = audio_path.replace(
            "C:\\auto-subtitle-service\\",
            str(PROJECT_ROOT) + "/",
        )
        audio_path = audio_path.replace("\\", "/")

    audio_path = str(Path(audio_path))

    # 오디오 로드
    audio_array, _ = librosa.load(audio_path, sr=16000, mono=True)

    # Whisper feature 추출
    batch["input_features"] = processor.feature_extractor(
        audio_array,
        sampling_rate=16000,
    )["input_features"][0]

    # text -> token ids
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

print("[INFO] preprocessing dataset...")
dataset = dataset.map(
    prepare_batch,
    remove_columns=dataset["train"].column_names,
)

# =========================
# Data collator
# =========================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 입력 feature padding
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        # label padding
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )

        # BOS token 제거
        bos_token_id = self.processor.tokenizer.bos_token_id
        if bos_token_id is not None and labels.shape[1] > 0:
            if (labels[:, 0] == bos_token_id).all():
                labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# =========================
# 학습 설정 (RTX 4090 24GB 기준 1차안)
# =========================
training_args = Seq2SeqTrainingArguments(
    output_dir=str(LOG_DIR),

    # 4090 기준 1차 설정
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,

    # 학습률
    learning_rate=1e-4,
    warmup_steps=100,

    # 실제 학습용
    num_train_epochs=3,

    # 평가 / 저장 / 로그
    eval_strategy="steps",
    eval_steps=250,
    save_strategy="steps",
    save_steps=250,
    logging_strategy="steps",
    logging_steps=20,

    # precision
    fp16=(torch.cuda.is_available() and not use_bf16),
    bf16=use_bf16,

    # 안정성 / 메모리
    max_grad_norm=1.0,
    gradient_checkpointing=True,
    remove_unused_columns=False,

    # 기타
    report_to=[],
    dataloader_num_workers=2,
    label_names=["labels"],
    predict_with_generate=False,
    save_total_limit=2,
    load_best_model_at_end=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    processing_class=processor,
)

# =========================
# 학습 실행
# =========================
print("[INFO] training start...")
trainer.train()

# =========================
# adapter 저장
# =========================
ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
print(f"[INFO] saving adapter to: {ADAPTER_DIR}")

model.save_pretrained(str(ADAPTER_DIR))
processor.save_pretrained(str(ADAPTER_DIR))

print("[INFO] done.")
print("[INFO] expected files:")
print(ADAPTER_DIR / "adapter_config.json")
print(ADAPTER_DIR / "adapter_model.safetensors")