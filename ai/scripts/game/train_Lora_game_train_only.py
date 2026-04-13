# ============================================
# 파일명: train_lora_game_train_only_lowmem.py
#
# 역할:
# - whisper-large-v3 base 모델 로드
# - game 도메인용 LoRA adapter 부착
# - train.jsonl 데이터만 사용하여 학습
# - 각 샘플의 audio 파일에서 start~end 구간만 잘라 사용
# - 전처리 결과를 메모리에 쌓지 않고 on-the-fly로 처리
# - 학습 완료 후 adapter 저장
# ============================================

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json
import time

import librosa
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from peft import LoraConfig, TaskType
from tqdm.auto import tqdm
from transformers import (
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
)

# =========================
# 실행 시간 측정 시작
# =========================
script_start_time = time.time()

# =========================
# 경로
# =========================
PROJECT_ROOT = Path("/home/user/SWPJ3/auto-subtitle-service")

MANIFEST_DIR = PROJECT_ROOT / "ai/data/processed/game"
RESULT_DIR = PROJECT_ROOT / "ai/data/results/game_lora"

TRAIN_JSONL = str(MANIFEST_DIR / "train.jsonl")
FILTERED_TRAIN_JSONL = str(MANIFEST_DIR / "train.filtered.jsonl")

ADAPTER_DIR = RESULT_DIR / "adapter"
LOG_DIR = RESULT_DIR / "logs"

MODEL_ID = "openai/whisper-large-v3"
ADAPTER_NAME = "default"
LANGUAGE = "korean"
TASK = "transcribe"

# =========================
# 하이퍼파라미터
# =========================
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
WARMUP_STEPS = 100
NUM_TRAIN_EPOCHS = 3
SAVE_STEPS = 1000
LOGGING_STEPS = 100

# 메모리 절약용
DATALOADER_NUM_WORKERS = 4
PIN_MEMORY = torch.cuda.is_available()

# =========================
# 유틸
# =========================
def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def print_stage_header(title: str):
    print("\n" + "=" * 70)
    print(f"[STAGE] {title}")
    print("=" * 70)


def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0

    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    pct = 100 * trainable_params / all_params
    print(
        f"[INFO] trainable params: {trainable_params:,d} || "
        f"all params: {all_params:,d} || "
        f"trainable%: {pct:.4f}"
    )

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
print(f"[INFO] load_dtype = {load_dtype}")

# =========================
# processor / model
# =========================
print_stage_header("Load processor and model")

print("[INFO] loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("[INFO] loading base model...")
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=load_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)

model.generation_config.language = LANGUAGE
model.generation_config.task = TASK
model.generation_config.forced_decoder_ids = None
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False

max_label_length = model.config.max_target_positions
print(f"[INFO] max_label_length = {max_label_length}")

# =========================
# LoRA 설정
# =========================
print_stage_header("Attach LoRA adapter")

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

print_trainable_parameters(model)

# =========================
# 데이터셋 로드
# =========================
print_stage_header("Load dataset")

dataset = load_dataset(
    "json",
    data_files={"train": TRAIN_JSONL},
)

raw_train_dataset = dataset["train"]
print(f"[INFO] raw train rows = {len(raw_train_dataset)}")

# =========================
# label 길이 필터링
# =========================
print_stage_header("Filter long-label samples")

filter_start_time = time.time()
kept_count = 0
removed_count = 0

with open(FILTERED_TRAIN_JSONL, "w", encoding="utf-8") as fout:
    for row in tqdm(raw_train_dataset, desc="Filtering train manifest", dynamic_ncols=True):
        input_ids = processor.tokenizer(row["text"]).input_ids
        label_length = len(input_ids)

        if label_length <= max_label_length:
            row = dict(row)
            row["label_length"] = label_length
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept_count += 1
        else:
            removed_count += 1

filter_end_time = time.time()

print(f"[INFO] kept rows    = {kept_count}")
print(f"[INFO] removed rows = {removed_count}")
print(f"[INFO] filtered manifest = {FILTERED_TRAIN_JSONL}")
print(f"[INFO] filter stage time = {format_seconds(filter_end_time - filter_start_time)}")

# =========================
# on-the-fly dataset
# =========================
print_stage_header("Build on-the-fly dataset")

class GameSpeechDataset(Dataset):
    def __init__(self, jsonl_path: str, processor: Any):
        self.jsonl_path = jsonl_path
        self.processor = processor
        self.rows = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]

        audio_path = str(Path(row["audio"]))
        text = row["text"]

        start = float(row.get("start", 0.0))
        end = float(row.get("end", 0.0))

        if end <= start:
            raise ValueError(
                f"Invalid segment range: start={start}, end={end}, file={audio_path}"
            )

        audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)

        start_sample = int(start * sr)
        end_sample = int(end * sr)
        audio_segment = audio_array[start_sample:end_sample]

        if audio_segment.size == 0:
            raise ValueError(
                f"Empty audio segment: start={start}, end={end}, file={audio_path}"
            )

        input_features = self.processor.feature_extractor(
            audio_segment,
            sampling_rate=16000,
        )["input_features"][0]

        labels = self.processor.tokenizer(text).input_ids

        return {
            "input_features": input_features,
            "labels": labels,
        }

train_dataset = GameSpeechDataset(
    jsonl_path=FILTERED_TRAIN_JSONL,
    processor=processor,
)

print(f"[INFO] final train rows = {len(train_dataset)}")

# =========================
# Data collator
# =========================
print_stage_header("Build data collator")

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )

        bos_token_id = self.processor.tokenizer.bos_token_id
        if bos_token_id is not None and labels.shape[1] > 0:
            if (labels[:, 0] == bos_token_id).all():
                labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# =========================
# 학습 설정
# =========================
print_stage_header("Configure trainer")

training_args = Seq2SeqTrainingArguments(
    output_dir=str(LOG_DIR),
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    eval_strategy="no",
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    logging_strategy="steps",
    logging_steps=LOGGING_STEPS,
    fp16=(torch.cuda.is_available() and not use_bf16),
    bf16=use_bf16,
    max_grad_norm=1.0,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    report_to=[],
    dataloader_num_workers=DATALOADER_NUM_WORKERS,
    dataloader_pin_memory=PIN_MEMORY,
    label_names=["labels"],
    predict_with_generate=False,
    save_total_limit=2,
    load_best_model_at_end=False,
    disable_tqdm=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    processing_class=processor,
)

# =========================
# 학습 실행
# =========================
print_stage_header("Train LoRA adapter")

train_start_time = time.time()
train_result = trainer.train()
train_end_time = time.time()

print(f"[INFO] training time = {format_seconds(train_end_time - train_start_time)}")

if hasattr(train_result, "metrics"):
    metrics = train_result.metrics
    if "train_runtime" in metrics:
        print(f"[INFO] trainer train_runtime = {metrics['train_runtime']:.2f} sec")
    if "train_samples_per_second" in metrics:
        print(f"[INFO] train samples/sec = {metrics['train_samples_per_second']:.4f}")
    if "train_steps_per_second" in metrics:
        print(f"[INFO] train steps/sec = {metrics['train_steps_per_second']:.4f}")

# =========================
# adapter 저장
# =========================
print_stage_header("Save adapter")

ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

save_start_time = time.time()

model.save_pretrained(str(ADAPTER_DIR))
processor.save_pretrained(str(ADAPTER_DIR))

save_end_time = time.time()

print("[INFO] expected files:")
print(ADAPTER_DIR / "adapter_config.json")
print(ADAPTER_DIR / "adapter_model.safetensors")
print(f"[INFO] save time = {format_seconds(save_end_time - save_start_time)}")

# =========================
# 전체 시간
# =========================
script_end_time = time.time()

print_stage_header("Total runtime summary")
print(f"[INFO] total script time = {format_seconds(script_end_time - script_start_time)}")
print(f"[INFO] filter stage      = {format_seconds(filter_end_time - filter_start_time)}")
print(f"[INFO] training stage    = {format_seconds(train_end_time - train_start_time)}")
print(f"[INFO] save stage        = {format_seconds(save_end_time - save_start_time)}")
print("[INFO] done.")